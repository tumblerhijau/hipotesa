#!/usr/bin/env python3
"""
Burst FIFO Analysis - STREAMING RAW (v8)

Cara pakai:
  python window_burst8.py <input.md> <start> <end>
  python window_burst8.py data.md 10 2000 --format raw
  python window_burst8.py data.md 10 2000 --format all

Kenapa v8 dibuat:
  - Di v7, format "raw" mengumpulkan SEMUA baris window dari SEMUA
    window_size (10..2000) ke satu list (all_raw_rows) sebelum ditulis
    di akhir. Untuk data ~5.400 baris, ini bisa jadi jutaan dict yang
    menumpuk di RAM -> proses mati (SIGKILL) sebelum sempat menulis apa pun.
  - v8 menulis baris raw setiap satu window_size SELESAI dihitung,
    langsung ke file (CSV) lalu langsung dibuang dari memori.
    Peak RAM raw jadi hanya sebesar 1 window_size saja, bukan akumulasi
    dari 10 sampai 2000.

Catatan:
  - Output md & csv ringkasan tetap sama seperti v7.
  - --format md|csv|both|raw|all
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from pathlib import Path
from collections import deque


LOCAL_TZ = timezone(timedelta(hours=8))


# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

NUMBER_OF_BINS = 10
SLIDING_RANGE_PERCENT = 0.10
OUTPUT_DIR = "output_txt"


# ============================================================
# HELPER & PARSER
# ============================================================

def parse_number(value):
    """Convert string number ke float."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def parse_epoch(value):
    """Epoch milliseconds -> datetime."""
    try:
        return datetime.fromtimestamp(
            float(value) / 1000,
            tz=LOCAL_TZ
        )
    except (ValueError, TypeError):
        return None


def parse_datetime(value):
    """Parse timestamp penuh: YYYY-MM-DD HH:MM:SS.mmm."""
    if not value:
        return None

    value = value.strip()

    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S"
    ):
        try:
            return datetime.strptime(
                value,
                fmt
            ).replace(tzinfo=LOCAL_TZ)
        except ValueError:
            pass

    return None


def parse_markdown_rows(filename):
    """Parse tabel Markdown format baru."""
    rows = []

    text = Path(filename).read_text(
        encoding="utf-8"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        # Jangan filter sel kosong.
        # Posisi kolom harus tetap benar.
        raw_parts = [
            x.strip()
            for x in line.split("|")
        ]

        # Buang elemen kosong di ujung kiri/kanan.
        if raw_parts and raw_parts[0] == "":
            raw_parts = raw_parts[1:]

        if raw_parts and raw_parts[-1] == "":
            raw_parts = raw_parts[:-1]

        if len(raw_parts) < 7:
            continue

        if (
            raw_parts[0].lower() == "tag_ts"
            or "---" in raw_parts[0]
        ):
            continue

        tag_ts_raw = raw_parts[0]
        ts_raw = raw_parts[1]
        startdealing_raw = raw_parts[2]
        gr_ts_raw = raw_parts[3]
        result_gr_raw = raw_parts[4]
        game_id = raw_parts[5]
        status = raw_parts[6]

        # Skip baris yang belum punya timestamp lengkap.
        if not all((
            tag_ts_raw,
            ts_raw,
            startdealing_raw,
            gr_ts_raw
        )):
            continue

        try:
            int(tag_ts_raw)
        except (ValueError, TypeError):
            continue

        epoch_dt = parse_epoch(tag_ts_raw)
        ts_dt = parse_datetime(ts_raw)
        startdealing_dt = parse_datetime(
            startdealing_raw
        )
        gr_ts_dt = parse_datetime(gr_ts_raw)
        result_gr = parse_number(result_gr_raw)

        if None in (
            epoch_dt,
            ts_dt,
            startdealing_dt,
            gr_ts_dt
        ):
            continue

        rows.append({
            "no": len(rows) + 1,
            "tag_ts_raw": tag_ts_raw,
            "epoch_dt": epoch_dt,
            "ts": ts_raw,
            "ts_dt": ts_dt,
            "startdealing": startdealing_raw,
            "startdealing_dt": startdealing_dt,
            "ts_gr": gr_ts_raw,
            "ts_gr_dt": gr_ts_dt,
            "result_gr": result_gr,
            "game_id": game_id,
            "status": status,
        })

    return rows


def find_first_live_complete(rows):
    """Cari titik Live pertama yang lengkap."""
    for i, row in enumerate(rows):
        if (
            row["status"].lower() == "live"
            and row["epoch_dt"] is not None
            and row["ts_gr_dt"] is not None
            and row["result_gr"] is not None
        ):
            return i

    return None


def prepare_rounds(rows, start_index):
    """Ambil semua ronde valid setelah titik start Live."""
    rounds = []

    for row in rows[start_index:]:
        if (
            row["epoch_dt"] is not None
            and row["ts_gr_dt"] is not None
            and row["result_gr"] is not None
        ):
            rounds.append(row)

    return rounds


# ============================================================
# PEMBENTUKAN WINDOW FIFO & STATISTIK
# ============================================================

THRESHOLDS = [2, 5, 10, 20, 50, 100]


def classify_result(value):
    """Klasifikasi result_gr."""
    if value is None:
        return None

    if 1.0 <= value < 2.0:
        return "low"

    if 2.0 <= value < 25.0:
        return "mid"

    if 25.0 <= value < 100.0:
        return "high"

    if value >= 100.0:
        return "extra"

    return None


def _rounds_since_ge(gr_list, threshold):
    """
    Dari ujung window mundur:
    berapa ronde sejak gr >= threshold.
    0 = ronde terakhir sendiri.
    """
    for i, g in enumerate(reversed(gr_list)):
        if g is not None and g >= threshold:
            return i

    return len(gr_list)


def build_windows_for_size(rounds, window_size):
    """
    OPTIMIZED VERSION.

    Membentuk window FIFO ukuran N.

    Output row tetap memiliki field yang sama
    dengan versi original.

    Optimasi:
      - mean menggunakan prefix sum
      - count_ge menggunakan prefix count
      - since_ge menggunakan posisi terakhir
      - min/max menggunakan monotonic deque
      - tidak membuat slice window baru setiap iterasi
      - tidak scan seluruh isi window berulang-ulang
    """

    n_rounds = len(rounds)

    if n_rounds < window_size:
        return []

    results = []

    # --------------------------------------------------------
    # ARRAY RESULT_GR
    # --------------------------------------------------------

    gr = [
        r["result_gr"]
        for r in rounds
    ]

    # --------------------------------------------------------
    # PREFIX SUM + PREFIX VALID COUNT
    # --------------------------------------------------------

    prefix_sum = [0.0] * (n_rounds + 1)
    prefix_valid = [0] * (n_rounds + 1)

    for i, value in enumerate(gr):

        prefix_sum[i + 1] = prefix_sum[i]
        prefix_valid[i + 1] = prefix_valid[i]

        if value is not None:
            prefix_sum[i + 1] += value
            prefix_valid[i + 1] += 1

    # --------------------------------------------------------
    # PREFIX COUNT UNTUK THRESHOLD
    # --------------------------------------------------------

    prefix_ge = {}

    for threshold in THRESHOLDS:

        prefix = [0] * (n_rounds + 1)
        running = 0

        for i, value in enumerate(gr):

            if (
                value is not None
                and value >= threshold
            ):
                running += 1

            prefix[i + 1] = running

        prefix_ge[threshold] = prefix

    # --------------------------------------------------------
    # POSISI TERAKHIR YANG >= THRESHOLD
    # --------------------------------------------------------

    last_pos_ge = {}

    for threshold in THRESHOLDS:

        positions = [-1] * n_rounds
        last_position = -1

        for i, value in enumerate(gr):

            if (
                value is not None
                and value >= threshold
            ):
                last_position = i

            positions[i] = last_position

        last_pos_ge[threshold] = positions

    # --------------------------------------------------------
    # MONOTONIC DEQUE UNTUK MIN / MAX
    # --------------------------------------------------------

    min_deque = deque()
    max_deque = deque()

    # --------------------------------------------------------
    # SLIDING WINDOW
    # --------------------------------------------------------

    for i in range(n_rounds):

        value = gr[i]

        if value is not None:

            # Minimum deque
            while (
                min_deque
                and gr[min_deque[-1]] >= value
            ):
                min_deque.pop()

            min_deque.append(i)

            # Maximum deque
            while (
                max_deque
                and gr[max_deque[-1]] <= value
            ):
                max_deque.pop()

            max_deque.append(i)

        window_start = i - window_size + 1

        if window_start < 0:
            continue

        # Buang index yang sudah keluar window.
        while (
            min_deque
            and min_deque[0] < window_start
        ):
            min_deque.popleft()

        while (
            max_deque
            and max_deque[0] < window_start
        ):
            max_deque.popleft()

        first_round = rounds[window_start]
        last_round = rounds[i]

        start_dt = first_round["epoch_dt"]
        end_dt = last_round["ts_gr_dt"]

        if end_dt is None:
            continue

        duration = (
            end_dt - start_dt
        ).total_seconds()

        if duration < 0:
            continue

        # ----------------------------------------------------
        # MEAN
        # ----------------------------------------------------

        valid_count = (
            prefix_valid[i + 1]
            - prefix_valid[window_start]
        )

        total_sum = (
            prefix_sum[i + 1]
            - prefix_sum[window_start]
        )

        # ----------------------------------------------------
        # LAST GR
        # ----------------------------------------------------

        last_gr = last_round["result_gr"]
        cat = classify_result(last_gr)

        # ----------------------------------------------------
        # ROW
        # ----------------------------------------------------

        row = {
            "window_size": window_size,
            "window_no": len(results) + 1,

            "window_start": first_round["no"],
            "window_end": last_round["no"],

            "first_game_id": first_round["game_id"],
            "last_game_id": last_round["game_id"],

            "first_tag_ts": first_round["tag_ts_raw"],

            "first_tag_ts_iso": (
                start_dt.strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3]
                if start_dt
                else ""
            ),

            "last_ts_gr": last_round["ts_gr"],

            "duration_sec": round(
                duration,
                3
            ),

            "duration_min": round(
                duration / 60,
                6
            ),

            "last_gr": last_gr,
            "last_cat": cat,

            "mean_gr": (
                round(
                    total_sum / valid_count,
                    4
                )
                if valid_count
                else None
            ),

            "max_gr": (
                gr[max_deque[0]]
                if max_deque
                else None
            ),

            "min_gr": (
                gr[min_deque[0]]
                if min_deque
                else None
            ),
        }

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            row[
                f"last_ge_{threshold}"
            ] = (
                1
                if (
                    last_gr is not None
                    and last_gr >= threshold
                )
                else 0
            )

            prefix = prefix_ge[threshold]

            row[
                f"count_ge_{threshold}"
            ] = (
                prefix[i + 1]
                - prefix[window_start]
            )

            last_position = (
                last_pos_ge[threshold][i]
            )

            if last_position >= window_start:
                row[
                    f"since_ge_{threshold}"
                ] = i - last_position
            else:
                row[
                    f"since_ge_{threshold}"
                ] = window_size

        results.append(row)

    return results


def make_equal_bins(windows, number_of_bins=10):
    """
    10 Pembagian Min-Max sama lebar.
    """

    values = [
        w["duration_min"]
        for w in windows
    ]

    minimum = min(values)
    maximum = max(values)

    total_range = maximum - minimum
    total = len(windows)

    def _agg(members):

        count = len(members)

        n_low = sum(
            1
            for w in members
            if w["last_cat"] == "low"
        )

        n_mid = sum(
            1
            for w in members
            if w["last_cat"] == "mid"
        )

        n_high = sum(
            1
            for w in members
            if w["last_cat"] == "high"
        )

        n_extra = sum(
            1
            for w in members
            if w["last_cat"] == "extra"
        )

        return {
            "count": count,

            "percent": (
                count / total
            ) * 100
            if total
            else 0,

            "n_low": n_low,
            "n_mid": n_mid,
            "n_high": n_high,
            "n_extra": n_extra,

            "pct_low": (
                n_low / count
            ) * 100
            if count
            else 0,

            "pct_mid": (
                n_mid / count
            ) * 100
            if count
            else 0,

            "pct_high": (
                n_high / count
            ) * 100
            if count
            else 0,

            "pct_extra": (
                n_extra / count
            ) * 100
            if count
            else 0,
        }

    if total_range == 0:

        agg = _agg(windows)

        return [{
            "bin_no": 1,
            "range_lo": minimum,
            "range_hi": maximum,
            **agg,
        }]

    width = (
        total_range
        / number_of_bins
    )

    bins = []

    for i in range(number_of_bins):

        bin_lo = (
            minimum
            + (i * width)
        )

        bin_hi = (
            maximum
            if i == number_of_bins - 1
            else minimum
            + ((i + 1) * width)
        )

        if i == number_of_bins - 1:

            members = [
                w
                for w in windows
                if (
                    bin_lo
                    <= w["duration_min"]
                    <= bin_hi
                )
            ]

        else:

            members = [
                w
                for w in windows
                if (
                    bin_lo
                    <= w["duration_min"]
                    < bin_hi
                )
            ]

        agg = _agg(members)

        bins.append({
            "bin_no": i + 1,
            "range_lo": bin_lo,
            "range_hi": bin_hi,
            **agg,
        })

    return bins

def find_sliding_range(
    values,
    percent=0.10
):
    """Mencari interval paling padat."""
    minimum = min(values)
    maximum = max(values)

    total_range = (
        maximum - minimum
    )

    if total_range == 0:

        return {
            "low": minimum,
            "high": maximum,
            "count": len(values),
            "percent": 100.0,
        }

    width = (
        total_range
        * percent
    )

    best = None

    for start in sorted(set(values)):

        end = start + width

        count = sum(
            1
            for v in values
            if start <= v <= end
        )

        candidate = {
            "low": start,
            "high": end,
            "count": count,
            "percent": (
                count / len(values)
            ) * 100,
        }

        if (
            best is None
            or candidate["count"]
            > best["count"]
        ):
            best = candidate

    return best


def fmt_duration(minutes):
    """Format menit ke bentuk M minute S second."""
    total_seconds = minutes * 60

    m = int(
        total_seconds // 60
    )

    s = (
        total_seconds % 60
    )

    return f"{m}m {s:.2f}s"


# ============================================================
# FORMATTING REPORT
# ============================================================

def generate_window_report_text(
    window_size,
    windows
):
    """Membentuk teks laporan."""

    values = [
        item["duration_min"]
        for item in windows
    ]

    minimum = min(values)
    maximum = max(values)
    average = mean(values)
    med = median(values)

    total_range = (
        maximum - minimum
    )

    # Lokasi Min & Max
    min_item = min(
        windows,
        key=lambda x:
        x["duration_min"]
    )

    max_item = max(
        windows,
        key=lambda x:
        x["duration_min"]
    )

    bins = make_equal_bins(
        windows,
        NUMBER_OF_BINS
    )

    max_bin = max(
        bins,
        key=lambda x:
        x["count"]
    )

    min_bin = min(
        bins,
        key=lambda x:
        x["count"]
    )

    sliding = find_sliding_range(
        values,
        SLIDING_RANGE_PERCENT
    )

    sliding_width = (
        total_range
        * SLIDING_RANGE_PERCENT
    )

    # Ringkasan kategori
    n = len(windows)

    total_low = sum(
        1
        for w in windows
        if w["last_cat"] == "low"
    )

    total_mid = sum(
        1
        for w in windows
        if w["last_cat"] == "mid"
    )

    total_high = sum(
        1
        for w in windows
        if w["last_cat"] == "high"
    )

    total_extra = sum(
        1
        for w in windows
        if w["last_cat"] == "extra"
    )

    len_str = f"{n:,}"
    min_str = f"{minimum:.3f} menit"
    max_str = f"{maximum:.3f} menit"
    mean_str = f"{average:.3f} menit"
    med_str = f"{med:.3f} menit"
    range_str = f"{total_range:.3f} menit"

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        f"UKURAN WINDOW (WINDOW SIZE): "
        f"{window_size} RONDE"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Jumlah window : "
        f"{len_str:<25} "
        f"Min           : {min_str}"
    )

    lines.append(
        f"Max           : "
        f"{max_str:<25} "
        f"Mean          : {mean_str}"
    )

    lines.append(
        f"Median        : "
        f"{med_str:<25} "
        f"Range total   : {range_str}"
    )

    lines.append(
        "-" * 80
    )

    lines.append(
        f"Low (1-1.99)   : "
        f"{total_low:>7,} "
        f"({100*total_low/n:.1f}%)   |  "
        f"Mid (2-24.99) : "
        f"{total_mid:>7,} "
        f"({100*total_mid/n:.1f}%)"
    )

    lines.append(
        f"High (25-99.99): "
        f"{total_high:>7,} "
        f"({100*total_high/n:.1f}%)   |  "
        f"Extra (≥100)  : "
        f"{total_extra:>7,} "
        f"({100*total_extra/n:.1f}%)"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        "  10 PEMBAGIAN MIN-MAX | "
        "Ronde terakhir window → "
        "kategori gr di bin durasi"
    )

    lines.append(
        "=" * 80
    )

    for b in bins:

        lines.append(
            f"{b['bin_no']:>2}. "
            f"{b['range_lo']:.3f} - "
            f"{b['range_hi']:.3f} menit | "
            f"{b['count']:>5} ronde | "
            f"{b['percent']:>6.2f}%"
        )

        lines.append(
            f"    Low: "
            f"{b['n_low']:>5} "
            f"({b['pct_low']:>5.1f}%)  |  "
            f"Mid: "
            f"{b['n_mid']:>5} "
            f"({b['pct_mid']:>5.1f}%)  |  "
            f"High: "
            f"{b['n_high']:>4} "
            f"({b['pct_high']:>5.1f}%)  |  "
            f"Extra: "
            f"{b['n_extra']:>3} "
            f"({b['pct_extra']:>5.1f}%)"
        )

    lines.append("")

    lines.append(
        f"TERBANYAK : "
        f"{max_bin['range_lo']:.3f} - "
        f"{max_bin['range_hi']:.3f} menit "
        f"({max_bin['count']} ronde / "
        f"{max_bin['percent']:.2f}%)"
    )

    lines.append(
        f"TERSEDIKIT: "
        f"{min_bin['range_lo']:.3f} - "
        f"{min_bin['range_hi']:.3f} menit "
        f"({min_bin['count']} ronde / "
        f"{min_bin['percent']:.2f}%)"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        "SLIDING RANGE"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Lebar sliding : "
        f"{sliding_width:.3f} menit "
        f"({int(SLIDING_RANGE_PERCENT * 100)}% "
        f"dari total range)"
    )

    r_best = (
        f"Range terbaik : "
        f"{sliding['low']:.3f} - "
        f"{sliding['high']:.3f} menit"
    )

    c_best = (
        f"Jumlah        : "
        f"{sliding['count']} window / "
        f"{sliding['percent']:.2f}%"
    )

    lines.append(
        f"{r_best:<55} {c_best}"
    )

    # Detail Min & Max
    lines.append(
        "=" * 80
    )

    lines.append(
        "DETAIL LOKASI MIN & MAX"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"[ MINIMUM ] : "
        f"{min_item['duration_min']:.3f} menit "
        f"({fmt_duration(min_item['duration_min'])})"
    )

    lines.append(
        f"  - Window Ke   : "
        f"#{min_item['window_no']}"
    )

    lines.append(
        f"  - Rentang No  : "
        f"Ronde {min_item['window_start']} "
        f"s/d {min_item['window_end']}"
    )

    lines.append(
        f"  - Game ID     : "
        f"Start [ {min_item['first_game_id']} ] "
        f"---> "
        f"End [ {min_item['last_game_id']} ]"
    )

    lines.append(
        f"  - Waktu       : "
        f"{min_item['first_tag_ts_iso']} "
        f"(Start) s/d "
        f"{min_item['last_ts_gr']} (End)"
    )

    lines.append(
        f"  - Ronde terakhir gr : "
        f"{min_item['last_gr']} "
        f"→ {min_item['last_cat']}"
    )

    lines.append("")

    lines.append(
        f"[ MAXIMUM ] : "
        f"{max_item['duration_min']:.3f} menit "
        f"({fmt_duration(max_item['duration_min'])})"
    )

    lines.append(
        f"  - Window Ke   : "
        f"#{max_item['window_no']}"
    )

    lines.append(
        f"  - Rentang No  : "
        f"Ronde {max_item['window_start']} "
        f"s/d {max_item['window_end']}"
    )

    lines.append(
        f"  - Game ID     : "
        f"Start [ {max_item['first_game_id']} ] "
        f"---> "
        f"End [ {max_item['last_game_id']} ]"
    )

    lines.append(
        f"  - Waktu       : "
        f"{max_item['first_tag_ts_iso']} "
        f"(Start) s/d "
        f"{max_item['last_ts_gr']} (End)"
    )

    lines.append(
        f"  - Ronde terakhir gr : "
        f"{max_item['last_gr']} "
        f"→ {max_item['last_cat']}"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        "\n\n"
    )

    return "\n".join(lines)


def generate_window_report_csv_rows(
    window_size,
    windows
):
    """Hasilkan list dict untuk CSV."""

    values = [
        item["duration_min"]
        for item in windows
    ]

    minimum = min(values)
    maximum = max(values)
    average = mean(values)
    med = median(values)

    total_range = (
        maximum - minimum
    )

    bins = make_equal_bins(
        windows,
        NUMBER_OF_BINS
    )

    n = len(windows)

    total_low = sum(
        1
        for w in windows
        if w["last_cat"] == "low"
    )

    total_mid = sum(
        1
        for w in windows
        if w["last_cat"] == "mid"
    )

    total_high = sum(
        1
        for w in windows
        if w["last_cat"] == "high"
    )

    total_extra = sum(
        1
        for w in windows
        if w["last_cat"] == "extra"
    )

    rows = []

    # Summary
    rows.append({
        "window_size": window_size,
        "row_type": "summary",
        "bin_no": "",
        "range_low": f"{minimum:.3f}",
        "range_high": f"{maximum:.3f}",
        "rounds": n,
        "percent": 100.0,
        "mean": f"{average:.3f}",
        "median": f"{med:.3f}",
        "range_total": f"{total_range:.3f}",
        "low": total_low,
        "mid": total_mid,
        "high": total_high,
        "extra": total_extra,
        "pct_low": (
            round(
                100 * total_low / n,
                2
            )
            if n
            else 0
        ),
        "pct_mid": (
            round(
                100 * total_mid / n,
                2
            )
            if n
            else 0
        ),
        "pct_high": (
            round(
                100 * total_high / n,
                2
            )
            if n
            else 0
        ),
        "pct_extra": (
            round(
                100 * total_extra / n,
                2
            )
            if n
            else 0
        ),
    })

    # Per bin
    for b in bins:

        rows.append({
            "window_size": window_size,
            "row_type": "bin",
            "bin_no": b["bin_no"],
            "range_low": f"{b['range_lo']:.3f}",
            "range_high": f"{b['range_hi']:.3f}",
            "rounds": b["count"],
            "percent": round(
                b["percent"],
                2
            ),
            "mean": "",
            "median": "",
            "range_total": "",
            "low": b["n_low"],
            "mid": b["n_mid"],
            "high": b["n_high"],
            "extra": b["n_extra"],
            "pct_low": round(
                b["pct_low"],
                2
            ),
            "pct_mid": round(
                b["pct_mid"],
                2
            ),
            "pct_high": round(
                b["pct_high"],
                2
            ),
            "pct_extra": round(
                b["pct_extra"],
                2
            ),
        })

    return rows

# ============================================================
# RAW OUTPUT (STREAMING - v8)
# ============================================================

def raw_fieldnames():
    """Daftar kolom raw, tetap/statis (tidak butuh pandas)."""

    base_cols = [
        "window_size",
        "window_no",
        "window_start",
        "window_end",
        "first_game_id",
        "last_game_id",
        "first_tag_ts",
        "first_tag_ts_iso",
        "last_ts_gr",
        "duration_sec",
        "duration_min",
        "last_gr",
        "last_cat",
        "mean_gr",
        "max_gr",
        "min_gr",
    ]

    flag_cols = [f"last_ge_{t}" for t in THRESHOLDS]
    count_cols = [f"count_ge_{t}" for t in THRESHOLDS]
    since_cols = [f"since_ge_{t}" for t in THRESHOLDS]

    return base_cols + flag_cols + count_cols + since_cols


class RawStreamWriter:
    """
    Menulis baris raw ke CSV LANGSUNG saat satu window_size selesai,
    tidak menunggu semua window_size (10..2000) selesai dulu.

    Ini yang menghilangkan lonjakan RAM di v7: tidak ada lagi
    list besar (all_raw_rows) yang menumpuk baris dari semua ukuran.
    """

    def __init__(self, base_name):
        self.csv_path = Path(f"{base_name}_raw.csv")
        self.fieldnames = raw_fieldnames()
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.total_rows = 0

    def write_window_size(self, windows):
        """Tulis semua baris untuk 1 window_size, lalu flush ke disk."""

        if not windows:
            return

        self.writer.writerows(windows)
        self.file.flush()
        self.total_rows += len(windows)

    def close(self):
        self.file.close()
        print(
            f"[+] Raw windows CSV     : "
            f"{self.csv_path.resolve()} "
            f"({self.total_rows:,} baris)"
        )


def convert_raw_csv_to_parquet(csv_path, chunksize=200_000):
    """
    Konversi CSV raw -> Parquet setelah semua selesai ditulis.

    Dibaca & ditulis per-chunk (bukan load semua sekaligus) supaya
    tidak menimbulkan lonjakan RAM yang sama seperti masalah semula.
    Butuh pandas + pyarrow. Kalau tidak tersedia, CSV tetap aman dipakai.
    """

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print(
            f"[!] Lewati Parquet, library tidak tersedia ({e}). "
            f"CSV raw tetap ada dan bisa dipakai langsung."
        )
        return

    pq_path = Path(str(csv_path).replace("_raw.csv", "_raw.parquet"))
    writer = None

    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(pq_path, table.schema)

            writer.write_table(table)

        print(f"[+] Raw windows Parquet : {pq_path.resolve()}")

    except Exception as e:
        print(f"[!] Parquet gagal ({e}). Pakai CSV saja.")

    finally:
        if writer is not None:
            writer.close()


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Burst FIFO Analysis - "
            "optimized sliding window"
        ),
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
        epilog="""
Contoh pemakaian:

  python window_burst7_optimized.py data.md 10 10 --format raw

  python window_burst7_optimized.py data.md 10 20 --format raw

  python window_burst7_optimized.py data.md 10 2000 --format raw

  python window_burst7_optimized.py data.md 10 2000 --format all


Format:

  md   = laporan ringkasan bin
  csv  = ringkasan bin dalam CSV
  raw  = 1 baris per window
         CSV + Parquet
  both = md + csv ringkasan
  all  = md + csv + raw


Nama file raw:

  <stem><start>_<end>_raw.csv
  <stem><start>_<end>_raw.parquet
"""
    )

    parser.add_argument(
        "input_file",
        help="File Markdown input"
    )

    parser.add_argument(
        "start",
        type=int,
        help="Ukuran window mulai"
    )

    parser.add_argument(
        "end",
        type=int,
        help="Ukuran window akhir"
    )

    parser.add_argument(
        "--format",
        choices=[
            "md",
            "csv",
            "both",
            "raw",
            "all"
        ],
        default="md",
        dest="fmt",
        help="Format output"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_file = args.input_file
    start_window = args.start
    end_window = args.end
    fmt = args.fmt

    if start_window > end_window:

        print(
            "ERROR: start harus <= end"
        )

        sys.exit(1)

    if start_window < 1:

        print(
            "ERROR: start harus >= 1"
        )

        sys.exit(1)

    is_single_size = (
        start_window == end_window
    )

    input_path = Path(
        input_file
    )

    stem = input_path.stem

    base_name = (
        f"{stem}"
        f"{start_window}_"
        f"{end_window}"
    )

    want_md = fmt in (
        "md",
        "both",
        "all"
    )

    want_csv = fmt in (
        "csv",
        "both",
        "all"
    )

    want_raw = fmt in (
        "raw",
        "all"
    )

    print("=" * 80)

    print(
        f"BURST FIFO ANALYSIS "
        f"OPTIMIZED "
        f"(WINDOW SIZE "
        f"{start_window} - "
        f"{end_window})"
    )

    print("=" * 80)

    print(
        f"Input file     : "
        f"{input_file}"
    )

    print(
        f"Format output  : "
        f"{fmt}"
    )

    if want_md or want_csv:

        print(
            f"Summary files  : "
            f"{base_name}.md / "
            f"{base_name}.csv"
        )

    if want_raw:

        print(
            f"Raw files      : "
            f"{base_name}_raw.csv / "
            f"{base_name}_raw.parquet"
        )

    print("=" * 80)

    try:

        rows = parse_markdown_rows(
            input_file
        )

    except FileNotFoundError:

        print(
            f"ERROR: File tidak ditemukan: "
            f"{input_file}"
        )

        sys.exit(1)

    print(
        f"Total row terbaca : "
        f"{len(rows):,}"
    )

    start_index = (
        find_first_live_complete(
            rows
        )
    )

    if start_index is None:

        print(
            "ERROR: Tidak menemukan "
            "titik Live pertama "
            "dengan data lengkap."
        )

        sys.exit(1)

    rounds = prepare_rounds(
        rows,
        start_index
    )

    print(
        f"Ronde valid setelah "
        f"titik start Live: "
        f"{len(rounds):,}\n"
    )

    all_md_reports = []
    all_csv_rows = []

    raw_writer = None

    if want_raw:
        raw_writer = RawStreamWriter(base_name)

    # --------------------------------------------------------
    # PROCESS SEMUA WINDOW SIZE
    # --------------------------------------------------------

    for w_size in range(
        start_window,
        end_window + 1
    ):

        if len(rounds) < w_size:

            print(
                f"Skipping Window Size "
                f"{w_size}: "
                f"Ronde valid tidak cukup "
                f"({len(rounds)} < {w_size})"
            )

            continue

        windows = build_windows_for_size(
            rounds,
            w_size
        )

        if not windows:

            print(
                f"Skipping Window Size "
                f"{w_size}: "
                f"Gagal menghitung "
                f"window FIFO."
            )

            continue

        if want_raw:

            # Tulis SEKARANG juga, jangan ditampung di list besar.
            raw_writer.write_window_size(windows)

        if want_md:

            report_txt = (
                generate_window_report_text(
                    w_size,
                    windows
                )
            )

            all_md_reports.append(
                report_txt
            )

            if is_single_size:

                out_dir = Path(
                    OUTPUT_DIR
                )

                out_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                (
                    out_dir
                    / f"window_{w_size}.md"
                ).write_text(
                    report_txt,
                    encoding="utf-8"
                )

        if want_csv:

            csv_rows = (
                generate_window_report_csv_rows(
                    w_size,
                    windows
                )
            )

            all_csv_rows.extend(
                csv_rows
            )

        print(
            f"✓ Selesai memproses "
            f"Window Size "
            f"{w_size:>4} | "
            f"Total Window FIFO: "
            f"{len(windows):,}"
        )

        # Buang dari memori sebelum lanjut ke window_size berikutnya.
        del windows

    if raw_writer is not None:
        raw_writer.close()

    # --------------------------------------------------------
    # SIMPAN MD
    # --------------------------------------------------------

    if (
        want_md
        and all_md_reports
    ):

        md_file = Path(
            f"{base_name}.md"
        )

        md_file.write_text(
            "\n".join(
                all_md_reports
            ),
            encoding="utf-8"
        )

        print(
            f"\n[+] Laporan MD gabungan : "
            f"{md_file.resolve()}"
        )

    # --------------------------------------------------------
    # SIMPAN CSV SUMMARY
    # --------------------------------------------------------

    if (
        want_csv
        and all_csv_rows
    ):

        csv_file = Path(
            f"{base_name}.csv"
        )

        fieldnames = [
            "window_size",
            "row_type",
            "bin_no",
            "range_low",
            "range_high",
            "rounds",
            "percent",
            "mean",
            "median",
            "range_total",
            "low",
            "mid",
            "high",
            "extra",
            "pct_low",
            "pct_mid",
            "pct_high",
            "pct_extra",
        ]

        with csv_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                all_csv_rows
            )

        print(
            f"[+] Laporan CSV gabungan: "
            f"{csv_file.resolve()}"
        )

    # --------------------------------------------------------
    # RAW: CSV sudah ditulis streaming di atas.
    # Parquet dibuat belakangan dari CSV, dibaca per-chunk.
    # --------------------------------------------------------

    if (
        want_raw
        and raw_writer is not None
        and raw_writer.total_rows > 0
    ):

        convert_raw_csv_to_parquet(
            raw_writer.csv_path
        )

    if (
        is_single_size
        and fmt in (
            "md",
            "both"
        )
    ):

        print(
            f"[+] File individu        : "
            f"{OUTPUT_DIR}/"
            f"window_{start_window}.md"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "PROSES BURST SELESAI"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
