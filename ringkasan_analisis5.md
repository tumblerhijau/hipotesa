# Ringkasan Analisis Data Crash Game

Dataset: `410_kalibrasi.md`

---

## 0. Tujuan & Aturan Project

**Tujuan:** cari apakah histori ribuan round bisa dipakai memperkirakan **range/kategori** multiplier round berikutnya (bukan angka exact) — fokus utama: mengenali kondisi yang bikin peluang **Low** (<2) berbeda secara konsisten, agar bisa dipakai sebagai filter.

**Kategori:** Low 1,00–1,99 | Mid 2,00–24,99 | High 25,00–99,99 | Extra ≥100

**Konsep dasar — Sliding FIFO Window:** window size N (mis. 30) bergeser 1 round setiap kali (round 1–30, lalu 2–31, dst) → antar-window **overlap berat** (W30 #100 & #101 berbagi 29/30 round). Karena itu ribuan window **tidak boleh dianggap sample independen** — validasi wajib pakai chronological split.

**Larangan keras (jangan dilakukan):**
- Cari pola hanya karena "terlihat menarik"
- Pilih window size terbaik berdasar hasil in-sample
- Anggap korelasi kecil = prediktabilitas
- Anggap sliding windows sebagai sample independen
- Pakai future data tanpa sengaja (data leakage)
- Pakai mean GR tanpa perhatikan heavy-tail
- Bikin model ML kompleks sebelum tahu signal dasarnya ada
- Simpulkan "bisa diprediksi" dari backtest satu periode saja
> Signal kecil tapi stabil lebih berharga daripada signal besar yang cuma muncul sekali.

**Peran AI yang diinginkan:** partner riset, bukan pembenar hipotesis. Tugas: bantu rumuskan hipotesis yang bisa diuji, tentukan fitur/baseline/eksperimen/metric, cegah leakage, cari counter-evidence, dan nilai apakah signal stabil atau cuma artefact. Kalau hasil tidak mendukung hipotesis user, katakan langsung — jangan mengarang kolom/struktur data yang tidak ada.

**Urutan kerja yang diinginkan:** validasi data → baseline → duration → relative duration → recency → frequency → duration+recency → duration+frequency → kombinasi sederhana → multi-window → walk-forward validation → baru modeling (kalau ada signal).

## A. Validasi Data & Baseline (Uji 1)

- Data mentah: **5.441 round** → dibuat jadi dataset sliding-window (ukuran 10–2000, total 1.991 ukuran window) → **8,8 juta baris**.
- Data bersih: tidak ada window duplikat, tidak ada `first/last_game_id` hilang.
- Baseline hasil round berikutnya (`target_low` = hasil < 2):
  - Low: **53,19%** | Non-Low (≥2): **46,81%**
  - ≥5: 19,05% | ≥10: 9,64% | ≥20: 4,65% | ≥50: 1,77% | ≥100: 0,69%
- Stabil dari waktu ke waktu (awal 53,79% → akhir 52,77%, beda tipis).
- **Kesimpulan:** struktur data valid, tapi ini baru fondasi, belum ada bukti bisa diprediksi.

## B. Uji Fitur Window: Duration, Ratio, Recency, Frequency, Interaksi, GR Stats (Uji 2, H1–H7 + H14–H15)

**Metode pengujian (berlaku utk semua H1–H7, H14, H15):**
Dijalankan lewat script (`single_feature_analysis.py`, terus dikembangkan sampai versi 5 + conditional analysis terpisah) yang membaca dataset window (8,8 juta baris, window size 10–2000) pakai DuckDB (tanpa load semua ke RAM). Caranya:
1. Tiap fitur dipecah jadi beberapa **bucket/kelompok** (bukan angka mentah) — untuk fitur bertipe kategori (recency, frequency) bucket dibuat manual (mis. `0`, `1-2`, `3-5`, dst); untuk fitur numerik heavy-tail (durasi, mean_gr) dipakai **desil (NTILE 10)** — data dibagi 10 kelompok sama rata berdasar urutan nilainya, supaya outlier ekstrem tidak merusak bucket.
2. Tiap bucket dihitung `low_rate` (persentase round berikutnya yang hasilnya Low) dan dibandingkan ke baseline keseluruhan (53,19%) → jadi `delta_low_rate`.
3. Bucket dengan sample terlalu kecil (< 100 baris, ditandai `enough_rows: False`) diabaikan karena rawan kebetulan.
4. Fitur interaksi (H5, H14, H15) dibuat dengan menggabungkan 2 bucket sekaligus (mis. `recent | frequent`) memakai kolom string gabungan, lalu dihitung dengan cara yang sama.

**Hasil per hipotesis** (semua terhadap baseline 53,19%, "sample besar" = jutaan baris kecuali disebutkan):

- **H1 — durasi absolut** (10 desil): selisih -0,25% s/d +0,21%, tidak ada tren.
- **H2 — rasio durasi** (durasi ÷ median durasi utk window_size-nya): bucket mayoritas (0,75–1 & 1–1,25, >8,8 juta baris) selisih ±0,5%. Bucket ekstrem kecil (198–1.650 baris) tidak reliable.
- **H3 — recency** (`since_ge_10`): selisih -1,5% s/d +1,5%, tidak konsisten.
- **H4 — frequency** (`count_ge_10`): selisih -1,1% s/d +0,7%, bucket mayoritas (">5", 8,5 juta) nyaris identik baseline.
- **H5 — recency × frequency**: bucket besar (jutaan baris) selisih -0,38% s/d +0,34%. Tidak ada sinyal.
- **H6 — mean_gr dalam window (10 desil)**: mayoritas desil kecil (-2% s/d +0,4%), **desil 10 (mean_gr tertinggi, 883 ribu baris) low_rate 55,8% → selisih +2,6%** — selisih terbesar dari semua fitur yang diuji. Pola antar desil tidak monoton (desil 8 malah -1,96%, desil 9 balik ~0%, baru desil 10 melonjak).
- **H7 — state HOT/COLD/NORMAL** (dari recency+frequency): HOT +0,34%, COLD -1,27%, NORMAL -0,12%. Kecil, seperti H5.
- **H14 — rasio durasi × recency**: bucket besar selisih -0,37% s/d +0,35%. Bucket kecil (`long`/`short`) tidak reliable.
- **H15 — rasio durasi × frequency**: bucket terbesar (`normal | frequent`, 8,7 juta baris) selisih nyaris nol (0,0002%).

**Status temuan (setelah uji lanjutan H6):**
- H1, H2, H3, H4, H5, H7, H14, H15 → **REJECTED / tidak ada sinyal** yang cukup besar & reliable.
- **H6 (mean_gr decile)** → **REJECTED sebagai fitur mandiri** (setelah H17 + conditional analysis).

**Detail uji lanjutan H6:**

1. **Stability lintas periode (H17)** — dari `h6_stability_by_period.csv`:
   - early: decile_10 = 57,53% (delta **+3,34%** vs period baseline, n=371k)
   - middle: decile_10 = 54,55% (delta **+2,19%**, n=496k)
   - late: decile_10 = 54,39% (delta hanya **+0,83%**, n=15k saja)
   - Efek melemah tajam di late + sample decile tinggi mengecil drastis → distribusi `mean_gr` tidak stasioner.

2. **Conditional analysis (mean_gr × max_gr)**:
   - Hampir seluruh massa mean_gr decile_10 (~727k dari 883k) juga berada di **max_gr decile_10** → low_rate 56,40%.
   - Ketika mean_gr tinggi **tapi max_gr rendah/sedang** (decile 1–4) → low_rate justru **49–52%** (di bawah baseline).
   - Ketika mean_gr tinggi **dan** max_gr tinggi → low_rate naik.
   - Kesimpulan: sinyal mean_gr hampir pasti **artefak dari outlier (max_gr)**, karena mean sangat didominasi 1–2 nilai ekstrem di heavy-tail distribution. Mean tidak menambah informasi independen yang berarti.

3. **Conditional vs count_ge_10**: pola serupa — efek positif hampir hanya muncul di bucket `count_ge_10 = >5`.

**Kesimpulan H6:** Tidak lolos sebagai sinyal mandiri. Lebih masuk akal menguji `max_gr` langsung (atau statistik robust lain) daripada mean.

## C. Analisis Bentuk Nilai `gr_result` (dataset 5.491 round)

**Statistik dasar:**
- Min 1,00 | Median 1,87 | Mean 8,04 | Max 5.000 | P95 = 19,17
- Mean jauh > median → distribusi miring kanan, ekor panjang.

**Pola distribusi (temuan utama):**
- Peluang hasil ≥ x mendekati rumus **0,95 / x** (cocok di banyak titik, dari x=1,1 sampai x=500).
- Exponent ekor ≈ 1 (0,997 keseluruhan; 1,004 di paruh awal; 0,990 di paruh akhir) → cukup stabil.
- Nilai rendah tetap dominan: 36% data ada di 1,00–1,49; 17% di 1,50–1,99.
- Nilai **1,00** paling sering muncul persis (5,90% dari semua data).

**Ketergantungan antar-round (apakah round sebelumnya pengaruhi berikutnya):**
- Korelasi log lag-1: **-0,0132** → nyaris nol, tidak ada hubungan berarti.
- Binary: sebelumnya LOW → 48,13% jadi HIGH; sebelumnya HIGH → 45,76% jadi HIGH (beda 2,37 poin, p≈0,084 → **belum signifikan**).
- "Mean reversion" (nilai ekstrem cenderung balik ke tengah) terlihat deskriptif tapi lemah, korelasinya tetap ~0.
- Base rate ≥2 stabil antara paruh awal (47,07%) dan akhir (46,98%).
- Streak: LOW rata-rata 2,08 round (maks 12), HIGH rata-rata 1,84 round (maks 16) — wajar untuk proses acak, bukan bukti dependency.

**Hipotesis kerja yang tersimpan:**
1. **H1** – Distribusi ekor ≈ 0,95/x, exponent ≈1 → didukung awal, perlu fitting formal.
2. **H2** – Parameter distribusi relatif stabil sepanjang waktu → indikasi awal.
3. **H3** – Ada mean reversion ringan → lemah, belum terbukti.
4. **H4** – Dependency antar-round sangat kecil → didukung.

**Yang BELUM boleh disimpulkan:** game bisa diprediksi, PRNG "punya memori", formula pasti 0,95/x, nilai ekstrem menentukan nilai berikutnya, atau streak bisa dipakai sebagai sinyal.

**Rencana lanjutan yang disarankan (belum dikerjakan):**
- Fitting distribusi formal (Pareto, log-normal, Weibull, vs model 0,95/x)
- Analisis residual terhadap model 0,95/x
- Rolling-window untuk cek apakah parameter berubah dari waktu ke waktu
- Conditional tail (P(X≥x | X sebelumnya≥y))
- Multi-lag (lag 1–20)
- Extreme-event response (efek nilai ekstrem ke beberapa round berikutnya)
- Semua pola baru wajib diuji ulang di data terpisah (discovery vs confirmation set) agar bukan kebetulan.

## C2. Roadmap Hipotesis Fitur Window (H1–H20, belum diuji semua)

Fitur per window: `window_size`, `window_duration`, `last_gr`, `mean/max/min GR`, `count_ge_X`, `since_ge_X`.

- **H1** duration window ↔ kategori hasil akhir — **sudah diuji (lihat bagian B), tidak ada sinyal**
- **H2** relative duration (`duration / median_duration_utk_window_size`) lebih informatif dari absolute — **sudah diuji (lihat bagian B), tidak ada sinyal di bucket bersample besar**
- **H3** recency (`since_ge_X`) — **sudah diuji (lihat bagian B), lemah**
- **H4** frequency (`count_ge_X`) — **sudah diuji (lihat bagian B), lemah**
- **H5** interaksi recency × frequency — **sudah diuji (lihat bagian B), tidak ada sinyal**
- **H6** statistik GR dalam window (mean/median/max/min) — hati-hati heavy-tail — **sudah diuji (mean_gr): overall +2,6% di decile tertinggi, tapi gagal stability (melemah di late) + conditional analysis menunjukkan artefak max_gr → REJECTED sebagai fitur mandiri. max_gr sendiri belum diuji formal.**
- **H7** state HOT/NORMAL/COLD dari recency+frequency — **sudah diuji (lihat bagian B), tidak ada sinyal**
- **H8** beda info antar window size (10–50, 51–100, ..., 1001–2000) — jangan buru-buru pilih "terbaik"
- **H9** konsistensi efek di banyak window size (bukan cuma satu)
- **H10** ensemble multi-window (majority/weighted vote, median probability)
- **H11** consensus score antar window (tetap ingat: window overlap, bukan independent votes)
- **H12** density = `window_size / duration`
- **H13** normalisasi: `count_ge_X / window_size`, `since_ge_X / window_size`
- **H14** interaksi duration_ratio × recency — **sudah diuji (lihat bagian B), tidak ada sinyal di bucket bersample besar**
- **H15** interaksi duration_ratio × frequency — **sudah diuji (lihat bagian B), tidak ada sinyal di bucket bersample besar**
- **H16** interaksi 3 arah (duration×recency×frequency) — risiko overfitting tinggi, jangan awal
- **H17** stability lintas waktu (awal/tengah/akhir) — signal cuma di 1 periode = tidak robust
- **H18** walk-forward validation (TRAIN→VALIDATION→TEST), bukan random split
- **H19** uji juga target selain ≥2 (≥5, ≥10, ≥20, ≥50, ≥100) — fitur bisa berguna untuk tail meski tidak untuk ≥2
- **H20** metric fokus LOW-filtering: bukan accuracy, tapi coverage, low rate vs base rate, lift, jumlah sample, stability out-of-sample

## D. Disiplin Metodologi (wajib dipakai di analisis selanjutnya)

1. **Kontrol False Discovery:** pisahkan data jadi *Discovery set* (cari pola) dan *Confirmation set* (uji pola), pembagian berdasarkan urutan waktu — bukan acak.
2. **Label status temuan** — jangan campur fakta dan dugaan:
   `OBSERVED` (terlihat langsung) → `CALCULATED` (hasil hitung) → `HYPOTHESIS` (dugaan) → `TESTED` (sudah diuji) → `CONFIRMED` (terbukti di data out-of-sample) / `REJECTED` (gagal uji).
3. **Urutan wajib:** observasi → pola → hipotesis → uji → replikasi → out-of-sample → kesimpulan. Tidak boleh loncat ke "kesimpulan" tanpa lewat tahap ini.

---

**Inti keseluruhan:** Struktur nilai `gr_result` (bentuk ekor ~0,95/x) adalah temuan paling kuat sejauh ini. Ketergantungan antar-round (fitur recency/frequency, korelasi lag-1, transisi low/high, serta mean_gr dalam window) semuanya masih sangat lemah atau terbukti artefak, dan belum bisa dipakai sebagai dasar prediksi. H6 (mean_gr) yang sempat menjanjikan telah ditolak setelah uji stabilitas dan conditional analysis terhadap max_gr.
