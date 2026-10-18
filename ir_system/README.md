# Multi-Dataset Information Retrieval System (CLI)

UTS Praktikum Penelusuran Informasi - Departemen Informatika FMIPA, Universitas Syiah Kuala

Sistem Information Retrieval berbasis CLI yang mencari dan meranking dokumen dari 5 dataset
(etd-usk, etd-ugm, kompas, tempo, mojok) memakai **Whoosh**, **CountVectorizer (Bag of Words)**,
dan **cosine similarity**.

## Anggota Tim
| Nama | NPM |
|------|-----|
| Andrea Widi | 2408107010083 |
| Khalisha Ufairah | 2408107010084 |
| Arsha Alifa Mahmud | 2408107010095 |

## Pipeline
```
CSV (5 dataset) -> Preprocessing -> BoW (CountVectorizer) ----------------+
                        |                                                  |
                        +-> Index Whoosh (BM25F)                           |
                                                                           v
Query -> Preprocessing -> Whoosh: 1000 kandidat -> Cosine(query BoW, dokumen BoW) -> Top 5
```
1. **Preprocessing** (`ir/preprocessing.py`): hapus URL/HTML, case folding, tokenisasi (hanya huruf),
   hapus stopword Indonesia, stemming ringan (opsional, default mati).
2. **Bag of Words** (`ir/indexer.py`): `CountVectorizer` dengan `min_df=2`, disimpan di `index_store/bow_model.pkl`.
3. **Index Whoosh** (`ir/indexer.py`): field `content` (teks bersih), `dataset`, `judul`, `snippet`, `doc_id`.
4. **Search** (`ir/searcher.py`): Whoosh mengambil kandidat, lalu kandidat di-rerank dengan
   `cosine_similarity` antara vektor BoW query dan dokumen. Ditampilkan 5 teratas.

Judul dokumen diulang 2x (`TITLE_WEIGHT`) sebelum preprocessing agar term judul lebih berpengaruh.

## Struktur Folder
```
ir_system/
|- main.py                 # CLI interaktif
|- config.py               # konfigurasi (path, top-N, stemming, dll.)
|- run_test_queries.py     # menjalankan query uji -> test_results.json
|- requirements.txt
|- ir/
|  |- data_loader.py       # load CSV (pandas)
|  |- preprocessing.py     # case folding, tokenisasi, stopword, stemming
|  |- stopwords_id.py      # daftar stopword
|  |- indexer.py           # BoW + index Whoosh
|  \- searcher.py          # Whoosh retrieval + cosine ranking
\- datasets/
   |- etd-usk/  (etd_usk.csv)
   |- etd-ugm/  (etd_ugm.csv)
   |- kompas/   (kompas.csv)
   |- tempo/    (tempo.csv)
   \- mojok/    (mojok.csv)
```
Setiap CSV memiliki kolom `judul` dan `konten`. Letakkan file CSV di folder dataset masing-masing
(`datasets/<nama>/`); loader membaca semua `*.csv` di dalamnya.

## Instalasi
Python 3.9+ (diuji pada Python 3.12).
```bash
pip install -r requirements.txt
```
Hanya library standar + scikit-learn, Whoosh, dan pandas yang dipakai.

## Cara Menjalankan
```bash
python main.py
```
```
=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================
```
1. Pilih **[1]** sekali untuk membangun index (membuat folder `index_store/`). Untuk ~49.5 ribu dokumen
   proses ini memakan waktu beberapa menit.
2. Pilih **[2]**, ketik query (mis. `harga BBM naik`). Opsional isi filter dataset
   (`etd-usk`, `etd-ugm`, `kompas`, `tempo`, `mojok`) atau Enter untuk semua.
   Ketik `back` untuk kembali ke menu.
3. Index yang sudah dibuat otomatis dimuat saat memilih [2] pada eksekusi berikutnya.

Query uji non-interaktif (setelah index dibuat):
```bash
python run_test_queries.py     # hasil tersimpan di test_results.json
```

## Contoh Output
```
Query > lagu anime
Token query: ['lagu', 'anime']
1. [mojok] 7 Lagu Anime yang Pasti Bikin Kamu Makin Semangat
   Cosine: 0.7822 | Whoosh(BM25F): ...
```

## Konfigurasi (`config.py`)
| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `TOP_N` | 5 | jumlah hasil ditampilkan |
| `CANDIDATES` | 1000 | kandidat dari Whoosh sebelum rerank cosine |
| `USE_STEMMING` | False | stemmer ringan berbasis aturan (perlu re-index bila diubah) |
| `BOW_MIN_DF` | 2 | term yang muncul < 2 dokumen dibuang |
| `TITLE_WEIGHT` | 2 | pengulangan judul |

## Keterbatasan
- BoW memakai frekuensi mentah (tanpa IDF), sehingga kata umum yang sering muncul dapat mendominasi skor.
- Tidak ada pemahaman sinonim/semantik (mis. "AI" vs "kecerdasan buatan").
- Stemmer opsional hanya aturan sederhana, bukan algoritma Nazief-Adriani lengkap.
