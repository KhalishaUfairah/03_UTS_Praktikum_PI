"""Konfigurasi global sistem IR."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
INDEX_DIR = BASE_DIR / "index_store"          # folder index Whoosh + model BoW
BOW_FILE = INDEX_DIR / "bow_model.pkl"        # CountVectorizer + matriks BoW

# Nama folder dataset (sesuai struktur soal)
DATASETS = ["etd-usk", "etd-ugm", "kompas", "tempo", "mojok"]

# Kolom CSV
COL_TITLE = "judul"
COL_CONTENT = "konten"

# Preprocessing
USE_STEMMING = False      # opsional (stemmer ringan berbasis aturan)
MIN_TOKEN_LEN = 2
TITLE_WEIGHT = 2          # judul diulang N kali agar lebih berpengaruh

# Bag of Words
BOW_MIN_DF = 2            # buang term yang muncul di < 2 dokumen

# Searching
TOP_N = 5                 # jumlah hasil akhir yang ditampilkan
CANDIDATES = 1000         # kandidat dari Whoosh sebelum di-rerank cosine
SNIPPET_CHARS = 220
