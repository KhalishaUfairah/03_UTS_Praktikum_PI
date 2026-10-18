"""Memuat kelima dataset dari folder datasets/ menggunakan pandas."""
import pandas as pd

import config


def load_all_datasets(verbose: bool = True) -> pd.DataFrame:
    """Return DataFrame berkolom: dataset, judul, konten (doc_id = index baris)."""
    frames = []
    for name in config.DATASETS:
        folder = config.DATASET_DIR / name
        files = sorted(folder.glob("*.csv")) if folder.exists() else []
        if not files:
            print(f"  [!] Dataset '{name}' tidak ditemukan di {folder}")
            continue
        for f in files:
            df = pd.read_csv(f, usecols=[config.COL_TITLE, config.COL_CONTENT])
            df = df.dropna(subset=[config.COL_TITLE, config.COL_CONTENT])
            df = df.drop_duplicates()
            df.insert(0, "dataset", name)
            frames.append(df)
            if verbose:
                print(f"  [+] {name:<8} {f.name:<14} {len(df):>6} dokumen")
    if not frames:
        raise FileNotFoundError(
            "Tidak ada dataset. Letakkan file CSV di datasets/<nama-dataset>/ "
            "(lihat README).")
    data = pd.concat(frames, ignore_index=True)
    data.index.name = "doc_id"
    return data
