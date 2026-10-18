"""CLI Information Retrieval System - UTS Praktikum Penelusuran Informasi."""
import time

import config
from ir.data_loader import load_all_datasets
from ir.indexer import build_index, index_exists

MENU = """
=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
===================================="""


def do_index():
    print("\nMemuat dataset ...")
    try:
        data = load_all_datasets()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    print(f"Total dokumen: {len(data)}\nMembangun index ...")
    t0 = time.time()
    stats = build_index(data)
    print("\nIndexing selesai.")
    print(f"  Dokumen            : {stats['n_docs']}")
    print(f"  Ukuran BoW         : {stats['bow_shape'][0]} dokumen x {stats['bow_shape'][1]} term")
    print(f"  Waktu preprocessing: {stats['preprocess_sec']:.1f} dtk")
    print(f"  Waktu BoW          : {stats['bow_sec']:.1f} dtk")
    print(f"  Waktu index Whoosh : {stats['whoosh_sec']:.1f} dtk")
    print(f"  Total              : {time.time() - t0:.1f} dtk")
    return stats


def do_search(searcher):
    print("\n(ketik 'back' untuk kembali ke menu)")
    while True:
        q = input("\nQuery > ").strip()
        if q.lower() in ("back", "exit", "quit"):
            return
        if not q:
            continue
        ds = input(f"Filter dataset {config.DATASETS} (Enter = semua) > ").strip() or None
        if ds and ds not in config.DATASETS:
            print("  Dataset tidak dikenal, filter diabaikan.")
            ds = None
        t0 = time.time()
        tokens, results = searcher.search(q, dataset=ds)
        dt = time.time() - t0
        print(f"\nToken query: {tokens}")
        if not results:
            print("Tidak ada dokumen yang cocok.")
            continue
        print(f"Top {len(results)} dokumen ({dt * 1000:.0f} ms):")
        print("-" * 70)
        for rank, r in enumerate(results, 1):
            print(f"{rank}. [{r['dataset']}] {r['judul']}")
            print(f"   Cosine: {r['cosine']:.4f} | Whoosh(BM25F): {r['bm25']:.2f}")
            print(f"   {r['snippet']}\n")


def main():
    searcher = None
    while True:
        print(MENU)
        choice = input("Pilih menu [1-3]: ").strip()
        if choice == "1":
            if do_index():
                from ir.searcher import Searcher
                searcher = Searcher()
        elif choice == "2":
            if searcher is None:
                if not index_exists():
                    print("\nIndex belum ada. Jalankan menu [1] terlebih dahulu.")
                    continue
                from ir.searcher import Searcher
                print("\nMemuat index ...")
                searcher = Searcher()
            do_search(searcher)
        elif choice == "3":
            print("Sampai jumpa!")
            break
        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nKeluar.")
