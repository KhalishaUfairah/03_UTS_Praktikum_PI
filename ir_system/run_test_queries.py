"""Menjalankan sejumlah query uji (non-interaktif) -> test_results.json + ringkasan di terminal.
Jalankan setelah index dibuat (menu [1] di main.py)."""
import json
import time

from ir.searcher import Searcher

QUERIES = [
    ("kecerdasan buatan", None),
    ("pembelajaran matematika siswa SMP", None),
    ("korupsi pejabat negara", None),
    ("harga BBM naik", None),
    ("lagu anime", None),
    ("dampak perubahan iklim terhadap pertanian", None),
    ("pemilihan presiden", None),
    ("sepak bola timnas", None),
    ("kecerdasan buatan", "kompas"),
    ("zzqxv plorf", None),
]


def main():
    s = Searcher()
    out = []
    for q, ds in QUERIES:
        t0 = time.time()
        tokens, res = s.search(q, dataset=ds)
        ms = (time.time() - t0) * 1000
        out.append({"query": q, "filter": ds, "tokens": tokens, "ms": round(ms),
                    "results": res})
        print(f"\nQuery: {q!r} filter={ds} tokens={tokens} ({ms:.0f} ms)")
        for i, r in enumerate(res, 1):
            print(f"  {i}. [{r['dataset']}] {r['cosine']:.4f} {r['judul'][:80]}")
        if not res:
            print("  (tidak ada hasil)")
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
