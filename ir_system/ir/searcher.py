"""Pencarian: kandidat dari Whoosh (BM25F) -> re-ranking cosine similarity (BoW)."""
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index, query

import config
from ir.preprocessing import preprocess


class Searcher:
    def __init__(self):
        self.ix = index.open_dir(str(config.INDEX_DIR))
        with open(config.BOW_FILE, "rb") as f:
            model = pickle.load(f)
        self.vectorizer = model["vectorizer"]
        self.matrix = model["matrix"]

    def search(self, raw_query: str, top_n: int = None, dataset: str = None):
        """Return (query_tokens, hasil). Hasil = list dict terurut cosine menurun."""
        top_n = top_n or config.TOP_N
        tokens = preprocess(raw_query)
        if not tokens:
            return tokens, []

        # 1) Retrieval kandidat dengan Whoosh (OR antar term query)
        q = query.Or([query.Term("content", t) for t in dict.fromkeys(tokens)])
        flt = query.Term("dataset", dataset) if dataset else None
        with self.ix.searcher() as s:
            hits = s.search(q, limit=config.CANDIDATES, filter=flt)
            cands = [{"doc_id": h["doc_id"], "dataset": h["dataset"], "judul": h["judul"],
                      "snippet": h["snippet"], "bm25": float(h.score)} for h in hits]
        if not cands:
            return tokens, []

        # 2) Representasi BoW query + cosine similarity terhadap kandidat
        q_vec = self.vectorizer.transform([" ".join(tokens)])
        if q_vec.nnz == 0:                      # semua term query tidak ada di vocabulary
            return tokens, []
        rows = [c["doc_id"] for c in cands]
        sims = cosine_similarity(q_vec, self.matrix[rows]).ravel()
        for c, sc in zip(cands, sims):
            c["cosine"] = float(sc)

        # 3) Ranking: cosine menurun (tie-break: skor Whoosh)
        cands.sort(key=lambda c: (c["cosine"], c["bm25"]), reverse=True)
        return tokens, cands[:top_n]
