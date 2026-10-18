"""Pembentukan index Whoosh + representasi Bag of Words (CountVectorizer)."""
import pickle
import shutil
import time

from sklearn.feature_extraction.text import CountVectorizer
from whoosh import analysis, fields, index

import config
from ir.preprocessing import preprocess_to_string, split_tokens


def build_schema():
    """Skema index Whoosh.
    - content : teks yang SUDAH dipreprocess, dipisah spasi (field yang dicari)
    - doc_id  : nomor baris pada matriks BoW (stored)
    - dataset : sumber dokumen (ID -> dapat dipakai untuk filter)
    - judul, snippet : disimpan untuk ditampilkan pada hasil
    """
    return fields.Schema(
        doc_id=fields.STORED,
        dataset=fields.ID(stored=True),
        judul=fields.STORED,
        snippet=fields.STORED,
        content=fields.TEXT(analyzer=analysis.RegexTokenizer(r"\S+"), phrase=False),
    )


def _make_snippet(text: str) -> str:
    text = " ".join(str(text).split())
    return text[: config.SNIPPET_CHARS] + ("..." if len(text) > config.SNIPPET_CHARS else "")


def build_index(data) -> dict:
    """Preprocess seluruh dokumen, buat matriks BoW, lalu index ke Whoosh."""
    stats = {}

    # 1) Preprocessing -------------------------------------------------------
    t0 = time.time()
    clean_docs = []
    n = len(data)
    for i, (title, body) in enumerate(zip(data[config.COL_TITLE], data[config.COL_CONTENT]), 1):
        text = (" " + str(title)) * config.TITLE_WEIGHT + " " + str(body)
        clean_docs.append(preprocess_to_string(text))
        if i % 5000 == 0 or i == n:
            print(f"  Preprocessing: {i}/{n}", end="\r")
    print()
    stats["preprocess_sec"] = time.time() - t0

    # 2) Bag of Words --------------------------------------------------------
    t0 = time.time()
    vectorizer = CountVectorizer(tokenizer=split_tokens, preprocessor=None,
                                 lowercase=False, token_pattern=None,
                                 min_df=config.BOW_MIN_DF)
    matrix = vectorizer.fit_transform(clean_docs).tocsr()
    stats["bow_sec"] = time.time() - t0
    stats["bow_shape"] = matrix.shape
    stats["bow_nnz"] = matrix.nnz

    # 3) Index Whoosh --------------------------------------------------------
    t0 = time.time()
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True)
    ix = index.create_in(str(config.INDEX_DIR), build_schema())
    writer = ix.writer(limitmb=512)
    for i, (ds, title, body, clean) in enumerate(
            zip(data["dataset"], data[config.COL_TITLE], data[config.COL_CONTENT], clean_docs)):
        writer.add_document(doc_id=i, dataset=ds, judul=str(title),
                            snippet=_make_snippet(body), content=clean)
        if (i + 1) % 5000 == 0 or i + 1 == n:
            print(f"  Indexing Whoosh: {i + 1}/{n}", end="\r")
    print("\n  Commit index ...")
    writer.commit()
    stats["whoosh_sec"] = time.time() - t0

    # 4) Simpan model BoW ----------------------------------------------------
    with open(config.BOW_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f, protocol=4)

    stats["n_docs"] = n
    return stats


def index_exists() -> bool:
    return (config.INDEX_DIR.exists() and index.exists_in(str(config.INDEX_DIR))
            and config.BOW_FILE.exists())
