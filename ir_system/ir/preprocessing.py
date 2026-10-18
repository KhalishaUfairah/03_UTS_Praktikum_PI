"""Text preprocessing: case folding, tokenization, stopword removal, (opsional) stemming."""
import re
from functools import lru_cache

import config
from ir.stopwords_id import STOPWORDS_ID

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z]+")          # hanya huruf (angka & tanda baca dibuang)

# ---------------------------------------------------------------------------
# Stemmer ringan berbasis aturan (opsional) - BUKAN Nazief-Adriani lengkap.
# Dibuat sendiri agar tetap hanya memakai library standar.
# ---------------------------------------------------------------------------
_PARTICLES = ("lah", "kah", "pun")
_POSSESSIVE = ("nya", "mu", "ku")
_SUFFIXES = ("kan", "an")
_PREFIXES = ("meng", "meny", "men", "mem", "me", "peng", "peny", "pen", "pem",
             "ber", "bel", "ter", "per", "di", "ke", "se")
_MIN_STEM = 4


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    w = word
    for group in (_PARTICLES, _POSSESSIVE, _SUFFIXES):
        for suf in group:
            if w.endswith(suf) and len(w) - len(suf) >= _MIN_STEM:
                w = w[: -len(suf)]
                break
    for pre in _PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= _MIN_STEM:
            w = w[len(pre):]
            break
    return w


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def case_folding(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS_ID]


def preprocess(text, use_stemming=None) -> list:
    """Teks mentah -> list token bersih."""
    if use_stemming is None:
        use_stemming = config.USE_STEMMING
    if not isinstance(text, str):
        return []
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = case_folding(text)
    tokens = tokenize(text)
    tokens = [t for t in tokens if len(t) >= config.MIN_TOKEN_LEN]
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = remove_stopwords([stem(t) for t in tokens])
    return tokens


def preprocess_to_string(text, use_stemming=None) -> str:
    return " ".join(preprocess(text, use_stemming))


def split_tokens(text: str) -> list:
    """Tokenizer untuk CountVectorizer (teks sudah bersih -> split spasi).
    Level modul agar bisa di-pickle."""
    return text.split()
