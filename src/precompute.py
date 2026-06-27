"""Offline embedding precompute (run on GPU, no time limit). Encodes every
candidate's profile text and the fixed JD, saving compact float16 artifacts that
the CPU-only ranking step consumes without any model or network."""
import argparse
import sys
from pathlib import Path
from typing import List
import numpy as np

from src import config as C
from src import io_utils

def save_artifacts(out_dir, emb: np.ndarray, ids: List[str], jd: np.ndarray) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb.astype(np.float16))
    np.save(out / "candidate_ids.npy", np.array(ids, dtype=object), allow_pickle=True)
    np.save(out / "jd_embedding.npy", jd.astype(np.float16))

def load_artifacts(out_dir):
    out = Path(out_dir)
    emb = np.load(out / "embeddings.npy")
    ids = np.load(out / "candidate_ids.npy", allow_pickle=True)
    jd = np.load(out / "jd_embedding.npy")
    return emb, ids, jd

def _read_jd_text(jd_path: Path) -> str:
    return jd_path.read_text(encoding="utf-8")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(C.DATA_FILE))
    ap.add_argument("--jd", required=True, help="Path to JD text file")
    ap.add_argument("--out", default=str(C.ARTIFACTS))
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer  # heavy import, offline-OK
    model = SentenceTransformer(C.EMBED_MODEL)

    ids, texts = [], []
    for c in io_utils.iter_candidates(Path(args.candidates)):
        ids.append(c["candidate_id"])
        texts.append(io_utils.build_profile_text(c))
    print(f"Encoding {len(texts)} candidates with {C.EMBED_MODEL} ...", file=sys.stderr)
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    jd = model.encode([_read_jd_text(Path(args.jd))], normalize_embeddings=True,
                      convert_to_numpy=True)[0]
    save_artifacts(args.out, emb, ids, jd)
    print(f"Saved artifacts to {args.out} (emb {emb.shape}, float16).", file=sys.stderr)

if __name__ == "__main__":
    main()
