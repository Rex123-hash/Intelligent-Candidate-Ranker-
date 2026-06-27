# tests/test_precompute.py
import numpy as np
from src import precompute

def test_save_and_load_roundtrip(tmp_path):
    emb = np.random.rand(5, 8).astype(np.float32)
    ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004", "CAND_0000005"]
    jd = np.random.rand(8).astype(np.float32)
    precompute.save_artifacts(tmp_path, emb, ids, jd)
    e2, i2, j2 = precompute.load_artifacts(tmp_path)
    assert e2.shape == (5, 8)
    assert e2.dtype == np.float16          # stored compact
    assert list(i2) == ids
    assert j2.shape == (8,)
    np.testing.assert_allclose(e2.astype(np.float32), emb, atol=1e-2)
