"""Replace the duplicate flip with the missing anti-diagonal TTA reflection."""
import re


def patch_true_d4(source: str) -> str:
    forward = re.compile(
        r"torch\.rot90\(\s*imgs,\s*1,\s*dims=\(-2,\s*-1\)\s*\)\.transpose\(-1,\s*-2\)"
    )
    inverse = re.compile(
        r"torch\.rot90\(\s*(det_at\[f\]|_u_at|secondary_det_at\[f\])"
        r"\.transpose\(-1,\s*-2\),\s*-1,\s*dims=\(-2,\s*-1\),?\s*\)"
    )
    source, n_forward = forward.subn("imgs.flip((-2, -1)).transpose(-1, -2)", source)
    source, n_inverse = inverse.subn(lambda m: m[1] + ".transpose(-1, -2).flip((-2, -1))", source)
    if (n_forward, n_inverse) != (2, 3):
        raise RuntimeError(f"Expected two forward and three inverse transforms, got {(n_forward, n_inverse)}")
    compile(source, "true_d4_predictor", "exec")
    return source


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    x = np.arange(35).reshape(5, 7)
    views = [x, x[:, ::-1], x[::-1], x[::-1, ::-1], np.rot90(x),
             np.rot90(x, 3), x.T, x[::-1, ::-1].T]
    assert len({(a.shape, a.tobytes()) for a in views}) == 8
    assert np.array_equal(views[-1].T[::-1, ::-1], x)
    source = Path("local_runs/frontier_20260909/harmonic/output/tracking_repo/scripts/predict_unet_transformer.py").read_text()
    patch_true_d4(source)
    print("Eight unique views, anti-diagonal inverse round-trip, five exact replacements, predictor compile: PASS")
