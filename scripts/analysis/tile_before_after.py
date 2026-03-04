"""
tile_before_after.py
====================
Pixel-level before/after yew probability reconstruction using Method C
(embedding-space KNN counterfactual) for a single tile.

For every logged pixel (VRI cat 2–4) in the chosen tile, the 10 nearest
spectrally-similar unlogged (cat 5) pixels are found in 64-dim cosine space
and their mean probability is used as the counterfactual "historic" value.

The KNN step is implemented as a batched matrix multiply so it runs on GPU
(via PyTorch CUDA) when available, falling back transparently to CPU numpy.
Typical runtimes on a 1k×1.5k tile with ~500k logged pixels:
  GPU (e.g. RTX 3080) : ~3 s
  CPU (16-core)        : ~30–60 s

Usage
-----
    python scripts/analysis/tile_before_after.py [--slug SLUG] [--k K] [--cpu]

Default tile: carmanah_walbran
Default K:    10
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TILE_CACHE = ROOT / "results" / "analysis" / \
    "cwh_spot_comparisons" / "tile_cache"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── VRI suppression factors (for reference only — not applied here) ────────
LOG_LABELS = {
    1: "water/non-forest",
    2: "logged <20yr",
    3: "logged 20-40yr",
    4: "logged 40-80yr",
    5: "forest >80yr",
    6: "alpine/barren",
}


def _cosine_knn_numpy(
    query_emb: np.ndarray,
    ref_norm: np.ndarray,
    ref_prob: np.ndarray,
    k: int,
    batch: int = 8_000,
) -> np.ndarray:
    """CPU numpy batched cosine-KNN — returns mean prob of K nearest neighbours."""
    n = len(query_emb)
    cf = np.empty(n, dtype=np.float32)
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        q = query_emb[b0:b1].astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
        sims = q @ ref_norm.T                          # (batch, R)
        top_k = np.argpartition(-sims, k, axis=1)[:, :k]
        cf[b0:b1] = ref_prob[top_k].mean(axis=1)
    return cf


def _cosine_knn_torch(
    query_emb: np.ndarray,
    ref_norm,           # torch.Tensor on device
    ref_prob,           # torch.Tensor on device
    k: int,
    device,
    target_tile_mb: int = 512,
) -> np.ndarray:
    """GPU (or CPU) PyTorch chunked cosine-KNN — never materialises the full sim matrix.

    Instead of one big (Q × R) matmul we use a double loop:
      outer : query chunks of size Q_chunk
      inner : reference chunks of size R_chunk
    Per-iteration allocation = Q_chunk × R_chunk × 4 bytes ≤ target_tile_mb.
    A running top-K is maintained on-device across reference chunks so the
    result is identical to the brute-force version.
    """
    import torch
    R = ref_norm.shape[0]
    D = ref_norm.shape[1]

    # Choose chunk sizes so one tile fits in target_tile_mb
    # Aim for roughly square tiles; clamp Q_chunk to ≤ 4096 to keep launch overhead low
    tile_floats = (target_tile_mb * 1024 * 1024) // 4
    R_chunk = max(k, min(R, int(tile_floats ** 0.5)))
    Q_chunk = max(1, min(4096, tile_floats // R_chunk))

    n = len(query_emb)
    cf = np.empty(n, dtype=np.float32)

    for q0 in range(0, n, Q_chunk):
        q1 = min(q0 + Q_chunk, n)
        q = torch.from_numpy(query_emb[q0:q1].astype(np.float32)).to(device)
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)   # (Qb, D)

        # Running top-K: scores and indices, initialised to -inf
        top_scores = torch.full((q1 - q0, k), -1.0, device=device)
        top_idx = torch.zeros((q1 - q0, k), dtype=torch.long, device=device)

        for r0 in range(0, R, R_chunk):
            r1 = min(r0 + R_chunk, R)
            sims = q @ ref_norm[r0:r1].T              # (Qb, Rb)

            # Merge with running top-K
            combined_scores = torch.cat(
                [top_scores, sims], dim=1)        # (Qb, k+Rb)
            combined_idx = torch.cat([
                top_idx,
                torch.arange(r0, r1, device=device).unsqueeze(
                    0).expand(q1-q0, -1)
            ], dim=1)
            top_scores, sel = combined_scores.topk(
                k, dim=1, largest=True, sorted=False)
            top_idx = combined_idx.gather(1, sel)

        cf[q0:q1] = ref_prob[top_idx].mean(dim=1).cpu().numpy()

    return cf


def run(slug: str = "carmanah_walbran", knn: int = 10, force_cpu: bool = False) -> None:
    # ── Device selection ────────────────────────────────────────────────
    try:
        import torch
        if not force_cpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU (torch)")
        use_torch = True
    except ImportError:
        use_torch = False
        print("PyTorch not found — using numpy (slower)")

    # ── Load tile arrays ────────────────────────────────────────────────
    emb_tile = np.load(TILE_CACHE / f"{slug}_emb.npy",     mmap_mode="r")
    grid_tile = np.load(TILE_CACHE / f"{slug}_grid.npy",    mmap_mode="r")
    log_tile = np.load(TILE_CACHE / f"{slug}_logging.npy", mmap_mode="r")

    H, W = grid_tile.shape
    log_flat = log_tile.ravel()
    grid_flat = grid_tile.ravel().astype(np.float32)
    emb_flat = emb_tile.reshape(-1, 64)

    mask_logged = np.isin(log_flat, [2, 3, 4])
    mask_unlogged = log_flat == 5
    logged_idx = np.where(mask_logged)[0]
    unlogged_idx = np.where(mask_unlogged)[0]

    print(f"Tile '{slug}': {H}×{W} = {H*W:,} pixels")
    print(
        f"  logged   (cat 2–4) : {mask_logged.sum():,}  ({100*mask_logged.mean():.1f} %)")
    print(
        f"  unlogged (cat  5)  : {mask_unlogged.sum():,}  ({100*mask_unlogged.mean():.1f} %)")

    if mask_unlogged.sum() == 0:
        raise ValueError(
            f"Tile '{slug}' has no unlogged pixels — cannot build reference pool.")

    # ── Normalize reference (unlogged) embeddings ───────────────────────
    uemb = emb_flat[unlogged_idx].astype(np.float32)
    uprob = grid_flat[unlogged_idx]
    unorm_np = uemb / np.linalg.norm(uemb, axis=1, keepdims=True).clip(1e-8)

    print(f"\nReference pool: {len(uemb):,} unlogged pixels, K={knn}")
    print(f"Querying {len(logged_idx):,} logged pixels …")
    t0 = time.time()

    # ── KNN query ───────────────────────────────────────────────────────
    if use_torch:
        import torch
        ref_norm = torch.from_numpy(unorm_np).to(device)
        ref_prob = torch.from_numpy(uprob).to(device)
        # target_tile_mb caps each (Q_chunk × R_chunk) sim-matrix allocation;
        # lower this if you still get OOM (e.g. target_tile_mb=128)
        cf_probs = _cosine_knn_torch(
            emb_flat[logged_idx], ref_norm, ref_prob, knn, device,
            target_tile_mb=512,
        )
    else:
        cf_probs = _cosine_knn_numpy(
            emb_flat[logged_idx], unorm_np, uprob, knn
        )

    print(f"  Done in {time.time()-t0:.1f} s")

    # ── Apply VRI suppression to current grid ────────────────────────────
    # All logged categories (2–4) and water/alpine are zeroed; only
    # old-growth / unlogged forest (cat 5) retains its raw probability.
    LOG_SUPPRESS_ARR = {1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 1.00, 6: 0.00}
    suppress_flat = np.vectorize(LOG_SUPPRESS_ARR.get)(
        log_flat, 0.0).astype(np.float32)
    current_flat = grid_flat * suppress_flat          # masked current
    current = current_flat.reshape(H, W)

    # ── Build historic grid ──────────────────────────────────────────────
    # Logged pixels → counterfactual (unlogged) probability, no suppression.
    # All other pixels keep their (suppressed) current value.
    historic_flat = current_flat.copy()
    historic_flat[logged_idx] = cf_probs
    historic = historic_flat.reshape(H, W)
    diff = (historic - current).astype(np.float32)

    mean_gain = diff[mask_logged.reshape(H, W)].mean()
    print(
        f"\nMean raw prob      (logged pixels) : {grid_flat[logged_idx].mean():.4f}")
    print(
        f"Mean masked current (logged pixels): {current_flat[logged_idx].mean():.4f}")
    print(f"Mean counterfactual (Method C)     : {cf_probs.mean():.4f}")
    print(f"Mean Δp (gain if unlogged)         : {mean_gain:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    LOG_CMAP_COLS = ["#4575b4", "#d73027",
                     "#fc8d59", "#fee090", "#1a9641", "#888888"]
    log_cmap = ListedColormap(LOG_CMAP_COLS)
    log_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], log_cmap.N)
    VMAX = float(max(current.max(), historic.max(), 0.1))

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle(
        f"{slug.replace('_', ' ').title()} — current vs historic yew probability  "
        f"(Method C embed-KNN, K={knn})\n"
        f"Logged pixels: {mask_logged.sum():,}  |  "
        f"Mean Δp on logged pixels = +{mean_gain:.3f}  |  "
        f"Colour scale 0 → {VMAX:.2f}",
        fontsize=11,
    )

    panel_data = [
        (log_tile,  "VRI logging category",
         dict(cmap=log_cmap, norm=log_norm, interpolation="nearest")),
        (current,   "Current (VRI-masked)",
         dict(cmap="RdYlGn", vmin=0, vmax=VMAX, interpolation="nearest")),
        (historic,  "Historic (pre-logging, Emb-KNN)",
         dict(cmap="RdYlGn", vmin=0, vmax=VMAX, interpolation="nearest")),
        (diff,      "Δ probability (historic − current)",
         dict(cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, interpolation="nearest")),
    ]

    for ax, (data, title, kw) in zip(axes, panel_data):
        im = ax.imshow(data, **kw)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        div = make_axes_locatable(ax)
        cax = div.append_axes("bottom", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="horizontal")

    cb0 = axes[0].images[0].colorbar
    cb0.set_ticks([1, 2, 3, 4, 5, 6])
    cb0.set_ticklabels(
        ["water/\nnon-forest", "logged\n<20yr", "logged\n20–40yr",
         "logged\n40–80yr", "forest\n>80yr", "alpine/\nbarren"],
        fontsize=6,
    )

    plt.tight_layout()
    out_path = OUT_DIR / f"{slug}_before_after.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {out_path}")

    # ── Save prediction arrays ───────────────────────────────────────────
    pred_dir = ROOT / "results" / "predictions" / "tile_before_after"
    pred_dir.mkdir(parents=True, exist_ok=True)

    np.save(pred_dir / f"{slug}_current.npy",  current)
    np.save(pred_dir / f"{slug}_historic.npy", historic)
    np.save(pred_dir / f"{slug}_diff.npy",     diff)

    # Per-logged-pixel summary CSV (row, col, current_prob, cf_prob, delta)
    import csv as _csv
    rows_idx, cols_idx = np.unravel_index(logged_idx, (H, W))
    csv_path = pred_dir / f"{slug}_logged_pixels.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(
            ["row", "col", "log_cat", "current_prob", "cf_prob", "delta"])
        for r, c, li, cur, cf, dp in zip(
            rows_idx, cols_idx,
            log_flat[logged_idx],
            current_flat[logged_idx],
            cf_probs,
            diff.ravel()[logged_idx],
        ):
            writer.writerow([int(r), int(c), int(li),
                             round(float(cur), 6), round(float(cf), 6), round(float(dp), 6)])

    print(
        f"Saved current  grid  → {pred_dir / f'{slug}_current.npy'}  {current.shape}")
    print(
        f"Saved historic grid  → {pred_dir / f'{slug}_historic.npy'}  {historic.shape}")
    print(
        f"Saved diff     grid  → {pred_dir / f'{slug}_diff.npy'}  {diff.shape}")
    print(f"Saved logged pixels  → {csv_path}  ({len(logged_idx):,} rows)")

    # ── Area × probability summary ────────────────────────────────────────
    # Each Sentinel-2 pixel = 10 m × 10 m = 0.01 ha.
    # "Index area" = Σ(probability × pixel_area) ≈ expected yew-habitat area
    # assuming uniform yew density within high-probability pixels.
    PIXEL_AREA_HA = 0.01  # 10 m resolution

    hist_index_ha = float(historic.sum()) * PIXEL_AREA_HA
    curr_index_ha = float(current.sum()) * PIXEL_AREA_HA
    delta_index_ha = hist_index_ha - curr_index_ha
    pct_decline = 100.0 * delta_index_ha / \
        hist_index_ha if hist_index_ha > 0 else float("nan")

    # Restrict to logged pixels for the "loss on logged footprint" figures
    curr_logged_ha = float(current_flat[logged_idx].sum()) * PIXEL_AREA_HA
    hist_logged_ha = float(cf_probs.sum()) * PIXEL_AREA_HA
    delta_logged_ha = hist_logged_ha - curr_logged_ha

    print(
        f"\n── Area × probability summary (pixel = {PIXEL_AREA_HA} ha) ──────────────────")
    print(
        f"  Historic yew index area (whole tile) : {hist_index_ha:>10,.1f} ha")
    print(
        f"  Current  yew index area (whole tile) : {curr_index_ha:>10,.1f} ha")
    print(
        f"  Δ index area (historic − current)    : {delta_index_ha:>10,.1f} ha  ({pct_decline:.1f}% decline)")
    print(f"\n  Logged pixels only:")
    print(
        f"    Historic (counterfactual)           : {hist_logged_ha:>10,.1f} ha")
    print(
        f"    Current  (zeroed by VRI mask)       : {curr_logged_ha:>10,.1f} ha")
    print(
        f"    Loss on logged footprint            : {delta_logged_ha:>10,.1f} ha")

    # Save summary JSON alongside arrays
    import json as _json
    summary = {
        "slug": slug,
        "tile_shape": [H, W],
        "pixel_area_ha": PIXEL_AREA_HA,
        "n_logged_pixels": int(mask_logged.sum()),
        "n_unlogged_pixels": int(mask_unlogged.sum()),
        "knn_k": knn,
        "whole_tile": {
            "historic_index_ha": round(hist_index_ha, 2),
            "current_index_ha":  round(curr_index_ha, 2),
            "delta_index_ha":    round(delta_index_ha, 2),
            "pct_decline":       round(pct_decline, 2),
        },
        "logged_pixels": {
            "historic_index_ha": round(hist_logged_ha, 2),
            "current_index_ha":  round(curr_logged_ha, 2),
            "delta_index_ha":    round(delta_logged_ha, 2),
        },
    }
    json_path = pred_dir / f"{slug}_summary.json"
    with open(json_path, "w") as fh:
        _json.dump(summary, fh, indent=2)
    print(f"\nSaved summary JSON   → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pixel-level before/after tile reconstruction.")
    parser.add_argument(
        "--slug", default="carmanah_walbran",
        help="Tile slug (default: carmanah_walbran). Run with --list to see all available tiles."
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Number of KNN neighbours (default: 10)")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if a GPU is available."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available tile slugs and exit."
    )
    args = parser.parse_args()

    if args.list:
        slugs = sorted(p.stem.replace("_grid", "")
                       for p in TILE_CACHE.glob("*_grid.npy"))
        print("Available tiles:")
        for s in slugs:
            log_path = TILE_CACHE / f"{s}_logging.npy"
            if log_path.exists():
                log = np.load(log_path, mmap_mode="r").ravel()
                n_logged = int(np.isin(log, [2, 3, 4]).sum())
                n_unlogged = int((log == 5).sum())
                print(
                    f"  {s:<35}  logged={n_logged:>7,}  unlogged={n_unlogged:>7,}")
            else:
                print(f"  {s}")
    else:
        run(slug=args.slug, knn=args.k, force_cpu=args.cpu)
