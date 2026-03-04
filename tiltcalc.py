#!/usr/bin/env python3
"""
tiltcalc.py
Compute octahedral tilt descriptors for 2D organic-inorganic metal-halide perovskites from CIF files.

The script reports:
  θ — M–X–M (metal–halide–metal) bridging angle
  φ — X–X–X (halide–halide–halide) rhomboid angle (~150°)
These angles serve as geometric descriptors of octahedral tilting.

The script automatically detects metal and halide species in the structure,
but users can override this behaviour via command-line options.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pymatgen.core import Structure


# ==============================
# Geometry helpers
# ==============================

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n != 0 else np.zeros_like(v)

def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    c = float(np.clip(np.dot(unit(v1), unit(v2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def min_image_cart_vec(struct: Structure, frac_from: np.ndarray, frac_to: np.ndarray) -> np.ndarray:
    """Cartesian vector from frac_from to nearest periodic image of frac_to."""
    _, img = struct.lattice.get_distance_and_image(frac_from, frac_to)
    frac_to_img = frac_to + np.array(img, dtype=float)
    return (struct.lattice.get_cartesian_coords(frac_to_img)
            - struct.lattice.get_cartesian_coords(frac_from))

def min_image_dist(struct: Structure, frac_a: np.ndarray, frac_b: np.ndarray) -> float:
    return float(struct.lattice.get_distance_and_image(frac_a, frac_b)[0])


# ==============================
# Atom detection (heuristics + overrides)
# ==============================

DEFAULT_HALIDES = {"Cl", "Br", "I"}
DEFAULT_ORGANIC = {"H", "C", "N", "O"}  # organic cations in organic-inorganic perovskites

def site_has_any_element(site, symbols: Iterable[str]) -> bool:
    syms = set(symbols)
    return any(el.symbol in syms for el in site.species.elements)

def detect_halide_symbols(struct: Structure,
                          halides_override: Optional[Sequence[str]] = None) -> List[str]:
    if halides_override and len(halides_override) > 0:
        return [h.strip() for h in halides_override]

    # Auto: only halides present in structure, intersect with DEFAULT_HALIDES
    present = {el.symbol for el in struct.composition.elements}
    halides = sorted(list(present.intersection(DEFAULT_HALIDES)))
    return halides

def detect_metal_symbols(struct: Structure,
                         halide_symbols: Sequence[str],
                         metals_override: Optional[Sequence[str]] = None,
                         organic_symbols: Optional[Sequence[str]] = None) -> List[str]:
    """
    Auto metal detection heuristic:
      Metals = elements that are not in (organic_symbols ∪ halides)
    Works well for organic-inorganic halide perovskites; users can override.
    """
    if metals_override and len(metals_override) > 0:
        return [m.strip() for m in metals_override]

    organic = set(organic_symbols) if organic_symbols else set(DEFAULT_ORGANIC)
    halides = set(halide_symbols)

    present = {el.symbol for el in struct.composition.elements}
    metals = sorted([sym for sym in present if sym not in organic and sym not in halides])

    return metals


# ==============================
# θ from M–X–M bridging angles
# ==============================

def enumerate_mxb_bridges(struct: Structure,
                          metal_symbols: Sequence[str],
                          halide_symbols: Sequence[str],
                          m_x_cutoff: float = 3.2) -> List[dict]:
    """
    Enumerate all M–X–M angles where X is within cutoff of two metals.
    Uses minimum-image vectors in Cartesian (correct for non-orthogonal cells).
    Works with mixed-occupancy sites.
    """
    metal_set = set(metal_symbols)
    halide_set = set(halide_symbols)

    metal_idx = [
        i for i, s in enumerate(struct)
        if any(el.symbol in metal_set for el in s.species.elements)
    ]

    x_idx = [
        i for i, s in enumerate(struct)
        if any(el.symbol in halide_set for el in s.species.elements)
    ]

    frac = np.array([s.frac_coords for s in struct], dtype=float)
    rows: List[dict] = []

    for xi in x_idx:
        near_metals: List[Tuple[int, float]] = []

        for mi in metal_idx:
            d = min_image_dist(struct, frac[mi], frac[xi])
            if d <= m_x_cutoff:
                near_metals.append((mi, d))

        if len(near_metals) < 2:
            continue

        for a in range(len(near_metals)):
            for b in range(a + 1, len(near_metals)):
                m1, d1 = near_metals[a]
                m2, d2 = near_metals[b]

                v1 = min_image_cart_vec(struct, frac[xi], frac[m1])  # X -> M1
                v2 = min_image_cart_vec(struct, frac[xi], frac[m2])  # X -> M2
                ang = angle_deg(v1, v2)

                rows.append({
                    "m1": m1, "x": xi, "m2": m2,
                    "d1": d1, "d2": d2,
                    "angle": ang
                })

    return rows

def select_theta_from_bridges(rows: List[dict], target: float = 150.0) -> Optional[float]:
    if not rows:
        return None

    angles = np.array([r["angle"] for r in rows], dtype=float)

    # If symmetry-equivalent, return mean
    if float(angles.max() - angles.min()) < 1e-3:
        return float(angles.mean())

    # Otherwise choose the candidate closest to expected target
    best = min(rows, key=lambda r: abs(r["angle"] - target))
    return float(best["angle"])

def print_mxb_summary(struct: Structure, rows: List[dict], top: int = 20) -> None:
    print(f"Found {len(rows)} M–X–M bridging candidates\n")
    if not rows:
        print("⚠ None found. Try increasing --mxb-cutoff slightly.")
        return

    rows_sorted = sorted(rows, key=lambda r: abs(r["angle"] - 180))

    print(" idx(M1) idx(X) idx(M2) |  d(M1-X)  d(M2-X) |  angle (deg)")
    print("-" * 70)

    for r in rows_sorted[:top]:
        print(f"{r['m1']:7d} {r['x']:6d} {r['m2']:7d} |"
              f"   {r['d1']:.3f}   {r['d2']:.3f} |   {r['angle']:.2f}")


# ==============================
# φ from X–X–X (halide rhomboid proxy)
# ==============================

def enumerate_xxx_triplets(struct: Structure,
                           halide_symbols: Sequence[str],
                           xx_cutoff: float = 4.25) -> List[dict]:
    """
    Enumerate unique X–X–X triplets (X2 central) where X1 and X3 are neighbors of X2.
    """
    halide_set = set(halide_symbols)

    x_idx = [
        i for i, site in enumerate(struct)
        if any(el.symbol in halide_set for el in site.species.elements)
    ]

    frac = np.array([site.frac_coords for site in struct], dtype=float)

    neigh: Dict[int, List[int]] = {i: [] for i in x_idx}

    for ai, a in enumerate(x_idx):
        for b in x_idx[ai + 1:]:
            d = min_image_dist(struct, frac[a], frac[b])
            if d <= xx_cutoff:
                neigh[a].append(b)
                neigh[b].append(a)

    triplets: List[dict] = []

    for x2 in x_idx:
        nbs = neigh[x2]
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                x1, x3 = nbs[i], nbs[j]

                v1 = min_image_cart_vec(struct, frac[x2], frac[x1])  # X2 -> X1
                v2 = min_image_cart_vec(struct, frac[x2], frac[x3])  # X2 -> X3
                ang = angle_deg(v1, v2)

                d21 = min_image_dist(struct, frac[x2], frac[x1])
                d23 = min_image_dist(struct, frac[x2], frac[x3])

                key = (min(x1, x3), x2, max(x1, x3))  # dedupe X1/X3 order
                triplets.append({
                    "key": key,
                    "x1": x1, "x2": x2, "x3": x3,
                    "angle": ang,
                    "d21": d21, "d23": d23
                })

    uniq: Dict[Tuple[int, int, int], dict] = {}
    for t in triplets:
        uniq.setdefault(t["key"], t)

    return list(uniq.values())

def mc_angle_for_triplet(struct: Structure,
                         x1: int, x2: int, x3: int,
                         sigma_frac: np.ndarray,
                         N: int = 10000,
                         seed: int = 0) -> Tuple[float, float]:
    """
    Monte Carlo propagation: perturb fractional coordinates by sigma_frac (3-vector),
    compute angle distribution, return (mean, std).
    """
    frac0 = np.array([site.frac_coords for site in struct], dtype=float)
    rng = np.random.default_rng(seed)
    vals = np.empty(N, dtype=float)

    for k in range(N):
        f1 = (frac0[x1] + rng.normal(0, sigma_frac, size=3)) % 1.0
        f2 = (frac0[x2] + rng.normal(0, sigma_frac, size=3)) % 1.0
        f3 = (frac0[x3] + rng.normal(0, sigma_frac, size=3)) % 1.0

        v1 = min_image_cart_vec(struct, f2, f1)  # X2 -> X1
        v2 = min_image_cart_vec(struct, f2, f3)  # X2 -> X3
        vals[k] = angle_deg(v1, v2)

    return float(vals.mean()), float(vals.std(ddof=1))

def select_xxx_obtuse(struct: Structure,
                      triplets: List[dict],
                      sigma_frac: np.ndarray,
                      target: float = 150.0,
                      N: int = 10000,
                      seed: int = 0,
                      dmin: float = 3.9,
                      dmax: float = 4.35) -> Tuple[float, float, dict]:
    """
    Select the X–X–X triplet whose MC mean is closest to target (~150°),
    optionally filtered by a distance window (dmin..dmax) for both X–X edges.
    """
    candidates = [t for t in triplets if (dmin <= t["d21"] <= dmax and dmin <= t["d23"] <= dmax)]
    if not candidates:
        candidates = triplets  # fallback if filter too strict

    best: Optional[Tuple[float, float, dict]] = None
    best_score = np.inf

    for t in candidates:
        mean, std = mc_angle_for_triplet(struct, t["x1"], t["x2"], t["x3"],
                                         sigma_frac=sigma_frac, N=N, seed=seed)
        score = abs(mean - target)
        if score < best_score:
            best_score = score
            best = (mean, std, t)

    if best is None:
        raise RuntimeError("No X–X–X triplets found.")

    return best


# ==============================
# Command-line interface
# ==============================

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute octahedral tilt descriptors from CIF: θ (M–X–M) and φ (X–X–X)."
    )
    p.add_argument("cif", help="Input CIF file path")

    p.add_argument("--halides", nargs="*", default=None,
                   help="Halide symbols to consider (default: auto-detect among Cl/Br/I). Example: --halides Br I")
    p.add_argument("--metals", nargs="*", default=None,
                   help="Metal symbols to consider (default: auto-detect as non-(organic,halide) elements). Example: --metals Pb Sn")
    p.add_argument("--organic", nargs="*", default=list(DEFAULT_ORGANIC),
                   help="Elements treated as organic (ignored for metal detection). Default: H C N O")

    p.add_argument("--mxb-cutoff", type=float, default=3.2,
                   help="Metal–halide distance cutoff (Å) for identifying bridging X. Default: 3.2")
    p.add_argument("--xx-cutoff", type=float, default=4.25,
                   help="Halide–halide cutoff (Å) for building X–X neighbor list. Default: 4.25")

    p.add_argument("--target-theta", type=float, default=150.0,
                   help="Target θ angle (deg) used if multiple M–X–M candidates exist. Default: 150")
    p.add_argument("--target-phi", type=float, default=150.0,
                   help="Target φ angle (deg) for selecting obtuse X–X–X. Default: 150")

    p.add_argument("--sigma-frac", nargs=3, type=float, default=[1e-4, 1e-4, 1e-4],
                   help="Fractional-coordinate sigma for MC on X–X–X angle (x y z). Default: 1e-4 1e-4 1e-4")
    p.add_argument("--mc", type=int, default=10000,
                   help="Monte Carlo samples. Default: 10000")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for MC. Default: 0")

    p.add_argument("--dmin", type=float, default=3.9, help="Min X–X distance (Å) filter for rhomboid selection. Default: 3.9")
    p.add_argument("--dmax", type=float, default=4.35, help="Max X–X distance (Å) filter for rhomboid selection. Default: 4.35")

    p.add_argument("--verbose", action="store_true", help="Print candidate tables.")
    return p.parse_args(list(argv))

def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    struct = Structure.from_file(args.cif)

    halides = detect_halide_symbols(struct, args.halides)
    metals = detect_metal_symbols(struct, halides, args.metals, args.organic)

    if len(halides) == 0:
        print("⚠ No halides detected. Use --halides to specify (e.g., --halides Br).")
        return 2
    if len(metals) == 0:
        print("⚠ No metals detected by heuristic. Use --metals to specify (e.g., --metals Pb Sn).")
        return 2

    sigma_frac = np.array(args.sigma_frac, dtype=float)

    print(f"\n=== {args.cif} ===\n")
    print(f"Detected halides: {halides}")
    print(f"Detected metals:  {metals}\n")

    # θ from M–X–M
    mxb_rows = enumerate_mxb_bridges(struct, metal_symbols=metals, halide_symbols=halides, m_x_cutoff=args.mxb_cutoff)
    if args.verbose:
        print_mxb_summary(struct, mxb_rows, top=30)

    theta = select_theta_from_bridges(mxb_rows, target=args.target_theta)
    if theta is None:
        print("θ (M–X–M): not found (no bridging candidates). Try increasing --mxb-cutoff or specify --metals/--halides.")
    else:
        print(f"θ (M–X–M) = {theta:.2f}°")

    # φ from X–X–X (obtuse near ~150°)
    triplets = enumerate_xxx_triplets(struct, halide_symbols=halides, xx_cutoff=args.xx_cutoff)
    print(f"\nFound {len(triplets)} unique X–X–X triplets with X–X cutoff {args.xx_cutoff:.2f} Å")

    phi_mean, phi_std, t = select_xxx_obtuse(
        struct, triplets,
        sigma_frac=sigma_frac,
        target=args.target_phi,
        N=args.mc,
        seed=args.seed,
        dmin=args.dmin,
        dmax=args.dmax
    )

    print(f"φ (X–X–X, obtuse ~{args.target_phi:.0f}°) = {phi_mean:.2f} ± {phi_std:.2f}°")
    print(f"Triplet indices: {t['x1']}-{t['x2']}-{t['x3']} (d={t['d21']:.3f}, {t['d23']:.3f} Å)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
