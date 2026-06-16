"""One-shot CLI: validate our SurfaceTexture pipeline against Mountains.

Walks ``35mm Sleeve Database/`` (or any folder layout with the same
``<part>/<measurement>.pro`` + ``<part>/*.csv`` shape), runs each
``.pro`` through :func:`batch_report.run_mountains_equivalent`, joins
the results with the matching Mountains template CSV on the studiable
name, and writes a single Excel comparison workbook.

Usage::

    python batch_validate.py
    python batch_validate.py --root "35mm Sleeve Database" --output out.xlsx
    python batch_validate.py --cutoff-in 0.03 --cref 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from batch_report import (
    PARAM_ORDER,
    build_comparison_workbook,
    parse_mountains_csv,
    run_mountains_equivalent,
)


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = WORKSPACE_ROOT / "35mm Sleeve Database"


def _find_mountains_csv(folder: Path) -> Path | None:
    """Return the first non-hidden CSV in ``folder`` or ``None``."""
    for cand in sorted(folder.glob("*.csv")):
        if not cand.name.startswith("."):
            return cand
    return None


def _gather_part_pros(root: Path) -> dict[str, list[Path]]:
    """Group ``.pro`` files by their immediate parent folder name."""
    parts: dict[str, list[Path]] = {}
    for pro in sorted(root.rglob("*.pro")):
        parts.setdefault(pro.parent.name, []).append(pro)
    # Sort each part's measurements by stem so the workbook ordering is
    # stable (9_1, 9_2, …).
    for name in parts:
        parts[name].sort(key=lambda p: p.stem)
    return parts


def _row_for_pro(
    pro_path: Path,
    mt_df: pd.DataFrame | None,
    *,
    lambda_c_in: float,
    cref_pct: float,
) -> dict:
    """Run our pipeline on ``pro_path`` and assemble a comparison row."""
    result = run_mountains_equivalent(
        pro_path, lambda_c_in=lambda_c_in, cref_pct=cref_pct
    )
    row: dict = {
        "part": result.part,
        "measurement": result.measurement,
        "file": str(result.file),
        "ours": dict(result.params),
        "ours_units": dict(result.units),
        "mountains": {},
        "units": {},
        "error": result.error,
    }
    if mt_df is not None and result.measurement in mt_df.index:
        mt_row = mt_df.loc[result.measurement]
        for param in PARAM_ORDER:
            if param in mt_df.columns:
                try:
                    row["mountains"][param] = float(mt_row[param])
                except (TypeError, ValueError):
                    row["mountains"][param] = float("nan")
        # Mountains units come back from parse_mountains_csv() in the
        # caller; we only need a mapping for the long format.
    return row


def _summarise_part(part: str, rows: list[dict]) -> str:
    """Return a 1-line stdout summary for ``part`` (mean %Δ Ra and Rz)."""
    def _mean_pct(param: str) -> float:
        diffs = []
        for r in rows:
            ours = r.get("ours", {}).get(param)
            mt = r.get("mountains", {}).get(param)
            if ours is None or mt is None or not mt or np.isnan(ours) or np.isnan(mt):
                continue
            diffs.append(100.0 * abs(ours - mt) / abs(mt))
        return float(np.mean(diffs)) if diffs else float("nan")

    ra = _mean_pct("Ra")
    rz = _mean_pct("Rz")
    return f"  part {part:>4}: n={len(rows):>2}  mean |%Δ| Ra={ra:6.2f}%  Rz={rz:6.2f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Database root containing per-part subfolders (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Excel workbook (default: <root>/batch_validation.xlsx).",
    )
    parser.add_argument(
        "--cutoff-in",
        type=float,
        default=0.03,
        help="Gaussian L-filter cutoff in inches (matches Mountains template).",
    )
    parser.add_argument(
        "--cref",
        type=float,
        default=5.0,
        help="Material-ratio reference Cref (percent).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: root folder not found: {root}", file=sys.stderr)
        return 2
    output = (args.output or (root / "batch_validation.xlsx")).resolve()

    parts = _gather_part_pros(root)
    if not parts:
        print(f"error: no .pro files found under {root}", file=sys.stderr)
        return 1

    print(f"Validating {sum(len(v) for v in parts.values())} .pro files "
          f"across {len(parts)} parts. Mountains λc = {args.cutoff_in} in, "
          f"Cref = {args.cref}%.")

    rows: list[dict] = []
    for part_name, pros in sorted(parts.items(), key=lambda kv: _safe_int(kv[0])):
        part_folder = pros[0].parent
        csv_path = _find_mountains_csv(part_folder)
        mt_df: pd.DataFrame | None = None
        if csv_path is not None:
            try:
                mt_df, _units, _mt_lc = parse_mountains_csv(csv_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  part {part_name}: WARN failed to parse {csv_path.name}: {exc}",
                      file=sys.stderr)
        else:
            print(f"  part {part_name}: WARN no Mountains CSV in {part_folder}",
                  file=sys.stderr)

        part_rows: list[dict] = []
        for pro in pros:
            part_rows.append(_row_for_pro(
                pro, mt_df,
                lambda_c_in=args.cutoff_in,
                cref_pct=args.cref,
            ))
        rows.extend(part_rows)
        print(_summarise_part(part_name, part_rows))

    settings_dump = {
        "root": str(root),
        "output": str(output),
        "lambda_c_in": args.cutoff_in,
        "cref_pct": args.cref,
        "filter": "Gaussian L-filter only (ISO 4287 path), no S-filter",
        "n_parts": len(parts),
        "n_files": sum(len(v) for v in parts.values()),
    }

    out_path = build_comparison_workbook(
        rows, output, settings=settings_dump, mountains_present=True
    )
    print(f"\nWorkbook written to: {out_path}")
    return 0


def _safe_int(s: str):
    try:
        return (0, int(s))
    except (TypeError, ValueError):
        return (1, str(s))


if __name__ == "__main__":
    raise SystemExit(main())
