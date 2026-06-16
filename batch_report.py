"""Shared helpers for batch validation and GUI batch xlsx export.

This module contains three pieces:

* :func:`parse_mountains_csv` — read a Digital Surf Mountains "Studies"
  CSV (the kind written by the user's ``TH_Template_10_2025`` template)
  into a tidy ``pandas.DataFrame`` indexed by the studiable name.
* :func:`run_mountains_equivalent` — drive :class:`SurfaceTexture` with
  Mountains-equivalent filter settings (Gaussian L-filter only, no S-
  filter) and return a parameter dict that can be diffed against the
  parser output.
* :func:`build_comparison_workbook` — write the per-measurement /
  per-part / settings sheets for either flow (CLI validation = with
  Mountains columns, GUI batch export = without).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Order in which we lay out parameter columns. These are the parameters
# :class:`SurfaceTexture` currently reports.
PARAM_ORDER: tuple[str, ...] = (
    "Ra", "Rq", "Rp", "Rv", "Rz", "Rt", "Rsk", "Rku", "Rmr",
)


# ---------------------------------------------------------------------------
# Mountains CSV parser
# ---------------------------------------------------------------------------

# Regex strips Mountains' parenthetical filter description from a column
# title so ``"Ra (Gaussian filter  0.03 in)"`` becomes ``"Ra"``.
_PAREN_RE = re.compile(r"\s*\(.*\)\s*$")
# Pulls a numeric cutoff in inches out of any title like
# ``"... Gaussian filter  0.03 in ..."``.
_CUTOFF_IN_RE = re.compile(r"([\d.]+)\s*in\b", re.IGNORECASE)


def _clean_title(raw: str) -> str:
    """Strip the parenthetical filter description from a Mountains column."""
    return _PAREN_RE.sub("", str(raw)).strip()


def parse_mountains_csv(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, str], float | None]:
    """Read a Mountains parameters-summary CSV.

    The expected layout (from inspection of the user's template export):
      * Row 0: numeric column IDs prefixed with ``#``.
      * Row 1: human-readable column titles, e.g.
        ``"Ra (Gaussian filter  0.03 in)"``.
      * Row 2: unit row (``"µin"``, ``"%"``, ``"in"`` …).
      * Rows 3..n: one data row per measurement. The studiable name
        (matching the ``.pro`` file stem) lives in the
        ``"Name of the studiable"`` column.
      * The file is encoded in ``cp1252`` (the µ glyph is 0xB5).
      * The trailing column is empty.

    Returns
    -------
    df : pandas.DataFrame
        Indexed by studiable name (``9_1``, ``9_2`` …) with one column
        per stripped title plus ``Length``. Numeric columns are cast to
        ``float``.
    units : dict[str, str]
        Cleaned title → unit token from row 2.
    lambda_c_in : float or None
        The first numeric inch cutoff scraped from the column titles
        (Mountains' ``"Gaussian filter 0.03 in"``), or ``None`` if not
        present.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, encoding="cp1252")

    # Drop columns that are entirely NaN (the trailing blank column).
    raw = raw.dropna(axis=1, how="all")

    titles_row = raw.iloc[1].tolist()
    units_row = raw.iloc[2].tolist()

    columns: list[str] = []
    for v in titles_row:
        text = str(v).strip()
        columns.append(_clean_title(text) if text != "#" else f"_meta_{len(columns)}")

    units: dict[str, str] = {}
    for col, u in zip(columns, units_row):
        if str(u).strip() not in {"#", "<no unit>", "nan"}:
            units[col] = str(u).strip()

    data = raw.iloc[3:].copy()
    data.columns = columns
    # Coerce parameter columns numerically; if a column has any non-numeric
    # entries (e.g. the studiable name) leave it untouched.
    for col in data.columns:
        if col.startswith("_meta_"):
            continue
        coerced = pd.to_numeric(data[col], errors="coerce")
        if coerced.notna().sum() >= max(1, int(0.5 * len(data))):
            data[col] = coerced

    if "Name of the studiable" not in data.columns:
        raise ValueError(
            f"CSV {path} does not contain a 'Name of the studiable' column."
        )
    data = data.set_index("Name of the studiable")
    # Drop the metadata banner columns (date / time / source path).
    data = data[[c for c in data.columns if not c.startswith("_meta_")]]

    # Scrape λc from any title containing a Gaussian filter inch cutoff.
    lambda_c_in: float | None = None
    for title in titles_row:
        m = _CUTOFF_IN_RE.search(str(title))
        if m:
            try:
                lambda_c_in = float(m.group(1))
                break
            except ValueError:
                pass

    return data, units, lambda_c_in


# ---------------------------------------------------------------------------
# Mountains-equivalent runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Outcome of running our pipeline on a single .pro file."""

    file: Path
    part: str
    measurement: str
    params: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    lambda_c: float = 0.0
    lambda_c_unit: str = ""
    error: str | None = None


def run_mountains_equivalent(
    pro_path: str | Path,
    *,
    lambda_c_in: float = 0.03,
    cref_pct: float = 5.0,
) -> RunResult:
    """Run :class:`SurfaceTexture` with Mountains-equivalent filter settings.

    Mountains' template uses a Gaussian L-filter only (no S-filter) at
    λc = 0.03 in (≈ 0.762 mm). Our :meth:`SurfaceTexture.compute_comparison_params`
    already populates an ``ISO 4287`` entry built from a Gaussian L-filter
    only, so we lean on that path and just choose the cutoff.

    Reports λc + parameters in **inches / µin** to match the Mountains
    output cell-for-cell.
    """
    # Local import keeps this module importable when SurfaceTexture's heavy
    # plotting dependencies are unavailable (e.g. in a headless CI env).
    from SurfaceTexture import SurfaceTexture

    pro_path = Path(pro_path)
    part = pro_path.parent.name
    measurement = pro_path.stem

    try:
        # We use compute_comparison_params()'s "ISO 4287" path which
        # applies only an L-filter (λc), so the value of short_cutoff
        # does not influence the numbers we read out. We still have to
        # pass *some* positive value so SurfaceTexture's constructor
        # can run — λc/1000 is small enough to be a no-op for the
        # ISO 21920 path it also computes internally.
        st = SurfaceTexture(
            str(pro_path),
            short_cutoff=lambda_c_in / 1000.0,
            long_cutoff=lambda_c_in,
            order=1,
            x_units="in",
            y_units="\u03bcin",
            setting_class=None,
        )
        st.compute_comparison_params(Cref=cref_pct)
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            file=pro_path, part=part, measurement=measurement, error=str(exc)
        )

    iso4287 = st.comparison_params.get("ISO 4287", {}) if hasattr(st, "comparison_params") else {}

    params: dict[str, float] = {}
    units: dict[str, str] = {}
    for name in PARAM_ORDER:
        if name in iso4287:
            value, unit = iso4287[name]
            params[name] = float(value)
            units[name] = str(unit)
    return RunResult(
        file=pro_path,
        part=part,
        measurement=measurement,
        params=params,
        units=units,
        lambda_c=float(st.long_cutoff),
        lambda_c_unit=str(st.x_units),
    )


# ---------------------------------------------------------------------------
# Comparison workbook writer
# ---------------------------------------------------------------------------

def _stats_block(values: pd.Series) -> dict[str, float]:
    """Return mean / stddev / min / max of a 1-D numeric Series."""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return {"mean": np.nan, "stddev": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(s.mean()),
        "stddev": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _group_by_part(
    rows: Iterable[dict], *, location_present: bool,
) -> list[tuple[tuple[str, str], list[dict]]]:
    """Group rows by Part (and Location, when present) in sort order."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["part"], (r.get("location") or "") if location_present else "")
        groups.setdefault(key, []).append(r)
    return sorted(
        groups.items(),
        key=lambda kv: (_sort_key(kv[0][0]), kv[0][1].lower()),
    )


def _row_unit_for(rows: Iterable[dict], param: str) -> str | None:
    """Pick the first unit reported for ``param`` across rows."""
    for r in rows:
        u = (r.get("ours_units") or {}).get(param) or (r.get("units") or {}).get(param)
        if u:
            return u
    return None


def _summary_sheet(
    rows: list[dict], *, location_present: bool,
) -> pd.DataFrame:
    """Per-part summary: rows = (Part, [Location], n), columns = Param means.

    A trailing 'Units' row exposes the unit each parameter was reported
    in, so the table reads top-to-bottom without needing a separate sheet.
    """
    grouped = _group_by_part(rows, location_present=location_present)
    records: list[dict] = []
    for (part, loc), part_rows in grouped:
        rec: dict = {"Part": part}
        if location_present:
            rec["Location"] = loc
        rec["n"] = len(part_rows)
        for param in PARAM_ORDER:
            series = pd.Series([r.get("ours", {}).get(param) for r in part_rows])
            rec[param] = _stats_block(series)["mean"]
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    units_row: dict = {"Part": "Units"}
    if location_present:
        units_row["Location"] = ""
    units_row["n"] = ""
    for param in PARAM_ORDER:
        units_row[param] = _row_unit_for(rows, param) or ""
    df = pd.concat([df, pd.DataFrame([units_row])], ignore_index=True)
    return df


def _stats_sheet(
    rows: list[dict], *, location_present: bool,
) -> pd.DataFrame:
    """Per-part stats with flattened ``"<Param> <stat>"`` column headers.

    The grid is wide: rows are Parts (and Locations); columns are
    ``Ra mean / Ra stddev / Ra min / Ra max / Rq mean / …`` so a single
    Excel sheet shows every parameter's full distribution.
    """
    grouped = _group_by_part(rows, location_present=location_present)
    if not grouped:
        return pd.DataFrame()

    stat_names = ("mean", "stddev", "min", "max")
    records: list[dict] = []
    for (part, loc), part_rows in grouped:
        n = len(part_rows)
        rec: dict = {"Part": part}
        if location_present:
            rec["Location"] = loc
        rec["n"] = n
        for param in PARAM_ORDER:
            series = pd.Series([r.get("ours", {}).get(param) for r in part_rows])
            stats = _stats_block(series)
            for s in stat_names:
                rec[f"{param} {s}"] = stats[s]
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _per_measurement_wide(
    rows: list[dict], *, location_present: bool,
) -> pd.DataFrame:
    """Per-measurement values in wide format.

    Rows are individual measurements; columns are Part, [Location],
    Measurement, then one column per parameter holding the measured value.
    """
    records: list[dict] = []
    for r in rows:
        rec: dict = {"Part": r["part"]}
        if location_present:
            rec["Location"] = r.get("location", "") or ""
        rec["Measurement"] = r["measurement"]
        for param in PARAM_ORDER:
            rec[param] = r.get("ours", {}).get(param)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _mountains_comparison(
    rows: list[dict], *, location_present: bool,
) -> pd.DataFrame:
    """Long-format Mountains-vs-result diff sheet (validation flow only).

    Rows: one per (measurement, param). Columns: Part, [Location],
    Measurement, Param, Unit, Mountains, Result, Δ, %Δ.
    """
    records: list[dict] = []
    for row in rows:
        for param in PARAM_ORDER:
            our_val = row.get("ours", {}).get(param)
            mt_val = row.get("mountains", {}).get(param)
            unit = (row.get("units") or {}).get(param) or (row.get("ours_units") or {}).get(param)
            delta = abs_pct = None
            if our_val is not None and mt_val is not None and not (
                np.isnan(our_val) or np.isnan(mt_val)
            ):
                delta = our_val - mt_val
                abs_pct = (
                    100.0 * delta / mt_val if mt_val != 0 else np.nan
                )
            rec: dict = {"Part": row["part"]}
            if location_present:
                rec["Location"] = row.get("location", "") or ""
            rec.update({
                "Measurement": row["measurement"],
                "Param": param,
                "Unit": unit,
                "Mountains": mt_val,
                "Result": our_val,
                "Δ": delta,
                "%Δ": abs_pct,
            })
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _sort_key(part: str):
    """Sort parts numerically when their names are integers, else lexically.

    ``"(root)"`` always sorts first so root-level files appear at the top
    of summary sheets.
    """
    if part == "(root)":
        return (0, 0, "")
    try:
        return (1, int(part), "")
    except (TypeError, ValueError):
        return (2, 0, str(part).lower())


def build_comparison_workbook(
    rows: list[dict],
    output_path: str | Path,
    *,
    settings: dict | None = None,
    mountains_present: bool = True,
) -> Path:
    """Write the batch-summary / validation workbook.

    ``rows`` is a list of dicts with keys:
      * ``part``       : str — folder / part identifier.
      * ``location``   : str — sub-folder identifier (optional).
      * ``measurement``: str — measurement label (e.g. ``9_9_1``).
      * ``file``       : str — original file path (recorded for audit).
      * ``ours``       : dict[str, float] — measured R-parameter values.
      * ``ours_units`` : dict[str, str] — units the values are in.
      * ``mountains``  : dict[str, float] | None — Mountains reference values.
      * ``units``      : dict[str, str] | None — Mountains units.
      * ``error``      : str | None — failure reason, if any.

    Sheets emitted (in order):
      1. ``Summary``           — Part × Param means; quick scan view.
      2. ``Stats``             — Part × ``"<Param> <stat>"`` columns.
      3. ``Per-measurement``   — every measurement's values, wide.
      4. ``Mountains comparison`` — long Δ/%Δ table (validation only).
      5. ``Settings``          — audit dump of run settings.
      6. ``Failures``          — files that errored during the run.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    valid_rows = [r for r in rows if not r.get("error")]
    failed_rows = [r for r in rows if r.get("error")]

    location_present = any((r.get("location") or "") for r in valid_rows)

    summary_df = _summary_sheet(valid_rows, location_present=location_present)
    stats_df = _stats_sheet(valid_rows, location_present=location_present)
    per_meas_df = _per_measurement_wide(valid_rows, location_present=location_present)
    if mountains_present and any(r.get("mountains") for r in valid_rows):
        comparison_df = _mountains_comparison(valid_rows, location_present=location_present)
    else:
        comparison_df = None

    settings_records: list[dict] = []
    if settings:
        for k, v in settings.items():
            settings_records.append({"Key": k, "Value": v})
    if failed_rows:
        settings_records.append({"Key": "_failures", "Value": len(failed_rows)})

    failed_df = pd.DataFrame(
        [{"Part": r.get("part"), "Measurement": r.get("measurement"),
          "File": str(r.get("file")), "Error": r.get("error")}
         for r in failed_rows]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        stats_df.to_excel(writer, sheet_name="Stats", index=False)
        per_meas_df.to_excel(writer, sheet_name="Per-measurement", index=False)
        if comparison_df is not None and not comparison_df.empty:
            comparison_df.to_excel(
                writer, sheet_name="Mountains comparison", index=False,
            )
        if settings_records:
            pd.DataFrame(settings_records).to_excel(
                writer, sheet_name="Settings", index=False,
            )
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="Failures", index=False)
    return output_path
