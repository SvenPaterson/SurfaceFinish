"""Reader for Digital Surf ``.pro`` profile files.

These are 1D profile studiables sharing the same on-disk layout as ``.sur``
surface files (magic ``DIGITAL SURF`` / ``DSCOMPRESSED``). The parser delegates
to :mod:`surfalize.file.sur` for the binary header / data block and adds:

* validation that the studiable is in fact a 1D profile,
* canonical-unit X/Z conversion (Metric mm/μm or Standard in/μin),
* a lightweight ``peek_pro_units`` helper for GUI auto-population.

Surfalize itself does not register ``.pro`` as a loadable suffix and only
exposes 2D surfaces via its public ``Surface`` API, hence this thin wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from surfalize.file.sur import (
    StudiableType,
    read_sur_header,
    read_sur_object,
)


# Map free-form unit strings found in DIGITAL SURF headers to the canonical
# unit names accepted elsewhere in this toolkit.
_UNIT_ALIASES = {
    "inch": "in",
    "inches": "in",
    "in": "in",
    "mm": "mm",
    "millimeter": "mm",
    "millimetre": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimetre": "cm",
    "m": "m",
    "meter": "m",
    "metre": "m",
    "um": "\u03bcm",      # μm
    "\u00b5m": "\u03bcm",  # µm (latin-1 micro)
    "\u03bcm": "\u03bcm",  # μm (Greek mu)
    "micrometer": "\u03bcm",
    "micrometre": "\u03bcm",
    "nm": "nm",
    "nanometer": "nm",
    "nanometre": "nm",
    "uin": "\u03bcin",
    "\u00b5in": "\u03bcin",
    "\u03bcin": "\u03bcin",
    "microinch": "\u03bcin",
    "microinches": "\u03bcin",
}

# Length of one of these units expressed in millimetres.
_MM_PER_UNIT = {
    "in": 25.4,
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "\u03bcm": 1e-3,
    "nm": 1e-6,
    "\u03bcin": 25.4e-6,
}

# Inch-family units that should map to the Standard display system.
_INCH_UNITS = {"in", "\u03bcin"}

# Canonical (X, Y) display pair per unit system.
_SYSTEM_PAIRS = {
    "Metric": ("mm", "\u03bcm"),
    "Standard": ("in", "\u03bcin"),
}


def _convert_length(value, from_unit: str, to_unit: str):
    """Scale ``value`` (numeric or array) from ``from_unit`` to ``to_unit``.

    Both units must appear in :data:`_MM_PER_UNIT`. Raises ``ValueError`` for
    unknown units, which signals an unexpected ``.pro`` header.
    """
    try:
        mm_from = _MM_PER_UNIT[from_unit]
        mm_to = _MM_PER_UNIT[to_unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported unit conversion: {from_unit!r} -> {to_unit!r}"
        ) from exc
    return value * (mm_from / mm_to)


def _natural_system(file_x_unit: str) -> str:
    """Return the unit-system name (``"Metric"`` or ``"Standard"``) the file
    naturally belongs to, based on its declared X unit.
    """
    return "Standard" if file_x_unit in _INCH_UNITS else "Metric"


def _normalise_unit(raw: str) -> str:
    """Return a canonical unit string, falling back to the trimmed input.

    Strings come from fixed-width C-style fields in the header and may carry
    trailing spaces or null bytes; surfalize strips those during decoding.
    """
    if raw is None:
        return ""
    cleaned = raw.strip().lower()
    if not cleaned:
        return ""
    return _UNIT_ALIASES.get(cleaned, cleaned)


def _ensure_profile_object(header: dict, path: str | Path) -> None:
    """Reject anything that is not a single 1D profile object."""
    studiable = header.get("studiable_type")
    if studiable != StudiableType.PROFILE:
        actual = getattr(studiable, "name", str(studiable))
        raise ValueError(
            f"{Path(path).name}: expected a 1D PROFILE studiable, but the "
            f"file declares studiable_type={actual}. This toolkit only "
            f"supports 1D profile traces; 2D surfaces (.sur) and other "
            f"studiable kinds are not supported."
        )
    n_objects = header.get("n_objects", 1)
    if n_objects and n_objects > 1:
        raise ValueError(
            f"{Path(path).name}: file contains {n_objects} stacked objects "
            f"(multi-layer or series studiable). Only single-profile files "
            f"are supported."
        )


def peek_pro_units(path: str | Path) -> Tuple[str, str]:
    """Return canonical display ``(x_unit, y_unit)`` for ``path``.

    Reads only the 512-byte header. The returned pair is the unit system the
    file naturally belongs to (``("mm", "μm")`` for metric files,
    ``("in", "μin")`` for inch-family files); callers may still convert into
    the opposite system at load time.

    Raises:
        ValueError: If the file is not a 1D profile studiable.
    """
    with open(path, "rb") as fh:
        header = read_sur_header(fh)
    _ensure_profile_object(header, path)
    file_x_unit = _normalise_unit(
        header.get("unit_step_x") or header.get("unit_x", "")
    )
    return _SYSTEM_PAIRS[_natural_system(file_x_unit)]


def read_pro_profile(
    path: str | Path,
    target_system: Optional[str] = None,
) -> Tuple[np.ndarray, str, str]:
    """Load a Digital Surf ``.pro`` profile in canonical display units.

    The profile is converted into the toolkit's canonical X/Y display units:

    * ``"Standard"`` system → X in inches, Y in micro-inches (μin).
    * ``"Metric"`` system → X in millimetres, Y in micrometres (μm).

    Args:
        path: Path to the ``.pro`` file.
        target_system: ``"Metric"`` or ``"Standard"``. ``None`` (default)
            picks whichever system matches the file header (inch-family →
            Standard; everything else → Metric).

    Returns:
        ``(xy, x_unit, y_unit)`` with ``xy`` shaped ``(2, N)``, ``xy[0]``
        carrying X positions and ``xy[1]`` Z heights, both in the chosen
        canonical units. Non-measured samples (sentinel ``min_point - 2``
        when ``non_measured_points`` is set) are dropped.

    Raises:
        ValueError: If the file is not a single 1D profile, has unrecognised
            units, or has no valid samples after sentinel filtering.
    """
    with open(path, "rb") as fh:
        sur_obj = read_sur_object(fh)

    header = sur_obj.header
    _ensure_profile_object(header, path)

    raw = sur_obj.data.reshape(-1)
    n_total = int(header["n_total_points"])
    if raw.size != n_total:
        raise ValueError(
            f"{Path(path).name}: header reports {n_total} points but data "
            f"block contains {raw.size}."
        )

    spacing_x = float(header["spacing_x"])
    offset_x = float(header.get("offset_x", 0.0))
    spacing_z = float(header["spacing_z"])
    offset_z = float(header.get("offset_z", 0.0))

    if not np.isfinite(spacing_x) or spacing_x == 0.0:
        raise ValueError(
            f"{Path(path).name}: invalid X spacing in header (spacing_x="
            f"{spacing_x!r})."
        )

    file_x_unit = _normalise_unit(
        header.get("unit_step_x") or header.get("unit_x", "")
    )
    file_y_unit = _normalise_unit(
        header.get("unit_step_z") or header.get("unit_z", "")
    )
    if not file_x_unit or not file_y_unit:
        raise ValueError(
            f"{Path(path).name}: header is missing X or Z unit (got "
            f"x_unit={file_x_unit!r}, y_unit={file_y_unit!r})."
        )

    if target_system is None:
        system = _natural_system(file_x_unit)
    elif target_system in _SYSTEM_PAIRS:
        system = target_system
    else:
        raise ValueError(
            f"target_system must be 'Metric' or 'Standard', got "
            f"{target_system!r}."
        )
    x_unit, y_unit = _SYSTEM_PAIRS[system]

    z_native = raw.astype(np.float64) * spacing_z + offset_z
    if header.get("non_measured_points") == 1:
        invalid_value = int(header["min_point"]) - 2
        nan_mask = raw == invalid_value
        if nan_mask.any():
            z_native[nan_mask] = np.nan

    x_native = np.arange(n_total, dtype=np.float64) * spacing_x + offset_x

    x = _convert_length(x_native, file_x_unit, x_unit)
    z = _convert_length(z_native, file_y_unit, y_unit)

    valid = np.isfinite(x) & np.isfinite(z)
    if not valid.any():
        raise ValueError(
            f"{Path(path).name}: no valid samples after dropping non-measured "
            f"points."
        )

    return np.vstack((x[valid], z[valid])), x_unit, y_unit
