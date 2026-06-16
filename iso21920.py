"""ISO 21920-3:2021 setting-class table and helpers.

ISO 21920-3:2021 (which supersedes ISO 4288:1996) specifies the *complete
specification operator* for areal-line (R-type) profile texture parameters.
A central concept is the **setting class** Sc1..Sc5 — a coupled tuple of
filter cutoffs, evaluation length, sectioning and stylus tip radius that
together define how the parameter is to be measured.

This module exposes:

* :class:`SettingClass` — immutable description of one ISO 21920-3 class.
* :data:`SETTING_CLASSES` — ordered mapping ``"Sc1"`` .. ``"Sc5"``.
* :func:`get_setting_class` — case-insensitive lookup with helpful errors.
* :func:`recommend_setting_class` — given a target parameter and tolerance,
  return the recommended Sc per ISO 21920-3 §4.4 / Tables 3–6 (placeholder
  thresholds; see note below).
* :func:`convert_mm_to` — convert a millimetre length into an arbitrary
  user x-unit (``mm``, ``cm``, ``m``, ``μm``/``um``, ``in``).

All canonical lengths are stored in **millimetres**; the stylus tip radius
``rtip`` is in **micrometres**. Use :func:`convert_mm_to` when the working
profile uses non-millimetre units.

NOTE on tolerance thresholds: the tolerance → setting-class boundaries in
ISO 21920-3 Tables 3–6 are populated here as a *first-pass interpretation*
suitable for engineering use. The exact boundaries are part of the
standard's normative tables and should be cross-checked against the
controlled document before relying on them for certification work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SettingClass:
    """An ISO 21920-3:2021 setting class (Table 1).

    Attributes
    ----------
    name : str
        Class identifier, ``"Sc1"`` .. ``"Sc5"``.
    lambda_s_mm : float
        S-filter (short-wavelength / noise) cutoff in millimetres.
    lambda_c_mm : float
        L-filter (long-wavelength / waviness) cutoff in millimetres. Equals
        the sampling length ``lsc``.
    dx_max_mm : float
        Maximum permitted sampling distance (point spacing) in millimetres.
    le_mm : float
        Evaluation length in millimetres. Equals ``nsc * lsc``.
    nsc : int
        Number of sampling sections (default 5).
    lsc_mm : float
        Sampling-section length in millimetres (= ``le_mm / nsc``).
    rtip_um : float
        Default stylus tip radius in micrometres.
    """

    name: str
    lambda_s_mm: float
    lambda_c_mm: float
    dx_max_mm: float
    le_mm: float
    nsc: int
    lsc_mm: float
    rtip_um: float

    @property
    def lambda_s_um(self) -> float:
        return self.lambda_s_mm * 1000.0

    @property
    def dx_max_um(self) -> float:
        return self.dx_max_mm * 1000.0


# ISO 21920-3:2021 Table 1 — canonical Sc1..Sc5 settings.
SETTING_CLASSES: Mapping[str, SettingClass] = {
    "Sc1": SettingClass("Sc1", 0.0025, 0.08, 0.0005, 0.40,  5, 0.08, 2.0),
    "Sc2": SettingClass("Sc2", 0.0025, 0.25, 0.0005, 1.25,  5, 0.25, 2.0),
    "Sc3": SettingClass("Sc3", 0.0025, 0.80, 0.0005, 4.00,  5, 0.80, 2.0),
    "Sc4": SettingClass("Sc4", 0.0080, 2.50, 0.0015, 12.5,  5, 2.50, 5.0),
    "Sc5": SettingClass("Sc5", 0.0250, 8.00, 0.0050, 40.0,  5, 8.00, 10.0),
}

DEFAULT_SETTING_CLASS = "Sc3"


def get_setting_class(name: str | None) -> SettingClass | None:
    """Look up a setting class by name (case-insensitive).

    Returns ``None`` if ``name`` is falsy. Raises ``KeyError`` if the
    supplied name is non-empty but does not match a known class.
    """
    if not name:
        return None
    key = str(name).strip()
    # Tolerate "sc3", "SC3", " Sc3 ", "Sc 3" → "Sc3"
    norm = key.replace(" ", "").replace("_", "")
    if not norm:
        return None
    norm = norm[:2].title() + norm[2:]
    if norm in SETTING_CLASSES:
        return SETTING_CLASSES[norm]
    raise KeyError(
        f"Unknown setting class {name!r}. "
        f"Expected one of: {', '.join(SETTING_CLASSES)}."
    )


# Conversion factors: how many target units fit in 1 millimetre.
_MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 0.1,
    "m":  0.001,
    "μm": 1000.0,
    "um": 1000.0,
    "in": 1.0 / 25.4,
    "inch": 1.0 / 25.4,
    "nm": 1_000_000.0,
    "uin": 1_000_000.0 / 25.4,
}


def _normalize_unit(unit: str) -> str:
    text = str(unit or "").strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    return {"um": "um", "umeter": "um", "umetre": "um",
            "uinch": "uin"}.get(text, text)


def convert_mm_to(length_mm: float, unit: str) -> float:
    """Convert a millimetre length to ``unit``.

    Recognised units (case-insensitive): ``mm``, ``cm``, ``m``, ``μm`` / ``um``,
    ``in`` / ``inch``, ``μin`` / ``uin``, ``nm``. Raises ``ValueError`` for
    unknown units.
    """
    norm = _normalize_unit(unit)
    if norm in {"um", "u m", "u-m"}:
        return length_mm * 1000.0
    if norm in {"in", "u-in", "u in"}:
        return length_mm / 25.4
    if norm == "uin":
        return length_mm * 1_000_000.0 / 25.4
    factor = _MM_PER_UNIT.get(norm)
    if factor is None:
        raise ValueError(f"Unsupported length unit for conversion: {unit!r}")
    return length_mm * factor


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` from one length unit to another.

    Pivots through millimetres so the same alias rules as
    :func:`convert_mm_to` apply to both sides. Recognised:
    ``mm``, ``cm``, ``m``, ``μm`` / ``um``, ``in`` / ``inch``,
    ``μin`` / ``uin``, ``nm``.
    """
    if value == 0:
        return 0.0
    if _normalize_unit(from_unit) == _normalize_unit(to_unit):
        return float(value)
    # value [from_unit] → mm
    mm = float(value) / convert_mm_to(1.0, from_unit)
    # mm → to_unit
    return convert_mm_to(mm, to_unit)


# ---------------------------------------------------------------------------
# Tolerance-driven recommendation (ISO 21920-3 §4.4 / Tables 3–6)
# ---------------------------------------------------------------------------
#
# The standard provides parameter-specific tables that map a tolerance value
# to a recommended setting class. The threshold structure below is a
# first-pass engineering interpretation; verify against the controlled
# ISO 21920-3:2021 document before normative use.

# Thresholds keyed by parameter; each list gives the *upper* bound (in µm)
# below which the corresponding class is sufficient. The last entry's bound
# is sentinel +inf to mean "use the largest class".
_RECOMMENDATION_TABLES_UM = {
    # Ra (ISO 21920-3 Table 3 — periodic and non-periodic profiles unified)
    "Ra":  [(0.02, "Sc1"), (0.10, "Sc2"), (2.00, "Sc3"), (10.0, "Sc4"), (float("inf"), "Sc5")],
    # Rz, Rzx, Rt — amplitude peak/valley parameters (Table 4)
    "Rz":  [(0.10, "Sc1"), (0.50, "Sc2"), (10.0, "Sc3"), (50.0, "Sc4"), (float("inf"), "Sc5")],
    "Rzx": [(0.10, "Sc1"), (0.50, "Sc2"), (10.0, "Sc3"), (50.0, "Sc4"), (float("inf"), "Sc5")],
    "Rt":  [(0.10, "Sc1"), (0.50, "Sc2"), (10.0, "Sc3"), (50.0, "Sc4"), (float("inf"), "Sc5")],
    # Rq (treat similarly to Ra; ISO 21920-3 Table 3 footnote)
    "Rq":  [(0.025, "Sc1"), (0.125, "Sc2"), (2.5, "Sc3"), (12.5, "Sc4"), (float("inf"), "Sc5")],
    # Rp, Rv — half-range amplitudes (use Rz/2 boundaries)
    "Rp":  [(0.05, "Sc1"), (0.25, "Sc2"), (5.0, "Sc3"), (25.0, "Sc4"), (float("inf"), "Sc5")],
    "Rv":  [(0.05, "Sc1"), (0.25, "Sc2"), (5.0, "Sc3"), (25.0, "Sc4"), (float("inf"), "Sc5")],
}


def recommend_setting_class(parameter: str,
                            tolerance_kind: str,
                            value_um: float | tuple[float, float]) -> dict:
    """Recommend a setting class for a parameter tolerance.

    Parameters
    ----------
    parameter : str
        ISO 21920-2 parameter symbol (``"Ra"``, ``"Rz"``, ``"Rt"``, ``"Rq"``,
        ``"Rp"``, ``"Rv"``, ``"Rzx"``).
    tolerance_kind : str
        One of ``"U"`` (upper-only), ``"L"`` (lower-only), ``"UL"`` /
        ``"bilateral"`` (upper + lower), ``"C"`` / ``"symmetric"`` (centered).
    value_um : float or (float, float)
        Tolerance value(s) in micrometres. For bilateral give ``(upper, lower)``.

    Returns
    -------
    dict
        ``{"setting_class": str, "rationale": str, "table": str}``.
    """
    sym = (parameter or "").strip()
    if sym not in _RECOMMENDATION_TABLES_UM:
        raise ValueError(
            f"No tolerance table available for parameter {parameter!r}. "
            f"Supported: {sorted(_RECOMMENDATION_TABLES_UM)}."
        )

    kind = (tolerance_kind or "").strip().upper()
    if kind in {"BILATERAL", "U+L", "UL"}:
        if not isinstance(value_um, (tuple, list)) or len(value_um) != 2:
            raise ValueError("Bilateral tolerance requires (upper_um, lower_um).")
        # Drive recommendation from the *upper* limit (worst-case roughness).
        ref = float(value_um[0])
        kind_label = f"bilateral U={value_um[0]:g}, L={value_um[1]:g}"
    elif kind in {"U", "UPPER"}:
        ref = float(value_um)
        kind_label = f"upper-only U={ref:g}"
    elif kind in {"L", "LOWER"}:
        ref = float(value_um)
        kind_label = f"lower-only L={ref:g} (smallest acceptable surface)"
    elif kind in {"C", "SYMMETRIC", "CENTRE", "CENTER"}:
        ref = float(value_um)
        kind_label = f"symmetric ±{ref:g}"
    else:
        raise ValueError(
            f"Unknown tolerance_kind {tolerance_kind!r}. "
            "Expected 'U', 'L', 'UL'/'bilateral', or 'C'/'symmetric'."
        )

    table = _RECOMMENDATION_TABLES_UM[sym]
    chosen = table[-1][1]
    bound_used = table[-1][0]
    for upper_um, sc_name in table:
        if ref <= upper_um:
            chosen = sc_name
            bound_used = upper_um
            break

    return {
        "setting_class": chosen,
        "rationale": (
            f"For {sym} {kind_label} µm: value {ref:g} µm ≤ {bound_used:g} µm "
            f"→ {chosen} (ISO 21920-3 Table for {sym})."
        ),
        "table": f"ISO 21920-3 §4.4 / Tables 3–6 ({sym})",
    }
