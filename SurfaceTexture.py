import time, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from scipy import signal, optimize
from functools import wraps
from blume.table import table

from iso21920 import (
    SETTING_CLASSES,
    DEFAULT_SETTING_CLASS,
    convert_mm_to,
    convert_length,
    get_setting_class,
)


class TraceTooShortError(ValueError):
    """Raised when the trace cannot accommodate one full λc sampling length.

    Carries structured context so callers (e.g. the GUI) can offer recovery
    options such as switching to a smaller setting class or forcing the run
    via the ``allow_short_trace`` override.
    """

    def __init__(
        self,
        message: str,
        *,
        suggested_class: str | None = None,
        current_class: str | None = None,
        short_cutoff: float | None = None,
        long_cutoff: float | None = None,
        x_units: str | None = None,
    ) -> None:
        super().__init__(message)
        self.suggested_class = suggested_class
        self.current_class = current_class
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.x_units = x_units


def _largest_fitting_setting_class(n_pts: int, T: float, x_units: str) -> str | None:
    """Return the largest ISO 21920-3 setting class whose λc + buffers fit.

    Walks ``SETTING_CLASSES`` in ascending order (Sc1 → Sc5) and returns
    the *last* class for which ``max_nsc_that_fits >= 1`` — i.e. the most
    conservative class the standard could still allow. Returns ``None``
    if even Sc1 does not fit.

    Args:
        n_pts: Number of samples in the trace.
        T: Sample spacing in the data's X distance unit.
        x_units: The data's X distance unit (used to convert canonical
            millimetre cutoffs).
    """
    best: str | None = None
    for name, sc in SETTING_CLASSES.items():
        try:
            ls = convert_mm_to(sc.lambda_s_mm, x_units)
            lc = convert_mm_to(sc.lambda_c_mm, x_units)
        except ValueError:
            continue
        edge_buff = int(ls / (2 * T)) + int(lc / (2 * T))
        lsc_samples = max(int(round(lc / T)), 1)
        usable = n_pts - 2 * edge_buff
        max_nsc_fits = usable // lsc_samples if lsc_samples > 0 else 0
        if max_nsc_fits >= 1:
            best = name
    return best


class SurfaceTexture():

    @staticmethod
    def _normalize_unit(unit):
        text = str(unit or "").strip().lower()
        text = text.replace("μ", "u").replace("µ", "u")
        return text

    def _display_decimals(self, param, unit):
        # Requested defaults:
        # - micro-inch params: 1 decimal place
        # - micro-meter params: 2 decimal places
        # - Bearing ratio (Rmr): always 1 decimal place
        # - Rsk: always 2 decimal places
        if param == "Rmr":
            return 1
        if param == "Rsk":
            return 2
        norm_unit = self._normalize_unit(unit)
        if norm_unit in {"uin", "uinch", "u-in", "u in"}:
            return 1
        if norm_unit in {"um", "umeter", "umetre", "u-m", "u m"}:
            return 2
        return 3

    def format_param_value(self, param, value, unit):
        if isinstance(value, (int, float, np.floating)):
            decimals = self._display_decimals(param, unit)
            return f"{float(value):.{decimals}f}"
        return str(value)

    @staticmethod
    def _parse_column_selector(selector):
        if isinstance(selector, int):
            return selector
        if isinstance(selector, str) and selector.strip().isdigit():
            return int(selector.strip())
        return selector

    @staticmethod
    def _read_text_profile(raw_data):
        try:
            return np.loadtxt(raw_data, delimiter=",", unpack=True)
        except ValueError:
            return np.loadtxt(raw_data, unpack=True)

    @classmethod
    def _read_excel_profile(cls, raw_data, x_col, y_col, sheet_name=0):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx files requires pandas and openpyxl. "
                "Install with: pip install pandas openpyxl"
            ) from exc

        x_selector = cls._parse_column_selector(x_col)
        y_selector = cls._parse_column_selector(y_col)

        # Pick the fastest available engine. ``python-calamine`` is roughly
        # 10-50x faster than ``openpyxl`` for large numeric workbooks.
        try:
            import python_calamine  # noqa: F401
            engine = "calamine"
        except ImportError:
            engine = "openpyxl"

        # If we have integer column indexes we can ask the engine to only
        # materialise the two columns of interest, which is dramatically
        # faster than parsing the whole sheet.
        usecols = None
        if isinstance(x_selector, int) and isinstance(y_selector, int):
            usecols = sorted({x_selector, y_selector})

        read_kwargs = {"sheet_name": sheet_name, "engine": engine}
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        df = pd.read_excel(raw_data, **read_kwargs)
        if isinstance(df, dict):
            first_sheet = next(iter(df))
            df = df[first_sheet]

        if isinstance(x_selector, str):
            if x_selector not in df.columns:
                raise ValueError(
                    f"X column '{x_selector}' was not found in sheet '{sheet_name}'."
                )
            x_series = df[x_selector]
        else:
            # When ``usecols`` was supplied the resulting frame has the
            # selected columns in their original order; map back via index.
            if usecols is not None:
                x_series = df.iloc[:, usecols.index(x_selector)]
            else:
                x_series = df.iloc[:, x_selector]

        if isinstance(y_selector, str):
            if y_selector not in df.columns:
                raise ValueError(
                    f"Y column '{y_selector}' was not found in sheet '{sheet_name}'."
                )
            y_series = df[y_selector]
        else:
            if usecols is not None:
                y_series = df.iloc[:, usecols.index(y_selector)]
            else:
                y_series = df.iloc[:, y_selector]

        x = x_series.to_numpy(dtype=float)
        y = y_series.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError(
                "No valid numeric X/Y pairs found in the selected worksheet columns."
            )

        return np.vstack((x[valid], y[valid]))

    @classmethod
    def _load_profile_data_with_units(
        cls, raw_data, x_col=0, y_col=1, sheet_name=0, pro_target_system=None
    ):
        """Load profile XY plus optional file-declared (x_unit, y_unit).

        Most formats (TXT/CSV/XLSX) carry no unit metadata, so they return
        ``(profile, None, None)``. Digital Surf ``.pro`` files declare X/Y
        units in their header and surface them here so the caller can adopt
        them before deriving filter cutoffs.

        ``pro_target_system`` (``"Metric"`` / ``"Standard"`` / ``None``)
        forwards to :func:`pro_reader.read_pro_profile`, allowing callers to
        force a ``.pro`` file into the opposite unit system.
        """
        ext = Path(raw_data).suffix.lower()
        if ext == ".xlsx":
            profile = cls._read_excel_profile(
                raw_data, x_col=x_col, y_col=y_col, sheet_name=sheet_name
            )
            return profile, None, None
        if ext == ".pro":
            from pro_reader import read_pro_profile
            profile, x_unit, y_unit = read_pro_profile(
                raw_data, target_system=pro_target_system
            )
            return profile, x_unit, y_unit
        return cls._read_text_profile(raw_data), None, None

    @classmethod
    def _load_profile_data(
        cls, raw_data, x_col=0, y_col=1, sheet_name=0, pro_target_system=None
    ):
        profile, _x_unit, _y_unit = cls._load_profile_data_with_units(
            raw_data,
            x_col=x_col,
            y_col=y_col,
            sheet_name=sheet_name,
            pro_target_system=pro_target_system,
        )
        return profile
    
    def timeit(method):
        @wraps(method)
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print(f"{method.__name__} took {te-ts:.3f} seconds")
            return result
        return timed

    @timeit
    def __init__(self, raw_data: str, short_cutoff=None,
                 long_cutoff=None, order=1,
                 x_units='mm', y_units='μm',
                 source_x_units=None, source_y_units=None,
                 x_col=0, y_col=1, sheet_name=0,
                 setting_class=None,
                 pro_target_system=None,
                 allow_short_trace=False,
                 **kwargs):
        """Process a surface profile per ISO 21920-3:2021 / ISO 21920-2:2021.

        The complete specification operator (S-filter, L-filter, F-operator,
        evaluation length, sectioning) is built from the *setting class*
        (ISO 21920-3 Table 1). The evaluation length and number of sampling
        sections are derived from the trace length and λc:

        * Default target ``nsc = 5`` sampling sections of length ``lsc = λc``.
        * If the trace cannot accommodate 5 sections plus the filter
          end-buffers, ``nsc`` is automatically reduced toward 1 and a
          warning is emitted on ``self.nsc_warning``.
        * If even one full ``λc`` section will not fit, a
          :class:`TraceTooShortError` is raised suggesting the largest
          setting class whose λc *will* fit.

        Args:
            raw_data: 2-column data file (.txt/.csv/.xlsx) or Digital Surf
                profile (.pro).
            short_cutoff: λs in the *report* X distance unit
                (``x_units``). Overrides the setting class.
            long_cutoff: λc in the *report* X distance unit
                (``x_units``). Overrides the setting class.
            order: Polynomial leveling order (0 skips, 1..3).
            x_units / y_units: **Report** units — the X / Y units the
                figure, parameter table, and cutoffs are expressed in.
                Profile data is converted into this frame at load time.
            source_x_units / source_y_units: Units the raw data file is in.
                For ``.pro`` files the loader populates these from the file
                header when not supplied. For TXT / CSV / XLSX, when
                ``None`` the source is assumed to match the report units
                (legacy behaviour).
            x_col / y_col / sheet_name: Column / sheet selectors. Ignored
                for ``.pro`` files.
            setting_class: ISO 21920-3 setting class name ("Sc1".."Sc5",
                or None to default to "Sc3"). Cutoffs inherit from the class
                unless explicitly overridden.
            pro_target_system: Deprecated. Retained for backward
                compatibility; superseded by the
                ``source_x_units`` / ``source_y_units`` + ``x_units`` /
                ``y_units`` pair (the loader now reads .pro header units
                and ``SurfaceTexture`` converts source → report uniformly).
            allow_short_trace: When True, bypass the trace-too-short
                guard. The filter end-buffers are clamped, ``nsc`` is forced
                to 1, and ``self.short_trace_override`` is populated with a
                warning. The figure banner flags the result as
                non-conformant. Use only when the user has explicitly
                acknowledged the compromise (edge artefacts will be
                significant).
            **kwargs: PLOT_LEVEL, PLOT_MR, PLOT_ROUGHNESS, PLOT_ALL flags.
        """

        kwargs.setdefault('PLOT_LEVEL', False)
        kwargs.setdefault('PLOT_MR', False)
        kwargs.setdefault('PLOT_ROUGHNESS', False)
        kwargs.setdefault('PLOT_ALL', False)

        self.x_units = x_units
        self.y_units = y_units
        self.raw_data = raw_data
        self.order = order

        # Load the profile first. For .pro files the header declares units;
        # we keep them as the *source* unit so we can convert into the
        # caller-requested report frame below.
        self.primary, file_x_unit, file_y_unit = self._load_profile_data_with_units(
            self.raw_data,
            x_col=x_col,
            y_col=y_col,
            sheet_name=sheet_name,
            pro_target_system=pro_target_system,
        )

        # Resolve effective source units.
        # Priority: explicit kwarg > file-declared (.pro header) > report unit.
        eff_source_x = source_x_units or file_x_unit or self.x_units
        eff_source_y = source_y_units or file_y_unit or self.y_units
        self.source_x_units = eff_source_x
        self.source_y_units = eff_source_y

        # Convert the profile from source units → report units. ``primary``
        # is a 2 × N ndarray where row 0 is X and row 1 is Y.
        try:
            x_factor = convert_length(1.0, eff_source_x, self.x_units)
        except ValueError:
            x_factor = 1.0
        try:
            y_factor = convert_length(1.0, eff_source_y, self.y_units)
        except ValueError:
            y_factor = 1.0
        if x_factor != 1.0 or y_factor != 1.0:
            self.primary = np.vstack((
                self.primary[0] * x_factor,
                self.primary[1] * y_factor,
            ))

        # ----- ISO 21920-3 setting-class resolution -----------------------
        # If the user did not name a class but supplied explicit cutoffs we
        # treat the configuration as "Custom"; otherwise we fall back to Sc3.
        if setting_class is None and short_cutoff is None and long_cutoff is None:
            setting_class = DEFAULT_SETTING_CLASS

        sc = get_setting_class(setting_class) if setting_class else None
        self.setting_class = sc            # SettingClass instance or None
        self.setting_class_name = sc.name if sc else "Custom"

        # All canonical class lengths are mm; convert into the data's X unit.
        if sc is not None:
            sc_lambda_s = convert_mm_to(sc.lambda_s_mm, self.x_units)
            sc_lambda_c = convert_mm_to(sc.lambda_c_mm, self.x_units)
            sc_target_nsc = sc.nsc
        else:
            sc_lambda_s = sc_lambda_c = None
            sc_target_nsc = 5

        self.short_cutoff = short_cutoff if short_cutoff is not None else sc_lambda_s
        self.long_cutoff = long_cutoff if long_cutoff is not None else sc_lambda_c

        if self.short_cutoff is None or self.long_cutoff is None:
            raise ValueError(
                "Filter cutoffs must be provided either explicitly "
                "(short_cutoff, long_cutoff) or via setting_class."
            )

        # ``target_nsc`` is the ISO-preferred number of sections (5). The
        # *actual* ``nsc`` may be reduced after the trace is loaded if the
        # measurement is too short to fit 5 full λc sampling lengths plus
        # the filter end-buffers (see the filter pipeline below).
        self.target_nsc = sc_target_nsc
        self.nsc = sc_target_nsc

        # Keep an immutable copy of the raw input for plotting/inspection.
        self.raw_data_xy = self.primary.copy()
        self.level_fit_x = None
        self.level_fit_y = None
        self.section_edges = None          # x positions of nsc+1 dividers
        self.section_Rz_values = []        # per-section peak-to-valley
        self.le_window = None              # (x_start, x_end) used for params
        self.dx_warning = None             # populated if dx exceeds Sc dx_max
        self.nsc_warning = None            # populated if nsc reduced below target
        self.short_trace_override = None   # populated if allow_short_trace forced run
        self.comparison_warnings = []      # populated by compute_comparison_params

        # if profile leveling is called for then fit to line/curve
        if order: 
            if order > 3:
                raise ValueError("order must be 1, 2 or 3 (0 to skip leveling)")
            self.order = order
            def first_order_func(x, a, b):
                return a * x + b
            def second_order_func(x, a, b, c):
                return a * x ** 2 + b * x + c
            def third_order_func(x, a, b, c, d):
                return a * x ** 3 + b * x ** 2 + c * x + d
            case = {1: first_order_func,
                    2: second_order_func,
                    3: third_order_func}
            func = case[self.order]

            x = self.primary[0]
            y = self.primary[1]
            popt, _ = optimize.curve_fit(func, x, y)

            # Save the leveling fit so it can be plotted alongside raw data later.
            self.level_fit_x = np.array(x)
            self.level_fit_y = func(x, *popt)

            if kwargs['PLOT_LEVEL'] or kwargs['PLOT_ALL']:
                if self.order == 1:
                    plt.plot(x, func(x, *popt), 'r-',
                            label='fit: a=%5.10f, b=%5.10f' % tuple(popt))
                else:
                    plt.plot(x, func(x, *popt), 'r-')
                plt.title("Initial Profile Leveling")
                plt.plot(x, y, 'b-', label='data')
                plt.legend()
                plt.show()
            
            self.primary = np.vstack((np.array(x),
                                      np.array(y - func(x, *popt))))


        # calc sampling frequency (robust to descending/non-uniform x spacing)
        diffs = np.diff(self.primary[0])
        diffs = np.abs(diffs[np.isfinite(diffs)])
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            raise ValueError("Invalid x spacing: unable to compute positive sample spacing from X data.")
        T = float(np.median(diffs))
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"Invalid x spacing: computed sample spacing T={T}.")
        self.Fs = 1.0 / T
        self.dx = T

        # ISO 21920-3 dx_max coupling — warn (do not block) if class is set
        if self.setting_class is not None:
            dx_max_in_x_units = convert_mm_to(self.setting_class.dx_max_mm, self.x_units)
            # Allow a small floating-point tolerance so traces exactly at the
            # spec value don't trigger a spurious warning.
            if T > dx_max_in_x_units * 1.001:
                self.dx_warning = (
                    f"Sample spacing dx = {T:.4g} {self.x_units} exceeds "
                    f"{self.setting_class.name} maximum "
                    f"{dx_max_in_x_units:.4g} {self.x_units} per ISO 21920-3 Table 1."
                )

        def gauss_filter(data, cutoff):
            # cutoff is a spatial frequency (1/length). The effective Gaussian
            # standard deviation in samples is Fs / (2*pi*cutoff).
            sigma = abs(self.Fs / (2 * np.pi * cutoff))
            sigma = max(sigma, np.finfo(float).eps)

            # Truncate the window where the Gaussian underflows in float64
            # (~38 sigma).
            half = max(int(np.ceil(38.0 * sigma)), 1)
            window_len = 2 * half + 1

            window = signal.windows.gaussian(window_len, sigma) / \
                    (sigma * np.sqrt(2 * np.pi))
            window = window[window > 0]
            return signal.fftconvolve(data, window, mode="same")

        self._gauss_filter = gauss_filter  # exposed for plotting helpers

        # Filter end-effect buffer (one half-cutoff per filter).
        prim_buff = int(self.short_cutoff / (2 * T))
        wav_buff = int(self.long_cutoff / (2 * T))
        edge_buff = prim_buff + wav_buff
        n_pts = self.primary[0].size

        # ----- Auto-derive evaluation length and nsc ----------------------
        # The sampling-section length lsc is, by ISO 21920-3 default, equal
        # to λc. We try to fit ``target_nsc`` (= 5 by default) full sections
        # inside the trace after subtracting the filter end-buffers; if the
        # trace is too short we reduce nsc down toward 1, recording a
        # warning. If even a single lsc + buffers won't fit we raise.
        lsc_samples = max(int(round(self.long_cutoff / T)), 1)
        usable = n_pts - 2 * edge_buff
        max_nsc_that_fits = max(usable // lsc_samples, 0)
        nsc_actual = min(self.target_nsc, max_nsc_that_fits)

        if nsc_actual < 1:
            # Build the structured "trace too short" message regardless of
            # path so both the override warning and the exception share
            # consistent wording.
            suggested = _largest_fitting_setting_class(n_pts, T, self.x_units)
            suggestion_txt = (
                f" Tip: try setting class {suggested}."
                if suggested else
                " Tip: no ISO 21920-3 setting class will fit this trace; "
                "the trace is shorter than Sc1's λc + filter buffer."
            )
            base_msg = (
                "Trace is too short for the requested ISO 21920-3 setting.\n"
                f"  Sample spacing dx = {T:.6g} {self.x_units}\n"
                f"  Cutoffs λs = {self.short_cutoff:g} {self.x_units}, "
                f"λc = {self.long_cutoff:g} {self.x_units}\n"
                f"  Filter buffer = {edge_buff} samples each end\n"
                f"  One sampling length lsc = λc requires {lsc_samples} samples, "
                f"but only {max(usable, 0)} samples are usable after the "
                f"filter end-buffers (trace has {n_pts} samples).{suggestion_txt}"
            )

            if not allow_short_trace:
                raise TraceTooShortError(
                    base_msg,
                    suggested_class=suggested,
                    current_class=self.setting_class_name,
                    short_cutoff=float(self.short_cutoff),
                    long_cutoff=float(self.long_cutoff),
                    x_units=self.x_units,
                )

            # ----- Override path ------------------------------------------
            # User has explicitly accepted edge artefacts. Clamp the filter
            # end-buffers down so we can extract at least one sampling
            # section. Keep λs/λc unchanged so the L-filter still produces
            # the requested separation between roughness and waviness.
            forced_edge = max(0, min(edge_buff, max((n_pts - 2) // 4, 0)))
            forced_usable = max(n_pts - 2 * forced_edge, 1)
            forced_lsc_samples = max(forced_usable, 1)
            edge_buff = forced_edge
            usable = forced_usable
            lsc_samples = forced_lsc_samples
            nsc_actual = 1
            self.short_trace_override = (
                "Short-trace override active — results are non-conformant.\n"
                f"  Requested λc = {self.long_cutoff:g} {self.x_units} kept; "
                f"nsc forced to 1.\n"
                f"  Filter end-buffer reduced from "
                f"{prim_buff + wav_buff} → {forced_edge} samples each end "
                f"to fit the trace (n = {n_pts}).\n"
                "  Edge artefacts from the L-filter will be significant; "
                "treat parameters as indicative only."
            )
            import sys as _sys
            print(
                f"WARNING: {self.short_trace_override}",
                file=_sys.stderr,
            )

        self.nsc = nsc_actual
        le_samples = lsc_samples * nsc_actual
        self.evaluation_length = le_samples * T
        self.lsc = float(self.long_cutoff)

        if (
            nsc_actual < self.target_nsc
            and self.short_trace_override is None
        ):
            # Soft warning — surface via plot banner, GUI log, and stderr.
            self.nsc_warning = (
                f"Trace is too short for the ISO 21920-3 default of "
                f"nsc = {self.target_nsc} sampling lengths at "
                f"{self.setting_class_name}; using nsc = {nsc_actual} "
                f"(le = {self.evaluation_length:g} {self.x_units} "
                f"instead of {self.target_nsc * self.long_cutoff:g} {self.x_units}). "
                f"Reported parameters are computed but should be flagged as "
                f"non-conformant with the default sampling-section count."
            )
            import sys as _sys
            print(f"WARNING: {self.nsc_warning}", file=_sys.stderr)

        # ----- ISO 21920-21 S- and L- filtering ---------------------------
        # Apply S-filter (λs) and L-filter (λc) to the *full* primary trace
        # so each filtered series has the same length as primary[0]. The
        # central evaluation-length window is then sliced consistently.
        denoised_primary_full = gauss_filter(self.primary[1], 1.0 / self.short_cutoff)
        waviness_full = gauss_filter(denoised_primary_full, 1.0 / self.long_cutoff)
        roughness_full = denoised_primary_full - waviness_full

        # Centre the evaluation-length window inside the usable region.
        start = edge_buff + (usable - le_samples) // 2
        end = start + le_samples

        # Store window indices for potential re-computation (e.g. comparison).
        self._le_start = start
        self._le_end = end
        self._lsc_samples = lsc_samples

        wav_x = self.primary[0][start:end]
        denoised_primary = denoised_primary_full[start:end]
        waviness = waviness_full[start:end]
        roughness = roughness_full[start:end]

        self.roughness = np.vstack((wav_x, roughness))
        self.waviness = np.vstack((wav_x, waviness))
        self.denoised_primary = np.vstack((wav_x, denoised_primary))
        self.le_window = (float(wav_x[0]), float(wav_x[-1]))
        # Buffer regions on the *full* primary, for shading on plots.
        self.le_buffer_regions = [
            (float(self.primary[0][0]), float(self.primary[0][start])),
            (float(self.primary[0][end - 1]), float(self.primary[0][-1])),
        ]
        # Section dividers (nsc + 1 boundaries spanning the le window).
        self.section_edges = np.array([
            float(wav_x[i * lsc_samples]) for i in range(self.nsc)
        ] + [float(wav_x[-1])])

        # Generate roughness parameters --------------------------------------
        len_wav_x = len(wav_x)
        if len_wav_x == 0:
            raise ValueError(
                "No samples remain after filtering. Cutoffs are too large for the "
                "available trace length. Reduce cutoffs or verify they are in the "
                f"same units as the X data ({self.x_units})."
            )
        self.R_params = self._compute_r_params(roughness, self.nsc, lsc_samples, self.y_units)
        # Expose per-section values for plotting overlays.
        section_Rp = []
        section_Rv = []
        section_Rz = []
        for i in range(self.nsc):
            seg = roughness[i * lsc_samples:(i + 1) * lsc_samples]
            sp = float(np.max(seg))
            sv = float(abs(np.min(seg)))
            section_Rp.append(sp)
            section_Rv.append(sv)
            section_Rz.append(sp + sv)
        self.section_Rp_values = section_Rp
        self.section_Rv_values = section_Rv
        self.section_Rz_values = section_Rz

        # ----- ISO 16610-31 robust Gaussian regression (2nd order) --------
        # Used as the L-operator for the Rk-family per ISO 21920-3 Table 1.
        try:
            self.robust_mean = self._robust_gauss_regression_2nd_order(
                denoised_primary, self.long_cutoff, T,
            )
            self.roughness_robust = denoised_primary - self.robust_mean
        except Exception:
            # If robust filter fails, fall back silently to linear-filtered roughness
            self.robust_mean = waviness.copy()
            self.roughness_robust = roughness.copy()

        if kwargs['PLOT_ROUGHNESS'] or kwargs['PLOT_ALL']:
            self.plot_roughness()

        if kwargs['PLOT_MR'] or kwargs['PLOT_ALL']:
            self.plot_material_ratio()

    # ------------------------------------------------------------------
    # R-parameter computation (reusable across filter pipelines).
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_r_params(roughness, nsc, lsc_samples, y_units, Cref=5.0):
        """Compute R-family parameters from a roughness array.

        Parameters
        ----------
        roughness : ndarray
            1-D roughness profile (evaluation length).
        nsc : int
            Number of sampling sections.
        lsc_samples : int
            Samples per sampling section.
        y_units : str
            Unit label for amplitude parameters.
        Cref : float
            Material ratio reference percentage for Rmr.

        Returns
        -------
        dict
            {param_name: (value, unit)} for Ra, Rq, Rsk, Rku, Rp, Rv, Rz, Rt, Rmr.
        """
        params = {}
        params['Ra'] = (float(np.mean(np.abs(roughness))), y_units)
        Rq = float(np.sqrt(np.mean(roughness ** 2)))
        params['Rq'] = (Rq, y_units)
        if Rq > 0:
            params['Rsk'] = (float(np.mean(roughness ** 3)) / (Rq ** 3), "")
            params['Rku'] = (float(np.mean(roughness ** 4)) / (Rq ** 4), "")
        else:
            params['Rsk'] = (0.0, "")
            params['Rku'] = (0.0, "")

        section_Rp = []
        section_Rv = []
        section_Rz = []
        for i in range(nsc):
            seg = roughness[i * lsc_samples:(i + 1) * lsc_samples]
            sp = float(np.max(seg))
            sv = float(abs(np.min(seg)))
            section_Rp.append(sp)
            section_Rv.append(sv)
            section_Rz.append(sp + sv)

        params['Rp'] = (float(np.mean(section_Rp)), y_units)
        params['Rv'] = (float(np.mean(section_Rv)), y_units)
        params['Rz'] = (float(np.mean(section_Rz)), y_units)
        params['Rt'] = (float(np.max(roughness) - np.min(roughness)), y_units)

        # Rmr: material ratio at (peak_at_Cref% - Rz/4)
        n = roughness.size
        Rz = params['Rz'][0]
        unit_label = f"% (@Rz/4, Cref={Cref:g}%)"
        if n > 0:
            sorted_desc = np.sort(roughness)[::-1]
            idx = max(0, min(n - 1, int(round(Cref / 100.0 * n))))
            c0 = sorted_desc[idx]
            target = c0 - Rz / 4.0
            above = int(np.sum(sorted_desc >= target))
            params['Rmr'] = (above / n * 100.0, unit_label)
        else:
            params['Rmr'] = (0.0, unit_label)
        return params

    # ------------------------------------------------------------------
    # ISO 4287:1997 R-parameters (per-sampling-length aggregation).
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_r_params_iso4287(roughness, nsc, lsc_samples, y_units, Cref=5.0):
        """Compute R-family parameters per ISO 4287:1997.

        Per §4.2 (Ra, Rq, Rsk, Rku) and §4.1.1–§4.1.3 (Rp, Rv, Rz) each
        parameter is evaluated on a single *sampling length* and then
        averaged across the N sampling lengths of the evaluation length
        (ISO 4288 reporting convention). This differs from a one-shot
        moment over the whole evaluation length for Rsk and Rku — the
        per-sampling-length Rq in the denominator changes the result.
        Ra and Rq are numerically equivalent under the two conventions
        when all sampling lengths are equal in size.

        Rt (§4.1.5) is the only amplitude parameter defined over the
        whole evaluation length — kept as ``max(roughness) - min(roughness)``.

        Peak / valley definition (§3.2.4 – §3.2.7): on a roughness
        profile whose mean line lies at z = 0, every positive-z portion
        between two consecutive mean-line crossings is a *profile peak*
        and every negative-z portion is a *profile valley*. The highest
        peak height ``Zp`` within a sampling length is therefore
        ``max(seg)`` clipped at 0, and the deepest valley depth ``Zv`` is
        ``|min(seg)|`` clipped at 0. The §3.2.7 Note (partial peaks /
        valleys at sampling-length boundaries still count) is satisfied
        automatically by the slice-and-max approach. A defensive check
        on the zero-mean assumption is included.

        Parameters
        ----------
        roughness : ndarray
            1-D roughness profile over the evaluation length, already
            band-pass filtered by λs (S-filter) + λc (L-filter) per
            §3.1.6 Note 1.
        nsc : int
            Number of sampling lengths within the evaluation length.
        lsc_samples : int
            Samples per sampling length.
        y_units : str
            Unit label for amplitude parameters.
        Cref : float
            Material ratio reference percentage for the Rmr heuristic.
            NOTE: this is *not* the literal §4.5.1 Rmr(c); it is the
            Cref/Rz-quarter heuristic re-used from the ISO 21920 path so
            the comparison column keeps a single Rmr semantics across
            standards. To be revisited in the ISO 21920 audit.

        Returns
        -------
        dict
            ``{param_name: (value, unit)}`` matching
            :meth:`_compute_r_params`: Ra, Rq, Rsk, Rku, Rp, Rv, Rz, Rt, Rmr.
        """
        params = {}

        # ISO 4287 §3.1.5 defines the mean line locally over the evaluation
        # length. The Gaussian L-filter produces a zero-mean output *over
        # the full trace*; the centred evaluation-length slice can retain
        # a small residual mean (typical Gaussian edge effect). Subtract
        # the local mean here so the §3.2.4 / §3.2.5 peak/valley
        # equivalence to ``max(seg)`` / ``|min(seg)|`` holds. ``info``
        # returns the raw offset so the caller can surface a warning if
        # the residual is unusually large (real filtering pathology).
        info = {"mean_offset": 0.0, "span": 0.0}
        if roughness.size:
            span = float(np.max(roughness) - np.min(roughness))
            mean_off = float(np.mean(roughness))
            info["mean_offset"] = mean_off
            info["span"] = span
            roughness = roughness - mean_off

        Ra_i, Rq_i, Rsk_i, Rku_i = [], [], [], []
        Rp_i, Rv_i, Rz_i = [], [], []
        for i in range(nsc):
            seg = roughness[i * lsc_samples:(i + 1) * lsc_samples]
            if seg.size == 0:
                continue
            # §4.2 amplitude parameters over one sampling length.
            Ra_i.append(float(np.mean(np.abs(seg))))
            rq = float(np.sqrt(np.mean(seg ** 2)))
            Rq_i.append(rq)
            if rq > 0:
                Rsk_i.append(float(np.mean(seg ** 3)) / (rq ** 3))
                Rku_i.append(float(np.mean(seg ** 4)) / (rq ** 4))
            else:
                Rsk_i.append(0.0)
                Rku_i.append(0.0)
            # §4.1.1 – §4.1.3 peak / valley parameters over one sampling
            # length. Clip at 0 so an all-positive or all-negative slice
            # still yields Rp ≥ 0 and Rv ≥ 0.
            sp = max(float(np.max(seg)), 0.0)
            sv = max(float(-np.min(seg)), 0.0)
            Rp_i.append(sp)
            Rv_i.append(sv)
            Rz_i.append(sp + sv)

        if Ra_i:
            params['Ra']  = (float(np.mean(Ra_i)),  y_units)
            params['Rq']  = (float(np.mean(Rq_i)),  y_units)
            params['Rsk'] = (float(np.mean(Rsk_i)), "")
            params['Rku'] = (float(np.mean(Rku_i)), "")
            params['Rp']  = (float(np.mean(Rp_i)),  y_units)
            params['Rv']  = (float(np.mean(Rv_i)),  y_units)
            params['Rz']  = (float(np.mean(Rz_i)),  y_units)
        else:
            for k in ('Ra', 'Rq', 'Rp', 'Rv', 'Rz'):
                params[k] = (0.0, y_units)
            params['Rsk'] = (0.0, "")
            params['Rku'] = (0.0, "")

        # §4.1.5 — Rt is defined over the whole evaluation length.
        if roughness.size:
            params['Rt'] = (float(np.max(roughness) - np.min(roughness)), y_units)
        else:
            params['Rt'] = (0.0, y_units)

        # Rmr — Cref/Rz-quarter heuristic kept from the ISO 21920 helper
        # (not the literal §4.5.1 Rmr(c)). See docstring note above.
        n = roughness.size
        Rz_eval = params['Rz'][0]
        unit_label = f"% (@Rz/4, Cref={Cref:g}%)"
        if n > 0:
            sorted_desc = np.sort(roughness)[::-1]
            idx = max(0, min(n - 1, int(round(Cref / 100.0 * n))))
            c0 = sorted_desc[idx]
            target = c0 - Rz_eval / 4.0
            above = int(np.sum(sorted_desc >= target))
            params['Rmr'] = (above / n * 100.0, unit_label)
        else:
            params['Rmr'] = (0.0, unit_label)
        return params, info

    # ------------------------------------------------------------------
    # Multi-standard comparison computation.
    # ------------------------------------------------------------------
    def compute_comparison_params(self, Cref=5.0):
        """Compute R-parameters under ISO 21920, ISO 4287, and ASME B46.1.

        Filter pipelines:

        - ISO 21920: S-filter (λs) + L-filter (λc) — already computed and
          stored in :attr:`R_params`.
        - ISO 4287: S-filter (λs) + L-filter (λc) per §3.1.6 Note 1 (the
          transmission band of the roughness profile is defined by *both*
          λs and λc). The filtered roughness profile therefore matches
          the ISO 21920 one — the two columns differ only in parameter
          math (per-sampling-length averaging here vs the ISO 21920
          section-based aggregation in :meth:`_compute_r_params`).
        - B46.1: identical math and filter to ISO 4287 under harmonised
          Gaussian filtering.

        Stores the result in :attr:`comparison_params` as
        ``{'ISO 21920': {...}, 'ISO 4287': {...}, 'B46.1': {...}}``.
        """
        # Recompute ISO 21920 Rmr with the requested Cref.
        self.compute_Rmr(Cref=Cref)
        # ISO 21920 values are already in self.R_params.
        iso21920_params = {k: self.R_params[k] for k in
                          ['Ra', 'Rq', 'Rp', 'Rv', 'Rz', 'Rt', 'Rsk', 'Rku', 'Rmr']}

        # ISO 4287 / B46.1 filter chain: S-filter (λs) then L-filter (λc),
        # per §3.1.6 Note 1. Identical to the ISO 21920 pipeline; the
        # difference between columns is purely in the parameter math.
        denoised_full = self._gauss_filter(self.primary[1], 1.0 / self.short_cutoff)
        waviness_4287_full = self._gauss_filter(denoised_full, 1.0 / self.long_cutoff)
        roughness_4287_full = denoised_full - waviness_4287_full

        # Slice to the same evaluation window used for ISO 21920.
        roughness_4287 = roughness_4287_full[self._le_start:self._le_end]

        iso4287_params, iso4287_info = self._compute_r_params_iso4287(
            roughness_4287, self.nsc, self._lsc_samples, self.y_units, Cref=Cref
        )
        # B46.1 with Gaussian filter is identical to ISO 4287.
        b461_params, _ = self._compute_r_params_iso4287(
            roughness_4287, self.nsc, self._lsc_samples, self.y_units, Cref=Cref
        )

        # Reset then repopulate any non-fatal comparison notes. Currently
        # only flags a large post-filter residual mean (> 1% of span);
        # normal Gaussian edge residuals stay silent.
        self.comparison_warnings = []
        span = iso4287_info.get("span", 0.0)
        mean_off = abs(iso4287_info.get("mean_offset", 0.0))
        if span > 0 and mean_off > 1e-2 * span:
            self.comparison_warnings.append(
                f"ISO 4287 / B46.1 roughness had residual mean "
                f"{mean_off:.3g} {self.y_units} "
                f"({mean_off / span * 100:.2f}% of span); demeaned before "
                "parameter extraction (ISO 4287 §3.1.5)."
            )

        self.comparison_params = {
            'ISO 21920': iso21920_params,
            'ISO 4287': iso4287_params,
            'B46.1': b461_params,
        }

    # ------------------------------------------------------------------
    # ISO 16610-31 robust Gaussian regression filter, second order.
    # ------------------------------------------------------------------
    @staticmethod
    def _robust_gauss_regression_2nd_order(y, lc, dx, max_iter=8, tol=5e-4):
        """ISO 16610-31:2010 robust Gaussian regression filter (2nd order).

        Iteratively reweighted local quadratic regression with Gaussian
        spatial weights of standard deviation σ = λc / (2π). The iterative
        weight uses Tukey's biweight on residuals scaled by 1.4826·MAD,
        with the ISO 16610-31 cut-off constant cb = 4.4478.

        Parameters
        ----------
        y : ndarray
            1-D profile (already S-filtered if applicable).
        lc : float
            Long-wavelength cutoff in distance units (same as ``dx``).
        dx : float
            Sample spacing in the same distance unit as ``lc``.

        Returns
        -------
        ndarray, shape ``y.shape``
            Robust mean line.
        """
        y = np.asarray(y, dtype=float)
        n = y.size
        if n < 5:
            return y.copy()

        sigma = lc / (2.0 * np.pi)
        sigma_samp = sigma / dx
        # 5σ truncation is sufficient when delta-weights down-rank the tail.
        half = max(int(np.ceil(5.0 * sigma_samp)), 2)
        k = np.arange(-half, half + 1, dtype=float)
        u = k * dx
        # Unnormalised Gaussian — proportional constants cancel in the
        # weighted-least-squares normal equations.
        g = np.exp(-0.5 * (u / sigma) ** 2)
        K0 = g
        K1 = g * u
        K2 = g * (u ** 2)
        K3 = g * (u ** 3)
        K4 = g * (u ** 4)

        # Initial estimate: ordinary Gaussian convolution.
        norm = g.sum()
        mean_line = signal.fftconvolve(y, g / norm, mode="same")

        cb = 4.4478  # ISO 16610-31 robust scale constant
        for _ in range(max_iter):
            residuals = y - mean_line
            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med)))
            scale = max(1.4826 * mad, 1e-12)
            ur = residuals / (cb * scale)
            delta = np.where(np.abs(ur) < 1.0, (1.0 - ur ** 2) ** 2, 0.0)

            # Weighted moments via FFT convolutions (zero-padded edges).
            M0 = signal.fftconvolve(delta, K0, mode="same")
            M1 = signal.fftconvolve(delta, K1, mode="same")
            M2 = signal.fftconvolve(delta, K2, mode="same")
            M3 = signal.fftconvolve(delta, K3, mode="same")
            M4 = signal.fftconvolve(delta, K4, mode="same")
            dy = delta * y
            T0 = signal.fftconvolve(dy, K0, mode="same")
            T1 = signal.fftconvolve(dy, K1, mode="same")
            T2 = signal.fftconvolve(dy, K2, mode="same")

            # Solve the 3x3 normal equations at every point via Cramer's rule.
            # | M0 M1 M2 | |a|   |T0|
            # | M1 M2 M3 | |b| = |T1|
            # | M2 M3 M4 | |c|   |T2|
            det_A = (M0 * (M2 * M4 - M3 * M3)
                     - M1 * (M1 * M4 - M3 * M2)
                     + M2 * (M1 * M3 - M2 * M2))
            det_a = (T0 * (M2 * M4 - M3 * M3)
                     - M1 * (T1 * M4 - M3 * T2)
                     + M2 * (T1 * M3 - M2 * T2))
            safe_det = np.where(np.abs(det_A) < 1e-300, 1.0, det_A)
            new_mean = np.where(np.abs(det_A) < 1e-300, mean_line, det_a / safe_det)

            denom = max(float(np.max(np.abs(y))), 1e-12)
            change = float(np.max(np.abs(new_mean - mean_line))) / denom
            mean_line = new_mean
            if change < tol:
                break
        return mean_line

    def wear_track_depth(self):
        return min(self.primary[1]), "μm"

    # ------------------------------------------------------------------
    # Plot annotation helpers (ISO 21920-3 conformance)
    # ------------------------------------------------------------------
    def _iso_banner_text(self) -> str:
        """Return a 1- or 2-line ISO conformance banner.

        The banner identifies the standards used and the active setting class
        so any rendered figure is self-describing.
        """
        line1 = "ISO 21920-3:2021 / ISO 21920-2:2021"
        sc_name = getattr(self, 'setting_class_name', 'Custom')
        sc_obj = getattr(self, 'setting_class', None)
        if sc_obj is not None:
            line2 = (
                f"Setting class: {sc_name}   "
                f"λc = {self.long_cutoff:g} {self.x_units}   "
                f"λs = {self.short_cutoff:g} {self.x_units}   "
                f"le = {self.evaluation_length:g} {self.x_units}   "
                f"nsc = {self.nsc}   "
                f"dx = {getattr(self, 'dx', float('nan')):.3g} {self.x_units}"
            )
        else:
            line2 = (
                f"Setting class: Custom   "
                f"λc = {self.long_cutoff:g} {self.x_units}   "
                f"λs = {self.short_cutoff:g} {self.x_units}   "
                f"le = {self.evaluation_length:g} {self.x_units}   "
                f"nsc = {self.nsc}"
            )
        warnings = []
        if getattr(self, 'short_trace_override', None):
            warnings.append(
                f"⚠ Short-trace override — nsc = 1, edge artefacts present; "
                f"results non-conformant with ISO 21920-3"
            )
        if getattr(self, 'nsc_warning', None):
            warnings.append(
                f"⚠ nsc reduced to {self.nsc} (ISO default {self.target_nsc}) "
                f"— trace too short for full le"
            )
        if getattr(self, 'dx_warning', None):
            warnings.append(f"⚠ {self.dx_warning}")
        for note in getattr(self, 'comparison_warnings', []) or []:
            warnings.append(f"⚠ {note}")
        # If source units differ from report units, flag the conversion so
        # exported figures self-document the unit transformation applied.
        src_x = getattr(self, 'source_x_units', None)
        src_y = getattr(self, 'source_y_units', None)
        if src_x and src_y and (src_x != self.x_units or src_y != self.y_units):
            warnings.append(
                f"Source units: {src_x} / {src_y} \u2192 "
                f"report: {self.x_units} / {self.y_units}"
            )
        if warnings:
            return "\n".join([line1, line2] + warnings)
        return f"{line1}\n{line2}"

    def _draw_section_overlays(self, ax, *, show_section_rz=True) -> None:
        """Overlay ISO 21920-3 sampling-section dividers on an axis.

        Draws ``nsc + 1`` thin dotted vertical lines at the section edges
        and (optionally) annotates each section with its peak-to-valley
        ``Rzi`` value.
        """
        if self.section_edges is None or len(self.section_edges) < 2:
            return
        for x_edge in self.section_edges:
            ax.axvline(x_edge, color='dimgrey', ls=':', lw=0.7, alpha=0.8,
                       zorder=2)
        # le label sits just inside the axes at the top centre.
        x0, xN = float(self.section_edges[0]), float(self.section_edges[-1])
        ax.text(
            0.5 * (x0 + xN), 0.98,
            f'le = {self.evaluation_length:g} {self.x_units}',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=7, color='dimgrey', alpha=0.9,
        )
        if show_section_rz and self.section_Rz_values:
            for i, Rz_i in enumerate(self.section_Rz_values):
                xc = 0.5 * (float(self.section_edges[i])
                            + float(self.section_edges[i + 1]))
                ax.text(
                    xc, 0.92, f"Rz{i+1}={Rz_i:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6, color='dimgrey',
                    alpha=0.85,
                )

    def _shade_le_buffer(self, ax) -> None:
        """Shade the trimmed end-buffer regions on the primary axis."""
        if not getattr(self, 'le_buffer_regions', None):
            return
        for x_lo, x_hi in self.le_buffer_regions:
            if x_hi > x_lo:
                ax.axvspan(x_lo, x_hi, color='lightgrey', alpha=0.25,
                           zorder=0, label='_nolegend_')

    def plot_roughness(self, y_lim=None):
        fig, axs = plt.subplots(2, 1, figsize=(9, 4))
        
        # plot primary and waviness together
        axs[0].plot(*self.primary, linewidth=0.5, color="blue")
        axs[0].plot(*self.waviness, linewidth=0.5, color="red")
        PW_title = (f"Primary + Waviness, λc = {self.long_cutoff} {self.x_units}, "
                    f"λs = {self.short_cutoff} {self.x_units}")
        axs[0].set_title(PW_title)

        # plot roughness
        axs[1].plot(*self.roughness, linewidth=0.5, color="green")
        R_title = (f"Roughness  (λc = {self.long_cutoff} {self.x_units}, "
                   f"λs = {self.short_cutoff} {self.x_units}, "
                   f"le = {self.evaluation_length:g} {self.x_units}, "
                   f"nsc = {self.nsc})")
        axs[1].set_title(R_title)

        # ISO 21920-3 sectioning + le window overlays
        self._draw_section_overlays(axs[1])
        self._shade_le_buffer(axs[0])

        minor_locator = AutoMinorLocator(2)
        # set x and y limits
        x_lim = (self.primary[0][0], self.primary[0][-1])
        for a in axs:
            a.set_xlabel(f"size, {self.x_units}")
            a.set_ylabel(f"height, {self.y_units}")
            a.set_xlim(x_lim)
            if y_lim: a.set_ylim(y_lim)
            a.axhline(0, color="black", linewidth=0.5)
            a.grid(True, which='both',
                   axis='both',
                   linewidth=0.5,
                   linestyle=(0, (5, 10)),
                   color='grey', 
                   alpha=0.5)
            #a.minorticks_on()
            a.yaxis.set_minor_locator(minor_locator)
            a.xaxis.set_minor_locator(minor_locator)

        # add R-Parameters to plot
        p = self._iso_banner_text() + '\n\n'
        for key in self.R_params:
            value, unit = self.R_params[key]
            p += f"{key} = {self.format_param_value(key, value, unit)}{unit}\n"
        fig.text(0.02, 0.05, p, 
                    transform=axs[1].transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', 
                    alpha=0.5))

        plt.tight_layout()
        plt.draw()

    @timeit
    def get_material_ratio(self, samples=1000, Pk_Offset=0.01, Vy_Offset=0.01):
        self.mr_params = {}
        self.Pk_Offset, self.Vy_Offset = Pk_Offset, Vy_Offset

        # ISO 21920-3 Table 1: Rk-family parameters use the ISO 16610-31
        # robust 2nd-order Gaussian regression filter as their L-operator.
        # Fall back to the linear-filtered roughness if the robust output
        # is unavailable for any reason.
        if hasattr(self, 'roughness_robust') and self.roughness_robust is not None:
            mr_source = np.vstack((self.roughness[0], self.roughness_robust))
        else:
            mr_source = self.roughness

        # sort the uniformly sampled profile in descending order
        self.material_ratio = np.sort(mr_source[1])[::-1]
        self.material_ratio_all = mr_source[:, mr_source[1].argsort()[::-1]]
        size = mr_source[0].size
        x = np.linspace(0, 100, size)
        self.material_ratio_all = np.vstack((self.material_ratio_all, x))

        # the sampling distance
        deltaX = self.material_ratio.size / samples
        interpolated_material_ratio = np.zeros((2, samples))
        for i in range(samples):
            # find the index of the profile that is closest to the 
            # interpolated value
            index = int(i * deltaX)
            interpolated_material_ratio[0][i] = 100 * i / samples
            interpolated_material_ratio[1][i] = self.material_ratio[index]
        self.material_ratio = np.asarray(interpolated_material_ratio)

        # calc best fit straight line which includes 40% of measured points
        delta40 = int(0.4 * samples)
        bf40_grad = float('inf') # best fit gradient of 40% kernel
        for i in range(samples - delta40):
            x = self.material_ratio[0][i:i + delta40]
            y = self.material_ratio[1][i:i + delta40]
            # use least square line instead of secant
            m, c = np.polyfit(x, y, 1)
            if abs(m) < bf40_grad:
                bf40_grad = abs(m)
                self.bf40_eq = (m, c)
            if abs(m) > bf40_grad: break

        # y = mx + c
        self.bf40at0 = self.bf40_eq[1]
        self.bf40at100 = self.bf40_eq[0] * 100 + self.bf40at0
        self.mr_params['Rvkx'] = self.bf40at100 - self.material_ratio_all[1][-1] 
        self.mr_params['Rk'] = self.bf40at0 - self.bf40at100
        # intersection of self.roughness and best fit line
        self.mr_params['Rpkx'] = self.material_ratio_all[1][0] - self.bf40at0

        # calculate Rmrk and Rak params
        self.mr_params['Rak1'] = 0
        self.mr_params['Rak2'] = 0
        self.Rak1_points = [], []
        self.Rak2_points = [], []

        # iterate from bottom of profile up to bf line x=100 intercept
        for i, y in enumerate(self.material_ratio[1][::-1]):
            x = self.material_ratio[0][::-1][i]
            if y > self.bf40at100:
                self.mr_params['Rmrk2'] = x
                self.Rmrk2_y = y
                break

        # calculate Rak values
        count = 0
        self.mr_params['Rmrk1'] = 0
        self.Rmrk1_y = 0
        for i in range(self.material_ratio_all[1].size - 1):
            dt = self.material_ratio_all[2][i + 1] - \
                 self.material_ratio_all[2][i]
            x = self.material_ratio_all[0][i]   # position
            x_p = self.material_ratio_all[2][i] # percentage
            y = self.material_ratio_all[1][i]   # height
            y_1 = self.material_ratio_all[1][i + 1] # next height

            if y > self.bf40at0:
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(y)
                self.mr_params['Rak1'] += dt * ((y + y_1) / 2 - self.bf40at0)
                count += 1
            else:
                if not self.mr_params['Rmrk1']:
                    self.mr_params['Rmrk1'] = x_p
                    self.Rmrk1_y = y
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(np.nan)

        count = 0
        self.mr_params['Rmrk2'] = 0
        self.Rmrk2_y = 0
        for i in range(self.material_ratio_all[1].size - 1):
            dt = self.material_ratio_all[2][i + 1] - \
                 self.material_ratio_all[2][i]
            x = self.material_ratio_all[0][::-1][i]   # position
            x_p = self.material_ratio_all[2][::-1][i] # percentage
            y = self.material_ratio_all[1][::-1][i]   # height
            y_1 = self.material_ratio_all[1][::-1][i + 1] # next height

            if y < self.bf40at100:
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(y)
                self.mr_params['Rak2'] -= dt * ((y + y_1) / 2 - self.bf40at100)
                count += 1
            else:
                if not self.mr_params['Rmrk2']:
                    self.mr_params['Rmrk2'] = x_p
                    self.Rmrk2_y = y
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(np.nan)

        # for plotting plateau and dale regions on top of the roughness profile
        self.Rak1_points = np.asarray(self.Rak1_points)
        self.Rak2_points = np.asarray(self.Rak2_points)
        self.Rak1_points = self.Rak1_points[:, self.Rak1_points[0, :].argsort()]
        self.Rak2_points = self.Rak2_points[:, self.Rak2_points[0, :].argsort()]

        
        self.mr_params['Rpk'] = 2 * self.mr_params['Rak1'] / \
                                self.mr_params['Rmrk1']
        self.mr_params['Rvk'] = 2 * self.mr_params['Rak2'] / \
                                (100 - self.mr_params['Rmrk2'])

        print(f'num_data_points: {self.material_ratio_all[1].size}')
        print(f'Rak1: {self.mr_params["Rak1"]:.2f}, Rmrk1: {self.mr_params["Rmrk1"]:.2f}')
        print(f'Rak2: {self.mr_params["Rak2"]:.2f}, Rmrk2: {self.mr_params["Rmrk2"]:.2f}')
        print(f'bf40at100: {self.bf40at100:.2f}, bf40_eq: {self.bf40at0:.2f}')
        print(f'Rpk: {self.mr_params["Rpk"]:.2f} Rvk: {self.mr_params["Rvk"]:.2f}')

    def plot_material_ratio(self):
        self.get_material_ratio()
        fig, axs = plt.subplots(nrows=2, ncols=2,
                                figsize=(12, 6),
                                gridspec_kw={'width_ratios': [2, 1],
                                             'height_ratios': [30, 1]})

        # combine bottom two axes and create space for R-Parameters
        gs = axs[1, 1].get_gridspec()
        for a in axs[1,:]:
            a.set_axis_off()
        
        # plot roughness + plateau and dale regions
        axs[0, 0].plot(*self.roughness, linewidth=0.5, color="blue")
        axs[0, 0].set_xlim(self.roughness[0][0], self.roughness[0][-1])
        axs[0, 0].plot(*self.Rak1_points, linewidth=0.75, color="green")
        axs[0, 0].plot(*self.Rak2_points, linewidth=0.75, color="green")

        
        #print(f'Rpk_line: {Rpk_line}, Rvk_line: {Rvk_line}')

        # plot material ratio
        axs[0, 1].plot(*self.material_ratio, color="red")
        axs[0, 1].set_ylim(0, 100)
        for xlabel_i in axs[0, 1].get_yticklabels():
            xlabel_i.set_visible(False)
        x = np.linspace(0, 100, 100)
        y = self.bf40_eq[0] * x + self.bf40at0
        axs[0, 1].plot(x, y, color="blue", linewidth=0.5)
        axs[0, 1].set_xlim(0, 100)
        #plot a dot at Rmrk1 & Rmrk2
        axs[0, 1].plot(self.mr_params['Rmrk1'], self.Rmrk1_y,
                       'x', color="green")
        axs[0, 1].plot(self.mr_params['Rmrk2'], self.Rmrk2_y,
                       'x', color="green")

        for a in axs[0, :]:
            a.minorticks_on()
            a.set_ylim(round(min(self.material_ratio[1]) - .2, 2), 
                       round(max(self.material_ratio[1]) + .2, 2))
            a.axhline(0, color="black", linewidth=0.5)
            # plot 40% kernel lines
            a.axhline(self.bf40at0, linestyle='--',
                      color="green", linewidth=0.5)
            a.axhline(self.bf40at100, linestyle='--',
                      color="green", linewidth=0.5)
            # plot Rpk and Rvk lines
            a.axhline(y=self.Rmrk1_y + self.mr_params['Rpk'], 
                      linestyle='dotted', color="k", linewidth=0.75)
            a.axhline(y=self.Rmrk2_y +-self.mr_params['Rvk'],
                      linestyle='dotted', color="k", linewidth=0.75)

        # create Rmr parameters table
        rows = ['Rk', 'Rpk', 'Rvk', 'Rmrk1', 'Rmrk2', 'Rak1', 'Rak2', 'Rpkx', 'Rvkx']
        units = ['μm', 'μm', 'μm', 'μm', 'μm', '%', '%', 'μm.%', 'μm.%']
        columns = ['ISO 21920-2:2021 Parameter', 'Value', 'Unit']
        n_rows = len(rows)
        
        cell_text = []
        for i in range(n_rows):
            cell_text.append([round(self.mr_params[rows[i]], 3), units[i]])
        mr_table = table(axs[1, 1], cellText=cell_text,
                                 rowLabels=rows,
                                 colLabels=columns,
                                 loc='top',
                                 fontsize=10)
        
        axs[0, 0].set_title(f"{os.path.split(self.raw_data)[1]} - Roughness")
        axs[0, 1].set_title("Material Ratio")
        plt.tight_layout()
        plt.draw()

    def compute_Rmr(self, Cref=5.0):
        """Material ratio at slicing level (peak_at_Cref% - Rz/4).

        Stores the result on ``self.R_params['Rmr']`` and saves the slicing
        level on ``self.Rmr_target_level`` for plotting.
        """
        rough = self.roughness[1]
        n = rough.size
        unit_label = f"% (@Rz/4, Cref={Cref:g}%)"
        if n == 0:
            self.R_params['Rmr'] = (0.0, unit_label)
            self.Rmr_target_level = 0.0
            self.Rmr_Cref = float(Cref)
            self.Rmr_Cref_level = 0.0
            self.Rmr_Rz4_drop = 0.0
            return
        Rz = self.R_params['Rz'][0]
        sorted_desc = np.sort(rough)[::-1]
        idx = max(0, min(n - 1, int(round(Cref / 100.0 * n))))
        c0 = sorted_desc[idx]
        target = c0 - Rz / 4.0
        above = int(np.sum(sorted_desc >= target))
        Rmr = above / n * 100.0
        self.R_params['Rmr'] = (Rmr, unit_label)
        self.Rmr_target_level = target
        self.Rmr_Cref = float(Cref)
        self.Rmr_Cref_level = c0
        self.Rmr_Rz4_drop = Rz / 4.0

    def build_overview_figure(self, Cref=5.0, y_lim=None, comparison=False):
        """Build a matplotlib ``Figure`` (no pyplot) suitable for GUI embedding.

        The figure contains four panels:
          * Raw data with the polynomial leveling fit overlaid
          * Roughness profile with the waviness curve overlaid (red)
          * Bearing ratio (Abbott) curve, depth on y-axis in y-units, % on x-axis
          * R-parameters table

        Parameters
        ----------
        comparison : bool
            If True, show multi-standard comparison table (ISO 21920 / ISO 4287 / B46.1).
        """
        from matplotlib.figure import Figure

        if not hasattr(self, 'material_ratio'):
            self.get_material_ratio()
        self.compute_Rmr(Cref=Cref)

        # Wider default figure to avoid cramped right-hand panels/table clipping.
        fig = Figure(figsize=(15.0, 7.5))
        gs = fig.add_gridspec(2, 2,
                  width_ratios=[2.25, 1.45],
                      hspace=0.40, wspace=0.30)
        # Reserve extra room for the multi-line ISO banner/suptitle.
        fig.subplots_adjust(top=0.82, bottom=0.08, left=0.06, right=0.98)
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_bear = fig.add_subplot(gs[0, 1])
        ax_rough = fig.add_subplot(gs[1, 0])
        ax_table = fig.add_subplot(gs[1, 1])

        # --- Raw data + leveling fit -------------------------------------
        ax_raw.plot(self.raw_data_xy[0], self.raw_data_xy[1],
                    lw=0.5, color='steelblue', label='raw')
        if self.level_fit_y is not None:
            ax_raw.plot(self.level_fit_x, self.level_fit_y,
                        color='red', lw=1.0,
                        label=f'order {self.order} fit')
            ax_raw.legend(fontsize=8, loc='best')
        ax_raw.set_title('Raw Data + Leveling Fit', pad=6)
        ax_raw.set_xlabel(f'size, {self.x_units}')
        ax_raw.set_ylabel(f'height, {self.y_units}')
        if y_lim is not None:
            ax_raw.set_ylim(y_lim)
        ax_raw.grid(True, ls=':', alpha=0.5)
        # Shade the trimmed end-buffer regions to communicate the le window.
        self._shade_le_buffer(ax_raw)

        # --- Roughness + waviness ---------------------------------------
        ax_rough.plot(*self.roughness, lw=0.5, color='green', label='roughness')
        ax_rough.plot(*self.waviness, lw=0.9, color='red', label='waviness')
        # Overlay ISO 16610-31 robust mean line used for the Rk-family.
        if hasattr(self, 'robust_mean') and self.robust_mean is not None:
            ax_rough.plot(
                self.roughness[0], self.robust_mean,
                color='darkorange', lw=0.9, ls='--',
                label='ISO 16610-31 robust mean (Rk-family L-filter)',
            )
        ax_rough.set_title(
            f'Roughness + Waviness  (\u03bbc = {self.long_cutoff:g} {self.x_units}, '
            f'\u03bbs = {self.short_cutoff:g} {self.x_units}, '
            f'le = {self.evaluation_length:g} {self.x_units}, nsc = {self.nsc})'
        )
        ax_rough.set_xlabel(f'size, {self.x_units}')
        ax_rough.set_ylabel(f'height, {self.y_units}')
        if y_lim is not None:
            ax_rough.set_ylim(y_lim)
        ax_rough.axhline(0, color='black', lw=0.5)
        ax_rough.legend(fontsize=7, loc='best')
        ax_rough.grid(True, ls=':', alpha=0.5)
        # Section dividers + per-section Rz markers.
        self._draw_section_overlays(ax_rough, show_section_rz=True)

        # --- Bearing ratio curve ----------------------------------------
        # self.material_ratio[0] is %; self.material_ratio[1] is depth
        ax_bear.plot(self.material_ratio[0], self.material_ratio[1],
                     color='blue', lw=1.0)
        ax_bear.set_title('Bearing Ratio Curve', pad=6)
        ax_bear.set_xlabel('Material Ratio %')
        ax_bear.set_ylabel(f'depth, {self.y_units}')
        ax_bear.set_xlim(0, 100)
        if y_lim is not None:
            ax_bear.set_ylim(y_lim)
        ax_bear.grid(True, ls=':', alpha=0.5)
        if hasattr(self, 'Rmr_target_level'):
            cref = getattr(self, 'Rmr_Cref', Cref)
            cref_level = getattr(self, 'Rmr_Cref_level', self.Rmr_target_level)
            rz4_drop = getattr(self, 'Rmr_Rz4_drop', 0.0)
            Rmr_val = self.R_params['Rmr'][0]

            # Construction lines for Rmr: start at Cref, drop by Rz/4,
            # then read the resulting material ratio.
            ax_bear.axvline(cref, color='darkorange', ls='--', lw=0.9,
                            label=f'Cref = {cref:g}%')
            ax_bear.plot(cref, cref_level, marker='o', ms=4,
                         color='darkorange', zorder=4)
            ax_bear.plot([cref, cref], [cref_level, self.Rmr_target_level],
                         color='darkorange', ls='-.', lw=1.0,
                         label=f'Rz/4 drop = {self.format_param_value("Rz", rz4_drop, self.y_units)} {self.y_units}')

            # Visual step guides: depth at Cref, then horizontal read to Rmr.
            ax_bear.axhline(cref_level, color='red', ls=':', lw=0.9,
                            label=f'Depth at Cref ({cref:g}%)')
            ax_bear.plot([cref, Rmr_val], [self.Rmr_target_level, self.Rmr_target_level],
                         color='red', ls=':', lw=1.0,
                         label='Read across to Rmr')

            ax_bear.axhline(self.Rmr_target_level, color='gray',
                            ls='--', lw=0.7,
                            label='Target level (Cref - Rz/4)')
            ax_bear.axvline(Rmr_val, color='magenta', ls=':', lw=0.8,
                            label=f"Rmr = {self.format_param_value('Rmr', Rmr_val, '%')}%")
            ax_bear.legend(fontsize=7, loc='best')

        # --- R-parameter table -------------------------------------------
        ax_table.set_axis_off()
        if comparison and hasattr(self, 'comparison_params'):
            rows = ['Ra', 'Rq', 'Rp', 'Rv', 'Rz', 'Rt', 'Rsk', 'Rku', 'Rmr']
            standards = ['ISO 21920', 'ISO 4287', 'B46.1']
            cells = []
            for r in rows:
                row_cells = []
                for std in standards:
                    v, u = self.comparison_params[std].get(r, ('-', ''))
                    if isinstance(v, (int, float)):
                        row_cells.append(self.format_param_value(r, v, u))
                    else:
                        row_cells.append(str(v))
                # Append unit from first standard (all share same units).
                _, u = self.comparison_params[standards[0]].get(r, ('-', ''))
                row_cells.append(u)
                cells.append(row_cells)
            col_labels = ['ISO 21920', 'ISO 4287', 'B46.1', 'Unit']
            tbl = ax_table.table(cellText=cells,
                         rowLabels=rows,
                         colLabels=col_labels,
                         loc='center', cellLoc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.15])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.08, 1.42)
            ax_table.set_title('R-Parameters (Multi-Standard)', fontsize=9, pad=12)
        else:
            rows = ['Ra', 'Rq', 'Rp', 'Rv', 'Rz', 'Rt', 'Rsk', 'Rku', 'Rmr']
            cells = []
            for r in rows:
                v, u = self.R_params.get(r, ('-', ''))
                if isinstance(v, (int, float)):
                    cells.append([self.format_param_value(r, v, u), u])
                else:
                    cells.append([str(v), u])
            tbl = ax_table.table(cellText=cells,
                         rowLabels=rows,
                         colLabels=['Value', 'Unit'],
                         loc='center', cellLoc='center',
                         colWidths=[0.43, 0.57])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.08, 1.42)
            ax_table.set_title('R-Parameters', fontsize=10, pad=12)

        # --- ISO conformance banner --------------------------------------
        banner = self._iso_banner_text()
        title = f"{os.path.basename(self.raw_data)}\n{banner}"
        fig.suptitle(title, fontsize=10, x=0.5, y=0.98, ha='center')
        return fig

    def __str__(self):
        p = (
            f'\nProcessed {self.raw_data}\n'
            f'  Setting class: {self.setting_class_name}\n'
            f'  \u03bbc = {self.long_cutoff:g} {self.x_units}, '
            f'\u03bbs = {self.short_cutoff:g} {self.x_units}, '
            f'le = {self.evaluation_length:g} {self.x_units}, nsc = {self.nsc}\n'
            f'\nParam\tValue'
        )
        for key in self.R_params:
            value, unit = self.R_params[key]
            p += f"\n{key}:\t{self.format_param_value(key, value, unit)}{unit}"
        p += "\n"
        return p


if __name__ == "__main__":
    data = "example/example_trace.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff,
                                     order=1)
    #surface_texture.plot_material_ratio()
    surface_texture.plot_roughness()
    surface_texture.plot_material_ratio()
    print(surface_texture.wear_track_depth())
    

    plt.show()

