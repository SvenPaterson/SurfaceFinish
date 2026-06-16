"""Tkinter front-end for the SurfaceFinish analysis pipeline.

Loads data, runs ``SurfaceTexture`` in a worker thread, and renders a live
matplotlib figure inside the window with raw+fit, roughness+waviness, the
bearing ratio (Abbott) curve, and an R-parameter table.

Run with::

    python gui.py
"""

from __future__ import annotations

import os
import queue
import re
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from SurfaceTexture import SurfaceTexture, TraceTooShortError  # noqa: E402
from iso21920 import (  # noqa: E402
    SETTING_CLASSES,
    DEFAULT_SETTING_CLASS,
    convert_mm_to,
    get_setting_class,
)


WORKSPACE_ROOT = Path(__file__).resolve().parent

# Source / Report unit-system selectors. The canonical X/Y unit pairs are
# derived from each system label via ``_unit_system_pair``.
UNIT_SYSTEM_CHOICES = ["Metric (mm / \u03bcm)", "Standard (in / \u03bcin)"]
UNIT_SYSTEM_METRIC = UNIT_SYSTEM_CHOICES[0]
UNIT_SYSTEM_STANDARD = UNIT_SYSTEM_CHOICES[1]


def _unit_system_pair(label: str) -> tuple[str, str]:
    """Return ``(x_unit, y_unit)`` for a unit-system label."""
    if label == UNIT_SYSTEM_STANDARD:
        return "in", "\u03bcin"
    return "mm", "\u03bcm"


def _unit_system_token(label: str) -> str:
    """Return the short token (``"Metric"`` / ``"Standard"``) for a label."""
    return "Standard" if label == UNIT_SYSTEM_STANDARD else "Metric"


def _unit_system_for_x_unit(x_unit: str) -> str:
    """Return the system label whose X unit matches ``x_unit``."""
    return UNIT_SYSTEM_STANDARD if str(x_unit).lower() == "in" else UNIT_SYSTEM_METRIC


def _col_letter(index: int) -> str:
    """Excel-style column letter for a 0-based ``index`` (A, B, ..., Z, AA)."""
    if index < 0:
        raise ValueError("Column index must be non-negative.")
    letters = ""
    n = index
    while True:
        n, rem = divmod(n, 26)
        letters = chr(ord("A") + rem) + letters
        if n == 0:
            break
        n -= 1
    return letters


def _letter_to_col(letter: str) -> int | None:
    """Inverse of :func:`_col_letter`. Returns ``None`` if not pure letters
    or if the token is too long to plausibly be a spreadsheet column ref
    (more than 3 characters), so values like ``"Time"`` keep being treated
    as bare column names instead of misread as a letter index.
    """
    s = str(letter or "").strip().upper()
    if not s or not s.isalpha() or len(s) > 3:
        return None
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


# Regex unit-marker tables for source-unit auto-detection. We avoid bare
# "m" (too easy to false-match "mean", "method", etc.) and bare "n"/"u".
_STANDARD_UNIT_RE = re.compile(
    r"\b(?:in|inch|inches|mil|\u03bcin|\u00b5in|uin)\b", re.IGNORECASE
)
_METRIC_UNIT_RE = re.compile(
    r"\b(?:mm|cm|\u03bcm|\u00b5m|um|nm)\b", re.IGNORECASE
)


def _classify_dx(dx: float) -> str | None:
    """Numeric magnitude heuristic: small dx → likely inches, large → mm."""
    if not (dx and dx > 0):
        return None
    if dx < 1e-4:
        return "Standard"
    if dx > 1e-3:
        return "Metric"
    return None


def _detect_source_system(path: str, ext: str) -> tuple[str | None, str]:
    """Best-effort guess at the source unit system for ``path``.

    Returns ``(detected, reason)`` where ``detected`` is ``"Metric"``,
    ``"Standard"``, or ``None`` (couldn't decide). ``reason`` is a short
    human-readable explanation suitable for a confirmation popup.

    Strategy:
      1. Scan a small sample of cells for explicit unit text
         (e.g. ``"X (in)"``, ``"mm"``, ``"\u03bcin"``).
      2. Fall back to the median sample spacing of the first ~200 X values
         using :func:`_classify_dx`.
    """
    ext = (ext or "").lower()
    text_blob = ""
    x_values: list[float] = []

    if ext == ".xlsx":
        try:
            import pandas as pd
            try:
                import python_calamine  # noqa: F401
                engine = "calamine"
            except ImportError:
                engine = "openpyxl"
            xls = pd.ExcelFile(path, engine=engine)
            chunks: list[str] = []
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(
                        path, sheet_name=sheet, header=None,
                        nrows=20, engine=engine,
                    )
                except Exception:  # noqa: BLE001
                    continue
                chunks.append(
                    " ".join(str(v) for v in df.values.flatten().tolist())
                )
            text_blob = " ".join(chunks)
            # Best-effort numeric fallback: read up to 200 rows from sheet 0
            # column 0, drop non-numeric.
            try:
                df0 = pd.read_excel(
                    path, sheet_name=0, header=None, nrows=200, engine=engine,
                )
                col0 = pd.to_numeric(df0.iloc[:, 0], errors="coerce").dropna()
                x_values = col0.tolist()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    elif ext in {".csv", ".txt"}:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head_lines = [next(fh, "") for _ in range(5)]
            text_blob = " ".join(head_lines)
        except Exception:  # noqa: BLE001
            pass
        # Numeric fallback over the file body.
        try:
            x_values = []
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.replace(",", " ").split()
                    if not parts:
                        continue
                    try:
                        x_values.append(float(parts[0]))
                    except ValueError:
                        continue
                    if len(x_values) >= 200:
                        break
        except Exception:  # noqa: BLE001
            pass

    # 1. Header-text match.
    has_std = bool(_STANDARD_UNIT_RE.search(text_blob))
    has_met = bool(_METRIC_UNIT_RE.search(text_blob))
    if has_std and not has_met:
        m = _STANDARD_UNIT_RE.search(text_blob)
        return "Standard", f"Found unit marker '{m.group(0)}' in file header text."
    if has_met and not has_std:
        m = _METRIC_UNIT_RE.search(text_blob)
        return "Metric", f"Found unit marker '{m.group(0)}' in file header text."
    if has_std and has_met:
        # Both present (e.g. "in/mm note"); break tie toward Metric.
        return "Metric", "Both metric and standard unit markers found; defaulting to Metric."

    # 2. Numeric magnitude fallback.
    if len(x_values) >= 3:
        import numpy as _np
        diffs = _np.abs(_np.diff(_np.asarray(x_values, dtype=float)))
        diffs = diffs[_np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dx = float(_np.median(diffs))
            cls = _classify_dx(dx)
            if cls:
                return cls, f"Median X spacing dx \u2248 {dx:.3g} \u2192 likely {cls}."
            return None, (
                f"Median X spacing dx \u2248 {dx:.3g} is ambiguous. "
                "Please confirm units."
            )

    return None, "No unit markers found and X spacing is unavailable. Please confirm units."


def _part_sort_key(part: str):
    """Sort parts numerically when their names are integers, else lexically.

    ``"(root)"`` always sorts first so root-level files appear at the top
    of summary trees and report sheets.
    """
    if part == "(root)":
        return (0, 0, "")
    try:
        return (1, int(part), "")
    except (TypeError, ValueError):
        return (2, 0, str(part).lower())


@dataclass
class _LayoutInfo:
    """Outcome of :meth:`SurfaceFinishGUI._scan_batch_root`.

    ``mode`` is one of ``flat`` / ``parts`` / ``parts_locations`` /
    ``mixed`` / ``empty``. ``files_by_part_loc`` maps a
    ``(part, location)`` tuple to the list of file paths under that
    bucket — ``location`` is ``""`` when the file lives directly in the
    Part folder, and ``part`` is ``"(root)"`` for files at the chosen
    root level.
    """
    mode: str
    root: Path
    ext: str
    files_by_part_loc: dict[tuple[str, str], list[Path]] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def files(self) -> list[Path]:
        """Flat list of files in deterministic (part, location, name) order."""
        out: list[Path] = []
        for key in sorted(
            self.files_by_part_loc.keys(),
            key=lambda kv: (_part_sort_key(kv[0]), kv[1].lower()),
        ):
            out.extend(self.files_by_part_loc[key])
        return out

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.files_by_part_loc.values())

    @property
    def has_location(self) -> bool:
        return any(loc for (_, loc) in self.files_by_part_loc.keys())

    @property
    def n_parts(self) -> int:
        return len({pl[0] for pl in self.files_by_part_loc.keys()})

    @property
    def n_locations(self) -> int:
        return len({pl for pl in self.files_by_part_loc.keys() if pl[1]})


class SurfaceFinishGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SurfaceFinish - Analysis")
        self.geometry("1912x900")
        self.minsize(1642, 760)

        self._msg_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        # Path that the source-units popup has already confirmed for. We
        # only prompt once per file load (sheet changes within the same
        # xlsx don't re-trigger the popup).
        self._source_prompted_path: str | None = None

        # ---- state vars --------------------------------------------------
        self.var_file = tk.StringVar()
        self.var_sheet = tk.StringVar()
        self.var_x_col = tk.StringVar()
        self.var_y_col = tk.StringVar()
        self.var_source_system = tk.StringVar(value=UNIT_SYSTEM_METRIC)
        self.var_report_system = tk.StringVar(value=UNIT_SYSTEM_METRIC)
        self.var_short_cutoff = tk.StringVar(value="0.0025")
        self.var_long_cutoff = tk.StringVar(value="0.8")
        self.var_setting_class = tk.StringVar(value=DEFAULT_SETTING_CLASS)
        self.var_order = tk.IntVar(value=1)
        self.var_cref = tk.StringVar(value="5")
        self.var_compare_standards = tk.BooleanVar(value=False)
        # Batch-only: whether to write a per-batch xlsx summary.
        self.var_batch_xlsx = tk.BooleanVar(value=True)

        self._build_ui()
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        self._paned = paned

        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=6)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)
        self.after(50, self._set_initial_plot_width)

    def _set_initial_plot_width(self) -> None:
        # Reserve a fixed control pane and give most width to plotting pane.
        try:
            self._paned.sashpos(0, 600)
        except Exception:
            pass

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        # --- Data Source ---
        src = ttk.LabelFrame(parent, text="Data Source")
        src.pack(fill="x", **pad)

        ttk.Label(src, text="File:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.ent_file = ttk.Entry(src, textvariable=self.var_file, width=44)
        self.ent_file.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(src, text="Browse...", command=self._browse_file).grid(
            row=0, column=2, padx=4, pady=3
        )
        self.lbl_sheet = ttk.Label(src, text="Sheet:")
        self.lbl_sheet.grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cmb_sheet = ttk.Combobox(
            src, textvariable=self.var_sheet, state="readonly", width=42
        )
        self.cmb_sheet.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=3)
        self.cmb_sheet.bind("<<ComboboxSelected>>", lambda _e: self._refresh_columns())
        src.columnconfigure(1, weight=1)

        # --- Columns ---
        cols = ttk.LabelFrame(parent, text="Columns")
        cols.pack(fill="x", **pad)
        self.frm_columns = cols
        self._cols_pack_kwargs = dict(fill="x", **pad)
        ttk.Label(cols, text="X column:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.cmb_x = ttk.Combobox(cols, textvariable=self.var_x_col, width=42)
        self.cmb_x.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(cols, text="Y column:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cmb_y = ttk.Combobox(cols, textvariable=self.var_y_col, width=42)
        self.cmb_y.grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        cols.columnconfigure(1, weight=1)

        # --- Units ---
        units = ttk.LabelFrame(parent, text="Units")
        units.pack(fill="x", **pad)
        self._units_frame = units
        ttk.Label(units, text="Source:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.cmb_source_system = ttk.Combobox(
            units,
            textvariable=self.var_source_system,
            values=UNIT_SYSTEM_CHOICES,
            state="readonly",
            width=22,
        )
        self.cmb_source_system.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        # Source change has no immediate UI effect; conversion happens at run.
        ttk.Label(units, text="Report:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cmb_report_system = ttk.Combobox(
            units,
            textvariable=self.var_report_system,
            values=UNIT_SYSTEM_CHOICES,
            state="readonly",
            width=22,
        )
        self.cmb_report_system.grid(row=1, column=1, sticky="w", padx=4, pady=3)
        self.cmb_report_system.bind(
            "<<ComboboxSelected>>", lambda _e: self._apply_setting_class()
        )

        # --- Filter Cutoffs ---
        cuts = ttk.LabelFrame(
            parent,
            text="ISO 21920-3 Filter / Evaluation (in same X distance unit as data)",
        )
        cuts.pack(fill="x", **pad)
        ttk.Label(cuts, text="Setting class:").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        sc_choices = sorted(SETTING_CLASSES.keys()) + ["Custom"]
        self.cmb_setting_class = ttk.Combobox(
            cuts, textvariable=self.var_setting_class,
            values=sc_choices, state="readonly", width=10,
        )
        self.cmb_setting_class.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        self.cmb_setting_class.bind(
            "<<ComboboxSelected>>", lambda _e: self._apply_setting_class()
        )
        ttk.Label(cuts, text="\u03bbs (short):").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(cuts, textvariable=self.var_short_cutoff, width=14).grid(
            row=1, column=1, sticky="w", padx=4, pady=3
        )
        ttk.Label(cuts, text="\u03bbc (long):").grid(row=1, column=2, sticky="w", padx=4, pady=3)
        ttk.Entry(cuts, textvariable=self.var_long_cutoff, width=14).grid(
            row=1, column=3, sticky="w", padx=4, pady=3
        )
        ttk.Label(
            cuts,
            text=("Evaluation length (le) and number of sampling sections (nsc) "
                  "are derived automatically from \u03bbc and the trace length."),
            wraplength=420, foreground="#555",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(3, 3))
        # Populate the cutoff fields with the default class once widgets exist.
        self._apply_setting_class()

        # --- Leveling + Cref ---
        params = ttk.LabelFrame(parent, text="Analysis Parameters")
        params.pack(fill="x", **pad)
        ttk.Label(params, text="Leveling order (0 skips):").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Spinbox(params, from_=0, to=3, textvariable=self.var_order, width=5).grid(
            row=0, column=1, sticky="w", padx=4, pady=3
        )
        ttk.Label(params, text="Cref % (Rmr ref level):").grid(
            row=1, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(params, textvariable=self.var_cref, width=10).grid(
            row=1, column=1, sticky="w", padx=4, pady=3
        )
        ttk.Checkbutton(
            params, text="Compare Standards", variable=self.var_compare_standards
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(
            params, text="Save Excel report (Batch only)",
            variable=self.var_batch_xlsx,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        # --- Action row ---
        actions = ttk.Frame(parent)
        actions.pack(fill="x", **pad)
        self.btn_run = ttk.Button(actions, text="Run Analysis", command=self._run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_batch = ttk.Button(
            actions, text="Batch Process", command=self._run_batch
        )
        self.btn_batch.pack(side="left", padx=4)
        ttk.Button(actions, text="Clear Output", command=self._clear_output).pack(
            side="left", padx=4
        )
        self.lbl_status = ttk.Label(actions, text="Idle")
        self.lbl_status.pack(side="right", padx=6)

        # --- Output panel ---
        out = ttk.LabelFrame(parent, text="Output")
        out.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(out, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt.configure(state="disabled")

        # Initial visibility: nothing selected yet, hide xlsx-only widgets.
        self._apply_file_visibility("")

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        plot_frame = ttk.LabelFrame(parent, text="Plot")
        plot_frame.pack(fill="both", expand=True)

        # Initial empty figure with friendly placeholder
        self._fig = Figure(figsize=(10, 7))
        ax = self._fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Choose a data file, configure options, and click Run Analysis.",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, plot_frame)
        self._toolbar.update()

    # ------------------------------------------------------------------
    # File / sheet / column wiring
    # ------------------------------------------------------------------
    def _browse_file(self) -> None:
        initial = WORKSPACE_ROOT
        if self.var_file.get():
            p = Path(self.var_file.get())
            if p.parent.exists():
                initial = p.parent
        path = filedialog.askopenfilename(
            initialdir=str(initial),
            title="Select data file",
            filetypes=[
                ("All files", "*.*"),
                ("Excel workbook", "*.xlsx"),
                ("Text/CSV", "*.txt *.csv"),
                ("Digital Surf profile", "*.pro"),
            ],
        )
        if not path:
            return
        self.var_file.set(path)
        self._refresh_sheets()

    def _refresh_sheets(self) -> None:
        self.cmb_sheet["values"] = ()
        self.var_sheet.set("")
        self.cmb_x["values"] = ()
        self.cmb_y["values"] = ()
        # New file load: forget any prior source-unit confirmation so the
        # popup can fire once for this path. (Sheet selection within an
        # already-loaded xlsx will not re-trigger it.)
        self._source_prompted_path = None

        path = self.var_file.get().strip()
        if not path:
            self._apply_file_visibility("")
            self.cmb_source_system.configure(state="readonly")
            return
        ext = Path(path).suffix.lower()
        self._apply_file_visibility(ext)
        if ext == ".pro":
            # Single-channel binary profile — no sheet, fixed X/Y columns.
            self.cmb_x["values"] = ("A",)
            self.cmb_y["values"] = ("B",)
            self.var_x_col.set("A")
            self.var_y_col.set("B")
            self._apply_pro_units(path)
            return
        # For non-.pro formats the user must declare the source units, so
        # re-enable the Source dropdown if a previous .pro file had locked it.
        self.cmb_source_system.configure(state="readonly")
        if ext != ".xlsx":
            self.cmb_x["values"] = ("A", "B")
            self.cmb_y["values"] = ("A", "B")
            self.var_x_col.set("A")
            self.var_y_col.set("B")
            self._maybe_prompt_source_units(path, ext)
            return

        try:
            import pandas as pd

            try:
                import python_calamine  # noqa: F401
                engine = "calamine"
            except ImportError:
                engine = "openpyxl"
            xls = pd.ExcelFile(path, engine=engine)
            sheets = list(xls.sheet_names)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Read error", f"Could not read workbook:\n{exc}")
            return

        self.cmb_sheet["values"] = sheets
        if sheets:
            self.var_sheet.set(sheets[0])
            self._refresh_columns()

    def _refresh_columns(self) -> None:
        path = self.var_file.get().strip()
        sheet = self.var_sheet.get().strip()
        if not path or not sheet or Path(path).suffix.lower() != ".xlsx":
            return
        try:
            import pandas as pd

            try:
                import python_calamine  # noqa: F401
                engine = "calamine"
            except ImportError:
                engine = "openpyxl"
            df_head = pd.read_excel(path, sheet_name=sheet, nrows=0, engine=engine)
            columns = [str(c) for c in df_head.columns]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Read error", f"Could not read columns from sheet '{sheet}':\n{exc}"
            )
            return

        indexed = [f"{_col_letter(i)}: {name}" for i, name in enumerate(columns)]
        values = tuple(indexed)
        self.cmb_x["values"] = values
        self.cmb_y["values"] = values
        if columns:
            self.var_x_col.set(indexed[0])
            self.var_y_col.set(indexed[1] if len(indexed) > 1 else indexed[0])
        # Confirm source units now that the sheet/column header text is
        # available — header rows are the most reliable unit-marker source.
        self._maybe_prompt_source_units(path, ".xlsx")

    # ------------------------------------------------------------------
    # Visibility / unit-aware helpers
    # ------------------------------------------------------------------
    def _apply_file_visibility(self, ext: str) -> None:
        """Show or hide the Sheet row and Columns frame based on file type.

        - ``.xlsx`` → both shown (sheet + column pickers needed).
        - ``.pro``  → both hidden (deterministic single-profile binary).
        - ``.csv`` / ``.txt`` / unknown → both hidden (fixed X=col0, Y=col1).
        """
        show = ext == ".xlsx"
        if show:
            self.lbl_sheet.grid()
            self.cmb_sheet.grid()
            # Re-pack BEFORE the Units frame so we preserve the original
            # Source → Columns → Units → Cutoffs ordering. A bare pack()
            # would append to the end of the parent's pack stack. We
            # ``pack_forget`` first (no-op if not packed) so the subsequent
            # ``pack(before=...)`` always lands in the correct slot.
            self.frm_columns.pack_forget()
            self.frm_columns.pack(before=self._units_frame, **self._cols_pack_kwargs)
        else:
            self.lbl_sheet.grid_remove()
            self.cmb_sheet.grid_remove()
            self.frm_columns.pack_forget()

    def _apply_pro_units(self, path: str) -> None:
        """Adopt the natural unit system declared by a ``.pro`` header.

        The user can still override the system afterwards (e.g. force a
        Metric report from an inch-native file) via the Units dropdown.
        """
        try:
            from pro_reader import peek_pro_units
            x_unit, _y_unit = peek_pro_units(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Read error",
                f"Could not read .pro header:\n{exc}",
            )
            return
        if x_unit == "in":
            self.var_source_system.set(UNIT_SYSTEM_STANDARD)
        elif x_unit == "mm":
            self.var_source_system.set(UNIT_SYSTEM_METRIC)
        # The .pro header is authoritative; lock the Source dropdown so the
        # user can't accidentally mis-declare the file's native units.
        self.cmb_source_system.configure(state="disabled")
        # Re-derive λs/λc defaults from the current report unit.
        self._apply_setting_class()

    def _prompt_source_units(self, detected: str | None, reason: str) -> None:
        """Confirm the source unit system with the user via a small modal.

        ``detected`` is ``"Metric"``, ``"Standard"``, or ``None``. The popup
        pre-selects ``detected`` (or the current Source value if ``None``)
        and lets the user accept or override before any analysis runs.
        Cancel keeps the current Source setting untouched.
        """
        # Map detected token to the dropdown label.
        if detected == "Standard":
            preset = UNIT_SYSTEM_STANDARD
        elif detected == "Metric":
            preset = UNIT_SYSTEM_METRIC
        else:
            preset = self.var_source_system.get() or UNIT_SYSTEM_METRIC

        win = tk.Toplevel(self)
        win.title("Confirm source units")
        win.transient(self)
        win.resizable(False, False)
        try:
            win.grab_set()
        except tk.TclError:
            # Toplevel may not yet be viewable in headless / test contexts.
            pass

        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        if detected:
            heading = f"Detected source units: {detected}"
        else:
            heading = "Could not determine source units automatically."
        ttk.Label(body, text=heading, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(body, text=reason, wraplength=380, foreground="#555").pack(
            anchor="w", pady=(0, 10)
        )
        ttk.Label(body, text="Source unit system:").pack(anchor="w")

        choice = tk.StringVar(value=preset)
        cmb = ttk.Combobox(
            body, textvariable=choice, values=UNIT_SYSTEM_CHOICES,
            state="readonly", width=24,
        )
        cmb.pack(anchor="w", pady=(2, 12))

        btns = ttk.Frame(body)
        btns.pack(fill="x")

        def on_ok() -> None:
            self.var_source_system.set(choice.get())
            win.destroy()

        def on_cancel() -> None:
            win.destroy()

        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btns, text="OK", command=on_ok).pack(side="right", padx=(0, 6))
        cmb.focus_set()
        win.bind("<Return>", lambda _e: on_ok())
        win.bind("<Escape>", lambda _e: on_cancel())

        # Center over the parent window.
        try:
            self.update_idletasks()
            x = self.winfo_rootx() + (self.winfo_width() // 2) - 200
            y = self.winfo_rooty() + (self.winfo_height() // 2) - 100
            win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except tk.TclError:
            pass

    def _maybe_prompt_source_units(self, path: str, ext: str) -> None:
        """Run source-unit detection and schedule the confirmation popup.

        Skipped for ``.pro`` (the header is authoritative) and when no path
        is set. Also skipped if the popup has already been answered for
        this exact ``path`` since the last file-pick — sheet/column changes
        within the same xlsx must not re-prompt. Detection itself scans
        every sheet, so a single pass is authoritative.
        """
        if not path or ext == ".pro":
            return
        if self._source_prompted_path == path:
            return
        self._source_prompted_path = path
        try:
            detected, reason = _detect_source_system(path, ext)
        except Exception as exc:  # noqa: BLE001
            detected, reason = None, f"Detection failed: {exc}"
        self.after(0, lambda: self._prompt_source_units(detected, reason))

    def _apply_setting_class(self) -> None:
        """Populate \u03bbs / \u03bbc / le from the selected ISO 21920-3 class.

        Lengths in the canonical class table are millimetres; convert into
        the data's X distance unit so the entry fields display values that
        the rest of the pipeline can use directly. Selecting "Custom" leaves
        the existing values untouched.
        """
        name = self.var_setting_class.get().strip()
        if not name or name == "Custom":
            return
        try:
            sc = get_setting_class(name)
        except KeyError:
            return
        if sc is None:
            return
        x_unit, _y_unit = _unit_system_pair(self.var_report_system.get())
        try:
            self.var_short_cutoff.set(f"{convert_mm_to(sc.lambda_s_mm, x_unit):g}")
            self.var_long_cutoff.set(f"{convert_mm_to(sc.lambda_c_mm, x_unit):g}")

        except ValueError:
            # Unknown unit \u2014 leave fields untouched.
            return

    # ------------------------------------------------------------------
    # Run handling
    # ------------------------------------------------------------------
    @staticmethod
    def _column_value(raw: str):
        """Parse a column-picker entry into the value SurfaceTexture expects.

        Accepts:
          * ``"A"`` / ``"BC"`` (Excel-style letter) → integer index.
          * ``"A: header"`` (xlsx labelled entry) → integer index.
          * ``"0"`` / ``"12"`` (legacy / numeric paths) → integer index.
          * ``"header"`` (a bare column name) → string passed through.
        """
        raw = raw.strip()
        if ":" in raw:
            head = raw.split(":", 1)[0].strip()
            idx = _letter_to_col(head)
            if idx is not None:
                return idx
            if head.isdigit():
                return int(head)
            return head
        if raw.isdigit():
            return int(raw)
        idx = _letter_to_col(raw)
        if idx is not None:
            return idx
        return raw

    @staticmethod
    def _sheet_value(raw: str):
        raw = raw.strip()
        if not raw:
            return 0
        if raw.isdigit():
            return int(raw)
        return raw

    def _collect_analysis_settings(self) -> dict | None:
        try:
            short_cutoff = float(self.var_short_cutoff.get())
            long_cutoff = float(self.var_long_cutoff.get())
        except ValueError:
            messagebox.showerror(
                "Invalid number",
                "Short cutoff and long cutoff must be valid numbers.",
            )
            return None
        try:
            cref = float(self.var_cref.get())
        except ValueError:
            messagebox.showerror("Invalid Cref", "Cref must be a number (percent).")
            return None
        if not (0 < cref < 100):
            messagebox.showerror("Invalid Cref", "Cref must be between 0 and 100.")
            return None
        order = int(self.var_order.get())
        if order < 0 or order > 3:
            messagebox.showerror("Invalid order", "Order must be 0, 1, 2, or 3.")
            return None

        # Evaluation length and nsc are derived by the backend from \u03bbc and
        # the trace length \u2014 no GUI control.
        sc_name = self.var_setting_class.get().strip()
        setting_class = None if sc_name == "Custom" else sc_name

        unit_system_label = self.var_report_system.get()
        x_unit, y_unit = _unit_system_pair(unit_system_label)
        report_token = _unit_system_token(unit_system_label)

        source_label = self.var_source_system.get()
        source_x_unit, source_y_unit = _unit_system_pair(source_label)
        source_token = _unit_system_token(source_label)

        return {
            "sheet": self._sheet_value(self.var_sheet.get()),
            "x_col": self._column_value(self.var_x_col.get() or "A"),
            "y_col": self._column_value(self.var_y_col.get() or "B"),
            "x_unit": x_unit,
            "y_unit": y_unit,
            "source_x_unit": source_x_unit,
            "source_y_unit": source_y_unit,
            "report_system": report_token,
            "source_system": source_token,
            "short_cutoff": short_cutoff,
            "long_cutoff": long_cutoff,
            "setting_class": setting_class,
            "order": order,
            "cref": cref,
            "compare_standards": self.var_compare_standards.get(),
        }

    def _validate_inputs(self) -> dict | None:
        file_path = self.var_file.get().strip()
        if not file_path:
            messagebox.showerror("Missing file", "Please choose a data file first.")
            return None
        if not Path(file_path).is_file():
            messagebox.showerror("File not found", f"File does not exist:\n{file_path}")
            return None
        settings = self._collect_analysis_settings()
        if settings is None:
            return None
        settings["file"] = file_path
        return settings

    def _set_busy(self, busy: bool, status: str) -> None:
        state = "disabled" if busy else "normal"
        self.btn_run.configure(state=state)
        self.btn_batch.configure(state=state)
        self.lbl_status.configure(text=status)

    def _run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "An analysis run is already in progress.")
            return
        params = self._validate_inputs()
        if params is None:
            return

        self._clear_output()
        self._append_output("Running analysis with:\n")
        for k, v in params.items():
            self._append_output(f"  {k} = {v!r}\n")
        self._append_output("\n")
        self._set_busy(True, "Running...")

        self._worker = threading.Thread(
            target=self._worker_run, args=(params,), daemon=True
        )
        self._worker.start()

    def _run_batch(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "An analysis run is already in progress.")
            return

        current_file = self.var_file.get().strip()
        if not current_file:
            messagebox.showerror(
                "Missing file",
                "Choose one source file first so batch can infer file format and settings.",
            )
            return
        current_path = Path(current_file)
        if not current_path.is_file():
            messagebox.showerror("File not found", f"File does not exist:\n{current_file}")
            return

        settings = self._collect_analysis_settings()
        if settings is None:
            return

        folder = filedialog.askdirectory(
            title="Select folder to batch process",
            initialdir=str(current_path.parent),
        )
        if not folder:
            return
        folder_path = Path(folder)

        ext = current_path.suffix.lower()
        # Batch supports .pro/.csv/.xlsx — .txt is intentionally omitted
        # from the batch path (we'd need a layout assumption per-file that
        # doesn't fit the bulk pattern).
        if ext not in {".xlsx", ".csv", ".pro"}:
            messagebox.showerror(
                "Unsupported file type",
                f"Batch processing currently supports .xlsx/.csv/.pro, not {ext or '[no extension]'}.\n"
                "Switch the loaded file to one of those types and try again.",
            )
            return

        layout = self._scan_batch_root(folder_path, ext)
        if layout.mode == "empty":
            messagebox.showinfo(
                "No files found",
                f"No {ext} files found in (or under, max 2 levels):\n{folder_path}",
            )
            return

        if not self._confirm_batch_plan(layout):
            return

        files = layout.files
        self._clear_output()
        self._append_output("Batch processing with settings:\n")
        for k, v in settings.items():
            self._append_output(f"  {k} = {v!r}\n")
        self._append_output(f"\nRoot folder: {folder_path}\n")
        self._append_output(
            f"Layout: {layout.mode}; {len(files)} *{ext} file(s) "
            f"across {layout.n_parts} part(s)"
            + (f", {layout.n_locations} location(s)\n\n" if layout.has_location else "\n\n")
        )
        self._set_busy(True, "Batch running...")

        self._worker = threading.Thread(
            target=self._worker_batch,
            args=(settings, layout),
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Recursive file gathering + summary popup
    # ------------------------------------------------------------------
    # Folders pruned from the recursive batch walk.
    _BATCH_SKIP_DIR_PREFIXES = ("TH_Template",)
    _BATCH_SKIP_DIR_NAMES = {"__pycache__", "node_modules", ".mypy_cache"}
    # Rough placeholder for the runtime estimate shown in the popup; we
    # can swap in a measured per-file timing later.
    _BATCH_SECONDS_PER_FILE = 0.5

    @classmethod
    def _scan_batch_root(cls, root: Path, ext: str) -> "_LayoutInfo":
        """Walk ``root`` (depth ≤ 2) and bucket files by ``(part, location)``.

        Depth interpretation:
          * depth 0 (root/file)         → part='(root)', location=''
          * depth 1 (root/A/file)       → part='A',     location=''
          * depth 2 (root/A/B/file)     → part='A',     location='B'

        Folders below depth 2 are not descended into. Hidden folders
        (any segment starting with '.'), Mountains scratch dirs
        (``TH_Template*``), build/cache dirs, and Office lock files
        (``~$*``) are skipped and tallied into ``skipped``.
        """
        ext = ext.lower()
        files_by_part_loc: dict[tuple[str, str], list[Path]] = {}
        skipped: dict[str, int] = {
            "hidden": 0, "TH_Template": 0, "build_cache": 0, "lock_file": 0,
        }
        root_resolved = Path(root).resolve()

        for dirpath, dirnames, filenames in os.walk(root_resolved):
            rel = Path(dirpath).relative_to(root_resolved)
            depth = 0 if rel == Path(".") else len(rel.parts)

            # Prune subdirs we don't want to descend into. We modify
            # ``dirnames`` in place so os.walk skips them.
            keep: list[str] = []
            for d in dirnames:
                if d.startswith("."):
                    skipped["hidden"] += 1
                elif any(d.startswith(p) for p in cls._BATCH_SKIP_DIR_PREFIXES):
                    skipped["TH_Template"] += 1
                elif d in cls._BATCH_SKIP_DIR_NAMES:
                    skipped["build_cache"] += 1
                else:
                    keep.append(d)
            # Cap at depth 2: don't descend below depth-2 folders.
            if depth >= 2:
                dirnames[:] = []
            else:
                dirnames[:] = keep

            for fname in filenames:
                if fname.startswith("~$"):
                    skipped["lock_file"] += 1
                    continue
                if not fname.lower().endswith(ext):
                    continue
                fpath = Path(dirpath) / fname
                if depth == 0:
                    key = ("(root)", "")
                elif depth == 1:
                    key = (rel.parts[0], "")
                else:  # depth == 2
                    key = (rel.parts[0], rel.parts[1])
                files_by_part_loc.setdefault(key, []).append(fpath)

        # Sort files within each (part, location) bucket case-insensitively.
        for k in list(files_by_part_loc.keys()):
            files_by_part_loc[k].sort(key=lambda p: p.name.lower())

        # Mode classification.
        if not files_by_part_loc:
            mode = "empty"
        else:
            keys = list(files_by_part_loc.keys())
            has_root = any(k[0] == "(root)" for k in keys)
            has_part_no_loc = any(k[0] != "(root)" and not k[1] for k in keys)
            has_part_loc = any(k[0] != "(root)" and k[1] for k in keys)
            non_root_kinds = sum(1 for x in (has_part_no_loc, has_part_loc) if x)
            if has_root and not has_part_no_loc and not has_part_loc:
                mode = "flat"
            elif has_part_loc and not has_root and not has_part_no_loc:
                mode = "parts_locations"
            elif has_part_no_loc and not has_root and not has_part_loc:
                mode = "parts"
            else:
                mode = "mixed"

        return _LayoutInfo(
            mode=mode, root=root_resolved, ext=ext,
            files_by_part_loc=files_by_part_loc, skipped=skipped,
        )

    def _confirm_batch_plan(self, layout: "_LayoutInfo") -> bool:
        """Show a layout-aware summary popup; return True on Proceed.

        The treeview shape adapts to ``layout.mode``:
          * 'flat'           → one '(root)' row with the file count.
          * 'parts'          → one row per Part with its file count.
          * 'parts_locations'→ Part rows with totals, expandable to
            Location children.
          * 'mixed'          → same as parts_locations plus a '(root)'
            and/or part rows for files without a location.
        """
        win = tk.Toplevel(self)
        win.title("Confirm batch plan")
        win.transient(self)
        try:
            win.grab_set()
        except tk.TclError:
            pass
        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        mode_blurbs = {
            "flat": "All files live directly in the chosen folder.",
            "parts": "Each subfolder is one Part; files inside are its measurements.",
            "parts_locations": "Two folder levels detected: Part → Location → measurements.",
            "mixed": "Some files have Location subfolders and some don't; "
                     "rows without a Location will use Part only.",
        }
        ttk.Label(
            body, text=f"Detected layout: {layout.mode}",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            body, text=mode_blurbs.get(layout.mode, ""),
            wraplength=560, foreground="#555",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            body, text=f"Root: {layout.root}",
            wraplength=560, foreground="#555",
        ).pack(anchor="w")

        total = layout.total
        n_parts = layout.n_parts
        n_locs = layout.n_locations
        eta_s = total * self._BATCH_SECONDS_PER_FILE
        eta_m, eta_sec = divmod(int(round(eta_s)), 60)
        if eta_m:
            eta_str = f"~{eta_m}m {eta_sec:02d}s"
        else:
            eta_str = f"~{eta_sec}s"
        summary = (
            f"{total} {layout.ext} file(s) — {n_parts} Part(s)"
            + (f", {n_locs} Location(s)" if layout.has_location else "")
            + f"  •  est. runtime {eta_str}"
        )
        ttk.Label(
            body, text=summary, wraplength=560,
        ).pack(anchor="w", pady=(2, 8))

        # Adaptive treeview.
        list_frame = ttk.Frame(body)
        list_frame.pack(fill="both", expand=True)
        if layout.has_location:
            cols = ("count",)
            tree = ttk.Treeview(
                list_frame, columns=cols, show="tree headings", height=14,
            )
            tree.heading("#0", text="Part / Location")
            tree.heading("count", text="Files")
            tree.column("#0", width=320, anchor="w")
            tree.column("count", width=80, anchor="e")
        else:
            cols = ("part", "count")
            tree = ttk.Treeview(
                list_frame, columns=cols, show="headings", height=14,
            )
            tree.heading("part", text="Part")
            tree.heading("count", text="Files")
            tree.column("part", width=320, anchor="w")
            tree.column("count", width=80, anchor="e")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Build per-part totals (sum across locations within a part).
        per_part: dict[str, int] = {}
        per_part_locs: dict[str, list[tuple[str, int]]] = {}
        for (part, loc), fs in layout.files_by_part_loc.items():
            per_part[part] = per_part.get(part, 0) + len(fs)
            per_part_locs.setdefault(part, []).append((loc, len(fs)))

        ordered_parts = sorted(per_part.keys(), key=_part_sort_key)
        if layout.has_location:
            for part in ordered_parts:
                pid = tree.insert("", "end", text=part, values=(per_part[part],), open=True)
                # Sort locations: empty first, then alpha-numeric.
                for loc, n in sorted(per_part_locs[part], key=lambda kv: (kv[0] == "", kv[0].lower())):
                    label = loc if loc else "(no location)"
                    tree.insert(pid, "end", text=label, values=(n,))
        else:
            for part in ordered_parts:
                tree.insert("", "end", values=(part, per_part[part]))

        # Skipped folders + filters info.
        skipped_total = sum(layout.skipped.values())
        if skipped_total:
            parts_skipped = []
            if layout.skipped["hidden"]:
                parts_skipped.append(f"hidden ×{layout.skipped['hidden']}")
            if layout.skipped["TH_Template"]:
                parts_skipped.append(f"TH_Template* ×{layout.skipped['TH_Template']}")
            if layout.skipped["build_cache"]:
                parts_skipped.append(f"cache dirs ×{layout.skipped['build_cache']}")
            if layout.skipped["lock_file"]:
                parts_skipped.append(f"~$ lock files ×{layout.skipped['lock_file']}")
            skipped_text = "Skipped: " + ", ".join(parts_skipped)
        else:
            skipped_text = (
                "Filters active: dot-folders, TH_Template*, "
                "__pycache__/node_modules/.mypy_cache, ~$* lock files."
            )
        ttk.Label(
            body, text=skipped_text, wraplength=560, foreground="#777",
        ).pack(anchor="w", pady=(8, 8))

        result = {"ok": False}
        btns = ttk.Frame(body)
        btns.pack(fill="x")

        def on_ok() -> None:
            result["ok"] = True
            win.destroy()

        def on_cancel() -> None:
            win.destroy()

        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btns, text="Proceed", command=on_ok).pack(
            side="right", padx=(0, 6)
        )
        win.bind("<Return>", lambda _e: on_ok())
        win.bind("<Escape>", lambda _e: on_cancel())

        # Center over parent.
        try:
            self.update_idletasks()
            x = self.winfo_rootx() + (self.winfo_width() // 2) - 320
            y = self.winfo_rooty() + (self.winfo_height() // 2) - 240
            win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except tk.TclError:
            pass

        win.wait_window()
        return result["ok"]

    def _worker_run(self, params: dict) -> None:
        try:
            ext = Path(params["file"]).suffix.lower()
            # For .pro the loader supplies source units from the file header,
            # so we don't pass source_*_units (let SurfaceTexture pick them up).
            extra: dict = {}
            if ext != ".pro":
                extra["source_x_units"] = params["source_x_unit"]
                extra["source_y_units"] = params["source_y_unit"]
            st = SurfaceTexture(
                params["file"],
                params["short_cutoff"],
                params["long_cutoff"],
                order=params["order"],
                x_units=params["x_unit"],
                y_units=params["y_unit"],
                x_col=params["x_col"],
                y_col=params["y_col"],
                sheet_name=params["sheet"],
                setting_class=params.get("setting_class"),
                allow_short_trace=params.get("allow_short_trace", False),
                **extra,
            )
            comparison = params.get("compare_standards", False)
            if comparison:
                st.compute_comparison_params(Cref=params["cref"])
            fig = st.build_overview_figure(Cref=params["cref"], comparison=comparison)
            summary = self._summarize(st)
            self._msg_queue.put(("done", (fig, summary)))
        except TraceTooShortError as exc:
            self._msg_queue.put((
                "trace_too_short",
                {
                    "params": params,
                    "message": str(exc),
                    "suggested": exc.suggested_class,
                    "current": exc.current_class,
                },
            ))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self._msg_queue.put(("error", f"{exc}\n\n{tb}"))

    def _worker_batch(self, settings: dict, layout: "_LayoutInfo") -> None:
        # Build (file, part, location) triples in deterministic order.
        triples: list[tuple[Path, str, str]] = []
        for key in sorted(
            layout.files_by_part_loc.keys(),
            key=lambda kv: (_part_sort_key(kv[0]), kv[1].lower()),
        ):
            part, loc = key
            for fp in layout.files_by_part_loc[key]:
                triples.append((fp, part, loc))
        files = [t[0] for t in triples]
        layout_meta = {
            "mode": layout.mode,
            "root": str(layout.root),
            "has_location": layout.has_location,
            "n_parts": layout.n_parts,
            "n_locations": layout.n_locations,
        }

        # Pass 1: load each file and determine a global y-range so all
        # generated plots are directly comparable.
        loaded: list[tuple[Path, SurfaceTexture]] = []
        y_mins: list[float] = []
        y_maxs: list[float] = []
        # Per-file rows for the optional xlsx export. Keyed shape matches
        # batch_report.build_comparison_workbook.
        report_rows: list[dict] = []

        success = 0
        failed = 0
        for idx, (file_path, part, loc) in enumerate(triples, start=1):
            try:
                file_ext = file_path.suffix.lower()
                extra: dict = {}
                if file_ext != ".pro":
                    extra["source_x_units"] = settings["source_x_unit"]
                    extra["source_y_units"] = settings["source_y_unit"]
                st = SurfaceTexture(
                    str(file_path),
                    settings["short_cutoff"],
                    settings["long_cutoff"],
                    order=settings["order"],
                    x_units=settings["x_unit"],
                    y_units=settings["y_unit"],
                    x_col=settings["x_col"],
                    y_col=settings["y_col"],
                    sheet_name=settings["sheet"],
                    setting_class=settings.get("setting_class"),
                    **extra,
                )

                # Ensure bearing curve data is available for y-limit analysis.
                if not hasattr(st, "material_ratio"):
                    st.get_material_ratio()

                y_min = float(
                    np.nanmin(
                        [
                            np.nanmin(st.raw_data_xy[1]),
                            np.nanmin(st.roughness[1]),
                            np.nanmin(st.waviness[1]),
                            np.nanmin(st.material_ratio[1]),
                        ]
                    )
                )
                y_max = float(
                    np.nanmax(
                        [
                            np.nanmax(st.raw_data_xy[1]),
                            np.nanmax(st.roughness[1]),
                            np.nanmax(st.waviness[1]),
                            np.nanmax(st.material_ratio[1]),
                        ]
                    )
                )
                y_mins.append(y_min)
                y_maxs.append(y_max)
                loaded.append((file_path, st))

                # Capture R-parameters for the optional xlsx report.
                report_rows.append(
                    self._extract_report_row(file_path, st, part=part, location=loc)
                )

                self._msg_queue.put(
                    (
                        "batch_progress",
                        f"[ANALYZE {idx}/{len(files)}] OK  {file_path.name}\n",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self._msg_queue.put(
                    (
                        "batch_progress",
                        f"[ANALYZE {idx}/{len(files)}] FAIL {file_path.name}: {exc}\n",
                    )
                )

        if not loaded:
            self._msg_queue.put((
                "batch_done",
                (success, failed, len(files), report_rows, dict(settings),
                 [str(f) for f in files], layout_meta),
            ))
            return

        global_min = float(np.nanmin(y_mins))
        global_max = float(np.nanmax(y_maxs))
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            global_min, global_max = -1.0, 1.0
        if global_max <= global_min:
            global_min -= 1.0
            global_max += 1.0
        span = global_max - global_min
        pad = max(0.05 * span, 1e-9)
        y_lim = (global_min - pad, global_max + pad)
        self._msg_queue.put(
            (
                "batch_progress",
                f"Using shared y-limits for all batch plots: [{y_lim[0]:.3f}, {y_lim[1]:.3f}] {settings['y_unit']}\n",
            )
        )

        # Pass 2: render and save all figures with the shared y-range.
        for idx, (file_path, st) in enumerate(loaded, start=1):
            try:
                fig = st.build_overview_figure(Cref=settings["cref"], y_lim=y_lim)
                out_path = file_path.with_name(f"{file_path.stem}_overview.png")
                fig.savefig(out_path, dpi=170, bbox_inches="tight")
                fig.clear()
                success += 1
                self._msg_queue.put(
                    (
                        "batch_progress",
                        f"[SAVE {idx}/{len(loaded)}] OK  {file_path.name} -> {out_path.name}\n",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self._msg_queue.put(
                    (
                        "batch_progress",
                        f"[SAVE {idx}/{len(loaded)}] FAIL {file_path.name}: {exc}\n",
                    )
                )

        self._msg_queue.put((
            "batch_done",
            (success, failed, len(files), report_rows, dict(settings),
             [str(f) for f in files], layout_meta),
        ))

    @staticmethod
    def _extract_report_row(
        file_path: Path,
        st: "SurfaceTexture",
        *,
        part: str | None = None,
        location: str = "",
    ) -> dict:
        """Build a comparison-workbook row from a finished SurfaceTexture run.

        ``part`` and ``location`` come from the layout scan; when ``part``
        is omitted (legacy callers, e.g. tests) we fall back to the
        immediate parent folder name. The Measurement key is always
        ``"<part>_<stem>"`` so two parts that share a stem (e.g.
        ``9/9_1.pro`` and ``20/9_1.pro``) never collide in the workbook;
        files at the chosen root use the bare stem.
        """
        ours: dict[str, float] = {}
        units: dict[str, str] = {}
        params = getattr(st, "R_params", {}) or {}
        for key, val in params.items():
            if isinstance(val, tuple) and len(val) == 2:
                value, unit = val
                try:
                    ours[key] = float(value)
                except (TypeError, ValueError):
                    continue
                units[key] = str(unit)
        if part is None:
            part = file_path.parent.name
        stem = file_path.stem
        if part and part != "(root)":
            measurement = f"{part}_{stem}"
        else:
            measurement = stem
        return {
            "part": part,
            "location": location,
            "measurement": measurement,
            "file": str(file_path),
            "ours": ours,
            "ours_units": units,
            "mountains": {},
            "units": {},
            "error": None,
        }

    @staticmethod
    def _summarize(st: SurfaceTexture) -> str:
        sc_name = getattr(st, "setting_class_name", "Custom")
        lines = [
            "Analysis complete (ISO 21920-3:2021 / ISO 21920-2:2021).",
            f"  Setting class: {sc_name}",
            f"  \u03bbc = {st.long_cutoff:g} {st.x_units}, "
            f"\u03bbs = {st.short_cutoff:g} {st.x_units}, "
            f"le = {st.evaluation_length:g} {st.x_units}, nsc = {st.nsc}",
        ]
        if getattr(st, "nsc_warning", None):
            target = getattr(st, "target_nsc", st.nsc)
            lines.append(
                f"  \u26a0 nsc reduced to {st.nsc} (ISO default {target}) "
                f"\u2014 trace too short for full le"
            )
        if getattr(st, "dx_warning", None):
            lines.append(f"  \u26a0 {st.dx_warning}")
        lines.append("")
        lines.append("R-Parameters:")
        for key in ["Ra", "Rq", "Rp", "Rv", "Rz", "Rzx", "Rt", "Rsk", "Rku", "Rmr"]:
            if key in st.R_params:
                v, u = st.R_params[key]
                if isinstance(v, (int, float)):
                    lines.append(
                        f"  {key:>5}: {st.format_param_value(key, v, u)} {u}"
                    )
                else:
                    lines.append(f"  {key:>5}: {v} {u}")
        return "\n".join(lines) + "\n"

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "done":
                    fig, summary = payload  # type: ignore[misc]
                    self._swap_figure(fig)
                    self._append_output(summary)
                    self._set_busy(False, "Done")
                elif kind == "batch_progress":
                    self._append_output(str(payload))
                elif kind == "batch_done":
                    # Payload (current): (success, failed, total, rows,
                    # settings, files, layout_meta). Older callers may
                    # still emit the 6-tuple without layout_meta or the
                    # 3-tuple summary-only form.
                    layout_meta: dict = {}
                    if isinstance(payload, tuple) and len(payload) == 7:
                        success, failed, total, rows, settings_used, files, layout_meta = payload
                    elif isinstance(payload, tuple) and len(payload) == 6:
                        success, failed, total, rows, settings_used, files = payload
                    else:
                        success, failed, total = payload  # type: ignore[misc]
                        rows, settings_used, files = [], {}, []
                    self._append_output(
                        f"\nBatch complete: {success}/{total} succeeded, {failed} failed.\n"
                    )
                    self._set_busy(False, "Done")
                    if rows and self.var_batch_xlsx.get():
                        self._maybe_save_batch_workbook(
                            rows, settings_used, files, layout_meta,
                        )
                elif kind == "error":
                    self._append_output("ERROR:\n" + str(payload) + "\n")
                    messagebox.showerror(
                        "Analysis failed", str(payload).split("\n\n")[0]
                    )
                    self._set_busy(False, "Error")
                elif kind == "trace_too_short":
                    self._prompt_trace_too_short(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # Batch xlsx export
    # ------------------------------------------------------------------
    def _maybe_save_batch_workbook(
        self,
        rows: list[dict],
        settings_used: dict,
        files: list[str],
        layout_meta: dict | None = None,
    ) -> None:
        """Prompt for a save path and write the batch summary workbook.

        Skips silently if the user cancels the save dialog or if writing
        fails (an error message is shown instead). Mountains comparison
        columns are omitted; this is a GUI-only summary of how the
        currently-configured pipeline scored each file.
        """
        if not files:
            return
        layout_meta = layout_meta or {}
        default_dir = Path(files[0]).parent
        save_path = filedialog.asksaveasfilename(
            title="Save batch Excel report",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
            initialdir=str(default_dir),
            initialfile="batch_results.xlsx",
        )
        if not save_path:
            self._append_output("Excel report skipped (no path selected).\n")
            return
        try:
            from batch_report import build_comparison_workbook
            settings_for_audit = {
                "setting_class": settings_used.get("setting_class"),
                "report_system": settings_used.get("report_system"),
                "source_system": settings_used.get("source_system"),
                "x_unit": settings_used.get("x_unit"),
                "y_unit": settings_used.get("y_unit"),
                "short_cutoff": settings_used.get("short_cutoff"),
                "long_cutoff": settings_used.get("long_cutoff"),
                "order": settings_used.get("order"),
                "cref_pct": settings_used.get("cref"),
                "n_files": len(files),
            }
            if layout_meta:
                settings_for_audit.update({
                    "layout_mode": layout_meta.get("mode"),
                    "layout_root": layout_meta.get("root"),
                    "n_parts": layout_meta.get("n_parts"),
                    "n_locations": layout_meta.get("n_locations"),
                })
            out = build_comparison_workbook(
                rows, save_path,
                settings=settings_for_audit,
                mountains_present=False,
            )
            self._append_output(f"Excel report written to: {out}\n")
        except Exception as exc:  # noqa: BLE001
            self._append_output(f"Excel report FAILED: {exc}\n")
            messagebox.showerror("Excel export failed", str(exc))

    # ------------------------------------------------------------------
    # Trace-too-short recovery dialog
    # ------------------------------------------------------------------
    def _prompt_trace_too_short(self, payload: dict) -> None:
        """Show a 3-button recovery dialog for ``TraceTooShortError``.

        ``payload`` carries ``params`` (the original analysis settings),
        ``message`` (rich diagnostic from the exception), ``suggested``
        (largest setting class that fits, or ``None``) and ``current``.
        """
        params = payload["params"]
        message = payload["message"]
        suggested = payload.get("suggested")
        current = payload.get("current")

        # Drop busy state while the user decides; we'll restore it if they
        # pick a recovery option.
        self._set_busy(False, "Awaiting input")
        self._append_output("ERROR:\n" + message + "\n")

        win = tk.Toplevel(self)
        win.title("Trace too short")
        win.transient(self)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: _close())

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill="both", expand=True)

        header = (
            "ISO 21920-3 trace-length check failed. Choose a recovery option:"
        )
        ttk.Label(outer, text=header, wraplength=520).pack(anchor="w", pady=(0, 8))

        txt = tk.Text(outer, width=72, height=10, wrap="word")
        txt.insert("1.0", message)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True, pady=(0, 8))

        if suggested:
            accept_label = f"Accept (switch to {suggested})"
            accept_state = "normal"
            hint = (
                f"Accept switches the setting class from {current} to "
                f"{suggested} and re-runs. Override keeps λc and forces "
                f"nsc = 1; the figure will be flagged non-conformant."
            )
        else:
            accept_label = "Accept (no class fits)"
            accept_state = "disabled"
            hint = (
                "No ISO 21920-3 setting class fits this trace. Use Override "
                "to force a run with nsc = 1 (non-conformant; edge artefacts "
                "will be significant)."
            )
        ttk.Label(outer, text=hint, wraplength=520, foreground="#555").pack(
            anchor="w", pady=(0, 8)
        )

        btns = ttk.Frame(outer)
        btns.pack(fill="x")

        def _close() -> None:
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()
            if self.lbl_status.cget("text") == "Awaiting input":
                self._set_busy(False, "Cancelled")

        def _on_accept() -> None:
            if not suggested:
                return
            self.var_setting_class.set(suggested)
            self._apply_setting_class()
            _close()
            # Re-run with the (now refreshed) cutoffs from the new class.
            self._run()

        def _on_override() -> None:
            new_params = dict(params)
            new_params["allow_short_trace"] = True
            _close()
            self._clear_output()
            self._append_output(
                "Re-running with short-trace override (non-conformant):\n"
            )
            for k, v in new_params.items():
                self._append_output(f"  {k} = {v!r}\n")
            self._append_output("\n")
            self._set_busy(True, "Running (override)...")
            self._worker = threading.Thread(
                target=self._worker_run, args=(new_params,), daemon=True
            )
            self._worker.start()

        accept_btn = ttk.Button(
            btns, text=accept_label, command=_on_accept, state=accept_state
        )
        accept_btn.pack(side="left", padx=(0, 6))
        override_btn = ttk.Button(btns, text="Override (force run)", command=_on_override)
        override_btn.pack(side="left", padx=6)
        cancel_btn = ttk.Button(btns, text="Cancel", command=_close)
        cancel_btn.pack(side="right")

        # Default focus: Accept when a class is suggested, otherwise Override.
        (accept_btn if suggested else override_btn).focus_set()

        win.update_idletasks()
        # Centre relative to the main window.
        rx = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        ry = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(rx, 0)}+{max(ry, 0)}")
        win.grab_set()

    def _swap_figure(self, new_fig: Figure) -> None:
        plot_frame = self._canvas.get_tk_widget().master
        self._canvas.get_tk_widget().destroy()
        self._toolbar.destroy()
        self._fig = new_fig
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, plot_frame)
        self._toolbar.update()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _append_output(self, text: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", text)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _clear_output(self) -> None:
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")


if __name__ == "__main__":
    SurfaceFinishGUI().mainloop()
