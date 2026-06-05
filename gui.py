"""Tkinter front-end for the SurfaceFinish analysis pipeline.

Loads data, runs ``SurfaceTexture`` in a worker thread, and renders a live
matplotlib figure inside the window with raw+fit, roughness+waviness, the
bearing ratio (Abbott) curve, and an R-parameter table.

Run with::

    python gui.py
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
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

from SurfaceTexture import SurfaceTexture  # noqa: E402
from iso21920 import (  # noqa: E402
    SETTING_CLASSES,
    DEFAULT_SETTING_CLASS,
    convert_mm_to,
    get_setting_class,
)


WORKSPACE_ROOT = Path(__file__).resolve().parent

X_UNIT_CHOICES = ["mm", "\u03bcm", "in", "cm", "m"]
Y_UNIT_CHOICES = ["\u03bcm", "\u03bcin", "mm", "nm", "in"]


class SurfaceFinishGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SurfaceFinish - Analysis")
        self.geometry("1720x900")
        self.minsize(1450, 760)

        self._msg_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._worker: threading.Thread | None = None

        # ---- state vars --------------------------------------------------
        self.var_file = tk.StringVar()
        self.var_sheet = tk.StringVar()
        self.var_x_col = tk.StringVar()
        self.var_y_col = tk.StringVar()
        self.var_x_unit = tk.StringVar(value="mm")
        self.var_y_unit = tk.StringVar(value="\u03bcm")
        self.var_short_cutoff = tk.StringVar(value="0.0025")
        self.var_long_cutoff = tk.StringVar(value="0.8")
        self.var_setting_class = tk.StringVar(value=DEFAULT_SETTING_CLASS)
        self.var_order = tk.IntVar(value=1)
        self.var_cref = tk.StringVar(value="5")

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
        ttk.Label(src, text="Sheet:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cmb_sheet = ttk.Combobox(
            src, textvariable=self.var_sheet, state="readonly", width=42
        )
        self.cmb_sheet.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=3)
        self.cmb_sheet.bind("<<ComboboxSelected>>", lambda _e: self._refresh_columns())
        src.columnconfigure(1, weight=1)

        # --- Columns ---
        cols = ttk.LabelFrame(parent, text="Columns")
        cols.pack(fill="x", **pad)
        ttk.Label(cols, text="X column:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.cmb_x = ttk.Combobox(cols, textvariable=self.var_x_col, width=42)
        self.cmb_x.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(cols, text="Y column:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cmb_y = ttk.Combobox(cols, textvariable=self.var_y_col, width=42)
        self.cmb_y.grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        cols.columnconfigure(1, weight=1)

        # --- Units ---
        units = ttk.LabelFrame(parent, text="Units (axis labels)")
        units.pack(fill="x", **pad)
        ttk.Label(units, text="X unit:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(
            units, textvariable=self.var_x_unit, values=X_UNIT_CHOICES, width=10
        ).grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(units, text="Y unit:").grid(row=0, column=2, sticky="w", padx=4, pady=3)
        ttk.Combobox(
            units, textvariable=self.var_y_unit, values=Y_UNIT_CHOICES, width=10
        ).grid(row=0, column=3, sticky="w", padx=4, pady=3)

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
        ttk.Button(cuts, text="mm/\u03bcm defaults", command=self._cutoffs_mm).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )
        ttk.Button(cuts, text="inch defaults", command=self._cutoffs_inch).grid(
            row=3, column=2, columnspan=2, sticky="w", padx=4, pady=3
        )
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
                ("Excel workbook", "*.xlsx"),
                ("Text/CSV", "*.txt *.csv"),
                ("All files", "*.*"),
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

        path = self.var_file.get().strip()
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext != ".xlsx":
            self.cmb_x["values"] = ("0", "1")
            self.cmb_y["values"] = ("0", "1")
            self.var_x_col.set("0")
            self.var_y_col.set("1")
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

        indexed = [f"{i}: {name}" for i, name in enumerate(columns)]
        values = tuple(indexed)
        self.cmb_x["values"] = values
        self.cmb_y["values"] = values
        if columns:
            self.var_x_col.set(indexed[0])
            self.var_y_col.set(indexed[1] if len(indexed) > 1 else indexed[0])

    # ------------------------------------------------------------------
    # Cutoff presets
    # ------------------------------------------------------------------
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
        x_unit = self.var_x_unit.get() or "mm"
        try:
            self.var_short_cutoff.set(f"{convert_mm_to(sc.lambda_s_mm, x_unit):g}")
            self.var_long_cutoff.set(f"{convert_mm_to(sc.lambda_c_mm, x_unit):g}")

        except ValueError:
            # Unknown unit \u2014 leave fields untouched.
            return

    def _cutoffs_mm(self) -> None:
        self.var_x_unit.set("mm")
        self.var_y_unit.set("\u03bcm")
        self.var_setting_class.set(DEFAULT_SETTING_CLASS)
        self._apply_setting_class()

    def _cutoffs_inch(self) -> None:
        self.var_x_unit.set("in")
        self.var_y_unit.set("\u03bcin")
        self.var_setting_class.set(DEFAULT_SETTING_CLASS)
        self._apply_setting_class()

    # ------------------------------------------------------------------
    # Run handling
    # ------------------------------------------------------------------
    @staticmethod
    def _column_value(raw: str):
        raw = raw.strip()
        if ":" in raw:
            head = raw.split(":", 1)[0].strip()
            if head.isdigit():
                return int(head)
            return head
        if raw.isdigit():
            return int(raw)
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

        return {
            "sheet": self._sheet_value(self.var_sheet.get()),
            "x_col": self._column_value(self.var_x_col.get() or "0"),
            "y_col": self._column_value(self.var_y_col.get() or "1"),
            "x_unit": self.var_x_unit.get() or "mm",
            "y_unit": self.var_y_unit.get() or "\u03bcm",
            "short_cutoff": short_cutoff,
            "long_cutoff": long_cutoff,
            "setting_class": setting_class,
            "order": order,
            "cref": cref,
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
        if ext not in {".xlsx", ".txt", ".csv"}:
            messagebox.showerror(
                "Unsupported file type",
                f"Batch processing currently supports .xlsx/.txt/.csv, not {ext or '[no extension]'}.",
            )
            return

        files = sorted(folder_path.glob(f"*{ext}"))
        if not files:
            messagebox.showinfo(
                "No files found",
                f"No {ext} files found in:\n{folder_path}",
            )
            return

        self._clear_output()
        self._append_output("Batch processing with settings:\n")
        for k, v in settings.items():
            self._append_output(f"  {k} = {v!r}\n")
        self._append_output(f"\nFolder: {folder_path}\n")
        self._append_output(f"Files detected ({len(files)}): *{ext}\n\n")
        self._set_busy(True, "Batch running...")

        self._worker = threading.Thread(
            target=self._worker_batch,
            args=(settings, files),
            daemon=True,
        )
        self._worker.start()

    def _worker_run(self, params: dict) -> None:
        try:
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
            )
            fig = st.build_overview_figure(Cref=params["cref"])
            summary = self._summarize(st)
            self._msg_queue.put(("done", (fig, summary)))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self._msg_queue.put(("error", f"{exc}\n\n{tb}"))

    def _worker_batch(self, settings: dict, files: list[Path]) -> None:
        # Pass 1: load each file and determine a global y-range so all
        # generated plots are directly comparable.
        loaded: list[tuple[Path, SurfaceTexture]] = []
        y_mins: list[float] = []
        y_maxs: list[float] = []

        success = 0
        failed = 0
        for idx, file_path in enumerate(files, start=1):
            try:
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
            self._msg_queue.put(("batch_done", (success, failed, len(files))))
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

        self._msg_queue.put(("batch_done", (success, failed, len(files))))

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
                    success, failed, total = payload  # type: ignore[misc]
                    self._append_output(
                        f"\nBatch complete: {success}/{total} succeeded, {failed} failed.\n"
                    )
                    self._set_busy(False, "Done")
                elif kind == "error":
                    self._append_output("ERROR:\n" + str(payload) + "\n")
                    messagebox.showerror(
                        "Analysis failed", str(payload).split("\n\n")[0]
                    )
                    self._set_busy(False, "Error")
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

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
