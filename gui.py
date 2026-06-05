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

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

from SurfaceTexture import SurfaceTexture  # noqa: E402


WORKSPACE_ROOT = Path(__file__).resolve().parent

X_UNIT_CHOICES = ["mm", "\u03bcm", "in", "cm", "m"]
Y_UNIT_CHOICES = ["\u03bcm", "\u03bcin", "mm", "nm", "in"]


class SurfaceFinishGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SurfaceFinish - Analysis")
        self.geometry("1320x820")
        self.minsize(1100, 700)

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

        left = ttk.Frame(paned, padding=6)
        right = ttk.Frame(paned, padding=6)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

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
            text="Filter Cutoffs (in same X distance unit as data)",
        )
        cuts.pack(fill="x", **pad)
        ttk.Label(cuts, text="Short cutoff:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(cuts, textvariable=self.var_short_cutoff, width=14).grid(
            row=0, column=1, sticky="w", padx=4, pady=3
        )
        ttk.Label(cuts, text="Long cutoff:").grid(row=0, column=2, sticky="w", padx=4, pady=3)
        ttk.Entry(cuts, textvariable=self.var_long_cutoff, width=14).grid(
            row=0, column=3, sticky="w", padx=4, pady=3
        )
        ttk.Button(cuts, text="mm/\u03bcm defaults", command=self._cutoffs_mm).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )
        ttk.Button(cuts, text="inch defaults", command=self._cutoffs_inch).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=4, pady=3
        )

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

            xls = pd.ExcelFile(path)
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

            df_head = pd.read_excel(path, sheet_name=sheet, nrows=0)
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
    def _cutoffs_mm(self) -> None:
        self.var_short_cutoff.set("0.0025")
        self.var_long_cutoff.set("0.8")
        self.var_x_unit.set("mm")
        self.var_y_unit.set("\u03bcm")

    def _cutoffs_inch(self) -> None:
        self.var_short_cutoff.set("9.8425e-5")
        self.var_long_cutoff.set("0.031496")
        self.var_x_unit.set("in")
        self.var_y_unit.set("\u03bcin")

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

    def _validate_inputs(self) -> dict | None:
        file_path = self.var_file.get().strip()
        if not file_path:
            messagebox.showerror("Missing file", "Please choose a data file first.")
            return None
        if not Path(file_path).is_file():
            messagebox.showerror("File not found", f"File does not exist:\n{file_path}")
            return None
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
        return {
            "file": file_path,
            "sheet": self._sheet_value(self.var_sheet.get()),
            "x_col": self._column_value(self.var_x_col.get() or "0"),
            "y_col": self._column_value(self.var_y_col.get() or "1"),
            "x_unit": self.var_x_unit.get() or "mm",
            "y_unit": self.var_y_unit.get() or "\u03bcm",
            "short_cutoff": short_cutoff,
            "long_cutoff": long_cutoff,
            "order": order,
            "cref": cref,
        }

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
        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="Running...")

        self._worker = threading.Thread(
            target=self._worker_run, args=(params,), daemon=True
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
            )
            fig = st.build_overview_figure(Cref=params["cref"])
            summary = self._summarize(st)
            self._msg_queue.put(("done", (fig, summary)))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self._msg_queue.put(("error", f"{exc}\n\n{tb}"))

    @staticmethod
    def _summarize(st: SurfaceTexture) -> str:
        lines = ["Analysis complete.", "", "R-Parameters:"]
        for key in ["Ra", "Rp", "Rt", "Rv", "Rz", "Rq", "Rsk", "Rmr"]:
            if key in st.R_params:
                v, u = st.R_params[key]
                if isinstance(v, (int, float)):
                    lines.append(f"  {key:>5}: {v:.4f} {u}")
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
                    self.lbl_status.configure(text="Done")
                    self.btn_run.configure(state="normal")
                elif kind == "error":
                    self._append_output("ERROR:\n" + str(payload) + "\n")
                    messagebox.showerror(
                        "Analysis failed", str(payload).split("\n\n")[0]
                    )
                    self.lbl_status.configure(text="Error")
                    self.btn_run.configure(state="normal")
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
