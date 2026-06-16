from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from SurfaceTexture import SurfaceTexture
from iso21920 import SETTING_CLASSES, DEFAULT_SETTING_CLASS


def _pick_root_xlsx(root_dir: Path) -> Path:
    xlsx_files = sorted(root_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {root_dir}")
    if len(xlsx_files) == 1:
        return xlsx_files[0]

    print("Select an .xlsx file:")
    for idx, f in enumerate(xlsx_files, start=1):
        print(f"{idx}. {f.name}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            pick = int(choice)
            if 1 <= pick <= len(xlsx_files):
                return xlsx_files[pick - 1]
        print("Invalid choice. Try again.")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Surface finish analysis for TXT/CSV/XLSX/PRO traces"
    )
    parser.add_argument("--file", help="Path to source data file (.txt/.csv/.xlsx/.pro)")
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or zero-based sheet index (default: 0)",
    )
    parser.add_argument("--x-col", default=0, help="X column index or name (default: 0)")
    parser.add_argument("--y-col", default=1, help="Y column index or name (default: 1)")
    parser.add_argument("--x-unit", default="mm", help="Label for X axis unit (default: mm)")
    parser.add_argument("--y-unit", default="μm", help="Label for Y axis unit (default: μm)")
    parser.add_argument(
        "--setting-class",
        choices=sorted(SETTING_CLASSES.keys()) + ["Custom"],
        default=DEFAULT_SETTING_CLASS,
        help=(f"ISO 21920-3 setting class (default: {DEFAULT_SETTING_CLASS}). "
              "Use 'Custom' to drive cutoffs purely from --short-cutoff/--long-cutoff."),
    )
    parser.add_argument("--short-cutoff", type=float, default=None,
                        help="Short wave cutoff λs (data X units). Overrides setting class.")
    parser.add_argument("--long-cutoff", type=float, default=None,
                        help="Long wave cutoff λc (data X units). Overrides setting class.")
    parser.add_argument("--order", type=int, default=1,
                        help="Leveling polynomial order: 0, 1, 2, or 3")
    parser.add_argument("--plot-level", action="store_true",
                        help="Show profile leveling preview plot.")
    parser.add_argument("--no-plot-roughness", action="store_true",
                        help="Disable the roughness plot.")
    parser.add_argument("--no-plot-mr", action="store_true",
                        help="Disable the material ratio plot.")
    return parser


def _parse_sheet(sheet):
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def _count_long_option(argv, opt):
    count = 0
    for arg in argv[1:]:
        if arg == opt or arg.startswith(opt + "="):
            count += 1
    return count


def _validate_duplicate_column_args(argv):
    if _count_long_option(argv, "--x-col") > 1:
        raise SystemExit("Error: --x-col specified multiple times. Did you mean --x-unit for units?")
    if _count_long_option(argv, "--y-col") > 1:
        raise SystemExit("Error: --y-col specified multiple times. Did you mean --y-unit for units?")

if __name__ == "__main__":
    _validate_duplicate_column_args(sys.argv)
    args = build_parser().parse_args()

    workspace_root = Path(__file__).resolve().parent
    data_file = Path(args.file) if args.file else _pick_root_xlsx(workspace_root)
    if not data_file.is_absolute():
        data_file = (workspace_root / data_file).resolve()

    surface_texture = SurfaceTexture(
        str(data_file),
        args.short_cutoff,
        args.long_cutoff,
        order=args.order,
        x_units=args.x_unit,
        y_units=args.y_unit,
        x_col=args.x_col,
        y_col=args.y_col,
        sheet_name=_parse_sheet(args.sheet),
        setting_class=None if args.setting_class == "Custom" else args.setting_class,
        PLOT_LEVEL=args.plot_level,
    )

    plotted = False
    if not args.no_plot_roughness:
        surface_texture.plot_roughness()
        plotted = True
    if not args.no_plot_mr:
        surface_texture.plot_material_ratio()
        plotted = True

    if plotted:
        plt.show()

