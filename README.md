# SurfaceFinish
Exploring the world of surface finish data analysis.

## Run

```bash
python main.py
```

If there are multiple `.xlsx` files in the project root, the script prompts you to pick one.

## Configure File, Sheet, Columns, Units

```bash
python main.py --file "48743-004 trell sample t1.xlsx" --sheet 0 --x-col 0 --y-col 1 --x-unit mm --y-unit um
```

You can pass column names instead of indexes:

```bash
python main.py --file "measurements.xlsx" --sheet "TraceData" --x-col "Distance" --y-col "Height" --x-unit mm --y-unit um
```

## Notes

- `--sheet` accepts either a zero-based index (`0`) or sheet name (`"TraceData"`).
- `--x-col` and `--y-col` accept either zero-based indexes or exact column names.
- `.txt`/`.csv` input still works as before (first two columns are used unless preprocessing is done externally).

## Dependencies For Excel Input

Excel support requires `pandas` and `openpyxl`:

```bash
pip install pandas openpyxl
```

## Environment Setup

Use one of these approaches:

### Option 1: pip + venv

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f SurfFin.yml
conda activate SurfFin
```

`SurfFin.yml` is a Conda environment file and cannot be installed with `pip install -r`.
