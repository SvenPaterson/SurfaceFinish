from pathlib import Path
from gui import SurfaceFinishGUI

root = Path("35mm Sleeve Database")
files = SurfaceFinishGUI._gather_batch_files(root, ".pro")
print("found", len(files), "pro files (expected 144)")
print("first 3:", [str(f.relative_to(root)) for f in files[:3]])
print("last 3:", [str(f.relative_to(root)) for f in files[-3:]])

from collections import Counter
parts = Counter(f.parent.name for f in files)
print("parts:", len(parts), "min/max files:", min(parts.values()), max(parts.values()))

# Filter test
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    tmpp = Path(tmp)
    (tmpp / "good").mkdir()
    (tmpp / "good" / "a.pro").write_text("x")
    (tmpp / ".hidden").mkdir()
    (tmpp / ".hidden" / "b.pro").write_text("x")
    (tmpp / "TH_Template_001").mkdir()
    (tmpp / "TH_Template_001" / "c.pro").write_text("x")
    (tmpp / "deep" / "sub").mkdir(parents=True)
    (tmpp / "deep" / "sub" / "d.pro").write_text("x")
    rels = sorted(str(f.relative_to(tmpp)) for f in SurfaceFinishGUI._gather_batch_files(tmpp, ".pro"))
    print("filter test:", rels)
    assert any("good" in r for r in rels)
    assert any("deep" in r for r in rels)
    assert not any(".hidden" in r for r in rels)
    assert not any("TH_Template" in r for r in rels)
print("OK")

# Render the popup briefly to confirm it builds
app = SurfaceFinishGUI(); app.update()
def close():
    win = next((w for w in app.winfo_children() if isinstance(w, __import__("tkinter").Toplevel)), None)
    if win is not None:
        print("popup title:", win.title())
        win.destroy()
    app.after(50, app.destroy)

# Schedule the popup with a quick close
app.after(100, lambda: print("plan:", app._confirm_batch_plan(root, ".pro", files[:8])) or app.destroy())
import threading
threading.Timer(0.3, lambda: [w.destroy() for w in app.winfo_children() if hasattr(w, "title")]).start()
try:
    app.mainloop()
except Exception:
    pass
print("DONE")
