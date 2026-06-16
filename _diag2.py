from gui import SurfaceFinishGUI
app = SurfaceFinishGUI()
def report():
    print("LIVE sashpos:", app._paned.sashpos(0))
    panes = app._paned.panes()
    for p in panes:
        w = app.nametowidget(p)
        print(" ", p, "size:", w.winfo_width(), "x", w.winfo_height())
    app.destroy()
app.after(800, report)
app.mainloop()
