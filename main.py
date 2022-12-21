from SurfaceTexture import SurfaceTexture

if __name__ == "__main__":
    data = "example/example_trace.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff, order=1)
    surface_texture.plot_material_ratio()

