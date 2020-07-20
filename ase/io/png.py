from ase.io.eps import EPS


class PNG(EPS):
    def write_header(self):
        from matplotlib.backends.backend_agg import RendererAgg
        dpi = 72
        self.renderer = RendererAgg(self.w, self.h, dpi)

    def write_trailer(self):
        import matplotlib.image
        rgba_buffer = self.renderer.buffer_rgba()

        matplotlib.image.imsave(
            self.filename, rgba_buffer, format="png")


def write_png(filename, atoms, **parameters):
    PNG(atoms, **parameters).write(filename)
