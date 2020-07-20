import numpy as np
from ase.io.eps import EPS


class PNG(EPS):
    def write_header(self):
        from matplotlib.backends.backend_agg import RendererAgg
        dpi = 72
        self.renderer = RendererAgg(self.w, self.h, dpi)

    def write_trailer(self):
        # The array conversion magic is necessary to make things work with
        # matplotlib 2.0.0, 3.2.x, and 3.3.0 at the same time.
        import matplotlib.image
        buf = self.renderer.buffer_rgba()
        # Buf is of type bytes (matplotlib < 3.3.0) or memoryview.
        # That might be an implementation detail.
        array = np.frombuffer(buf, dtype=np.uint8).reshape(
            int(self.h), int(self.w), 4)
        matplotlib.image.imsave(
            self.filename, array, format="png")


def write_png(filename, atoms, **parameters):
    PNG(atoms, **parameters).write(filename)
