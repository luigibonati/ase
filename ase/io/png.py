from distutils.version import LooseVersion

import numpy as np

from ase.io.eps import EPS


class PNG(EPS):
    def write_header(self):
        from matplotlib.backends.backend_agg import RendererAgg

        dpi = 72
        self.renderer = RendererAgg(self.w, self.h, dpi)

    def write_trailer(self):
        from matplotlib import _png
        import matplotlib
        x = self.renderer.buffer_rgba()
        try:
            _png.write_png(x, self.w, self.h, self.filename, 72)
        except (TypeError, ValueError):
            x = np.frombuffer(x, np.uint8).reshape(
                (int(self.h), int(self.w), 4))
            _png.write_png(x, self.filename, 72)


def write_png(filename, atoms, **parameters):
    PNG(atoms, **parameters).write(filename)
