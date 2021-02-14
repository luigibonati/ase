from math import pi, sin, cos
import numpy as np

from ase.cell import Cell


def bz_vertices(icell):
    """See https://xkcd.com/1421 ..."""
    from scipy.spatial import Voronoi
    icell = Cell.ascell(icell)


    # Replace zero cell vectors with "almost zero" vectors:
    cellmask = icell.any(1)
    icell = icell.complete()
    for i in range(3):
        if not cellmask[i]:
            icell[i] *= 1e-3

    I = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
    G = np.dot(icell.T, I).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 13 in points:
            normal = G[points].sum(0)
            normal /= (normal**2).sum()**0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1


def bz_plot(cell, vectors=False, paths=None, points=None,
            elev=None, scale=1, interactive=False,
            pointstyle=None, ax=None, show=False):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.gcf()

    cell = Cell.new(cell)
    dimensions = cell.rank
    if dimensions == 0:
        raise ValueError('No BZ for 0D cell')

    verysmall = 1e-6

    if dimensions == 3:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d
        from matplotlib.patches import FancyArrowPatch
        Axes3D  # silence pyflakes

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d,
                                                   zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)
        azim = pi / 5
        elev = elev or pi / 6
        x = sin(azim)
        y = cos(azim)
        view = [x * cos(elev), y * cos(elev), sin(elev)]
        if ax is None:
            ax = fig.gca(projection='3d')
    elif dimensions == 2:
        # 2d in xy
        assert (all(abs(cell[2][0:2]) < verysmall) and
                all(abs(cell.T[2][0:2]) < verysmall))
        ax = plt.gca()
        cell = cell.copy()
    else:
        # 1d in x
        assert (all(abs(cell[2][:2]) < verysmall) and
                all(abs(cell.T[2][:2]) < verysmall) and
                abs(cell[0][1]) < verysmall and abs(cell[1][0]) < verysmall)
        ax = plt.gca()
        cell = cell.copy()

    icell = cell.reciprocal()
    kpoints = points
    vertices = bz_vertices(icell)

    maxp = 0.0
    minp = 0.0
    if dimensions == 1:
        length = icell[0, 0]
        x = 0.5 * length * np.array([-1, 1, -1])
        y = np.zeros(3)
        ax.plot(x, y, c='k', ls='-')
        maxp = length
    else:
        for points, normal in vertices:
            x, y, z = np.concatenate([points, points[:1]]).T
            if dimensions == 3:
                if np.dot(normal, view) < 0 and not interactive:
                    ls = ':'
                else:
                    ls = '-'
                ax.plot(x, y, z, c='k', ls=ls)
            elif dimensions == 2:
                ax.plot(x, y, c='k', ls='-')
            maxp = max(maxp, points.max())
            minp = min(minp, points.min())

    if vectors:
        if dimensions == 3:
            for i in range(3):
                ax.add_artist(Arrow3D([0, icell[i, 0]],
                                      [0, icell[i, 1]],
                                      [0, icell[i, 2]],
                                      mutation_scale=20, lw=1,
                                      arrowstyle='-|>', color='k'))

            maxp = max(maxp, 0.6 * icell.max())
        elif dimensions == 2:
            for i in range(2):
                ax.arrow(0, 0, icell[i, 0], icell[i, 1],
                         lw=1, color='k',
                         length_includes_head=True,
                         head_width=0.03, head_length=0.05)
            maxp = max(maxp, icell.max())
        else:
            ax.arrow(0, 0, icell[0, 0], 0,
                     lw=1, color='k',
                     length_includes_head=True,
                     head_width=0.03, head_length=0.05)
            maxp = max(maxp, icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            if dimensions == 3:
                ax.plot(x, y, z, c='r', ls='-', marker='.')
            elif dimensions in [1, 2]:
                ax.plot(x, y, c='r', ls='-')

            for name, point in zip(names, points):
                x, y, z = point
                if name == 'G':
                    name = '\\Gamma'
                elif len(name) > 1:
                    import re
                    m = re.match(r'^(\D+?)(\d*)$', name)
                    if m is None:
                        raise ValueError('Bad label: {}'.format(name))
                    name, num = m.group(1, 2)
                    if num:
                        name = '{}_{{{}}}'.format(name, num)
                if dimensions == 3:
                    ax.text(x, y, z, '$\\mathrm{' + name + '}$',
                            ha='center', va='bottom', color='g')
                elif dimensions == 2:
                    ha_s = ['right', 'left', 'right']
                    va_s = ['bottom', 'bottom', 'top']

                    ha = ha_s[int(np.sign(x))]
                    va = va_s[int(np.sign(y))]
                    if abs(z) < 1e-6:
                        ax.text(x, y, '$\\mathrm{' + name + '}$',
                                ha=ha, va=va, color='g',
                                zorder=5)
                else:
                    if abs(y) < 1e-6 and abs(z) < 1e-6:
                        ax.text(x, y, '$\\mathrm{' + name + '}$',
                                ha='center', va='bottom', color='g',
                                zorder=5)

    if kpoints is not None:
        kw = {'c': 'b'}
        if pointstyle is not None:
            kw.update(pointstyle)
        for p in kpoints:
            if dimensions == 3:
                ax.scatter(p[0], p[1], p[2], **kw)
            elif dimensions == 2:
                ax.scatter(p[0], p[1], zorder=4, **kw)
            else:
                ax.scatter(p[0], 0, zorder=4, **kw)

    ax.set_axis_off()

    if dimensions in [1, 2]:
        ax.autoscale_view(tight=True)
        s = maxp * 1.05
        ax.set_xlim(-s, s)
        ax.set_ylim(-s, s)
        ax.set_aspect('equal')

    if dimensions == 3:
        # ax.set_aspect('equal') <-- won't work anymore in 3.1.0
        ax.view_init(azim=azim / pi * 180, elev=elev / pi * 180)
        # We want aspect 'equal', but apparently there was a bug in
        # matplotlib causing wrong behaviour.  Matplotlib raises
        # NotImplementedError as of v3.1.0.  This is a bit unfortunate
        # because the workarounds known to StackOverflow and elsewhere
        # all involve using set_aspect('equal') and then doing
        # something more.
        #
        # We try to get square axes here by setting a square figure,
        # but this is probably rather inexact.
        fig = ax.get_figure()
        xx = plt.figaspect(1.0)
        fig.set_figheight(xx[1])
        fig.set_figwidth(xx[0])

        # oldlibs tests with matplotlib 2.0.0 say we have no set_proj_type:
        if hasattr(ax, 'set_proj_type'):
            ax.set_proj_type('ortho')

        minp0 = 0.9 * minp  # Here we cheat a bit to trim spacings
        maxp0 = 0.9 * maxp
        ax.set_xlim3d(minp0, maxp0)
        ax.set_ylim3d(minp0, maxp0)
        ax.set_zlim3d(minp0, maxp0)

    if show:
        plt.show()

    return ax
