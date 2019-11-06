# creates: rattled.png, translated_2.png, translated_4.csv, reduced_2.png, clustered_2.png, clustered_4.png, reduced_4.png
import itertools
import numpy as np
import matplotlib.pyplot as plt
from ase.geometry.rmsd import find_crystal_reductions
from ase.build import graphene


def plot_atoms(cell, positions=None, c='C0', components=None):
    x = [0, cell[0, 0], cell[0, 0] + cell[1, 0], cell[1, 0], 0]
    y = [0, cell[0, 1], cell[0, 1] + cell[1, 1], cell[1, 1], 0]
    plt.plot(x, y, c='k', ls=['-', '--'][positions is None])
    if positions is None:
        return

    s = 150
    x, y, _ = positions.T
    if components is None:
        plt.scatter(x, y, s=s, edgecolor='k', facecolor='none')
        plt.scatter(x, y, s=s, edgecolor='none', facecolor=c)
        return

    for c in np.unique(components):
        indices = np.where(components == c)
        plt.scatter(x[indices], y[indices], s=s, edgecolor='k',
                    facecolor='none', zorder=1)
        plt.scatter(x[indices], y[indices], s=s, edgecolor='none',
                    facecolor='C%d' % c)


def save_figure(name, factor=None):
    plt.axis('equal')
    plt.axis('off')
    if factor:
        path = "{0}_{1}.png".format(name, factor)
    else:
        path = "{0}.png".format(name)

    plt.savefig(path)
    plt.clf()


atoms = graphene(size=(2, 2, 1))
atoms.positions += (-0.1, 0.6, 0)
atoms.rattle(seed=0, stdev=0.1)
atoms.wrap()

figsize=(8, 6)
plt.figure(figsize=figsize)
plot_atoms(atoms.cell[:2, :2], atoms.get_positions(), c='gray')
save_figure("rattled")

result = find_crystal_reductions(atoms)
for reduced in result[:2]:
    print("factor: {}  rmsd: {}".format(reduced.factor, reduced.rmsd))

    plt.figure(figsize=figsize)
    plot_atoms(atoms.cell[:2, :2], atoms.get_positions(),
               components=reduced.components)
    plot_atoms(reduced.atoms.cell[:2, :2])
    save_figure("clustered", reduced.factor)

    plt.figure(figsize=figsize)
    r = range(reduced.factor)
    for i, j in itertools.product(r, r):
        translated = atoms.copy()
        translated.positions += i * reduced.atoms.cell[0]
        translated.positions += j * reduced.atoms.cell[1]
        translated.wrap()
        plot_atoms(translated.cell[:2, :2], translated.get_positions(),
                   components=reduced.components)
    plot_atoms(reduced.atoms.cell[:2, :2])
    save_figure("translated", reduced.factor)

    plt.figure(figsize=figsize)
    plot_atoms(reduced.atoms.cell[:2, :2], reduced.atoms.get_positions(),
               components=np.arange(len(reduced.atoms)))
    plot_atoms(atoms.cell[:2, :2])
    save_figure("reduced", reduced.factor)
