from ase.build import graphene
from ase.visualize import view
from ase.geometry.rmsd import find_crystal_reductions


size = 2
atoms = graphene(size=(2, 2, 1))
atoms.positions += (-0.1, 0.6, 0)
atoms.wrap()
atoms.rattle(seed=0, stdev=0.1)

result = find_crystal_reductions(atoms)
for reduced in result:
    print("factor: {}  rmsd: {}".format(reduced.factor, reduced.rmsd))
    view(reduced.atoms)
