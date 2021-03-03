from ase.io.opls import OPLSff, OPLSStructure

s = OPLSStructure('172_ext.xyz')
with open("pclpars.par") as fd:
    opls = OPLSff(fd)
opls.write_lammps(s, prefix='lmp')
