from ase.io.opls import OPLSff, OPLSStructure

s = OPLSStructure('172_ext.xyz')
fileObj = open('172_defs.par')
opls = OPLSff(fileObj)
opls.write_lammps(s, prefix='lmp')
