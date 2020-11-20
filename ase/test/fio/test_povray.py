from ase.io.pov import write_pov
from ase.build import molecule
from subprocess import check_call,DEVNULL
def test_povray_io(povray_executable):
    H2 = molecule('H2')
    write_pov('H2', H2)
    check_call([povray_executable, 'H2.pov'], stderr=DEVNULL)
    
