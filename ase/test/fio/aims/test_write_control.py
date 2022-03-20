import re
from pathlib import Path
from io import StringIO
from ase.io.aims import write_control
from ase.build import bulk
from ase.calculators.aims import AimsCube

parent = Path(__file__).parent


def test_control():
    atoms = bulk("Au")
    cube = AimsCube(plots=("delta_density",))
    parameters = {
        "xc": "LDA",
        "kpts": [2, 2, 2],
        "smearing": ("gaussian", 0.1),
        "output": ["dos 0.0 10.0 101 0.05", "hirshfeld"],
        "dos_kgrid_factors": [21, 21, 21],
        "vdw_correction_hirshfeld": True,
        "compute_forces": True,
        "output_level": "MD_light",
        "charge": 0.0,
        "species_dir": f"{parent}/species_dir/",
        "cubes": cube,
    }
    fd = StringIO()
    write_control(fd, atoms, parameters)
    txt = fd.getvalue()

    def contains(pattern):
        return re.search(pattern, txt, re.M)

    assert contains(r"xc\s+pw-lda")
    assert contains(r"k_grid\s+2 2 2")
    assert contains(r"k_offset\s+0.250000 0.250000 0.250000")
    assert contains(r"occupation_type\s+gaussian 0.1")
    assert contains(r"output\s+dos 0.0 10.0 101 0.05")
    assert contains(r"output\s+hirshfeld")
    assert contains(r"dos_kgrid_factors\s+21 21 21")
    assert contains(r"vdw_correction_hirshfeld")
    assert contains(r"compute_forces\s+.true.")
    assert contains(r"output_level\s+MD_light")
    assert contains(r"charge\s+0.0")
    assert contains("output cube delta_density")
    assert contains("   cube origin 0 0 0 ")
    assert contains("   cube edge 50 0.1 0.0 0.0 ")
    assert contains("   cube edge 50 0.0 0.1 0.0")
    assert contains("   cube edge 50 0.0 0.0 0.1")
