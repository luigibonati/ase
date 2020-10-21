from io import StringIO
import ase.io

def test_lammpsdata_write(datadir):
    lammps_data_path = datadir / 'lammpsdata_input.data'
    lammps_data = ase.io.read(lammps_data_path, format="lammps-data",
                              units="metal")

    expected_output_path = datadir / 'lammpsdata_expected_output.data'
    with open(expected_output_path) as f:
        expected_output = f.read().splitlines()

    buf = StringIO()
    ase.io.write(buf, lammps_data, format="lammps-data", atom_style="full")

    lines = [line.strip() for line in buf.getvalue().split("\n")]
    assert lines == expected_output

