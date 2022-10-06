"""Test file for exciting file input and output methods."""

import logging
import os
import xml.etree.ElementTree as ET

import pytest
import numpy as np

import ase
from ase.units import Bohr


LOGGER = logging.getLogger(__name__)


try:
    __import__('excitingtools')
    import ase.io.exciting
except ModuleNotFoundError:
    MESSAGE = "exciting tests are skipped if excitingtools not installed."
    LOGGER.info(MESSAGE)


LDA_VWN_AR_INFO_OUT = """
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Starting initialization                                                      +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 Lattice vectors (cartesian) :
     15.0000000000      0.0000000000      0.0000000000
      0.0000000000     15.0000000000      0.0000000000
      0.0000000000      0.0000000000     15.0000000000
 
 Reciprocal lattice vectors (cartesian) :
      0.4188790205      0.0000000000      0.0000000000
      0.0000000000      0.4188790205      0.0000000000
      0.0000000000      0.0000000000      0.4188790205
 
 Unit cell volume                           :    3375.0000000000
 Brillouin zone volume                      :       0.0734963595
 
 Species :    1 (Ar)
     parameters loaded from                 :    Ar.xml
     name                                   :    argon
     nuclear charge                         :     -18.00000000
     electronic charge                      :      18.00000000
     atomic mass                            :   72820.74920000
     muffin-tin radius                      :       6.00000000
     # of radial points in muffin-tin       :    1000
 
     atomic positions (lattice) :
       1 :   0.00000000  0.00000000  0.00000000
 
 Total number of atoms per unit cell        :       1
 
 Spin treatment                             :    spin-unpolarised
 
 Number of Bravais lattice symmetries       :      48
 Number of crystal symmetries               :      48
 
 k-point grid                               :       1    1    1
 Total number of k-points                   :       1
 k-point set is reduced with crystal symmetries
 
 R^MT_min * |G+k|_max (rgkmax)              :      10.00000000
 Species with R^MT_min                      :       1 (Ar)
 Maximum |G+k| for APW functions            :       1.66666667
 Maximum |G| for potential and density      :       7.50000000
 Polynomial order for pseudochg. density    :       9
 
 G-vector grid sizes                        :      36    36    36
 Total number of G-vectors                  :   23871
 
 Maximum angular momentum used for
     APW functions                          :       8
     computing H and O matrix elements      :       4
     potential and density                  :       4
     inner part of muffin-tin               :       2
 
 Total nuclear charge                       :     -18.00000000
 Total electronic charge                    :      18.00000000
 Total core charge                          :      10.00000000
 Total valence charge                       :       8.00000000
 
 Effective Wigner radius, r_s               :       3.55062021
 
 Number of empty states                     :       5
 Total number of valence states             :      10
 
 Maximum Hamiltonian size                   :     263
 Maximum number of plane-waves              :     251
 Total number of local-orbitals             :      12
 
 Exchange-correlation type                  :     100
     libxc; exchange: Slater exchange; correlation: Vosko, Wilk & Nusair (VWN5) (see libxc for references)
 
 Smearing scheme                            :    Gaussian
 Smearing width                             :       0.00100000
 
 Using multisecant Broyden potential mixing
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Ending initialization                                                        +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    1                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.82493279
 _______________________________________________________________
 Fermi energy                               :        -0.20111449
 Kinetic energy                             :       530.56137212
 Coulomb energy                             :     -1029.02167746
 Exchange energy                            :       -27.93377198
 Correlation energy                         :        -1.43085548
 Sum of eigenvalues                         :      -305.07886015
 Effective potential energy                 :      -835.64023227
 Coulomb potential energy                   :      -796.81322609
 xc potential energy                        :       -38.82700618
 Hartree energy                             :       205.65681157
 Electron-nuclear energy                    :     -1208.12684923
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61506441
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00183897
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99816103
     total charge in muffin-tins            :        17.99816103
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36071248
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
                                                          
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Convergency criteria checked for the last 2 iterations                       +
| Convergence targets achieved. Performing final SCF iteration                 +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81796101
 _______________________________________________________________
 Fermi energy                               :        -0.20044598
 Kinetic energy                             :       530.57303096
 Coulomb energy                             :     -1029.02642037
 Exchange energy                            :       -27.93372809
 Correlation energy                         :        -1.43084350
 Sum of eigenvalues                         :      -305.07413840
 Effective potential energy                 :      -835.64716936
 Coulomb potential energy                   :      -796.82023455
 xc potential energy                        :       -38.82693481
 Hartree energy                             :       205.65454603
 Electron-nuclear energy                    :     -1208.12932661
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61630310
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36095838
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Self-consistent loop stopped                                                 +
| EXCITING NITROGEN-14 stopped                                                 =
"""

@pytest.fixture
def excitingtools():
    """If we cannot import excitingtools we skip tests with this fixture."""
    return pytest.importorskip('excitingtools')


@pytest.fixture
def nitrogen_trioxide_atoms():
    """Helper fixture to create a NO3 ase atoms object for tests."""
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 3, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)


def structure_xml_to_ase_atoms(fileobj) -> ase.Atoms:
    """Helper function to read structure from input.xml file.

    Args:
        fileobj: File handle from which data should be read.
    Returns:
        ASE atoms object with all the relevant fields filled.
    """
    # Parse file into element tree
    doc = ET.parse(fileobj)
    root = doc.getroot()
    species_nodes = root.find('structure').iter('species')  # type: ignore

    symbols = []
    positions = []
    basevects = []

    # Collect data from tree
    for species_node in species_nodes:
        symbol = species_node.get('speciesfile').split('.')[0]  # type: ignore
        natoms = species_node.iter('atom')
        for atom in natoms:
            x_pos, y_pos, z_pos = atom.get('coord').split()  # type: ignore
            positions.append([float(x_pos), float(y_pos), float(z_pos)])
            symbols.append(symbol)

    # scale unit cell according to scaling attributes
    if 'scale' in doc.find('structure/crystal').attrib:  # type: ignore
        scale = float(str(
            doc.find('structure/crystal').attrib['scale']))  # type: ignore
    else:
        scale = 1

    if 'stretch' in doc.find('structure/crystal').attrib:  # type: ignore
        a_stretch, b_stretch, c_stretch = doc.find(  # type: ignore
            'structure/crystal').attrib['stretch'].text.split()
        stretch = np.array(
            [float(a_stretch), float(b_stretch), float(c_stretch)])
    else:
        stretch = np.array([1.0, 1.0, 1.0])

    basevectsn = root.findall('structure/crystal/basevect')
    for basevect in basevectsn:
        x_mag, y_mag, z_mag = basevect.text.split()  # type: ignore
        basevects.append(np.array([float(x_mag) * Bohr * stretch[0],
                                   float(y_mag) * Bohr * stretch[1],
                                   float(z_mag) * Bohr * stretch[2]
                                   ]) * scale)  # type: ignore
    atoms = ase.Atoms(symbols=symbols, cell=basevects)

    atoms.set_scaled_positions(positions)
    if 'molecule' in root.find('structure').attrib.keys():  # type: ignore
        if root.find('structure').attrib['molecule']:  # type: ignore
            atoms.set_pbc(False)
    else:
        atoms.set_pbc(True)

    return atoms


def test_write_input_xml_file(tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test writing input.xml file using write_input_xml_file()."""
    file_path = os.path.join(tmp_path, 'input.xml')
    input_param_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        input_parameters=input_param_dict,
        species_path=(
            "/home/dts/Documents/theory/ase-exciting/ase/ase/test/fio/"
            "exciting"),
        title=None)
    assert os.path.exists(file_path)
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = structure_xml_to_ase_atoms(file_path)
    print(atoms_obj.get_chemical_symbols())
    assert atoms_obj.get_chemical_symbols() == ['N', 'O', 'O', 'O']
    input_xml_tree = ET.parse(file_path).getroot()
    assert list(input_xml_tree)[2].get("xctype") == "GGA_PBE_SOL"
    assert list(input_xml_tree)[2].get("rgkmax") == '8.0'
    assert list(input_xml_tree)[2].get("tforce") == 'true'


def test_parse_info_out_xml_bad_path(tmp_path, excitingtools):
    """Tests parse method raises error when info.out file doesn't exist."""
    output_file_path = os.path.join(tmp_path, 'info.out')

    with pytest.raises(FileNotFoundError):
        ase.io.exciting.parse_output(
            output_file_path)


def test_parse_info_out_energy(tmp_path, excitingtools):
    """Test parsing the INFO.OUT output from exciting using parse_output()."""
    file = tmp_path / "INFO.OUT"
    file.write_text(LDA_VWN_AR_INFO_OUT)
    assert file.exists(), "INFO.OUT not written to tmp_path"

    results = ase.io.exciting.parse_output(file.as_posix())

    # Finally ensure that we that the final SCL cycle is what we expect and
    # the final SCL results can be accessed correctly:
    final_scl_iteration = list(results["scl"].keys())[-1]
    assert np.round(
        float(results["scl"][final_scl_iteration][
            "Hartree energy"]) - 205.65454603, 6) == 0.
    assert np.round(
        float(results["scl"][final_scl_iteration][
            "Estimated fundamental gap"]) - 0.36095838, 6) == 0.
    assert np.round(float(results["scl"][
        final_scl_iteration]["Hartree energy"]) - 205.65454603, 6) == 0.
