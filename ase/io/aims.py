import time
import warnings

import numpy as np

from ase import Atoms, Atom
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.units import Ang, fs
from ase.utils import reader, writer


v_unit = Ang / (1000.0 * fs)


class AimsOutError(IOError):
    pass


class AimsOutChunk:
    """A part of the aims.out file correponding to a single structure"""

    def __init__(self, lines, n_atoms=None, constraints=None):
        self.lines = lines
        self._n_atoms = n_atoms
        self._constraints = constraints
        self._atoms = None
        self._forces = None
        self._stresses = None
        self._stress = None
        self._energy = None
        self._free_energy = None
        self._n_iter = None
        self._magmom = None
        self._E_f = None
        self._dipole = None
        self._is_md = None
        self._is_metallic = None
        self._hirshfeld_charges = None
        self._hirshfeld_atomic_dipoles = None
        self._hirshfeld_volumes = None
        self._hirshfeld_dipole = None

    def reverse_search_for(self, keys, line_start=0):
        """Find the last time one of the keys appears in self.lines

        Parameters
        ----------
        keys: list of str
            The key strings to search for in self.lines
        line_start: int
            The lowest index to search for in self.lines

        Returns
        -------
        int
            The last time one of the keys appears in self.lines
        """
        for ll, line in enumerate(self.lines[-1:line_start:-1]):
            if any([key in line for key in keys]):
                return len(self.lines) - ll

        return len(self.lines)

    def _parse_constraints(self):
        """Parse the constraints from the aims.out file"""

        line_start = self.reverse_search_for(["Found relaxation constraint for atom"])
        if line_start == len(self.lines):
            return None

        fix = []
        fix_cart = []
        for line in self.lines[line_start:]:
            xyz = [0, 0, 0]
            ind = int(line.split()[5][:-1]) - 1
            if "All coordinates fixed" in line:
                if ind not in fix:
                    fix.append(ind)
            if "coordinate fixed" in line:
                coord = line.split()[6]
                if coord == "x":
                    xyz[0] = 1
                elif coord == "y":
                    xyz[1] = 1
                elif coord == "z":
                    xyz[2] = 1
                keep = True
                for n, c in enumerate(fix_cart):
                    if ind == c.a:
                        keep = False
                if keep:
                    fix_cart.append(FixCartesian(ind, xyz))
                else:
                    fix_cart[n].mask[xyz.index(1)] = 0
        if len(fix) > 0:
            fix_cart.append(FixAtoms(indices=fix))

        return fix_cart

    def _parse_initial_cell(self):
        """Parse the initial cell from the aims.out file"""
        line_start = self.reverse_search_for(["| Unit cell:"])
        if line_start < len(self.lines):
            cell = [
                [float(inp) for inp in line.split()[-3:]]
                for line in self.lines[line_start : line_start + 3]
            ]
        else:
            cell = None
        return cell

    def _parse_initial_atoms(self):
        """Create an atoms object for the initial geometry.in structure from the aims.out file"""
        line_start = self.reverse_search_for(["| Number of atoms"]) - 1

        # Get the initial geometries
        self._is_md = len(self.lines) > self.reverse_search_for(
            ["Complete information for previous time-step:"]
        )
        if self._is_md:
            return self._parse_atoms()

        self._n_atoms = int(self.lines[line_start].split()[5])
        cell = self._parse_cell()

        atoms = Atoms()
        line_start = self.reverse_search_for(["Atomic structure:"]) + 1
        for line in self.lines[line_start : line_start + self._n_atoms]:
            inp = line.split()
            atoms.append(Atom(inp[3], (float(inp[4]), float(inp[5]), float(inp[6]))))
        assert len(atoms) == self._n_atoms

        if cell:
            atoms.set_cell(cell)
        atoms.set_constraint(self.constraints)

        return atoms

    def _parse_atoms(self):
        """Create an atoms object for the subsequent structures calculated in the aims.out file"""
        start_keys = [
            "Atomic structure (and velocities) as used in the preceding time step",
            "Updated atomic structure",
            "Atomic structure that was used in the preceding time step of the wrapper",
        ]
        line_start = self.reverse_search_for(start_keys)
        cell = []
        velocities = []
        atoms = Atoms()
        for line in self.lines[line_start:]:
            if "lattice_vector   " in line:
                cell.append([float(inp) for inp in line.split()[1:]])
            elif "atom   " in line:
                line_split = line.split()
                atoms.append(
                    Atom(line_split[4], tuple([float(inp) for inp in line_split[1:4]]))
                )
            elif "velocity   " in line:
                velocities.append([float(inp) for inp in line.split()[1:]])

        assert len(atoms) == self._n_atoms
        assert (len(velocities) == self._n_atoms) or (len(velocities) == 0)
        if len(cell) > 0:
            atoms.set_cell(np.array(cell))
        if len(velocities) > 0:
            atoms.set_velocities(np.array(velocities))
        atoms.set_constraint(self.constraints)

        return atoms

    def _parse_forces(self):
        """Parse the forces from the aims.out file"""
        line_start = self.reverse_search_for(["Total atomic forces"])
        if line_start == len(self.lines):
            return

        return np.array(
            [
                [float(inp) for inp in line.split()[-3:]]
                for line in self.lines[line_start : line_start + self._n_atoms]
            ]
        )

    def _parse_stresses(self):
        """Parse the stresses from the aims.out file"""
        line_start = self.reverse_search_for(
            ["Per atom stress (eV) used for heat flux calculation"]
        )
        if line_start == len(self.lines):
            return
        line_start = self.reverse_search_for(["-------------"], line_start)
        self._stresses = []
        for line in self.lines[line_start : line_start + self._n_atoms]:
            xx, yy, zz, xy, xz, yz = [float(d) for d in line.split()[2:8]]
            stresses.append([xx, yy, zz, yz, xz, xy])

        return np.array(stresses)

    def _parse_stress(self):
        """Parse the stress from the aims.out file"""
        from ase.stress import full_3x3_to_voigt_6_stress

        line_start = self.reverse_search_for(
            ["Analytical stress tensor - Symmetrized"]
        )  # Offest to relevant lines
        if line_start == len(self.lines):
            return

        stress = [
            [float(inp) for inp in line.split()[2:5]]
            for line in self.lines[line_start + 4 : line_start + 7]
        ]
        return full_3x3_to_voigt_6_stress(stress)

    def _parse_metallic(self):
        """Checks the outputfile to see if the chunk corresponds to a metallic system"""
        line_start = self.reverse_search_for(
            "material is metallic within the approximate finite broadening function (occupation_type)"
        )
        return line_start != len(self.lines)

    def _parse_energy(self):
        """Parse the energy from the aims.out file"""
        if np.any(self._atoms.pbc):
            line = self.lines[
                -1 + self.reverse_search_for(["Total energy uncorrected"])
            ]
        else:
            line = self.lines[-1 + self.reverse_search_for(["Total energy corrected"])]
        return float(line.split()[5])

    def _parse_free_energy(self):
        """Parse the free energy from the aims.out file"""
        line = self.lines[
            2 + self.reverse_search_for(["Energy and forces in a compact form"])
        ]
        return float(line.split()[5])

    def _parse_number_of_iterations(self):
        """Parse the number of iterations an SCF cycle took from the aims.out file"""
        line = self.lines[
            -1 + self.reverse_search_for(["| Number of self-consistency cycles"])
        ]
        return int(line.split(":")[-1].strip())

    def _parse_magnetic_moment(self):
        """Parse the magnetic moment from the aims.out file"""
        line_start = self.reverse_search_for(["N_up - N_down"])
        if line_start == len(self.lines):
            return None

        line = self.lines[line_start - 1]
        return float(line.split(":")[-1].strip())

    def _parse_fermi_level(self):
        """Parse the Fermi Level from the aims.out file"""
        line_start = self.reverse_search_for(
            ["| Chemical potential (Fermi level) in eV"]
        )
        if line_start == len(self.lines):
            return None

        line = self.lines[line_start - 1]
        return float(line.split(":")[-1].strip())

    def _parse_dipole(self):
        """Parse the electric dipole moment from the aims.out file file."""
        line_start = self.reverse_search_for("Total dipole moment [eAng]")
        if line_start == len(self.lines):
            return

        line = self.lines[line_start - 1]
        return np.array([float(inp) for inp in line.split()[6:9]])

    def _parse_hirshfled(self):
        """Parse the Hirshfled charges and dipole moments from the ouput"""
        line_start = self.reverse_search_for(
            "Performing Hirshfeld analysis of fragment charges and moments."
        )
        if line_start == len(self.lines):
            return

        line_start = self.reverse_search_for("Hirshfeld charge", line_start)
        self._hirshfeld_charges = np.array(
            [
                float(line.split(":")[1])
                for line in self.lines[
                    line_start : line_start + (self._n_atoms - 1) * 10 : 10
                ]
            ]
        )

        line_start = self.reverse_search_for("Hirshfeld volume", line_start)
        self._hirshfeld_volumes = np.array(
            [
                float(line.split(":")[1])
                for line in self.lines[
                    line_start : line_start + (self._n_atoms - 1) * 10 : 10
                ]
            ]
        )

        line_start = self.reverse_search_for("Hirshfeld dipole vector", line_start)
        self._hirshfeld_atomic_dipoles = np.array(
            [
                [float(inp) for inp in line.split(":")[1].split()]
                for line in self.lines[
                    line_start : line_start + (self._n_atoms - 1) * 10 : 10
                ]
            ]
        )

        self._hirshfeld_dipole = np.sum(
            self._hirshfeld_charges.rehsape((-1, 1)) * self._atoms.get_positions(),
            axis=1,
        )

    @property
    def atoms(self):
        """Convert AimsOutChunk to Atoms object"""
        if self._n_atoms is None and self._atoms is None:
            self._atoms = self._parse_initial_atoms()
        elif self._atoms is None:
            self._atoms = self._parse_atoms()

        self._atoms.calc = SinglePointDFTCalculator(
            self._atoms,
            energy=self.energy,
            free_energy=self.free_energy,
            forces=self.forces,
            stress=self.stress,
            stresses=self.stresses,
            magmom=self.magmom,
            dipole=self.dipole,
        )
        return self._atoms

    @property
    def results(self):
        """Convert an AimsOutChunk to a Results Dictionary"""
        results = {
            "energy": self.energy,
            "free_energy": self.free_energy,
            "forces": self.forces,
            "stress": self.stress,
            "stresses": self.stresses,
            "magmom": self.magmom,
            "dipole": self.dipole,
            "fermi_energy": self._E_f,
            "n_iter": self.n_iter,
            "hirshfeld_charges": self.hirshfeld_charges,
            "hirshfeld_dipole": self.hirshfeld_atomic_dipoles,
            "hirshfeld_volumes": self.hirshfeld_volumes,
            "hirshfeld_atomic_dipoles": self.hirshfeld_atomic_dipoles,
        }

        for key, val in results.items():
            if val is None:
                del results[key]

        return results

    @lazyproperty
    def constraints(self):
        """The relaxation constraints for the calculation"""
        return self._parse_constraints()

    @lazyproperty
    def forces(self):
        """The forces for the chunk"""
        return self._parse_forces()

    @lazyproperty
    def stresses(self):
        """The atomic stresses for the chunk"""
        return self._parse_stresses()

    @lazyproperty
    def stress(self):
        """The stress for the chunk"""
        return self._parse_stress()

    @lazyproperty
    def energy(self):
        """The energy for the chunk"""
        return self._parse_energy()

    @lazyproperty
    def free_energy(self):
        """The free energy for the chunk"""
        return self._parse_free_energy()

    @lazyproperty
    def n_iter(self):
        """The number of SCF iterations needed to converge the SCF cycle for the chunk"""
        return self._parse_number_of_iterations()

    @lazyproperty
    def magmom(self):
        """The magnetic moment for the chunk"""
        return self._parse_magnetic_moment()

    @lazyproperty
    def E_f(self):
        """The Fermi energy for the chunk"""
        return self._parse_E_f()

    @lazyproperty
    def dipole(self):
        """The electronic dipole moment for the chunk"""
        return self._parse_dipole()

    @lazyproperty
    def is_metallic(self):
        """True if the chunk is for a metallic material"""
        return self._parse_metallic()

    @property
    def n_atoms(self):
        """The number of atoms for the material"""
        if self._n_atoms is None:
            self._parse_initial_atoms()
        return self._n_atoms

    @property
    def hirshfeld_charges(self):
        """The Hirshfeld charges for the chunk"""
        if self._hirshfeld_charges is None:
            self._parse_hirshfeld()
        return self._hirshfeld_charges

    @property
    def hirshfeld_atomic_dipole(self):
        """The Hirshfeld atomic dipole moments for the chunk"""
        if self._hirshfeld_atomic_dipoles is None:
            self._parse_hirshfeld()
        return self._hirshfeld_atomic_dipoles

    @property
    def hirshfeld_volume(self):
        """The Hirshfeld volume for the chunk"""
        if self._hirshfeld_volumes is None:
            self._parse_hirshfeld()
        return self._hirshfeld_volumes

    @property
    def hirshfeld_dipole(self):
        """The Hirshfeld systematic dipole moment for the chunk"""
        if self._hirshfeld_dipole is None:
            self._parse_hirshfeld()
        return self._hirshfeld_dipole


def get_aims_out_chunks(fd, n_atoms=None, constraints=None):
    """Yield unprocessed chunks (header, lines) for each AimsOutChunk image."""
    try:
        line = next(fd).strip()  # Raises StopIteration on empty file
    except StopIteration:
        break

    # Get to the control.in section
    while "Reading file control.in." not in line:
        line = next(fd).strip()

    # Check if calculation is a relaxation
    control_in = []
    while "Reading geometry description geometry.in." not in line:
        line = next(fd).strip()
        control_in.append(line)

    # If the calculation is a relaxation start a new chunk upon geometry optimization; else upon Re-initialization of the SCF cycle
    chunk_end_line = "Begin self-consistency loop: Re-initialization"
    for line in control_in:
        if "Geometry relaxation:" in line:
            chunk_end_line = (
                "Geometry optimization: Attempting to predict improved coordinates."
            )

    ignore_chunk_end_line = False
    while True:
        try:
            line = next(fd).strip()  # Raises StopIteration on empty file
        except StopIteration:
            break

        lines = []
        while not ignore_chunk_end_line and chunk_end_line not in line:
            lines.append(line)
            # If SCF cycle not converged, don't end chunk on next Re-initialization
            if (
                "Self-consistency cycle not yet converged - restarting mixer to attempt better convergence."
                in line
            ):
                ignore_chunk_end_line = True
            elif "Begin self-consistency loop: Re-initialization" in line:
                ignore_chunk_end_line = False

            try:
                line = next(fd).strip()
            except StopIteration:
                break

        yield AimsOutChunk(lines, n_atoms, constraints)


@reader
def read_aims(fd, apply_constraints=True):
    """Import FHI-aims geometry type files.

    Reads unitcell, atom positions and constraints from
    a geometry.in file.

    If geometric constraint (symmetry parameters) are in the file
    include that information in atoms.info["symmetry_block"]
    """

    lines = fd.readlines()
    return parse_geometry_lines(lines, apply_constraints=apply_constraints)


def parse_geometry_lines(lines, apply_constraints=True):

    from ase import Atoms
    from ase.constraints import (
        FixAtoms,
        FixCartesian,
        FixScaledParametricRelations,
        FixCartesianParametricRelations,
    )

    atoms = Atoms()

    positions = []
    cell = []
    symbols = []
    velocities = []
    magmoms = []
    symmetry_block = []
    charges = []
    fix = []
    fix_cart = []
    xyz = np.array([0, 0, 0])
    i = -1
    n_periodic = -1
    periodic = np.array([False, False, False])
    cart_positions, scaled_positions = False, False
    for line in lines:
        inp = line.split()
        if inp == []:
            continue
        if inp[0] in ["atom", "atom_frac"]:

            if inp[0] == "atom":
                cart_positions = True
            else:
                scaled_positions = True

            if xyz.all():
                fix.append(i)
            elif xyz.any():
                fix_cart.append(FixCartesian(i, xyz))
            floatvect = float(inp[1]), float(inp[2]), float(inp[3])
            positions.append(floatvect)
            symbols.append(inp[4])
            magmoms.append(0.0)
            charges.append(0.0)
            xyz = np.array([0, 0, 0])
            i += 1

        elif inp[0] == "lattice_vector":
            floatvect = float(inp[1]), float(inp[2]), float(inp[3])
            cell.append(floatvect)
            n_periodic = n_periodic + 1
            periodic[n_periodic] = True

        elif inp[0] == "initial_moment":
            magmoms[-1] = float(inp[1])

        elif inp[0] == "initial_charge":
            charges[-1] = float(inp[1])

        elif inp[0] == "constrain_relaxation":
            if inp[1] == ".true.":
                fix.append(i)
            elif inp[1] == "x":
                xyz[0] = 1
            elif inp[1] == "y":
                xyz[1] = 1
            elif inp[1] == "z":
                xyz[2] = 1

        elif inp[0] == "velocity":
            floatvect = [v_unit * float(l) for l in inp[1:4]]
            velocities.append(floatvect)

        elif inp[0] in [
            "symmetry_n_params",
            "symmetry_params",
            "symmetry_lv",
            "symmetry_frac",
        ]:
            symmetry_block.append(" ".join(inp))

    if xyz.all():
        fix.append(i)
    elif xyz.any():
        fix_cart.append(FixCartesian(i, xyz))

    if cart_positions and scaled_positions:
        raise Exception(
            "Can't specify atom positions with mixture of "
            "Cartesian and fractional coordinates"
        )
    elif scaled_positions and periodic.any():
        atoms = Atoms(symbols, scaled_positions=positions, cell=cell, pbc=periodic)
    else:
        atoms = Atoms(symbols, positions)

    if len(velocities) > 0:
        if len(velocities) != len(positions):
            raise Exception("Number of positions and velocities have to coincide.")
        atoms.set_velocities(velocities)

    fix_params = []

    if len(symmetry_block) > 5:
        params = symmetry_block[1].split()[1:]

        lattice_expressions = []
        lattice_params = []

        atomic_expressions = []
        atomic_params = []

        n_lat_param = int(symmetry_block[0].split(" ")[2])

        lattice_params = params[:n_lat_param]
        atomic_params = params[n_lat_param:]

        for ll, line in enumerate(symmetry_block[2:]):
            expression = " ".join(line.split(" ")[1:])
            if ll < 3:
                lattice_expressions += expression.split(",")
            else:
                atomic_expressions += expression.split(",")

        fix_params.append(
            FixCartesianParametricRelations.from_expressions(
                list(range(3)),
                lattice_params,
                lattice_expressions,
                use_cell=True,
            )
        )

        fix_params.append(
            FixScaledParametricRelations.from_expressions(
                list(range(len(atoms))), atomic_params, atomic_expressions
            )
        )

    if any(magmoms):
        atoms.set_initial_magnetic_moments(magmoms)
    if any(charges):
        atoms.set_initial_charges(charges)

    if periodic.any():
        atoms.set_cell(cell)
        atoms.set_pbc(periodic)
    if len(fix):
        atoms.set_constraint([FixAtoms(indices=fix)] + fix_cart + fix_params)
    else:
        atoms.set_constraint(fix_cart + fix_params)

    if fix_params and apply_constraints:
        atoms.set_positions(atoms.get_positions())
    return atoms


@writer
def write_aims(
    fd,
    atoms,
    scaled=False,
    geo_constrain=False,
    velocities=False,
    ghosts=None,
    info_str=None,
    wrap=False,
):
    """Method to write FHI-aims geometry files.

    Writes the atoms positions and constraints (only FixAtoms is
    supported at the moment).

    Args:
        fd: file object
            File to output structure to
        atoms: ase.atoms.Atoms
            structure to output to the file
        scaled: bool
            If True use fractional coordinates instead of Cartesian coordinates
        symmetry_block: list of str
            List of geometric constraints as defined in:
            https://arxiv.org/abs/1908.01610
        velocities: bool
            If True add the atomic velocity vectors to the file
        ghosts: list of Atoms
            A list of ghost atoms for the system
        info_str: str
            A string to be added to the header of the file
        wrap: bool
            Wrap atom positions to cell before writing
    """

    from ase.constraints import FixAtoms, FixCartesian

    if geo_constrain:
        if not scaled:
            warnings.warn(
                "Setting scaled to True because a symmetry_block is detected."
            )
            scaled = True

    fd.write("#=======================================================\n")
    if hasattr(fd, "name"):
        fd.write("# FHI-aims file: " + fd.name + "\n")
    fd.write("# Created using the Atomic Simulation Environment (ASE)\n")
    fd.write("# " + time.asctime() + "\n")

    # If writing additional information is requested via info_str:
    if info_str is not None:
        fd.write("\n# Additional information:\n")
        if isinstance(info_str, list):
            fd.write("\n".join(["#  {}".format(s) for s in info_str]))
        else:
            fd.write("# {}".format(info_str))
        fd.write("\n")

    fd.write("#=======================================================\n")

    i = 0
    if atoms.get_pbc().any():
        for n, vector in enumerate(atoms.get_cell()):
            fd.write("lattice_vector ")
            for i in range(3):
                fd.write("%16.16f " % vector[i])
            fd.write("\n")
    fix_cart = np.zeros([len(atoms), 3])

    # else aims crashes anyways
    # better be more explicit
    # write_magmoms = np.any([a.magmom for a in atoms])

    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_cart[constr.index] = [1, 1, 1]
            elif isinstance(constr, FixCartesian):
                fix_cart[constr.a] = -constr.mask + 1

    if ghosts is None:
        ghosts = np.zeros(len(atoms))
    else:
        assert len(ghosts) == len(atoms)

    if geo_constrain:
        wrap = False
    scaled_positions = atoms.get_scaled_positions(wrap=wrap)

    for i, atom in enumerate(atoms):
        if ghosts[i] == 1:
            atomstring = "empty "
        elif scaled:
            atomstring = "atom_frac "
        else:
            atomstring = "atom "
        fd.write(atomstring)
        if scaled:
            for pos in scaled_positions[i]:
                fd.write("%16.16f " % pos)
        else:
            for pos in atom.position:
                fd.write("%16.16f " % pos)
        fd.write(atom.symbol)
        fd.write("\n")
        # (1) all coords are constrained:
        if fix_cart[i].all():
            fd.write("    constrain_relaxation .true.\n")
        # (2) some coords are constrained:
        elif fix_cart[i].any():
            xyz = fix_cart[i]
            for n in range(3):
                if xyz[n]:
                    fd.write("    constrain_relaxation %s\n" % "xyz"[n])
        if atom.charge:
            fd.write("    initial_charge %16.6f\n" % atom.charge)
        if atom.magmom:
            fd.write("    initial_moment %16.6f\n" % atom.magmom)

        # Write velocities if this is wanted
        if velocities and atoms.get_velocities() is not None:
            fd.write(
                "    velocity {:.16f} {:.16f} {:.16f}\n".format(
                    *atoms.get_velocities()[i] / v_unit
                )
            )

    if geo_constrain:
        for line in get_sym_block(atoms):
            fd.write(line)


def get_sym_block(atoms):
    """Get symmetry block for Parametric constraints in atoms.constraints"""
    from ase.constraints import (
        FixScaledParametricRelations,
        FixCartesianParametricRelations,
    )

    # Initialize param/expressions lists
    atomic_sym_params = []
    lv_sym_params = []
    atomic_param_constr = np.zeros((len(atoms),), dtype="<U100")
    lv_param_constr = np.zeros((3,), dtype="<U100")

    # Populate param/expressions list
    for constr in atoms.constraints:
        if isinstance(constr, FixScaledParametricRelations):
            atomic_sym_params += constr.params

            if np.any(atomic_param_constr[constr.indices] != ""):
                warnings.warn(
                    "multiple parametric constraints defined for the same "
                    "atom, using the last one defined"
                )

            atomic_param_constr[constr.indices] = [
                ", ".join(expression) for expression in constr.expressions
            ]
        elif isinstance(constr, FixCartesianParametricRelations):
            lv_sym_params += constr.params

            if np.any(lv_param_constr[constr.indices] != ""):
                warnings.warn(
                    "multiple parametric constraints defined for the same "
                    "lattice vector, using the last one defined"
                )

            lv_param_constr[constr.indices] = [
                ", ".join(expression) for expression in constr.expressions
            ]

    if np.all(atomic_param_constr == "") and np.all(lv_param_constr == ""):
        return []

    # Check Constraint Parameters
    if len(atomic_sym_params) != len(np.unique(atomic_sym_params)):
        warnings.warn(
            "Some parameters were used across constraints, they will be "
            "combined in the aims calculations"
        )
        atomic_sym_params = np.unique(atomic_sym_params)

    if len(lv_sym_params) != len(np.unique(lv_sym_params)):
        warnings.warn(
            "Some parameters were used across constraints, they will be "
            "combined in the aims calculations"
        )
        lv_sym_params = np.unique(lv_sym_params)

    if np.any(atomic_param_constr == ""):
        raise IOError(
            "FHI-aims input files require all atoms have defined parametric "
            "constraints"
        )

    cell_inds = np.where(lv_param_constr == "")[0]
    for ind in cell_inds:
        lv_param_constr[ind] = "{:.16f}, {:.16f}, {:.16f}".format(*atoms.cell[ind])

    n_atomic_params = len(atomic_sym_params)
    n_lv_params = len(lv_sym_params)
    n_total_params = n_atomic_params + n_lv_params

    sym_block = []
    if n_total_params > 0:
        sym_block.append("#" + "=" * 55 + "\n")
        sym_block.append("# Parametric constraints\n")
        sym_block.append("#" + "=" * 55 + "\n")
        sym_block.append(
            "symmetry_n_params {:d} {:d} {:d}\n".format(
                n_total_params, n_lv_params, n_atomic_params
            )
        )
        sym_block.append(
            "symmetry_params %s\n" % " ".join(lv_sym_params + atomic_sym_params)
        )

        for constr in lv_param_constr:
            sym_block.append("symmetry_lv {:s}\n".format(constr))

        for constr in atomic_param_constr:
            sym_block.append("symmetry_frac {:s}\n".format(constr))
    return sym_block


@reader
def read_aims_output(fd, index=-1):
    """Import FHI-aims output files with all data available, i.e.
    relaxations, MD information, force information etc etc etc."""
    chunks = get_aims_out_chunks(fd)
    initial_chunk = next(chunks)
    chunks = [initial_chunk] + list(
        get_aims_out_chunks(fd, initial_chunk.n_atoms, initial_chunk.constraints)
    )
    images = [chunk.atoms for chunk in chunks]
    return images[index]


@reader
def read_aims_results(fd, index=-1):
    chunks = get_aims_out_chunks(fd)
    initial_chunk = next(chunks)
    chunks = [initial_chunk] + list(
        get_aims_out_chunks(fd, initial_chunk.n_atoms, initial_chunk.constraints)
    )
    return chunks[index].results
