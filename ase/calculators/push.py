import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii, atomic_numbers


class Push(Calculator):
    """
    Simple potential to push apart atoms to minimal distance.
    Based on TS ASE tools by Henkelman et al.
    Here we use the atomic radii of each element to determine the bond length.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, attractive_force=0.0, repulsive_force=5.0,
                 bond_factor=2., **kwargs):
        Calculator.__init__(self, **kwargs)
        self.attractive_force = attractive_force
        self.repulsive_force = repulsive_force
        self.bond_factor = bond_factor
        self.atoms = None

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        rf = self.repulsive_force
        af = self.attractive_force
        r2 = rf/2
        r22 = (rf**2)/2
        r24 = (rf**2)/4
        a2 = af/2
        a22 = (af**2)/2
        a24 = (af**2)/4
        f = np.zeros((len(self.atoms), 3))
        u = 0
        for i in range(len(self.atoms)):
            for j in range(len(self.atoms)):
                if i == j:
                    continue
                # Get the vector and distance between the two atoms.
                radii_i = covalent_radii[atomic_numbers[atoms[i].symbol]]
                radii_j = covalent_radii[atomic_numbers[atoms[j].symbol]]
                print('atom i', atoms[i].symbol)
                print('atom j', atoms[j].symbol)
                print('radii i', radii_i)
                print('radii j', radii_j)
                z = (radii_i + radii_j) / self.bond_factor
                print('z', z)
                v = self.atoms.positions[i] - self.atoms.positions[j]
                vr = np.linalg.solve(self.atoms.get_cell().T, v)
                v = np.dot(vr - np.round(vr) * self.atoms.get_pbc(),
                           self.atoms.get_cell())
                d = np.linalg.norm(v)
                if d == 0:
                    raise ValueError("Push potential cannot handle zero distance between atoms.")
                # The repulsive regime:
                if d < -r2 + z:
                    u += -rf*(d-z) - r22 + r24
                    f[i] += rf * (v/d)
                # The attractive regime:
                elif d > a2 + z:
                    u += af*(d-z) - a22 + a24
                    f[i] += -af * (v/d)
                # The harmonic regime:
                else:
                    u += (d-z)**2
                    f[i] += (-2*(d-z)) * (v/d)

        self.results['energy'] = u / 2.
        self.results['free_energy'] = u / 2.
        self.results['forces'] = f
