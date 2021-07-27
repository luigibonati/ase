from abc import ABC, abstractmethod


class StaticPolarizabilityCalculator(ABC):
    @abstractmethod
    def calculate(self, atoms):
        """Calculate the polarizability tensor

        atoms: Atoms object
        
        Returns:
          Polarizabilty tensor (3x3 matrix) in units (e^2 Angstrom^2 / eV)
          Can be multiplied by Bohr * Ha to get (Angstrom^3)
        """
        pass
