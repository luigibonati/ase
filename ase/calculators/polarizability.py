from abc import ABC


class StaticPolarizabilityCalculator(ABC):
    def __call__(self, atoms):
        return self.calculate(atoms)

    def calculate(self, atoms):
        pass
