from ase import Atoms
from ase.calculators.acemolecule import ACE

def dict_is_subset(d1, d2):
    """True if all the key-value pairs in dict 1 are in dict 2"""
    for key, value in d1.items():
        if key not in d2:
            return False
        elif d2[key] != value:
            return False
    else:
        return True



def test_acemolecule_calculator():
    import pytest
    from ase import Atoms
    from ase.units import Hartree, Bohr

    #ace_cmd = "mpirun -np ncores /PATH/TO/ace PREFIX.inp > PREFIX.log"
    ace_cmd = "mpirun -np 2 /home/khs/hs_file/programs/ACE-Molecule/ace PREFIX.inp > PREFIX.log"

    basis = dict(Scaling='0.5', Cell=7.0, Grid='Basic', Centered=0,Pseudopotential={'Pseudopotential':1,'Format':'upf','PSFilenames':'/PATH/TO/He.pbe.UPF'} )
    guess = dict(InitialGuess=1,InitialFilenames='/PATH/TO/He.pbe.UPF')
    scf = dict(IterateMaxCycle=50,ConvergenceType='Energy',ConvergenceTolerance=0.000001,EnergyDecomposition=1, 
            ExchangeCorrelation={'XFunctional':'LDA_X','CFunctional':'LDA_C_PW'}, 
            Diagonalize={'Tolerance':0.000000001}, Mixing={'MixingType':'Density','MixingParameter':0.3,'MixingMethod':1})
    he = Atoms("He", positions = [[0.0, 0.0, 0.0]])
    he.calc = ACE(command = ace_cmd, BasicInformation=basis,Guess=guess,Scf=scf)
    sample_parameters =he.calc.parameters
    assert dict_is_subset(basis, sample_parameters['BasicInformation'][0])
    assert dict_is_subset(guess, sample_parameters['Guess'][0])
    assert dict_is_subset(scf, sample_parameters['Scf'][0])
    he.calc.set(BasicInformation={"Pseudopotential": {"UsingDoubleGrid": 1}})
    sample_parameters =he.calc.parameters
    assert dict_is_subset({"UsingDoubleGrid": 1}, sample_parameters['BasicInformation'][0]["Pseudopotential"])


if __name__ == "__main__":
    test_acemolecule_calculator()
