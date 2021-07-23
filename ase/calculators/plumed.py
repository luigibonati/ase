from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.parallel import broadcast
from ase.parallel import world
import numpy as np
from os.path import exists


class Plumed(Calculator):
    '''Plumed calculator for simulations of enhanced sampling methods
    plumed.org
    [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
    [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi, 
    Comput. Phys. Commun. 185, 604 (2014)'''

    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calc, input, timestep, atoms=None, kT=1., log='', prev_traj=None, prev_steps=None):
        '''Plumed calculator. Plumed settings from input according to:
        plumed.org

        Parameters
        ----------  
        calc: ASE-calculator
               It  calculates MD unbias forces
        
        input: List of strings
               It contains the setup of plumed actions.

        timestep: float
               Timestep of the simulated dynamics
        
        atoms: Atoms
               For this case, the calculator is defined strictly with the
               object atoms inside. This is necessary for initializing the
               Plumed object.

        kT: float. Default 1.
               Value of the thermic energy in eV units. I is important for
               some of the methods of plumed like Well-Tempered Metadynamics.
        
        log: string
               Log file of the plumed calculations
              
        prev_traj: .traj file
               the trajectory of the previous run. This parameter is necessary
               for restarting simulations. Since some enhanced sampling 
               methods are history-dependent, it is necessary that the files
               created by plumed in the previous simulation stay in the 
               same directory.

        prev_steps: int
               The number of timesteps simulated previously. This parameter is
               necessary for restarting simulations. In case prev_traj is
               defined and this does not, it will be set to the number of
               steps in the trajectory.
               '''
        from plumed import Plumed as pl

        if atoms is None:
            raise TypeError('plumed calculator has to be defined with the object atoms inside.')
        if prev_traj is not None:
            with Trajectory(prev_traj) as traj:
                if prev_steps is None:
                    self.istep = len(traj) - 1
                else:
                    self.istep = prev_steps
                atoms.set_positions(traj[-1].get_positions())
                atoms.set_momenta(traj[-1].get_momenta())
        else:
            self.istep = 0
        Calculator.__init__(self, atoms=atoms)

        self.input = input
        self.calc = calc
        self.name = '{}+Plumed'.format(self.calc.name)
        
        if world.rank == 0:
            natoms = len(atoms.get_positions())
            self.plumed = pl()
            self.plumed.cmd("setNatoms", natoms)
            self.plumed.cmd("setMDEngine", "ASE")
            self.plumed.cmd("setLogFile", log)
            self.plumed.cmd("setTimestep", float(timestep))
            self.plumed.cmd("setRestart", prev_traj is not None)
            self.plumed.cmd("setKbT", float(kT))
            self.plumed.cmd("init")
            for line in input:
                self.plumed.cmd("readInputLine", line)
        self.atoms = atoms

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(self.atoms.get_positions(), self.istep)
        self.istep += 1
        self.results['energy'], self. results['forces'] = energy, forces

    def compute_energy_and_forces(self, pos, istep):
        if world.rank == 0:
            ener_forc = self.compute_bias(pos, istep)
        else:
            ener_forc = None
        energy_bias, forces_bias = broadcast(ener_forc)
        energy = self.calc.get_potential_energy(self.atoms) + energy_bias
        forces = self.calc.get_forces(self.atoms) + forces_bias
        return energy, forces

    def compute_bias(self, pos, istep):
        self.plumed.cmd("setStep", istep)
        self.plumed.cmd("setPositions", pos)
        self.plumed.cmd("setMasses", self.atoms.get_masses())
        forces_bias = np.zeros((self.atoms.get_positions()).shape)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)
        return [energy_bias, forces_bias]
    
    def write_plumed_files(self, trajectory):
        """ This function computes what is required in
        plumed input for some trajectory.
        
        The outputs are saved in the typical files of
        plumed such as COLVAR, HILLS """
        for i in range(len(trajectory)):
            pos = trajectory[i].get_positions()
            self.compute_energy_and_forces(pos, i)
        return self.read_plumed_files()

    def read_plumed_files(self, directory=''):
        read_files = {}
        for line in self.input:
            if line.find('FILE') != -1:
                ini = line.find('FILE')
                end = line.find(' ', ini)
                if end == -1:
                    file_name = line[ini+5:]
                else:
                    file_name = line[ini+5:end]
                read_files[file_name] = np.loadtxt(file_name, unpack=True)
    
        if len(read_files) == 0:
            if exists('COLVAR'):
                read_files['COLVAR'] = np.loadtxt('COLVAR', unpack=True)
            if exists('HILLS'):
                read_files['HILLS'] = np.loadtxt('HILLS', unpack=True)
        
        assert not len(read_files) == 0, "There are not files for reading"
        return read_files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plumed.finalize()
