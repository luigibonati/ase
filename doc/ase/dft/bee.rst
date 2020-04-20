.. module:: ase.dft.bee
   :synopsis: Bayesian error estimation

=========================
Bayesian error estimation
=========================

The major approximation within (well converged) density functional theory
is the exchange-correlation (XC) functional. The choice of functional will
give rise to errors. Countless benchmark studies of functional accuracy have
been performed and can sometimes help you make a qualified guess of the
uncertainty of your own calculations. 

Bayesian ensembles provide an alternative and systematic approach to 
uncertainty quantification\ [#BEEF_ens]_. Using Baysian statistics an ensemble of
XC functionals are constructed. The ensemble functionals all belong to the same
functional model space (e.g. meta-GGA functionals). The BEEF family of
functionals are all constructed using an expansions of terms with fitted
expansion coefficients. This construction makes it very suitable for ensemble
generation by varying the expansion coefficients and allows highly efficient
non-selfconsistent calculation of ensemble energies. The uncertainty is
quantified by calculating the ensemble standard deviation.

Below is an example which calculates the BEEF-vdW binding energy of molecular
H\ :sub:`2` (E_bind), as well as an ensemble estimate of the binding energy error
(dE_bind). The example requires the GPAW_ calculator (or the VASP calcualtor).

.. _GPAW: https://wiki.fysik.dtu.dk/gpaw

>>> from ase import Atoms
>>> from gpaw import GPAW
>>> # from ase.calculators.vasp import Vasp
>>> from ase.dft.bee import BEEFEnsemble
>>> h2 = Atoms('H2', [[0., 0., 0.], [0., 0., 0.75]])
>>> h2.center(vacuum=3)
>>> cell = h2.get_cell()
>>> calc = GPAW(xc='BEEF-vdW')
>>> # calc = Vasp(xc='beef-vdw', lbeefens=True)
>>> h2.set_calculator(calc)
>>> e_h2 = h2.get_potential_energy()
>>> ens = BEEFEnsemble(calc)
>>> de_h2 = ens.get_ensemble_energies()
>>> del h2, calc, ens
>>> h = Atoms('H')
>>> h.set_cell(cell)
>>> h.center()
>>> calc = GPAW(xc='BEEF-vdW')
>>> # calc = Vasp(xc='beef-vdw', lbeefens=True)
>>> h.set_calculator(calc)
>>> e_h = h.get_potential_energy()
>>> ens = BEEFEnsemble(calc)
>>> de_h = ens.get_ensemble_energies()
>>> E_bind = 2 * e_h - e_h2
>>> dE_bind = 2 * de_h[:] - de_h2[:]
>>> dE_bind = dE_bind.std()
>>> print('Binding energy: %s +- %s' % (E_bind, dE_bind))


The default number of ensemble XC functionals is 2000, for which
well-converged error estimates should be ensured. Therefore, "de_h2" and
"de_h" in the example are both arrays of 2000 perturbations of a BEEF-vdW
total energy. The syntax "ens.get_ensemble_energies(N)" changes this number
to N. The calculator object input to the BEEFEnsemble class could of course
stem from a restarted calculation.

It is very important to calculate the ensemble statistics correctly.
Computing the standard deviation of each array of total energy perturbations
makes little sense. Only the standard deviation of the relative energy
perturbations should be used for the ensemble error estimates on a quantity.
The ensembles are constructed to exploit the error cancellation obtained in
relative energies as is standard in DFT. 

The BEEFEnsemble module can be used to obtain ensembles and calculate
ensemble statistics for calculations using the BEEF-vdW\ [#BEEF-vdW]_,
mBEEF\ [#mBEEF]_, or mBEEF-vdW\ [#mBEEF-vdW]_ functionals and the GPAW or VASP
calculators. 

More details can be found in the general 2017 ASE paper (see :ref:`cite`)


.. [#BEEF_ens] R. Christensen, T. Bligaard, K. W. Jacobsen,
    `Chap. 3 - Bayesian error estimation in density functional theory`__,
    In *Uncertainty Quantification in Multiscale Materials Modeling*, 77-91
    (2020)

    __ https://doi.org/10.1016/B978-0-08-102941-1.00003-1

.. [#BEEF-vdW] J. Wellendorff, K. T. Lundgaard, A. Mogelhoj,
   V. Petzold, D. D. Landis, J. K. Norskov, T. Bligard, and K. W. Jacobsen,
   `Density functionals for surface science: Exchange-correlation model
   development with Bayesian error estimation`__ Physical Review B, 85, 235149
   (2012)

   __ https://doi.org/10.1103/PhysRevB.85.235149

.. [#mBEEF] J. Wellendorff, K. T. Lundgaard, K. W. Jacobsen, T. Bligaard,
    `mBEEF: An accurate semi-local Bayesian error estimation density
    functional`__, The Journal of Chemical Physics 140, 14, 144107 (2014)

    __ https://doi.org/10.1063/1.4870397

.. [#mBEEF-vdW] K. T. Lundgaard, J. Wellendorff, J. Voss, K. W. Jacobsen,
   T. Bligaard, `mBEEF-vdW: Robust fitting of error estimation density
   functionals`__, Physical Review B, 93, 235162 (2016)

   __ https://doi.org/10.1103/PhysRevB.93.235162
