========================
Setting up point defects
========================

.. currentmodule:: ase.build

The DefectBuilder
=================

The DefectBuilder :class: incorporates tools to set up single
point defects in supercells. Currently, it includes the creation of
vacancy, substitutional defects, interstitial defects and adsorption
sites.

Example - Vacancies, subst. defects, interstitials
--------------------------------------------------

First set up a bulk Ag structure::

  from ase.build import bulk
  atoms = bulk('Ag')

Using the Ag bulk structure, we can now initialize the DefectBuilder::

  from ase.build.defects import DefectBuilder
  builder = DefectBuilder(atoms)

and afterwards, we can easily create the vacancies via::

  vacancies = builder.get_vacancy_structures()

Since there is only one unique vacancy in bulk Ag, it will
return a list with only one Atoms object of the vacancy structure in
a supercell. In general, for more complex structures it will return
a list of all vacancy defects. By default, the defect will be set
up in a 3x3x3 supercell (3x3x1 for 2D structures), but the user can
specify the number of repititions, i.e. define N in NxNxN with passing
``sc=N`` to the ``get_vacancy_structures()`` method. Furthermore,
one can also define a physical size of the supercell in Angstrom with
the ``size`` argument and the method will automatically set up a
suitable integer repitition of the input structure.

Similarly, one can generate substitutional defects via::

  substitutions = builder.get_substitution_structures(extrinsic=['C', 'O'])

By default, it will set up intrinsic subst. defects (set ``intrinsic=False``
if this is not desired), and the user can pass a list of extrinsic elements
which will be used to create the defect structures. For this particular example,
``substitutions`` will contain two defects (C:sub:`Ag`, and O:sub:`Ag`).
Defining the supercell works in the same way as for the vacancy case.

Lastly, interstitial defects with elements of ``kindlist`` can be created
in the following way::

  interstitials = builder.get_interstitial_structures(kindlist=['C', 'O'])

which returns a list of interstitial defects. Note, that the number of
interstitials can be controlled by the ``min_dist`` parameter (minimum
distance between interstitial sites) that is 1 Angstrom
by default and can be defined when initializing the :class: `DefectBuilder`.
In addition, by setting ``Nsites`` in ``get_interstitial_structures()`` one can
define the exact number of interstitials that will be created. The respective
sites will be chosen based on the largest distance to atoms of the pristine
structure. Note, that this interstitial site creation can create a lot of
structures in one go. It might be useful to look at the general interstitial
sites before actually creating the structures. This is done by::

  unique, equivalent = builder.create_interstitials()

returning two mock structures incorporating all possible unique, and equivalent
(by symmetry) interstitial structures in the primitive unit cell. For the case of
Ag, the unique and equivalent interstitial positions can be viewed with the ASE
GUI and look like this:

 * unique interstitials: |unique|
 * equivalent interstitials: |equivalent|

.. |unique|    image:: unique.png
.. |equivalent|   image:: equivalent.png
