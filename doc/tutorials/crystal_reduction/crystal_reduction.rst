.. _reductiontutorial:

=================
Crystal reduction
=================

The translational symmetry of a structure can be analyzed using the
:func:`ase.geometry.rmsd.find_crystal_reductions` function. This is
useful for finding candidates for a primitive unit cell in a crystal structure
whose atoms do not lie on the exact sites prescribed by the symmetry
conditions. For each crystal reduction, a symmetrized structure is calculated.
The root-mean-square distance (RMSD) from the input structure to the
symmetrized structure shows the cost of a reduction.

.. autofunction:: ase.geometry.rmsd.find_crystal_reductions

The example below creates a single-layer graphene structure with perturbed
atomic positions, and finds all reductions of the crystal.

.. literalinclude:: reduction_example.py

In the above example A (2 x 2) graphene monolayer is created with perturbed
positions:

.. image:: rattled.png
   :scale: 50%

.. |img1| image:: clustered_2.png
   :scale: 50%
   :align: middle
.. |img2| image:: translated_2.png
   :scale: 50%
   :align: top
.. |img3| image:: reduced_2.png
   :scale: 50%
   :align: top

The first reduction has an RMSD cost of 0.124 and has a reduction factor of 2.

+-----------------------------+---------------------+-----------------------------+
| Find atoms which belong     | Apply translational | Calculate average positions |
|                             |                     |                             |
| together in reduced crystal | symmetry            | and use primitive cell      |
+-----------------------------+---------------------+-----------------------------+
| |img1|                      | | |img2|            | |img3|                      |
+-----------------------------+---------------------+-----------------------------+

.. |img4| image:: clustered_4.png
   :scale: 50%
   :align: middle
.. |img5| image:: translated_4.png
   :scale: 50%
   :align: top
.. |img6| image:: reduced_4.png
   :scale: 50%
   :align: top

The second reduction has an RMSD cost of 0.146 and has a reduction factor of 4.

+-----------------------------+---------------------+-----------------------------+
| Find atoms which belong     | Apply translational | Calculate average positions |
|                             |                     |                             |
| together in reduced crystal | symmetry            | and use primitive cell      |
+-----------------------------+---------------------+-----------------------------+
| |img4|                      | | |img5|            | |img6|                      |
+-----------------------------+---------------------+-----------------------------+

This reduction produces the primitive cell. A more aggresive reduction with
RMSD cost 0.3584 and reduction factor of 8 is also possible. This is an overly
aggressive reduction though, and produces a nonsense structure.
