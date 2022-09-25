"""Test file for exciting ASE calculator."""
import os
import numpy as np
import pytest
import ase
# from ase.build import bulk
import ase.calculators.exciting.exciting as exciting
import xml.etree.ElementTree as ET
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Bohr, Hartree


LDA_VWN_Ar_INFO_OUT = """================================================================================
| EXCITING NITROGEN-14 started                                                 =
| version hash id: 1775bff4453c84689fb848894a9224f155377cfc                    =
|                                                                              =
|                                                                              =
| Date (DD-MM-YYYY) : 10-12-2020                                               =
| Time (hh:mm:ss)   : 20:02:27                                                 =
|                                                                              =
| All units are atomic (Hartree, Bohr, etc.)                                   =
================================================================================
 
********************************************************************************
* Ground-state run starting from atomic densities                              *
********************************************************************************
 
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
 
********************************************************************************
* Groundstate module started                                                   *
********************************************************************************
 Output level for this task is set to high
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Self-consistent loop started                                                 +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Density and potential initialised from atomic data
 
 
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
 
 Wall time (seconds)                        :         1.05
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    2                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.82194513
 _______________________________________________________________
 Fermi energy                               :        -0.20093816
 Kinetic energy                             :       530.56518416
 Coulomb energy                             :     -1029.02256672
 Exchange energy                            :       -27.93371266
 Correlation energy                         :        -1.43084992
 Sum of eigenvalues                         :      -305.07718688
 Effective potential energy                 :      -835.64237105
 Coulomb potential energy                   :      -796.81544990
 xc potential energy                        :       -38.82692115
 Hartree energy                             :       205.65547702
 Electron-nuclear energy                    :     -1208.12640394
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61484177
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00183981
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99816019
     total charge in muffin-tins            :        17.99816019
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36074458
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         1.46
 
 RMS change in effective potential (target) :  0.587126E-02  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.298766E-02  ( 0.100000E-06)
 Charge distance                   (target) :  0.233904E-04  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    3                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.82022198
 _______________________________________________________________
 Fermi energy                               :        -0.20079854
 Kinetic energy                             :       530.56785510
 Coulomb energy                             :     -1029.02353844
 Exchange energy                            :       -27.93369195
 Correlation energy                         :        -1.43084669
 Sum of eigenvalues                         :      -305.07613298
 Effective potential energy                 :      -835.64398808
 Coulomb potential energy                   :      -796.81709801
 xc potential energy                        :       -38.82689007
 Hartree energy                             :       205.65480064
 Electron-nuclear energy                    :     -1208.12669929
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61498944
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184026
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815974
     total charge in muffin-tins            :        17.99815974
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36078432
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         1.82
 
 RMS change in effective potential (target) :  0.433810E-02  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.172315E-02  ( 0.100000E-06)
 Charge distance                   (target) :  0.150521E-04  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    4                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.82282966
 _______________________________________________________________
 Fermi energy                               :        -0.20048984
 Kinetic energy                             :       530.57007281
 Coulomb energy                             :     -1029.02813987
 Exchange energy                            :       -27.93391070
 Correlation energy                         :        -1.43085190
 Sum of eigenvalues                         :      -305.07638893
 Effective potential energy                 :      -835.64646174
 Coulomb potential energy                   :      -796.81927455
 xc potential energy                        :       -38.82718719
 Hartree energy                             :       205.65722553
 Electron-nuclear energy                    :     -1208.13372561
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61850260
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00183901
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99816099
     total charge in muffin-tins            :        17.99816099
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36103015
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         2.15
 
 RMS change in effective potential (target) :  0.115090E-03  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.260768E-02  ( 0.100000E-06)
 Charge distance                   (target) :  0.157915E-04  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    5                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81254947
 _______________________________________________________________
 Fermi energy                               :        -0.20040531
 Kinetic energy                             :       530.57593246
 Coulomb energy                             :     -1029.02412325
 Exchange energy                            :       -27.93352379
 Correlation energy                         :        -1.43083489
 Sum of eigenvalues                         :      -305.07152179
 Effective potential energy                 :      -835.64745425
 Coulomb potential energy                   :      -796.82080096
 xc potential energy                        :       -38.82665329
 Hartree energy                             :       205.65168249
 Electron-nuclear energy                    :     -1208.12416595
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61372277
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184168
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815832
     total charge in muffin-tins            :        17.99815832
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36089040
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         2.86
 
 RMS change in effective potential (target) :  0.144178E-03  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.102802E-01  ( 0.100000E-06)
 Charge distance                   (target) :  0.556814E-04  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    6                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81742129
 _______________________________________________________________
 Fermi energy                               :        -0.20044051
 Kinetic energy                             :       530.57339029
 Coulomb energy                             :     -1029.02626113
 Exchange energy                            :       -27.93370794
 Correlation energy                         :        -1.43084251
 Sum of eigenvalues                         :      -305.07389872
 Effective potential energy                 :      -835.64728901
 Coulomb potential energy                   :      -796.82038212
 xc potential energy                        :       -38.82690689
 Hartree energy                             :       205.65423921
 Electron-nuclear energy                    :     -1208.12886054
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61607007
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184054
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815946
     total charge in muffin-tins            :        17.99815946
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36094948
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         3.25
 
 RMS change in effective potential (target) :  0.117020E-04  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.487182E-02  ( 0.100000E-06)
 Charge distance                   (target) :  0.245233E-04  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    7                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81797586
 _______________________________________________________________
 Fermi energy                               :        -0.20044613
 Kinetic energy                             :       530.57302094
 Coulomb energy                             :     -1029.02642463
 Exchange energy                            :       -27.93372865
 Correlation energy                         :        -1.43084353
 Sum of eigenvalues                         :      -305.07414492
 Effective potential energy                 :      -835.64716586
 Coulomb potential energy                   :      -796.82023028
 xc potential energy                        :       -38.82693558
 Hartree energy                             :       205.65455455
 Electron-nuclear energy                    :     -1208.12933939
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61630949
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36095863
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         3.58
 
 RMS change in effective potential (target) :  0.323152E-06  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.554576E-03  ( 0.100000E-06)
 Charge distance                   (target) :  0.346569E-05  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    8                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81796070
 _______________________________________________________________
 Fermi energy                               :        -0.20044598
 Kinetic energy                             :       530.57303118
 Coulomb energy                             :     -1029.02642030
 Exchange energy                            :       -27.93372808
 Correlation energy                         :        -1.43084350
 Sum of eigenvalues                         :      -305.07413828
 Effective potential energy                 :      -835.64716946
 Coulomb potential energy                   :      -796.82023466
 xc potential energy                        :       -38.82693479
 Hartree energy                             :       205.65454584
 Electron-nuclear energy                    :     -1208.12932634
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61630297
 Core-electron kinetic energy               :         0.00000000
 
 DOS at Fermi energy (states/Ha/cell)       :         0.00000000
 
 Electron charges :
     core                                   :        10.00000000
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36095838
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         3.90
 
 RMS change in effective potential (target) :  0.740517E-08  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.151658E-04  ( 0.100000E-06)
 Charge distance                   (target) :  0.967001E-07  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    9                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81796102
 _______________________________________________________________
 Fermi energy                               :        -0.20044598
 Kinetic energy                             :       530.57303095
 Coulomb energy                             :     -1029.02642037
 Exchange energy                            :       -27.93372809
 Correlation energy                         :        -1.43084350
 Sum of eigenvalues                         :      -305.07413841
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
 
 Wall time (seconds)                        :         4.23
 
 RMS change in effective potential (target) :  0.117161E-09  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.316730E-06  ( 0.100000E-06)
 Charge distance                   (target) :  0.220818E-08  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :   10                                                  +
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
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36095838
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         4.62
 
 RMS change in effective potential (target) :  0.250465E-10  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.419402E-08  ( 0.100000E-06)
 Charge distance                   (target) :  0.314056E-10  ( 0.100000E-04)
 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :   11                                                  +
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
     core leakage                           :        -0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000
 
 Estimated fundamental gap                  :         0.36095838
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000
 
 Wall time (seconds)                        :         4.95
 
 RMS change in effective potential (target) :  0.141030E-10  ( 0.100000E-05)
 Absolute change in total energy   (target) :  0.662567E-09  ( 0.100000E-06)
 Charge distance                   (target) :  0.430772E-11  ( 0.100000E-04)
                                                                                
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
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 STATE.OUT is written
 
********************************************************************************
* Groundstate module stopped                                                   *
********************************************************************************

================================================================================
| EXCITING NITROGEN-14 stopped                                                 =
================================================================================
"""


@pytest.fixture
def nitrogen_trioxide_atoms():
    """Pytest fixture that creates ASE Atoms cell for other tests."""
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 3, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)


def test_ExcitingProfile_init():
    """Test initializing an ExcitingProfile object."""
    exciting_root = 'testdir/nowhere/'
    species_path = 'testdir/species/'
    # A fake exciting root should cause a FileNotFoundError when the init
    # method tries to query what exciting version is present.
    with pytest.raises(FileNotFoundError):
        exciting.ExcitingProfile(
            exciting_root=exciting_root, species_path=species_path)


def test_ExcitingGroundStateTemplate_init():
    gs_template_obj = exciting.ExcitingGroundStateTemplate()
    assert gs_template_obj.name == 'exciting'
    assert len(gs_template_obj.implemented_properties) == 2
    assert 'energy' in gs_template_obj.implemented_properties


def test_write_input(tmp_path, nitrogen_trioxide_atoms):
    """Test the write input method of ExcitingGroundStateTemplate.
    
    Args:
        tmp_path: This tells pytest to create a temporary directory
             in which we will store the exciting input file.
        nitrogen_trioxide_atoms: pytest fixture to create ASE Atoms
            unit cell composed of NO3.
    """
    expected_path = os.path.join(tmp_path, 'input.xml')

    gs_template_obj = exciting.ExcitingGroundStateTemplate()
    assert nitrogen_trioxide_atoms.cell is not None
    gs_template_obj.write_input(
        directory=tmp_path, atoms=nitrogen_trioxide_atoms,
        input_parameters = {
            "title": None,
            "species_path": tmp_path,
            "ground_state_input": {
                "rgkmax": 8.0,
                "do": "fromscratch",
                "ngridk": [6, 6, 6],
                "xctype": "GGA_PBE_SOL",
                "vkloff": [0, 0, 0]},
        })
    # Let's assert the file we just wrote exists.
    os.path.exists(expected_path)
    # Let's assert it's what we expect.
    element_tree = ET.parse(expected_path)
    # Ensure the coordinates of the atoms in the unit cell is correct.
    # We could test the other parts of the input file related coming from
    # the ASE Atoms object like species data but this is tested already in
    # test/io/exciting/test_exciting.py.
    expected_coords = [
        [0, 0, 0], [1.0, 3.0, 0], [0, 0, 1.], [0.5, 0.5, 0.5]]
    coords_list = element_tree.findall('./structure/species/atom')

    for i in range(len(coords_list)):
        coords_vect_float = [
            float(x) for x in coords_list[i].get('coord').split()]
        assert len(coords_vect_float) == len(expected_coords[i])
        assert all([np.round(a - b, 14) == 0 for a, b in zip(
            coords_vect_float, expected_coords[i])])

    # Ensure that the exciting calculator properites (e.g. functional type have
    # been set).
    assert element_tree.findall('input') is not None
    assert element_tree.getroot().tag == 'input'
    assert element_tree.getroot()[2].attrib['xctype'] == 'GGA_PBE_SOL'
    assert element_tree.getroot()[2].attrib['rgkmax'] == '8.0'


def test_read_results(tmp_path):
    """Test the read result method of ExcitingGroundStateTemplate."""

    # ASE doesn't want us to store any other files for test, so instead let's write
    # a temporary XML file. We could write it with XML but easier to copy a good known
    # XML file as a string above and write it directly to a file.
    output_file_path = os.path.join(tmp_path, 'info.xml')
    with open(output_file_path, "w") as xml_file:
        xml_file.write(LDA_VWN_Ar_INFO_OUT)
    assert os.path.exists(output_file_path)  # Ensure file has been written.

    gs_template_obj = exciting.ExcitingGroundStateTemplate()
    results = gs_template_obj.read_results(tmp_path)
    final_scl_iteration = list(results["scl"].keys())[-1]
    assert np.round(float(results["scl"][
        final_scl_iteration]["Hartree energy"]) - 205.65454603, 6) == 0.

#TODO(dts): Write a test for when the species path is missing.

#TODO(dts): Write a test to get properties.