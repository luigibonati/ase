def test_lammpslib_change_cell_bcs():
    # test that a change in unit cell boundary conditions is
    # handled correctly by lammpslib
    import os
    import numpy as np
    from ase.calculators.lammpslib import LAMMPSlib
    from ase.lattice.cubic import FaceCenteredCubic

    # If the potential file NiAlH_jea.eam.alloy isn't present in the current
    # directory, create a symlink for it.  This is done using the installation
    # path for the potential files in the ase-dockers lammps image,
    # /usr/local/share/lammps/potentials
    POTENTIAL_FILE = "NiAlH_jea.eam.alloy"
    if not os.path.isfile(POTENTIAL_FILE):
        LAMMPS_POTENTIALS_DIR = os.path.join(os.sep, "usr", "local", "share",
                                             "lammps", "potentials")
        os.symlink(os.path.join(LAMMPS_POTENTIALS_DIR, POTENTIAL_FILE),
                   os.path.join(os.getcwd(), POTENTIAL_FILE))

    cmds = ["pair_style eam/alloy",
            "pair_coeff * * NiAlH_jea.eam.alloy Ni H"]
    lammps = LAMMPSlib(lmpcmds=cmds,
                       atom_types={'Ni': 1, 'H': 2},
                       log_file='test.log', keep_alive=True)
    atoms = FaceCenteredCubic(size=(2, 2, 2), latticeconstant=3.52, symbol="Ni",
                              pbc=True)
    atoms.calc = lammps

    energy_ppp_ref = -142.400000403
    energy_ppp = atoms.get_potential_energy()
    print("Computed energy with boundary ppp = {}".format(energy_ppp))
    np.testing.assert_allclose(energy_ppp, energy_ppp_ref, atol=1e-4, rtol=1e-4)

    atoms.set_pbc((False, False, True))
    energy_ssp_ref = -114.524625705
    energy_ssp = atoms.get_potential_energy()
    print("Computed energy with boundary ssp = {}".format(energy_ssp))
    np.testing.assert_allclose(energy_ssp, energy_ssp_ref, atol=1e-4, rtol=1e-4)
