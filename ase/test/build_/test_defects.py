def test_spg():
    from ase.build import bulk
    from ase.build.defects import get_spg_cell
    from ase.build.defects import DefectBuilder
    import spglib as spg

    Fe = bulk('Fe')
    spg_cell = get_spg_cell(Fe)
    dataset = spg.get_symmetry_dataset(spg_cell,
                                       symprec=1e-2)

    assert dataset['number'] == 229


def test_construction_cell():
    from ase.build import mx2
    from ase.build.defects import DefectBuilder

    MoS2 = mx2(formula='MoS2', kind='2H')
    builder = DefectBuilder(MoS2)
    N = len(MoS2)
    MoS2_con = builder._set_construction_cell(MoS2)

    assert len(MoS2_con) == 9 * N


def test_create_vacancies():
    from ase.build import mx2, bulk
    from ase.build.defects import DefectBuilder
    import numpy as np

    MoS2 = mx2(formula='MoS2', kind='2H', vacuum=10)
    Fe = bulk('Fe')
    structures = [MoS2, Fe]

    for structure in structures:
        builder = DefectBuilder(structure)
        vacancies = builder.create_vacancies()
        if np.sum(structure.get_pbc()) == 2:
            N = 9
        elif np.sum(structure.get_pbc()) == 3:
            N = 27
        for vacancy in vacancies:
            assert len(vacancy) == len(structure) * N - 1


def test_create_substitutional():
    from ase.build import mx2, bulk
    from ase.build.defects import DefectBuilder
    import numpy as np

    MoS2 = mx2(formula='MoS2', kind='2H', vacuum=10)
    Fe = bulk('Fe')
    structures = [MoS2, Fe]
    numbers = [8, 3]
    element_list = ['Fe', 'Mo', 'S', 'C', 'Nb', 'He']

    for i, structure in enumerate(structures):
        builder = DefectBuilder(structure)
        substitutions = builder.create_substitutions(
            intrinsic=True,
            extrinsic=['C', 'Nb', 'He'])
        if np.sum(structure.get_pbc()) == 2:
            N = 9
        elif np.sum(structure.get_pbc()) == 3:
            N = 27

        assert len(substitutions) == numbers[i]

        for sub in substitutions:
            assert len(sub) == len(structure) * N
            symbols = sub.get_chemical_symbols()
            for sym in symbols:
                assert sym in element_list

