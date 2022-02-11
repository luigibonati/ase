def test_input_structure():
    from ase.build import bulk, mx2
    from ase.build.defects import DefectBuilder
    structures = [mx2(formula='MoS2'),
                  mx2(formula='WSSe'),
                  bulk('Fe'),
                  bulk('Ag')]

    primitive = [3, 3, 1, 1]
    construction_list = [27, 27, 27, 27]
    for i, structure in enumerate(structures):
        builder = DefectBuilder(structure)
        prim = builder.get_primitive_structure()
        construction = builder.get_input_structure()
        assert len(prim) == primitive[i]
        assert len(construction) == construction_list[i]


def test_spg():
    from ase.build import bulk
    from ase.build.defects import get_spg_cell
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


def test_get_vacancy_structures():
    from ase.build import mx2, bulk
    from ase.build.defects import DefectBuilder
    import numpy as np

    MoS2 = mx2(formula='MoS2', kind='2H', vacuum=10)
    Fe = bulk('Fe')
    structures = [MoS2, Fe]

    for structure in structures:
        builder = DefectBuilder(structure)
        vacancies = builder.get_vacancy_structures()
        if np.sum(structure.get_pbc()) == 2:
            N = 9
        elif np.sum(structure.get_pbc()) == 3:
            N = 27
        for vacancy in vacancies:
            assert len(vacancy) == len(structure) * N - 1


def test_get_substitution_structures():
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
        substitutions = builder.get_substitution_structures(
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


def test_create_interstitials():
    from ase.build import mx2
    from ase.build.defects import DefectBuilder

    MoS2 = mx2(formula='MoS2', kind='2H', vacuum=10)
    builder = DefectBuilder(MoS2)

    unique, equivalent = builder.create_interstitials()

    assert len(MoS2) + 7 == len(unique)


def test_create_adsorption():
    from ase.build import mx2
    from ase.build.defects import DefectBuilder

    formulas = ['MoS2', 'MoSSe']
    lengths = [6, 12]
    structures = []
    for formula in formulas:
        structure = mx2(formula=formula, kind='2H',
                        vacuum=10)
        structures.append(structure)

    for i, structure in enumerate(structures):
        builder = DefectBuilder(structure)
        adsorbates = builder.get_adsorbate_structures(
            kindlist=['H', 'He'], Nsites=2)
        assert len(adsorbates) == lengths[i]

        for ad in adsorbates:
            symbols = ad.get_chemical_symbols()
            for sym in symbols:
                assert sym in ['Mo', 'S', 'Se', 'H', 'He']


def test_get_layer():
    from ase.build import mx2
    from ase.build.defects import (DefectBuilder,
                                   has_same_kind)
    structures = [mx2(formula='MoS2'),
                  mx2(formula='WSSe')]
    lengths = [(3, 1), (3, 1)]
    same_kind = [True, False]

    for i, structure in enumerate(structures):
        builder = DefectBuilder(structure)
        top_bot = []
        for loc in ['top', 'bottom']:
            layer = builder.get_layer(kind=loc)
            top_bot.append(layer)
            assert len(structure) == lengths[i][0]
            assert len(layer) == lengths[i][1]
        assert has_same_kind(top_bot[0],
                             top_bot[1]) == same_kind[i]


def test_true_interstitial():
    from ase.build import mx2
    import numpy as np
    from ase.build.defects import DefectBuilder

    vacs = np.arange(5, 20, 1)
    position = [0, 0, 7]
    for vac in vacs:
        structure = mx2(formula='WS2', vacuum=vac)
        boundaries = ((vac) / structure.get_cell()[2, 2],
                      (vac + 2 * 1.595) / structure.get_cell()[2, 2])
        rel_pos = position / structure.get_cell()[2, 2]
        if (rel_pos[2] > boundaries[0]
           and rel_pos[2] < boundaries[1]):
            ref = True
        else:
            ref = False
        builder = DefectBuilder(structure)
        assert builder.is_true_interstitial(rel_pos) == ref


def test_intrinsic_types():
    from ase.build import mx2, bulk
    from ase.build.defects import DefectBuilder

    structures = [mx2(formula='MoS2'),
                  mx2(formula='WSSe'),
                  bulk('Fe'),
                  bulk('Ag')]

    ref_dict = {'MoS2': ['Mo', 'S'],
                'WSSe': ['W', 'S', 'Se'],
                'Fe': ['Fe'],
                'Ag': ['Ag']}

    for structure in structures:
        formula = structure.get_chemical_formula(mode='metal')
        ref = ref_dict[f'{formula}']
        builder = DefectBuilder(structure)
        symbols = builder.get_intrinsic_types()
        for symbol in symbols:
            assert symbol in ref
