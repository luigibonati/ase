from ase.optimize.activelearning.gp.kernel import SquaredExponential, BondExponential
import numpy as np

def test_implementation():
    D = 6
    params = {"weight": 1.0, "scale": 0.4}
    X = np.asarray([np.random.rand(6) for i in range(3)])

    bk = BondExponential(D)
    sq_k = SquaredExponential(D)

    bk.set_params(params)
    sq_k.set_params(params)

    bk.G = np.eye(D)
    bk.F = np.eye(D)

    assert np.allclose(bk.kernel_matrix(X), sq_k.kernel_matrix(X))
    assert np.allclose(bk.gradient(X),      sq_k.gradient(X))


def one_point_sample(Natoms=2):
    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)

    r = [1.2]*Natoms
    kernel.init_metric(r)

    X = np.asarray([np.random.rand(3*Natoms)])
    K = kernel.kernel_matrix(X)
    # print(K)
    o = np.zeros(3*Natoms).reshape(1, -1)
    true_K = np.block([[np.eye(1), o], [o.T, kernel.G/kernel.l**2]])
    # print(true_K)

    assert np.allclose(K, true_K)


def test_g_eigh(Natoms=2):
    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)
    r = [1.2]*Natoms
    kernel.init_metric(r)

    assert np.allclose(kernel.g.sum(axis=1), np.zeros(Natoms))
    assert np.allclose(kernel.G.sum(axis=1), np.zeros(3*Natoms))


def test_is_symmetric(Natoms=2, Nsamples=1, gradient=False):

    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)

    r = [1.2]*Natoms
    kernel.init_metric(r)

    X = np.asarray([np.random.rand(3*Natoms) for i in range(Nsamples)])
    K = kernel.kernel_matrix(X)
    # print(K)
    # print(kernel.G)
    # print(kernel.K(X,X))
    assert np.allclose(K, K.T)

    if gradient:
        g = kernel.gradient(X)
        assert np.allclose(g, g.T)


def first_step(Nattempts=1, l=0.1):

    from ase.data import covalent_radii, atomic_numbers
    from ase.calculators.emt import EMT

    from ase.build import bulk

    from ase.optimize.activelearning.gp.calculator import GPCalculator
    from ase.optimize.activelearning.aidmin import AIDMin, SP

    # Define atoms object
    rc = covalent_radii[atomic_numbers['C']]
    atoms = bulk('C', 'fcc', 2*np.sqrt(2)*rc)
    atoms *= (2, 2, 2)
    atoms.rattle(0.2)
    atoms.set_calculator(EMT())

    # Define kernel
    N = len(atoms)
    interaction = lambda x, y: 1. / (x * y)
    radii = N * [rc]

    kernel = BondExponential(3*N)
    params = {"weight": 1.0, "scale": l}
    kernel.set_params(params)
    kernel.init_metric(radii, interaction, normalize=True)

    # Define ml calculator
    calcparams = {'noise': 0.001,
                  'kernel': kernel,
                  'kernel_params': params,
                  'update_prior_strategy': 'maximum',
                  'train_images': [],
                  'calculate_uncertainty': False,
                  'mask_constraints': False}

    ml_calculator = GPCalculator(**calcparams)

    # Optimizer
    opt_params = {'model_calculator': ml_calculator,
                  'optimizer': SP,
                  'use_previous_observations': False,
                  'surrogate_starting_point': 'min',
                  'trainingset': [],
                  'print_format': 'ASE',
                  'fit_to': 'constraints',
                  'optimizer_kwargs': {'fmax': 'scipy default',
                                       'method': 'L-BFGS-B'}}

    atoms0 = atoms.copy()
    optimizer = AIDMin(atoms, logfile=None, **opt_params)
    optimizer.run(fmax=0.01, steps=1)

    x0 = atoms0.get_positions().flatten()
    x1 = atoms.get_positions().flatten()

    # print(np.sqrt(((x0-x1)**2).sum()))
    # print(rc*l)

    assert np.abs(np.sqrt(((x0-x1)**2).sum()) - rc*l) < 0.005

if __name__ == "__main__":

    test_implementation()

    one_point_sample()

    test_is_symmetric()

    first_step(l=0.1)
