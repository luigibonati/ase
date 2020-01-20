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


def test_graph_metric(debug=False):
    Natoms = 12
    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)

    r = [1.2]*Natoms
    kernel.init_metric(r, eps = 1e-14)

    if debug:
        print(kernel.g)

    assert np.allclose(kernel.G, np.dot(kernel.F.T, kernel.F))
    
    if debug:
        X = np.asarray([np.random.rand(3*Natoms) for i in range(3)])
        print(kernel.kernel_matrix(X))
        print(kernel.gradient(X))

def one_point_sample(Natoms=2):
    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)

    r = [1.2]*Natoms
    kernel.init_metric(r, eps = 1e-14)

    X = np.asarray([np.random.rand(3*Natoms)])
    K = kernel.kernel_matrix(X)
    #print(K)
    o = np.zeros(3*Natoms).reshape(1,-1)
    true_K = np.block([[np.eye(1), o],[o.T, kernel.G/kernel.l**2]])
    #print(true_K)

    assert np.allclose(K,true_K)


def test_is_symmetric(Natoms=2, Nsamples=1, gradient = False):

    kernel = BondExponential(3*Natoms)
    params = {"weight": 1.0, "scale": 0.4}
    kernel.set_params(params)

    r = [1.2]*Natoms
    kernel.init_metric(r, eps = 1e-14)

    X = np.asarray([np.random.rand(3*Natoms) for i in range(Nsamples)])
    K = kernel.kernel_matrix(X)
    #print(K)
    #print(kernel.G)
    #print(kernel.K(X,X))
    assert np.allclose(K, K.T)

    if gradient:
        g = kernel.gradient(X)
        assert np.allclose(g, g.T)



if __name__ == "__main__":

    test_implementation()

    test_graph_metric()

    one_point_sample()

    test_is_symmetric()
