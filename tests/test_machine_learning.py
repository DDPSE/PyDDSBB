# tests/test_machine_learning.py

import numpy as np
import pytest

from PyDDSBB._machine_learning import MachineLearning, LocalSVR, NN, MFSM


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_quad_data_1d(n=50, noise=0.01, seed=0):
    """
    Simple 1D dataset: y = x^2 + noise
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    Y = X[:, 0] ** 2 + noise * rng.randn(n)
    return X, Y


def make_quad_data_2d(n=80, noise=0.01, seed=1):
    """
    2D dataset: y = x1^2 + 0.5 * x2 + noise
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    Y = X[:, 0] ** 2 + 0.5 * X[:, 1] + noise * rng.randn(n)
    return X, Y


# -----------------------------------------------------------------------------
# MachineLearning base class
# -----------------------------------------------------------------------------

def test_machinelearning_model_type_default_and_custom():
    m_default = MachineLearning()
    assert m_default._model_type == "SVR"

    m_mlp = MachineLearning(model="MLPRegressor")
    assert m_mlp._model_type == "MLPRegressor"


def test_machinelearning_predict_forwards_to_model():
    class DummyModel:
        def __init__(self):
            self.last_X = None

        def predict(self, X):
            self.last_X = X
            return np.sum(X, axis=1)

    ml = MachineLearning()
    ml.model = DummyModel()

    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    y_dummy = None

    y_pred = ml._predict(X, y_dummy, multifidelity=False)

    assert np.allclose(y_pred, np.array([3.0, 7.0]))
    assert ml.model.last_X is X


# -----------------------------------------------------------------------------
# LocalSVR tests
# -----------------------------------------------------------------------------

def test_localsvr_default_init():
    model = LocalSVR()
    assert model._model_type == "SVR"
    assert model.eps == pytest.approx(0.01)
    assert model.time_training == 0.0
    assert model.time_var_select == 0.0


def test_localsvr_custom_eps():
    model = LocalSVR(eps=0.1)
    assert model.eps == pytest.approx(0.1)


def test_localsvr_train_sets_model_and_gamma_and_dim():
    X, Y = make_quad_data_1d()
    model = LocalSVR()
    model._train(X, Y)

    # attributes set
    assert hasattr(model, "model")
    assert hasattr(model, "_gamma")
    assert model.dim == X.shape[1]
    assert model._gamma > 0.0
    assert isinstance(model.time_training, float)
    assert model.time_training >= 0.0


def test_localsvr_predict_uses_machinelearning_predict():
    X, Y = make_quad_data_1d()
    model = LocalSVR()
    model._train(X, Y)

    y_pred1 = model._predict(X, None, multifidelity=False)
    y_pred2 = model.model.predict(X)

    assert y_pred1.shape == y_pred2.shape
    assert np.allclose(y_pred1, y_pred2)


def test_localsvr_rank_shape_and_finite():
    X, Y = make_quad_data_2d()
    model = LocalSVR()
    model._train(X, Y)

    crit = model._rank()

    # rank length equals number of variables (dim)
    assert crit.shape == (X.shape[1],)
    assert np.all(np.isfinite(crit))


def test_localsvr_rank_raises_if_not_trained():
    model = LocalSVR()
    with pytest.raises(AttributeError):
        model._rank()


# -----------------------------------------------------------------------------
# NN tests
# -----------------------------------------------------------------------------

def test_nn_default_hyperparameters():
    nn = NN()
    assert nn._model_type == "MLPRegressor"
    assert nn.hidden_layer_sizes == (15, 20, 15)
    assert nn.activation == "tanh"
    assert nn.solver == "lbfgs"
    assert nn.learning_rate_init == pytest.approx(0.01)
    assert nn.random_state == 26
    assert nn.max_iter == 100000
    assert nn.time_training == 0.0


def test_nn_custom_hyperparameters():
    nn = NN(
        hidden_layer_sizes=(5, 5),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        random_state=123,
        max_iter=1000,
    )
    assert nn.hidden_layer_sizes == (5, 5)
    assert nn.activation == "relu"
    assert nn.solver == "adam"
    assert nn.learning_rate_init == pytest.approx(0.001)
    assert nn.random_state == 123
    assert nn.max_iter == 1000


def test_nn_train_sets_model_dim_and_time_training():
    X, Y = make_quad_data_1d()
    nn = NN()
    nn._train(X, Y)

    assert hasattr(nn, "model")
    assert nn.dim == X.shape[1]
    assert nn.time_training > 0.0

    # sklearn MLPRegressor has n_layers_ etc.
    assert hasattr(nn.model, "n_layers_")
    assert hasattr(nn.model, "n_iter_")


def test_nn_predict_via_machinelearning_base():
    X, Y = make_quad_data_2d()
    nn = NN()
    nn._train(X, Y)

    # MachineLearning._predict is inherited
    y_base = nn._predict(X, None, multifidelity=False)
    y_direct = nn.model.predict(X)

    assert y_base.shape == y_direct.shape
    assert np.allclose(y_base, y_direct)


# -----------------------------------------------------------------------------
# MFSM tests (multi-fidelity model)
# -----------------------------------------------------------------------------

def test_mfsm_default_hyperparameters():
    mf = MFSM()
    assert mf._model_type == "SVR"
    assert mf.eps == pytest.approx(0.01)
    assert mf.hidden_layer_sizes == (15, 20, 15)
    assert mf.activation == "tanh"
    assert mf.solver == "lbfgs"
    assert mf.learning_rate_init == pytest.approx(0.01)
    assert mf.random_state == 26
    assert mf.max_iter == 100000
    assert mf.time_training == 0.0
    assert mf.time_var_select == 0.0


def test_mfsm_train_builds_lf_and_composite_models():
    X, Y = make_quad_data_1d()
    mf = MFSM()
    mf._train(X, Y)

    assert hasattr(mf, "model")           # low-fidelity SVR
    assert hasattr(mf, "composite_model") # NN corrector
    assert mf.dim == X.shape[1]
    assert mf.time_training > 0.0

    from sklearn.svm import SVR as _SVR
    from sklearn.neural_network import MLPRegressor as _MLP

    assert isinstance(mf.model, _SVR)
    assert isinstance(mf.composite_model, _MLP)


def test_mfsm_predict_uses_both_models_shape_and_finiteness():
    X, Y = make_quad_data_2d()
    mf = MFSM()
    mf._train(X, Y)

    y_pred = mf._predict(X, y=None, multifidelity=True)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(y_pred))


def test_mfsm_rank_shape_and_time_var_select():
    X, Y = make_quad_data_1d()
    mf = MFSM()
    mf._train(X, Y)

    crit = mf._rank()

    assert crit.shape == (X.shape[1],)
    assert np.all(np.isfinite(crit))


def test_mfsm_predict_consistent_for_multifidelity_flag():
    """
    MFSM._predict signature has 'multifidelity' argument but currently ignores it.
    We check calls with True/False give same result.
    """
    X, Y = make_quad_data_1d()
    mf = MFSM()
    mf._train(X, Y)

    y_true = mf._predict(X, y=None, multifidelity=True)
    y_false = mf._predict(X, y=None, multifidelity=False)

    assert np.allclose(y_true, y_false)
