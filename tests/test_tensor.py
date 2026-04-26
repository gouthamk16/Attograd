import pytest
import numpy as np
from attograd import Tensor


# ── creation ──────────────────────────────────────────────────────────────────

def test_tensor_creation():
    assert Tensor([1, 2, 3]).data.shape == (3,)
    assert Tensor(np.array([[1, 2], [3, 4]])).data.shape == (2, 2)
    t = Tensor(5)
    assert t.data.shape == ()
    assert t.data.item() == 5


# ── arithmetic ────────────────────────────────────────────────────────────────

def test_addition():
    np.testing.assert_array_equal((Tensor([1, 2, 3]) + Tensor([4, 5, 6])).data, [5, 7, 9])

def test_rsub():
    x = Tensor(3.0)
    y = 5.0 - x
    y.backward()
    assert y.data == pytest.approx(2.0)
    assert x.grad == pytest.approx(-1.0)

def test_rtruediv():
    x = Tensor(4.0)
    y = 1.0 / x
    y.backward()
    assert y.data == pytest.approx(0.25)
    assert x.grad == pytest.approx(-1 / 16)

def test_neg():
    x = Tensor(3.0)
    y = -x
    y.backward()
    assert y.data == pytest.approx(-3.0)
    assert x.grad == pytest.approx(-1.0)


# ── gradients ─────────────────────────────────────────────────────────────────

def test_scalar_gradient():
    x = Tensor(2.0)
    y = x * x
    y.backward()
    assert x.grad == pytest.approx(4.0)

def test_chain_rule():
    x = Tensor(1.0)
    y = (x ** 2).tanh().sigmoid()
    y.backward()
    import math
    t = math.tanh(1.0)
    s = 1 / (1 + math.exp(-t))
    expected = s * (1 - s) * (1 - t ** 2) * 2.0
    assert x.grad == pytest.approx(expected, rel=1e-5)

def test_backward_non_scalar_raises():
    with pytest.raises(RuntimeError):
        Tensor([1.0, 2.0]).backward()


# ── activations ───────────────────────────────────────────────────────────────

def test_sigmoid_value_and_grad():
    x = Tensor(0.0)
    y = x.sigmoid()
    y.backward()
    assert y.data == pytest.approx(0.5)
    assert x.grad == pytest.approx(0.25)

def test_tanh_value_and_grad():
    x = Tensor(0.0)
    y = x.tanh()
    y.backward()
    assert y.data == pytest.approx(0.0)
    assert x.grad == pytest.approx(1.0)

def test_relu_positive():
    x = Tensor(2.0); x.relu().backward(); assert x.grad == pytest.approx(1.0)

def test_relu_negative():
    x = Tensor(-2.0); x.relu().backward(); assert x.grad == pytest.approx(0.0)

def test_scalar_only_ops_raise_on_array():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]).tanh()
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]).sigmoid()

def test_log_domain_check():
    with pytest.raises(ValueError):
        Tensor(-1.0).log()


# ── tensor ops with grad ──────────────────────────────────────────────────────

def test_sum_forward_and_grad():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    s = x.sum()
    s.backward()
    assert s.data == pytest.approx(6.0)
    np.testing.assert_allclose(x.grad, [1.0, 1.0, 1.0])

def test_flatten_restores_grad_shape():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    x.flatten().sum().backward()
    assert x.grad.shape == (2, 2)
    np.testing.assert_allclose(x.grad, np.ones((2, 2)))

def test_reshape_restores_grad_shape():
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    x.reshape((2, 2)).sum().backward()
    assert x.grad.shape == (4,)
    np.testing.assert_allclose(x.grad, np.ones(4))

def test_matmul_grad():
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    A.matmul(B).sum().backward()
    expected_dA = np.ones((2, 2)) @ np.array([[5, 7], [6, 8]])
    expected_dB = np.array([[1, 3], [2, 4]]) @ np.ones((2, 2))
    np.testing.assert_allclose(A.grad, expected_dA)
    np.testing.assert_allclose(B.grad, expected_dB)


# ── loss functions ────────────────────────────────────────────────────────────

def test_mse_loss():
    from attograd.loss_functions import mseLoss
    preds = [Tensor(2.0), Tensor(3.0)]
    tgts  = [Tensor(1.0), Tensor(1.0)]
    loss = mseLoss(preds, tgts)
    assert loss.data == pytest.approx(5.0)  # (2-1)^2 + (3-1)^2

def test_loss_length_mismatch_raises():
    from attograd.loss_functions import mseLoss
    with pytest.raises(ValueError):
        mseLoss([Tensor(1.0)], [Tensor(1.0), Tensor(2.0)])

def test_bce_range_raises():
    from attograd.loss_functions import bceLoss
    with pytest.raises(ValueError):
        bceLoss([Tensor(1.5)], [Tensor(1.0)])


# ── optimizers ────────────────────────────────────────────────────────────────

def test_sgd_step():
    from attograd.nn import SGD
    x = Tensor(1.0); x.grad = 2.0
    SGD([x], lr=0.1).step()
    assert x.data == pytest.approx(0.8)

def test_adam_step_decreases_loss():
    from attograd.nn import Adam
    x = Tensor(1.0)
    opt = Adam([x], lr=0.1)
    for _ in range(10):
        x.grad = float(x.data)
        opt.step()
    assert abs(float(x.data)) < 1.0

def test_rmsprop_step():
    from attograd.nn import RMSProp
    x = Tensor(1.0)
    opt = RMSProp([x], lr=0.1)
    x.grad = 1.0
    opt.step()
    assert float(x.data) < 1.0

def test_optimizer_empty_params_raises():
    from attograd.nn import SGD
    with pytest.raises(ValueError):
        SGD([], lr=0.01)


# ── layers ────────────────────────────────────────────────────────────────────

def test_linear_invalid_args_raise():
    from attograd.nn import Linear
    with pytest.raises(ValueError):
        Linear(0, 4)
    with pytest.raises(ValueError):
        Linear(4, -1)
    with pytest.raises(ValueError):
        Linear(4, 4, activation='gelu')

def test_flatten_layer():
    from attograd.nn import Flatten
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = Flatten()(x)
    assert out.data.shape == (4,)

def test_full_train_cycle():
    from attograd.nn import Linear, Sequential
    from attograd.loss_functions import mseLoss
    net = Sequential([Linear(2, 4, activation='tanh'), Linear(4, 1)])
    x = [Tensor(1.0), Tensor(-1.0)]
    y = [Tensor(0.0)]
    losses = []
    for _ in range(30):
        out = net(x)
        loss = mseLoss(out, y)
        net.zero_grad()
        loss.backward()
        net.update(lr=0.1)
        losses.append(float(loss.data))
    assert losses[-1] < losses[0]
