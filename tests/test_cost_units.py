"""Cost/unit sanity: smooth positive part, bill accounting, negative prices."""
import numpy as np

from src.costs import bill_increment, psi_eps, running_cost, terminal_cost


def test_psi_limits():
    z = np.linspace(-500, 500, 10001)
    eps = 0.5
    psi = psi_eps(z, eps)
    assert np.all(psi >= 0)
    assert np.all(psi >= z)                      # psi >= z and >= 0
    assert np.allclose(psi[z > 50], z[z > 50], rtol=1e-4)
    assert np.all(psi[z < -50] < 1e-2)
    # psi(0) = eps/2
    assert np.isclose(psi_eps(0.0, eps), eps / 2)


def test_bill_import_export(params):
    # import 100 kW for 1 h at 0.2 EUR/kWh -> 20 EUR
    assert np.isclose(bill_increment(0.2, 100.0, 1.0, params), 20.0)
    # export 100 kW for 1 h at 0.2 -> revenue alpha_s*0.2*100 = 10 EUR
    assert np.isclose(bill_increment(0.2, -100.0, 1.0, params), -10.0)


def test_negative_price_preserved(params):
    # negative price: importing PAYS the consumer (bill negative)
    assert bill_increment(-0.05, 100.0, 1.0, params) < 0


def test_running_cost_components(params, profile):
    # with a = 0 and y such that G > 0, cost approx C*G (+peak term)
    t = np.array([12.0])
    y = np.array([100.0])
    pz = np.array([0.0])
    s = np.array([0.5])
    ell = running_cost(t, s, y, pz, np.array([0.0]), params, profile)
    G = profile.n_bar(t) + y
    C = profile.c_bar(t)
    approx = C * G + params.lam_pk * G ** 2
    assert np.isclose(ell[0], approx[0], rtol=1e-3)


def test_terminal_cost(params):
    assert terminal_cost(params.s_tar, params) == 0.0
    assert np.isclose(terminal_cost(params.s_tar + 0.1, params),
                      params.lam_T * 0.01)


def test_degradation_at_zero_action(params, profile):
    # lam1 term vanishes exactly at a = 0
    t = np.array([6.0]); s = np.array([0.5])
    y = np.array([0.0]); pz = np.array([0.0])
    e0 = running_cost(t, s, y, pz, np.array([0.0]), params, profile)
    params2 = params
    lam1 = params2.lam1
    params2.lam1 = 0.0
    e1 = running_cost(t, s, y, pz, np.array([0.0]), params2, profile)
    params2.lam1 = lam1
    assert np.isclose(e0[0], e1[0], atol=1e-12)
