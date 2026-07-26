"""O teste mais importante do repositório.

Gera rebaixamento pela solução **exata** de Theis (função de poço, W(u), via
``scipy.special.exp1``) com T e S conhecidos, restrito à janela onde
u < 0.01, e verifica que ``cooper_jacob_analysis`` recupera T e S dentro de
uma tolerância apertada. Isso protege toda a cadeia
``linregress -> T -> t0 -> S`` de qualquer regressão futura.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import exp1

from hydropump.domain import math as hm


def _theis_drawdown(t_sec: np.ndarray, T: float, S: float, Q: float, r: float) -> np.ndarray:
    """Solução exata de Theis: s = Q/(4*pi*T) * W(u), u = r^2 S / (4 T t)."""
    u = r**2 * S / (4.0 * T * t_sec)
    return Q / (4.0 * np.pi * T) * exp1(u)


def _log_t_window_with_valid_u(T: float, S: float, r: float, n: int = 40) -> np.ndarray:
    """Janela de tempo cujo início já satisfaz u < 0.01 (com folga de 0.2 dec)."""
    t_min_needed = r**2 * S / (4.0 * T * hm.U_VALIDITY_THRESHOLD)
    log_start = np.log10(t_min_needed) + 0.2
    return np.logspace(log_start, log_start + 3.0, n)


@pytest.mark.parametrize(
    "T_true,S_true,Q,r",
    [
        (1.5e-3, 2.0e-4, 0.01, 10.0),   # aquífero confinado típico
        (5.0e-4, 1.0e-4, 0.005, 5.0),   # T baixa, poço de observação próximo
        (8.0e-3, 1.5e-3, 0.03, 25.0),   # T alta, r grande
    ],
)
def test_cooper_jacob_recupera_T_e_S_de_theis_exato(T_true, S_true, Q, r):
    t = _log_t_window_with_valid_u(T_true, S_true, r)  # janela com u pequeno
    s = _theis_drawdown(t, T_true, S_true, Q, r)

    res = hm.cooper_jacob_analysis(t, s, Q, r)

    assert res.transmissivity_m2_s == pytest.approx(T_true, rel=0.02)
    assert res.storativity == pytest.approx(S_true, rel=0.10)
    assert res.approximation_is_valid
    assert res.r_squared > 0.999


def test_cooper_jacob_detecta_janela_com_u_invalido():
    """Incluindo tempos muito curtos (u grande), o u_max deve refletir isso."""
    T_true, S_true, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    t = np.logspace(-1, 5.5, 60)  # inclui tempos com u >> 0.01
    s = _theis_drawdown(t, T_true, S_true, Q, r)

    res = hm.cooper_jacob_analysis(t, s, Q, r)
    # com a série toda, o trecho inicial contamina o ajuste linear
    assert res.transmissivity_m2_s != pytest.approx(T_true, rel=0.02)


def test_janela_de_ajuste_restaura_a_estimativa():
    """Restringir a janela ao trecho log-linear recupera T e S corretos."""
    T_true, S_true, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    t = np.logspace(-1, 5.5, 60)
    s = _theis_drawdown(t, T_true, S_true, Q, r)

    res = hm.cooper_jacob_analysis(t, s, Q, r, t_start_sec=1000.0)
    assert res.transmissivity_m2_s == pytest.approx(T_true, rel=0.03)


def test_suggest_fit_window_encontra_o_trecho_valido():
    T_true, S_true, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    t = np.logspace(-1, 5.5, 60)
    s = _theis_drawdown(t, T_true, S_true, Q, r)

    suggestion = hm.suggest_fit_window(t, s, Q, r, min_points=5)
    assert suggestion is not None
    assert suggestion.u_max < hm.U_VALIDITY_THRESHOLD
    assert suggestion.r_squared > 0.99

    res = hm.cooper_jacob_analysis(
        t, s, Q, r,
        t_start_sec=suggestion.t_start_sec,
        t_end_sec=suggestion.t_end_sec,
    )
    assert res.transmissivity_m2_s == pytest.approx(T_true, rel=0.05)
