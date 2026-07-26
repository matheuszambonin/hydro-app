"""Testes unitários das funções puras de ``hydropump.domain.math``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hydropump.domain import math as hm


# --------------------------------------------------------------------------
# Vazão
# --------------------------------------------------------------------------
def test_calculate_flow_rate_escalar():
    res = hm.calculate_flow_rate(200.0, 18.0)
    assert res.q_m3_h == pytest.approx(40.0, rel=1e-9)
    assert res.q_l_s == pytest.approx(200.0 / 18.0, rel=1e-9)


def test_calculate_flow_rate_rejeita_tempo_nao_positivo():
    with pytest.raises(hm.NonPositiveTimeError):
        hm.calculate_flow_rate(200.0, 0.0)


def test_calculate_flow_rate_rejeita_volume_nao_positivo():
    with pytest.raises(hm.InvalidGeometryError):
        hm.calculate_flow_rate(-1.0, 10.0)


# --------------------------------------------------------------------------
# Correção de Jacob — direta e inversa
# --------------------------------------------------------------------------
def test_jacob_correction_sem_b_retorna_s_inalterado():
    out = hm.jacob_unconfined_correction([1.0, 2.0, 3.0], None)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_jacob_correction_apply_false_retorna_s_inalterado():
    out = hm.jacob_unconfined_correction([1.0, 2.0], 20.0, apply_correction=False)
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_jacob_correction_rejeita_dessaturacao_total():
    with pytest.raises(hm.InvalidGeometryError):
        hm.jacob_unconfined_correction([25.0], 20.0)


@given(
    s=st.floats(min_value=0.0, max_value=19.9, allow_nan=False),
    b=st.floats(min_value=20.0, max_value=200.0, allow_nan=False),
)
def test_jacob_correction_e_inversa_sao_mutuamente_inversas(s, b):
    """Round-trip: inverse(direct(s)) == s, para s < b."""
    s_corr = hm.jacob_unconfined_correction(s, b, strict=False)
    s_back = hm.inverse_jacob_correction(np.asarray([s_corr]), b)
    assert s_back[0] == pytest.approx(s, abs=1e-6)


@given(s=st.floats(min_value=0.0, max_value=19.0, allow_nan=False))
def test_jacob_correction_e_monotonica_em_s(s):
    """s' deve crescer (fracamente) com s, para s < b — física exige isso."""
    b = 20.0
    delta = 0.05
    s_lo = hm.jacob_unconfined_correction(s, b, strict=False)
    s_hi = hm.jacob_unconfined_correction(min(s + delta, b - 1e-6), b, strict=False)
    assert float(s_hi) >= float(s_lo) - 1e-9


def test_inverse_jacob_correction_rejeita_dessaturacao():
    with pytest.raises(hm.InvalidGeometryError):
        hm.inverse_jacob_correction([15.0], 20.0)  # s' > b/2, discriminante < 0


# --------------------------------------------------------------------------
# Tempo de armazenamento no poço (Schafer, 1978)
# --------------------------------------------------------------------------
def test_wellbore_storage_time_diminui_com_T():
    t_low_T = hm.wellbore_storage_time(2.0, 0.10)
    t_high_T = hm.wellbore_storage_time(200.0, 0.10)
    assert t_high_T < t_low_T


def test_wellbore_storage_time_rejeita_tubo_maior_que_revestimento():
    with pytest.raises(hm.InvalidGeometryError):
        hm.wellbore_storage_time(50.0, 0.10, pump_pipe_radius_m=0.15)


# --------------------------------------------------------------------------
# Plausibilidade de S
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "s,unconfined,expected",
    [
        (0.15, True, True),
        (0.005, True, False),
        (0.5, True, False),
        (2.0e-4, False, True),
        (0.5, False, False),
    ],
)
def test_storativity_is_plausible(s, unconfined, expected):
    assert hm.storativity_is_plausible(s, unconfined=unconfined) is expected


# --------------------------------------------------------------------------
# Regressão: erros esperados de cooper_jacob_analysis
# --------------------------------------------------------------------------
def test_cooper_jacob_rejeita_menos_de_3_pontos():
    with pytest.raises(hm.RegressionError):
        hm.cooper_jacob_analysis([10.0, 20.0], [0.1, 0.2], Q=0.01, r=1.0)


def test_cooper_jacob_rejeita_inclinacao_negativa():
    # rebaixamento decrescente com o tempo -> fisicamente inconsistente
    t = np.array([10.0, 100.0, 1000.0, 10000.0])
    s = np.array([2.0, 1.5, 1.0, 0.5])
    with pytest.raises(hm.RegressionError):
        hm.cooper_jacob_analysis(t, s, Q=0.01, r=1.0)
