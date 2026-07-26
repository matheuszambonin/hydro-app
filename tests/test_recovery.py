"""Testes de ``theis_recovery_analysis`` e do serviço de recuperação."""

from __future__ import annotations

import numpy as np
import pytest

from hydropump.domain import math as hm
from hydropump.domain.models import AnalysisConfig, RecoveryConfig, WellConfiguration
from hydropump.services.pumping_test import run_recovery_analysis


def _synthetic_recovery(T: float, Q: float, t_p_sec: float, n: int = 25):
    """s'' = 2.303 Q / (4 pi T) * log10(t/t'), t = t_p + t'."""
    t_rec = np.logspace(1, 4, n)  # t' de 10 s a 10^4 s
    t_total = t_p_sec + t_rec
    ratio = t_total / t_rec
    s2 = (hm.LN10 * Q) / (4.0 * np.pi * T) * np.log10(ratio)
    return t_rec, s2


def test_theis_recovery_recupera_T():
    T_true, Q, t_p = 1.2e-3, 0.012, 7200.0
    t_rec, s2 = _synthetic_recovery(T_true, Q, t_p)

    res = hm.theis_recovery_analysis(t_p, t_rec, s2, Q)

    assert res.transmissivity_m2_s == pytest.approx(T_true, rel=0.01)
    assert res.r_squared > 0.999
    assert res.ratio_at_zero == pytest.approx(1.0, abs=0.05)


def test_theis_recovery_rejeita_inclinacao_negativa():
    # s'' crescente com t/t' é fisicamente invertido
    t_rec = np.array([10.0, 100.0, 1000.0])
    s2 = np.array([0.1, 0.5, 1.0])
    with pytest.raises(hm.RegressionError):
        hm.theis_recovery_analysis(3600.0, t_rec, s2, Q=0.01)


def test_run_recovery_analysis_aplica_correcao_de_jacob_quando_configurado():
    T_true, Q, t_p = 1.2e-3, 0.012, 7200.0
    t_rec, s2 = _synthetic_recovery(T_true, Q, t_p)

    cfg_sem_correcao = AnalysisConfig(
        static_level_m=10.0, radius_m=0.076, Q_m3_s=Q,
        configuration=WellConfiguration.SINGLE_WELL, apply_jacob=False,
    )
    cfg_com_correcao = AnalysisConfig(
        static_level_m=10.0, radius_m=0.076, Q_m3_s=Q,
        configuration=WellConfiguration.SINGLE_WELL,
        saturated_thickness_m=50.0, apply_jacob=True,
    )
    rec_cfg = RecoveryConfig(pumping_duration_min=t_p / 60.0)

    res_sem = run_recovery_analysis(t_p, t_rec, s2, cfg_sem_correcao, rec_cfg)
    res_com = run_recovery_analysis(t_p, t_rec, s2, cfg_com_correcao, rec_cfg)

    # com b=50 (grande frente a s''), a correção deve ter efeito pequeno mas
    # mensurável — os dois T não são idênticos bit-a-bit.
    assert res_sem.transmissivity_m2_s != res_com.transmissivity_m2_s
    assert res_sem.transmissivity_m2_s == pytest.approx(T_true, rel=0.02)
