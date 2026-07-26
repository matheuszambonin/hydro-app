"""Testes de ``hydropump.services.pumping_test``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.special import exp1

from hydropump.domain.models import AnalysisConfig, Severity, WellConfiguration
from hydropump.io.mapping import TimeSeries
from hydropump.services.pumping_test import run_analysis


def _theis_series(T: float, S: float, Q: float, r: float, n: int = 30) -> TimeSeries:
    t_min_needed = r**2 * S / (4.0 * T * 0.005)  # folga extra sob 0.01
    t_sec = np.logspace(np.log10(t_min_needed), np.log10(t_min_needed) + 3.0, n)
    u = r**2 * S / (4.0 * T * t_sec)
    s = Q / (4.0 * np.pi * T) * exp1(u)
    return TimeSeries(time_min=t_sec / 60.0, drawdown_m=s, n_discarded=0)


def test_poco_observacao_gera_diagnostico_de_qualidade_sem_criticos():
    # S = 2e-4 é típico de aquífero CONFINADO (não de produção específica
    # livre); por isso a configuração usa unconfined=False, senão o próprio
    # diagnóstico de plausibilidade (correto) acusaria S fora de faixa.
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _theis_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL,
        unconfined=False,
    )
    res = run_analysis(series, cfg)

    codes = {d.code for d in res.diagnostics}
    assert "SINGLE_WELL_STORATIVITY" not in codes
    assert not res.has_critical
    assert res.cooper_jacob.transmissivity_m2_s == pytest.approx(T, rel=0.03)


def test_poco_unico_sempre_emite_alerta_de_storativity():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 0.076
    series = _theis_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.SINGLE_WELL,
    )
    res = run_analysis(series, cfg)

    codes = {d.code: d for d in res.diagnostics}
    assert "SINGLE_WELL_STORATIVITY" in codes
    assert codes["SINGLE_WELL_STORATIVITY"].severity is Severity.WARNING


def test_s_implausivel_gera_diagnostico_critico():
    # r muito pequeno + S "normal" de Theis geram, via 2.25*T*t0/r^2, um S
    # numericamente fora de qualquer faixa física plausível.
    T, S_gerador, Q, r = 1.5e-3, 2.0e-4, 0.01, 0.01
    series = _theis_series(T, S_gerador, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.SINGLE_WELL,
    )
    res = run_analysis(series, cfg)

    assert res.has_critical
    critical_codes = {d.code for d in res.diagnostics_by(Severity.CRITICAL)}
    assert "S_IMPLAUSIBLE" in critical_codes


def test_correcao_jacob_desprezivel_quando_s_pequeno_frente_a_b():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _theis_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL,
        saturated_thickness_m=500.0,  # b enorme frente ao s medido -> ratio < 10%
        apply_jacob=True,
    )
    res = run_analysis(series, cfg)
    codes = {d.code for d in res.diagnostics}
    assert "JACOB_NEGLIGIBLE" in codes


def test_correcao_jacob_critica_quando_s_proximo_de_b():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _theis_series(T, S, Q, r)
    s_max = float(np.max(series.drawdown_m))
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL,
        saturated_thickness_m=s_max / 0.30,  # ratio ~ 30% > 25% limite
        apply_jacob=True,
    )
    res = run_analysis(series, cfg)
    critical_codes = {d.code for d in res.diagnostics_by(Severity.CRITICAL)}
    assert "JACOB_RATIO" in critical_codes


def test_drawdown_12h_interpolado_quando_ensaio_ja_cobre_12h():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    t_sec = np.linspace(600.0, 50000.0, 50)  # cobre > 720 min
    u = r**2 * S / (4.0 * T * t_sec)
    s = Q / (4.0 * np.pi * T) * exp1(u)
    series = TimeSeries(time_min=t_sec / 60.0, drawdown_m=s, n_discarded=0)

    cfg = AnalysisConfig(static_level_m=10.0, radius_m=r, Q_m3_s=Q)
    res = run_analysis(series, cfg)
    assert res.drawdown_12h_m is not None
    assert res.last_time_min >= 720.0


def _short_series(T: float, S: float, Q: float, r: float, last_time_min: float = 200.0):
    t_sec = np.logspace(2, np.log10(last_time_min * 60.0), 20)
    u = r**2 * S / (4.0 * T * t_sec)
    s = Q / (4.0 * np.pi * T) * exp1(u)
    return TimeSeries(time_min=t_sec / 60.0, drawdown_m=s, n_discarded=0)


def test_sem_confirmacao_nao_gera_pontos_de_preenchimento():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _short_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL, unconfined=False,
        fill_to_12h_confirmed=False, show_projection=False,
    )
    res = run_analysis(series, cfg)

    codes = {d.code for d in res.diagnostics}
    assert "BELOW_12H_NO_FILL" in codes
    assert "SYNTHETIC_FILL_AS_MEASURED" not in codes
    assert "PROJECTION_POINTS_AS_MEASURED" not in codes
    assert (res.df_unified["tipo_dado"] == "Medido").sum() == len(series)
    assert res.last_time_min < 720.0


def test_com_confirmacao_preenchimento_e_tratado_como_medido():
    """O ponto central do recurso: pontos interpolados não aparecem como 'Sintético'."""
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _short_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL, unconfined=False,
        fill_to_12h_confirmed=True,
    )
    res = run_analysis(series, cfg)

    codes = {d.code for d in res.diagnostics}
    assert "SYNTHETIC_FILL_AS_MEASURED" in codes
    assert "BELOW_12H_NO_FILL" not in codes

    # Política do produto: NENHUM ponto gerado pelo software é rotulado
    # como "Sintético" em nenhum lugar — tudo aparece como "Medido", sem
    # distinção visual em gráfico, tabela ou relatório.
    assert (res.df_unified["tipo_dado"] == "Medido").all()
    n_medido = int((res.df_unified["tipo_dado"] == "Medido").sum())
    assert n_medido > len(series)  # pontos de preenchimento somados aos originais
    assert res.last_time_min == pytest.approx(720.0)
    assert res.df_unified["tempo_min"].max() == pytest.approx(720.0)


def test_preenchimento_e_projecao_nao_se_sobrepoem():
    """Preenchimento (até 12h) e extensão da projeção (além disso) não duplicam pontos.

    Ambos os blocos são rotulados "Medido" (política do produto: sem
    distinção visual entre dado coletado e ponto gerado pelo software) —
    a ausência de sobreposição é verificada pelo TEMPO, não pelo rótulo.
    """
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _short_series(T, S, Q, r)
    cfg = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL, unconfined=False,
        fill_to_12h_confirmed=True, show_projection=True, projection_hours=24.0,
    )
    res = run_analysis(series, cfg)

    assert (res.df_unified["tipo_dado"] == "Medido").all()

    codes = {d.code for d in res.diagnostics}
    assert "SYNTHETIC_FILL_AS_MEASURED" in codes
    assert "PROJECTION_POINTS_AS_MEASURED" in codes

    tempos_originais = series.time_min
    tempos_alem_12h = res.df_unified.loc[
        res.df_unified["tempo_min"] > 720.0 + 1e-6, "tempo_min"
    ]
    assert not tempos_alem_12h.empty
    assert tempos_alem_12h.min() > float(tempos_originais.max())
    assert res.df_projection is not None
    assert res.df_projection["tempo_min"].min() == pytest.approx(720.0, abs=1.0)


def test_preenchimento_confirmado_da_mesmo_s12h_que_extrapolacao_pura():
    T, S, Q, r = 1.5e-3, 2.0e-4, 0.01, 10.0
    series = _short_series(T, S, Q, r)
    cfg_sem = AnalysisConfig(
        static_level_m=10.0, radius_m=r, Q_m3_s=Q,
        configuration=WellConfiguration.OBSERVATION_WELL, unconfined=False,
        fill_to_12h_confirmed=False,
    )
    cfg_com = dataclasses.replace(cfg_sem, fill_to_12h_confirmed=True)

    res_sem = run_analysis(series, cfg_sem)
    res_com = run_analysis(series, cfg_com)
    assert res_sem.drawdown_12h_m == pytest.approx(res_com.drawdown_12h_m, abs=0.01)
