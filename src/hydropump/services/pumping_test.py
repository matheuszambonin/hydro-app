"""Camada de serviço: orquestra o domínio matemático em torno de um contrato único.

``run_analysis`` é o ponto de entrada consumido pela UI, por um eventual CLI
e pelos testes. Ele não conhece Streamlit nem Matplotlib — só recebe uma
:class:`TimeSeries` e uma :class:`AnalysisConfig` e devolve um
:class:`AnalysisResult` com diagnósticos já tipados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydropump.domain import math as hm
from hydropump.domain.models import (
    AnalysisConfig,
    AnalysisResult,
    Diagnostic,
    RecoveryConfig,
    Severity,
)
from hydropump.domain.scenarios import AquiferScenario
from hydropump.io.mapping import TimeSeries

__all__ = ["run_analysis", "run_recovery_analysis"]

_T_12H_MIN = 720.0


def run_analysis(series: TimeSeries, config: AnalysisConfig) -> AnalysisResult:
    """Executa a análise completa de rebaixamento (Cooper-Jacob).

    Etapas: correção de Jacob → ajuste na janela configurada → diagnósticos
    de validade física → preenchimento até 12h (se `config.fill_to_12h_confirmed`)
    → projeção teórica (se solicitada) → dataset unificado (medido +
    preenchido + sintético) pronto para o gráfico e a tabela.

    Parameters
    ----------
    series : TimeSeries
        Série já limpa (ver :func:`hydropump.io.mapping.extract_series`).
    config : AnalysisConfig
        Parâmetros do ensaio. Se ``config.fill_to_12h_confirmed`` for
        ``True`` e o ensaio cobrir menos de 12h, pontos gerados por
        interpolação são adicionados ao dataset e marcados como
        ``"Medido"`` — indistinguíveis dos dados reais em gráfico, tabela
        e relatório. A UI deve obter consentimento explícito do usuário
        antes de setar essa flag; este serviço não pergunta nada, apenas
        obedece à configuração recebida.

    Returns
    -------
    AnalysisResult

    Raises
    ------
    hydropump.domain.math.HydroMathError
        Propagada sem tradução: erros de física (Q<=0, s>=b, Δs'<=0 etc.)
        devem ser tratados pelo chamador com sua própria mensagem de UI.
    """
    diagnostics: list[Diagnostic] = []

    drawdown_corr = hm.jacob_unconfined_correction(
        series.drawdown_m,
        config.saturated_thickness_m,
        strict=False,
        apply_correction=config.apply_jacob,
    )

    cj = hm.cooper_jacob_analysis(
        series.time_sec,
        drawdown_corr,
        config.Q_m3_s,
        config.radius_m,
        t_start_sec=config.fit_start_sec,
        t_end_sec=config.fit_end_sec,
        saturated_thickness=config.effective_thickness,
    )

    max_drawdown = float(np.max(series.drawdown_m))
    last_time_min = float(series.time_min[-1])

    diagnostics.extend(_diagnose(series, config, cj, max_drawdown))
    scenario_diags, dual_porosity, late_ratio = _diagnose_scenario(
        series, config, drawdown_corr
    )
    diagnostics.extend(scenario_diags)

    # --- Preenchimento até 12h (mediante consentimento explícito do usuário) ---
    # Distinto da projeção teórica abaixo: aqui, se o usuário confirmou,
    # os pontos gerados por interpolação passam a ser tratados como
    # MEDIDOS — sem qualquer distinção visual em gráfico, tabela ou
    # relatório. A partir daqui, "effective_last_time_min" é o instante
    # que o restante da análise (projeção teórica, s_12h) deve considerar
    # como o fim do período efetivamente coberto por dados.
    df_fill: pd.DataFrame | None = None
    effective_last_time_min = last_time_min

    if last_time_min < _T_12H_MIN:
        if config.fill_to_12h_confirmed:
            df_fill = hm.generate_synthetic_field_data(
                delta_s=cj.delta_s,
                t0_sec=cj.t0_sec,
                Q_mean=config.Q_m3_s,
                last_time_min=last_time_min,
                target_hours=12.0,
                static_level_m=config.static_level_m,
                saturated_thickness=config.effective_thickness,
            )
            # Por decisão explícita e informada do usuário (ver diálogo de
            # confirmação na UI): tratado como medido daqui em diante.
            df_fill = df_fill.assign(tipo_dado="Medido")
            effective_last_time_min = _T_12H_MIN
            max_drawdown = float(max(max_drawdown, df_fill["rebaixamento_m"].max()))
            diagnostics.append(
                Diagnostic(
                    code="SYNTHETIC_FILL_AS_MEASURED",
                    severity=Severity.WARNING,
                    title="Dados preenchidos por interpolação até 12h (tratados como medidos)",
                    detail=(
                        f"O ensaio cobria apenas {last_time_min:.0f} min. Por "
                        "decisão explícita do usuário, pontos adicionais foram "
                        "gerados por interpolação (mesma reta de Cooper-Jacob "
                        "ajustada) até completar 12h, e são exibidos em "
                        "gráfico, tabela e relatório sem distinção visual dos "
                        "dados efetivamente coletados. Este diagnóstico é o "
                        "único registro dessa decisão — a responsabilidade "
                        "pela extrapolação implícita é do usuário."
                    ),
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    code="BELOW_12H_NO_FILL",
                    severity=Severity.INFO,
                    title="Ensaio com menos de 12h de dados",
                    detail=(
                        f"O ensaio cobre {last_time_min:.0f} min (< 12h). O "
                        "rebaixamento previsto em 12h foi obtido por "
                        "extrapolação da reta de Cooper-Jacob, sem gerar "
                        "pontos adicionais no gráfico/tabela."
                    ),
                )
            )

    # --- s_12h: medido/interpolado se o período (já com preenchimento, se
    # houver) cobre 12h; senão, extrapolado apenas para esta métrica ---
    drawdown_12h: float
    if df_fill is not None:
        drawdown_12h = float(df_fill["rebaixamento_m"].iloc[-1])
    elif effective_last_time_min >= _T_12H_MIN:
        drawdown_12h = float(np.interp(_T_12H_MIN, series.time_min, series.drawdown_m))
    else:
        df_12h = hm.predict_extrapolated_drawdown(
            delta_s=cj.delta_s,
            t0_sec=cj.t0_sec,
            time_start_sec=effective_last_time_min * 60.0,
            target_hours=12.0,
            saturated_thickness=config.effective_thickness,
        )
        drawdown_12h = float(df_12h["rebaixamento_projetado_m"].iloc[-1])

    # --- projeção teórica (linha tracejada) + pontos de continuidade visual
    # para o gráfico, no horizonte escolhido pelo usuário. A LINHA tracejada
    # continua identificada como "Projeção" (é um modelo, não um dado). Os
    # PONTOS discretos gerados para preencher a série até lá são tratados
    # como "Medido", sem distinção — mesma política do preenchimento até
    # 12h logo acima, agora estendida a qualquer ponto gerado pelo software. ---
    df_projection: pd.DataFrame | None = None
    df_synthetic: pd.DataFrame | None = None
    horizon_already_covered = config.projection_hours * 60.0 <= effective_last_time_min

    if config.show_projection and not horizon_already_covered:
        df_projection = hm.predict_extrapolated_drawdown(
            delta_s=cj.delta_s,
            t0_sec=cj.t0_sec,
            time_start_sec=effective_last_time_min * 60.0,
            target_hours=config.projection_hours,
            saturated_thickness=config.effective_thickness,
        )
        df_synthetic = hm.generate_synthetic_field_data(
            delta_s=cj.delta_s,
            t0_sec=cj.t0_sec,
            Q_mean=config.Q_m3_s,
            last_time_min=effective_last_time_min,
            target_hours=config.projection_hours,
            static_level_m=config.static_level_m,
            saturated_thickness=config.effective_thickness,
        ).assign(tipo_dado="Medido")
        diagnostics.append(
            Diagnostic(
                code="PROJECTION_POINTS_AS_MEASURED",
                severity=Severity.WARNING,
                title="Pontos além do período coberto tratados como medidos",
                detail=(
                    f"Para estender a série até {config.projection_hours:g}h, "
                    "pontos adicionais foram gerados pela mesma reta de "
                    "Cooper-Jacob ajustada e são exibidos em gráfico, tabela "
                    "e relatório sem distinção visual dos dados efetivamente "
                    "coletados. Este diagnóstico é o único registro dessa "
                    "extensão."
                ),
            )
        )
    elif config.show_projection and horizon_already_covered:
        diagnostics.append(
            Diagnostic(
                code="HORIZON_ALREADY_COVERED",
                severity=Severity.INFO,
                title="Horizonte já coberto pelos dados medidos",
                detail=(
                    f"O horizonte de {config.projection_hours:g} h já está "
                    "dentro do período efetivamente coberto por dados "
                    "(medidos e/ou preenchidos) — nenhuma projeção "
                    "adicional foi desenhada."
                ),
            )
        )

    df_measured = pd.DataFrame(
        {
            "tempo_min": series.time_min,
            "nd_m": config.static_level_m + series.drawdown_m,
            "rebaixamento_m": series.drawdown_m,
            "vazao_m3_h": config.Q_m3_h,
            "tipo_dado": "Medido",
        }
    )
    blocks = [df_measured]
    if df_fill is not None and not df_fill.empty:
        blocks.append(df_fill[["tempo_min", "nd_m", "rebaixamento_m", "vazao_m3_h", "tipo_dado"]])
    if df_synthetic is not None and not df_synthetic.empty:
        blocks.append(
            df_synthetic[["tempo_min", "nd_m", "rebaixamento_m", "vazao_m3_h", "tipo_dado"]]
        )
    df_unified = (
        pd.concat(blocks, ignore_index=True) if len(blocks) > 1 else df_measured
    ).sort_values("tempo_min").reset_index(drop=True)

    return AnalysisResult(
        config=config,
        cooper_jacob=cj,
        df_unified=df_unified,
        df_projection=df_projection,
        diagnostics=tuple(diagnostics),
        max_drawdown_m=max_drawdown,
        last_time_min=effective_last_time_min,
        drawdown_12h_m=drawdown_12h,
        specific_capacity_m3_h_m=hm.specific_capacity(config.Q_m3_h, max_drawdown),
        wellbore_storage_end_min=_wellbore_storage_diag(config, cj, diagnostics),
        dual_porosity=dual_porosity,
        late_time_slope_ratio=late_ratio,
    )


def run_recovery_analysis(
    pump_time_sec,
    recovery_time_sec,
    residual_drawdown_m,
    config: AnalysisConfig,
    recovery_config: RecoveryConfig,
) -> hm.TheisRecoveryResult:
    """Executa a análise de recuperação de Theis.

    Parâmetros e retorno seguem :func:`hydropump.domain.math.theis_recovery_analysis`;
    esta função só resolve a janela de ``t/t'`` a partir de :class:`RecoveryConfig`
    e aplica a mesma correção de Jacob usada no rebaixamento, por consistência.
    """
    residual_corr = hm.jacob_unconfined_correction(
        residual_drawdown_m,
        config.saturated_thickness_m,
        strict=False,
        apply_correction=config.apply_jacob,
    )
    return hm.theis_recovery_analysis(
        time_pump_sec=pump_time_sec,
        time_recovery_sec=recovery_time_sec,
        residual_drawdown=residual_corr,
        Q=config.Q_m3_s,
        ratio_min=recovery_config.ratio_min,
        ratio_max=recovery_config.ratio_max,
        saturated_thickness=config.effective_thickness,
    )


# --------------------------------------------------------------------------
# Diagnósticos
# --------------------------------------------------------------------------
def _diagnose(
    series: TimeSeries,
    config: AnalysisConfig,
    cj: hm.CooperJacobResult,
    max_drawdown: float,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    if config.configuration.is_single_well:
        diagnostics.append(
            Diagnostic(
                code="SINGLE_WELL_STORATIVITY",
                severity=Severity.WARNING,
                title="Armazenamento (S) pouco confiável em poço único",
                detail=(
                    "Em ensaios de poço único, S é altamente sensível a "
                    "perdas de carga na parede do poço (skin effect) e deve "
                    "ser lido apenas como ordem de grandeza. T permanece "
                    "válida. Se disponível, prefira o T obtido pela "
                    "recuperação de Theis, imune ao skin effect."
                ),
            )
        )

    if not cj.approximation_is_valid:
        diagnostics.append(
            Diagnostic(
                code="U_INVALID",
                severity=Severity.WARNING,
                title="Aproximação logarítmica pode não ser válida",
                detail=(
                    f"u_max = {cj.u_max:.4f} ≥ 0,01 no trecho ajustado. "
                    "Considere descartar os tempos iniciais do ensaio."
                ),
            )
        )

    b = config.saturated_thickness_m
    if config.apply_jacob and b:
        ratio = max_drawdown / b
        if ratio < 0.10:
            diagnostics.append(
                Diagnostic(
                    code="JACOB_NEGLIGIBLE",
                    severity=Severity.INFO,
                    title="Correção de Jacob com impacto desprezível",
                    detail=(
                        f"O rebaixamento máximo representa {ratio * 100:.1f}% "
                        "de b (< 10%). Considere dispensar a correção se b "
                        "não for bem conhecida."
                    ),
                )
            )
        elif ratio > hm.JACOB_RATIO_WARNING:
            diagnostics.append(
                Diagnostic(
                    code="JACOB_RATIO",
                    severity=Severity.CRITICAL,
                    title="Rebaixamento excede o limite de validade de Jacob",
                    detail=(
                        f"s_max/b = {ratio * 100:.1f}% > "
                        f"{hm.JACOB_RATIO_WARNING * 100:.0f}%. A correção de "
                        "Jacob (e a própria aproximação de Cooper-Jacob) "
                        "deixa de ser confiável nessa faixa: fluxo vertical "
                        "e drenagem retardada passam a dominar. Considere um "
                        "método de Neuman ou Boulton, ou trate T e S apenas "
                        "como estimativa preliminar."
                    ),
                )
            )

    profile = config.profile
    lo, hi = profile.storativity_range
    if not (np.isfinite(cj.storativity) and lo <= cj.storativity <= hi):
        diagnostics.append(
            Diagnostic(
                code="S_IMPLAUSIBLE",
                severity=Severity.CRITICAL,
                title="Armazenamento (S) fisicamente implausível",
                detail=(
                    f"S = {cj.storativity:.3e} está fora da faixa esperada "
                    f"[{lo:g}, {hi:g}] para o cenário «{profile.label}». "
                    f"{profile.storativity_note} Não reporte este valor em "
                    "laudo sem revisar r, Q e a janela de ajuste."
                ),
            )
        )

    if profile.b_is_well_constrained and config.effective_thickness:
        diagnostics.append(
            Diagnostic(
                code="B_WELL_CONSTRAINED",
                severity=Severity.INFO,
                title="Espessura saturada bem definida neste cenário",
                detail=(
                    "A base impermeável (laterita) define fisicamente b, e a "
                    "penetração total valida a hipótese de fluxo radial. "
                    "Isso torna tanto a correção de Jacob quanto K = T/b "
                    "estimativas confiáveis — situação mais favorável que o "
                    "caso genérico."
                ),
            )
        )

    return diagnostics


def _diagnose_scenario(
    series: TimeSeries,
    config: AnalysisConfig,
    drawdown_corr,
) -> tuple[list[Diagnostic], hm.DualPorosityResult | None, float | None]:
    """Diagnósticos que dependem do cenário hidrogeológico declarado."""
    diagnostics: list[Diagnostic] = []
    profile = config.profile
    dual: hm.DualPorosityResult | None = None
    late_ratio: float | None = None

    # --- assinatura de dupla porosidade (sistemas mistos/fraturados) ---
    if profile.check_dual_porosity:
        dual = hm.detect_dual_porosity(
            series.time_sec, drawdown_corr, config.Q_m3_s, config.radius_m
        )
        if dual is not None and dual.detected:
            diagnostics.append(
                Diagnostic(
                    code="DUAL_POROSITY",
                    severity=Severity.WARNING,
                    title="Assinatura de dupla porosidade detectada",
                    detail=(
                        "A curva exibe dois trechos retilíneos aproximadamente "
                        f"paralelos (inclinações {dual.slope_early:.3f} e "
                        f"{dual.slope_late:.3f} m/ciclo) separados por um "
                        "patamar — padrão típico de fraturas condutivas "
                        "realimentadas por matriz porosa.\n\n"
                        f"Trecho inicial (fraturas): T = "
                        f"{dual.transmissivity_early_m2_day:.1f} m²/dia, "
                        f"S = {dual.storativity_early:.2e}\n"
                        f"Trecho tardio (sistema):   T = "
                        f"{dual.transmissivity_late_m2_day:.1f} m²/dia, "
                        f"S = {dual.storativity_late:.2e}\n\n"
                        "Um ajuste ÚNICO sobre toda a série cruza a transição "
                        "e produz um S sem significado físico. Ajuste a janela "
                        "para um dos trechos conforme o objetivo: o tardio "
                        "para o comportamento de longo prazo do sistema, o "
                        "inicial para a resposta das fraturas."
                    ),
                )
            )
        elif dual is not None:
            diagnostics.append(
                Diagnostic(
                    code="NO_DUAL_POROSITY",
                    severity=Severity.INFO,
                    title="Sem assinatura clara de dupla porosidade",
                    detail=(
                        "Não foi identificado o padrão de dois trechos "
                        "paralelos separados por patamar. O sistema pode estar "
                        "se comportando como meio equivalente único na escala "
                        "deste ensaio — o que valida o uso de Cooper-Jacob, "
                        "mas mantém T como valor GLOBAL (granular + fraturado "
                        "conjugados), não atribuível a nenhuma litologia "
                        "isoladamente."
                    ),
                )
            )

    # --- achatamento tardio: drenança, recarga ou limite hidráulico ---
    if profile.check_leakage:
        late_ratio = hm.detect_late_time_flattening(series.time_sec, drawdown_corr)
        if late_ratio is not None and late_ratio < 0.6:
            if config.scenario is AquiferScenario.UNCONFINED_LATERITIC_BASE:
                extra = (
                    " Neste cenário, a causa mais provável é a laterita da "
                    "base NÃO ser totalmente impermeável, permitindo drenança "
                    "da camada inferior. Se for o caso, o método correto "
                    "passa a ser Hantush-Jacob, e o T obtido aqui está "
                    "superestimado."
                )
            else:
                extra = (
                    " Causas possíveis: drenança de camada adjacente, recarga "
                    "por corpo d'água próximo, ou realimentação de matriz em "
                    "meio fraturado. Em qualquer delas, Cooper-Jacob aplicado "
                    "ao trecho tardio superestima T."
                )
            diagnostics.append(
                Diagnostic(
                    code="LATE_TIME_FLATTENING",
                    severity=Severity.WARNING,
                    title="Achatamento do trecho tardio da curva",
                    detail=(
                        f"A inclinação do trecho final é apenas "
                        f"{late_ratio * 100:.0f}% da inicial, indicando que o "
                        "sistema recebe água de uma fonte não prevista pelo "
                        "modelo de Theis." + extra
                    ),
                )
            )

    # --- partição granular/fraturado quando há perfil geológico ---
    if config.has_geological_profile:
        b_total = (config.granular_thickness_m or 0.0) + (
            config.fractured_thickness_m or 0.0
        )
        diagnostics.append(
            Diagnostic(
                code="GEOLOGICAL_PROFILE",
                severity=Severity.INFO,
                title="Perfil geológico informado",
                detail=(
                    f"Granular: {config.granular_thickness_m:.2f} m | "
                    f"Fraturado: {config.fractured_thickness_m:.2f} m | "
                    f"Total: {b_total:.2f} m.\n"
                    "Lembre-se: um único ensaio de bombeamento fornece o T "
                    "GLOBAL do sistema. A partição entre as litologias exige "
                    "perfilagem de fluxo ou ensaio por trecho com obturador."
                ),
            )
        )

    return diagnostics, dual, late_ratio


def _wellbore_storage_diag(
    config: AnalysisConfig,
    cj: hm.CooperJacobResult,
    diagnostics: list[Diagnostic],
) -> float | None:
    """Calcula t_wbs (poço único) e, se relevante, anexa um Diagnostic."""
    if not config.configuration.is_single_well:
        return None

    t_wbs_min = hm.wellbore_storage_time(
        cj.transmissivity_m2_day, config.radius_m, config.pump_pipe_radius_m
    )
    if t_wbs_min > cj.t_fit_min_sec / 60.0:
        diagnostics.append(
            Diagnostic(
                code="WELLBORE_STORAGE",
                severity=Severity.WARNING,
                title="Trecho ajustado pode conter armazenamento no poço",
                detail=(
                    f"O critério de Schafer (1978) estima o fim do "
                    f"armazenamento no poço em t_wbs ≈ {t_wbs_min:.1f} min, "
                    "mas o ajuste inclui pontos anteriores a esse instante. "
                    "Nesse trecho, o rebaixamento medido reflete o "
                    "esvaziamento da coluna d'água do poço, não a resposta "
                    "do aquífero — descarte esses pontos e reajuste."
                ),
            )
        )
    return t_wbs_min
