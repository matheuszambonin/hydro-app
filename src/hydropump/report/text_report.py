"""Geração do relatório textual da análise.

Separado da UI para que o mesmo texto sirva ao botão de download do
Streamlit, a um CLI e a testes — o app.py original montava essa f-string
gigante inline, tornando-a impossível de testar isoladamente.
"""

from __future__ import annotations

import textwrap
from datetime import datetime

from hydropump.domain.models import AnalysisResult, Severity

__all__ = ["build_text_report"]


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Quebra texto longo preservando a indentação, para o relatório."""
    return textwrap.wrap(
        text, width=width, initial_indent=indent, subsequent_indent=indent
    ) or [indent.rstrip()]


_SEVERITY_LABEL = {
    Severity.INFO: "INFO",
    Severity.WARNING: "ATENÇÃO",
    Severity.CRITICAL: "CRÍTICO",
}


def build_text_report(result: AnalysisResult, *, generated_at: datetime | None = None) -> str:
    """Monta o relatório em texto plano da análise de rebaixamento.

    Parameters
    ----------
    result : AnalysisResult
        Saída de :func:`hydropump.services.pumping_test.run_analysis`.
    generated_at : datetime | None
        Timestamp a exibir no cabeçalho; padrão ``datetime.now()``.

    Returns
    -------
    str
        Relatório pronto para ``.encode("utf-8")`` e download.
    """
    cfg = result.config
    cj = result.cooper_jacob
    when = generated_at or datetime.now()

    config_label = (
        "Poço Único (medição no poço bombeado)"
        if cfg.configuration.is_single_well
        else "Poço de Observação (medição em piezômetro vizinho)"
    )

    lines: list[str] = []
    lines.append("RELATÓRIO — ANÁLISE DE ENSAIO DE BOMBEAMENTO")
    lines.append("=" * 64)
    lines.append(f"Gerado em: {when:%d/%m/%Y %H:%M}")
    lines.append("")

    # ------------------------------------------------------------------
    # Identificação do poço
    # ------------------------------------------------------------------
    well = cfg.well
    lines.append("Identificação do poço")
    lines.append("-" * 21)
    lines.append(f"Poço:                           {(well.well_id if well and well.well_id else cfg.well_id) or 'não informado'}")
    if well is not None:
        if well.datetime_utc:
            lines.append(f"Data/hora do ensaio:            {well.datetime_utc:%d/%m/%Y %H:%M}")
        if well.location:
            lines.append(f"Local:                          {well.location}")
        if well.purpose:
            lines.append(f"Finalidade:                     {well.purpose}")
        if well.well_depth_m is not None:
            lines.append(f"Profundidade do poço:           {well.well_depth_m:.2f} m")
        if well.weather:
            lines.append(f"Clima durante o ensaio:         {well.weather}")
        lines.append("")
        lines.append(well.coordinate_summary())
        lines.append("")
        lines.append("Referência das medições de nível:")
        for chunk in _wrap(well.level_note, 74, indent="  "):
            lines.append(chunk)
        if well.notes:
            lines.append("")
            lines.append("Observações:")
            for chunk in _wrap(well.notes, 74, indent="  "):
                lines.append(chunk)
    lines.append("")

    # ------------------------------------------------------------------
    # Bomba
    # ------------------------------------------------------------------
    if cfg.pump is not None:
        pump = cfg.pump
        lines.append("Bomba instalada")
        lines.append("-" * 15)
        if pump.pump_type:
            lines.append(f"Tipo:                           {pump.pump_type}")
        if pump.brand:
            lines.append(f"Marca/modelo:                   {pump.brand}")
        lines.append(f"Profundidade do crivo:          {pump.intake_depth_m:.2f} m")
        lines.append(f"Vazão máxima nominal:           {pump.max_flow_m3_h:.2f} m³/h")
        disponivel = pump.available_drawdown_m(cfg.static_level_m)
        lines.append(f"Rebaixamento disponível:        {disponivel:.2f} m  (crivo − NE)")
        uso = pump.flow_utilization(cfg.Q_m3_h)
        lines.append(f"Uso da capacidade da bomba:     {uso * 100:.1f}% da vazão nominal")
        if uso > 1.0:
            lines.append(
                "  ATENÇÃO: a vazão do ensaio SUPERA a vazão máxima nominal "
                "informada para a bomba. Verifique a vazão máxima cadastrada "
                "ou a medição volumétrica — um dos dois está inconsistente."
            )
        if result.max_drawdown_m >= disponivel:
            lines.append(
                "  ATENÇÃO: o rebaixamento observado já atinge ou supera a "
                "profundidade do crivo."
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Metodologia
    # ------------------------------------------------------------------
    profile = cfg.profile
    lines.append("Metodologia")
    lines.append("-" * 11)
    lines.append(f"Cenário declarado: {profile.label}")
    lines.append(f"Configuração do teste: {config_label}")
    for chunk in _wrap(profile.methodology, 74, indent="  "):
        lines.append(chunk)
    lines.append("")
    lines.append("Resultados")
    lines.append("-" * 10)
    lines.append(f"Vazão (Q):                     {cfg.Q_m3_h:.3f} m³/h  ({cfg.Q_l_s:.3f} L/s)")
    lines.append(f"Transmissividade (T):          {cj.transmissivity_m2_day:.3f} m²/dia")
    ci_lo, ci_hi = cj.transmissivity_ci95_m2_day
    ci_txt = f"{ci_lo:.3f} – {ci_hi:.3f} m²/dia" if ci_hi != float("inf") else "não calculável (poucos pontos)"
    lines.append(f"  IC 95% de T:                 {ci_txt}")
    lines.append(f"Armazenamento (S):              {cj.storativity:.4e}")
    if cj.conductivity_m_day is not None:
        lines.append(f"Condutividade (K):              {cj.conductivity_m_day:.4f} m/dia")
    lines.append(f"Coeficiente de determinação R²: {cj.r_squared:.5f}")
    lines.append(f"u_max:                           {cj.u_max:.5f}  (aproximação válida: {cj.approximation_is_valid})")
    lines.append(f"Pontos usados no ajuste:        {cj.n_points}")
    if result.wellbore_storage_end_min is not None:
        lines.append(f"t_wbs (armazenamento no poço):  {result.wellbore_storage_end_min:.2f} min")
    lines.append("")
    lines.append(f"Rebaixamento Máximo Real:       {result.max_drawdown_m:.3f} m  (em t = {result.last_time_min:.1f} min)")
    if result.drawdown_12h_m is not None:
        lines.append(f"Rebaixamento em 12h:            {result.drawdown_12h_m:.3f} m")
    if result.specific_capacity_m3_h_m is not None:
        sc = result.specific_capacity_m3_h_m
        lines.append("")
        lines.append("Capacidade específica")
        lines.append("-" * 21)
        lines.append(f"Q/s (no rebaixamento máximo):   {sc:.3f} m³/h/m")
        lines.append(f"                                {sc / 3.6:.4f} L/s/m")
        if result.drawdown_12h_m:
            sc12 = cfg.Q_m3_h / result.drawdown_12h_m
            lines.append(f"Q/s projetada em 12h:           {sc12:.3f} m³/h/m")
        if cfg.pump is not None:
            disponivel = cfg.pump.available_drawdown_m(cfg.static_level_m)
            if disponivel > 0:
                q_max = sc * disponivel
                lines.append(
                    f"Vazão no rebaixamento disponível: {q_max:.2f} m³/h  "
                    "(estimativa linear Q/s × rebaixamento até o crivo)"
                )
                for chunk in _wrap(
                    "Nota: extrapolação LINEAR. Em vazões altas as perdas no "
                    "poço crescem com Q², de modo que a vazão real sustentável "
                    "é MENOR que este valor. Um teste de degraus quantifica "
                    "esse efeito (coeficiente C).",
                    74, indent="  ",
                ):
                    lines.append(chunk)
                if q_max > cfg.pump.max_flow_m3_h:
                    for chunk in _wrap(
                        f"ATENÇÃO: este valor ({q_max:.1f} m³/h) excede a vazão "
                        f"máxima da bomba instalada ({cfg.pump.max_flow_m3_h:.1f} "
                        "m³/h). O fator limitante da exploração passa a ser o "
                        "equipamento, não o aquífero.",
                        74, indent="  ",
                    ):
                        lines.append(chunk)
                if result.step_test is not None and result.step_test.C > 0:
                    st = result.step_test
                    q_lin_m3s = q_max / 3600.0
                    s_real = st.predicted_drawdown(q_lin_m3s)
                    s_real_val = float(s_real if s_real.ndim == 0 else s_real.reshape(-1)[0])
                    for chunk in _wrap(
                        f"Correção pelo teste de degraus: em {q_max:.1f} m³/h o "
                        f"rebaixamento previsto por s = BQ + CQ² seria "
                        f"{s_real_val:.2f} m, contra os {disponivel:.2f} m "
                        "disponíveis — compare os dois antes de fixar a vazão "
                        "de exploração.",
                        74, indent="  ",
                    ):
                        lines.append(chunk)
        for chunk in _wrap(
            "A capacidade específica é a grandeza mais estável de um ensaio de "
            "poço único e a mais usada em outorga e dimensionamento de bomba — "
            "ao contrário de S, não é contaminada pelo skin effect.",
            74, indent="  ",
        ):
            lines.append(chunk)

    if result.recovery is not None:
        rec = result.recovery
        lines.append("")
        lines.append("Recuperação (Theis)")
        lines.append("-" * 19)
        lines.append(f"Transmissividade de recuperação (T'): {rec.transmissivity_m2_day:.3f} m²/dia")
        lines.append(f"R² (recuperação):                      {rec.r_squared:.5f}")
        lines.append(f"Razão S/S' (t/t' em s''=0):            {rec.ratio_at_zero:.3f}")
        lines.append(
            "Nota: T' é imune ao skin effect (bomba desligada) — em poço "
            "único, prefira T' a T do rebaixamento para fins de projeto."
        )

    # ------------------------------------------------------------------
    # Teste de degraus
    # ------------------------------------------------------------------
    if result.step_test is not None:
        st = result.step_test
        lines.append("")
        lines.append("Teste de degraus (Jacob, 1947)")
        lines.append("-" * 30)
        lines.append(f"Degraus utilizados:             {st.n_steps}")
        lines.append(f"B (perda no aquífero):          {st.B:.4g} s/m²")
        lines.append(f"C (perda no poço):              {st.C:.4g} s²/m⁵")
        lines.append(f"R² do ajuste s/Q = B + C·Q:     {st.r_squared:.5f}")
        eff = st.well_efficiency(cfg.Q_m3_s)
        lines.append(f"Eficiência do poço em {cfg.Q_m3_h:.1f} m³/h: {eff * 100:.1f}%")
        if st.C < 0:
            lines.append(
                "  ATENÇÃO: C negativo não tem sentido físico. Indica degraus "
                "que não estabilizaram ou erro de leitura."
            )
        elif eff < 0.6:
            lines.append(
                "  ATENÇÃO: eficiência abaixo de 60% sugere poço mal "
                "desenvolvido, colmatado ou com pré-filtro inadequado."
            )

    # ------------------------------------------------------------------
    # Dupla porosidade
    # ------------------------------------------------------------------
    dp = result.dual_porosity
    if dp is not None and dp.detected:
        lines.append("")
        lines.append("Assinatura de dupla porosidade")
        lines.append("-" * 30)
        lines.append(
            f"Trecho inicial (fraturas): T = {dp.transmissivity_early_m2_day:.2f} m²/dia, "
            f"S = {dp.storativity_early:.3e}"
        )
        lines.append(
            f"Trecho tardio (sistema):   T = {dp.transmissivity_late_m2_day:.2f} m²/dia, "
            f"S = {dp.storativity_late:.3e}"
        )
        lines.append(f"Razão entre inclinações:   {dp.slope_ratio:.3f} (≈1 no padrão clássico)")
        for chunk in _wrap(
            "Os dois trechos retilíneos paralelos indicam fraturas condutivas "
            "realimentadas por matriz porosa. Um ajuste único cruzando a "
            "transição produz um S sem significado físico. Estimar os "
            "parâmetros de dupla porosidade (ω, λ) exigiria ajuste de "
            "curvas-tipo de Warren-Root/Moench, fora do escopo deste software.",
            74, indent="  ",
        ):
            lines.append(chunk)

    if result.diagnostics:
        lines.append("")
        lines.append("Diagnósticos")
        lines.append("-" * 12)
        for d in result.diagnostics:
            lines.append(f"[{_SEVERITY_LABEL[d.severity]}] {d.title}")
            lines.append(f"  {d.detail}")

    lines.append("")
    lines.append("Parâmetros de entrada")
    lines.append("-" * 22)
    lines.append(f"Nível Estático (NE):            {cfg.static_level_m:.3f} m")
    lines.append(f"Correção de Jacob aplicada:     {'Sim' if cfg.apply_jacob else 'Não'}")
    b_txt = f"{cfg.saturated_thickness_m:.3f} m" if cfg.saturated_thickness_m else "N/A (correção não aplicada)"
    lines.append(f"Espessura Saturada (b):         {b_txt}")
    lines.append(f"{cfg.configuration.radius_label}:          {cfg.radius_m:.3f} m")

    lines.append("")
    lines.append("Ressalvas do método")
    lines.append("-" * 19)
    for caveat in profile.caveats:
        for chunk in _wrap(f"- {caveat}", 74, indent="  "):
            lines.append(chunk)

    if result.has_critical:
        lines.append("")
        lines.append(
            "*** ATENÇÃO: este resultado contém diagnóstico(s) CRÍTICO(S). "
            "Não utilize os valores marcados como críticos em laudo sem "
            "revisão adicional. ***"
        )

    return "\n".join(lines) + "\n"
