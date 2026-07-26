"""Contratos de dados entre a camada de serviço e as camadas de UI/relatório.

O ponto central deste módulo é o :class:`Diagnostic`: em vez de espalhar
``st.warning(...)`` pelo script da interface, a regra de negócio "quando
alertar o usuário" vive no serviço e é devolvida como **dado tipado**. Isso
torna cada alerta testável em pytest, reutilizável no relatório e
independente do Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from hydropump.domain.metadata import PumpMetadata, WellMetadata
from hydropump.domain.scenarios import AquiferScenario, ScenarioProfile, profile_for

if TYPE_CHECKING:  # evita import circular em runtime
    from hydropump.domain.math import (
        CooperJacobResult,
        DualPorosityResult,
        StepDrawdownResult,
        TheisRecoveryResult,
    )

__all__ = [
    "Severity",
    "Diagnostic",
    "WellConfiguration",
    "AnalysisConfig",
    "RecoveryConfig",
    "AnalysisResult",
]


class Severity(StrEnum):
    """Gravidade de um diagnóstico, do menos ao mais restritivo."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Alerta hidrogeológico ou numérico emitido durante a análise.

    Attributes
    ----------
    code : Identificador estável, usado em testes (ex.: ``"JACOB_RATIO"``).
    severity : Gravidade.
    title : Título curto, em linguagem de usuário.
    detail : Explicação e ação recomendada.
    """

    code: str
    severity: Severity
    title: str
    detail: str


class WellConfiguration(StrEnum):
    """Onde o rebaixamento foi medido."""

    SINGLE_WELL = "single"
    OBSERVATION_WELL = "observation"

    @property
    def is_single_well(self) -> bool:
        return self is WellConfiguration.SINGLE_WELL

    @property
    def radius_label(self) -> str:
        return (
            "Raio do poço, r_w"
            if self.is_single_well
            else "Distância ao poço de observação, r"
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Parâmetros de entrada de uma análise de rebaixamento.

    Attributes
    ----------
    static_level_m : Nível estático NE [m].
    radius_m : r_w (poço único) ou r (poço de observação) [m].
    Q_m3_s : Vazão constante [m³/s].
    configuration : Poço único ou de observação.
    saturated_thickness_m : Espessura saturada b [m], ou ``None``.
    apply_jacob : Aplicar a correção de aquífero livre.
    unconfined : Aquífero livre (define a faixa plausível de S).
    fit_start_sec / fit_end_sec : Janela do trecho retilíneo [s].
    projection_hours : Horizonte da projeção teórica [h].
    show_projection : Desenhar a projeção no gráfico.
    pump_pipe_radius_m : Raio externo do recalque [m], para t_wbs.
    well_id : Identificação do poço, exibida em gráfico e relatório.
    fill_to_12h_confirmed : Consentimento explícito do usuário para gerar
        pontos por interpolação até 12h quando o ensaio for mais curto.
        Quando ``True``, esses pontos são tratados como **medidos** — sem
        distinção visual — em gráficos, tabela e relatório; a
        responsabilidade pela extrapolação implícita é do usuário.
    """

    static_level_m: float
    radius_m: float
    Q_m3_s: float
    configuration: WellConfiguration = WellConfiguration.SINGLE_WELL
    saturated_thickness_m: float | None = None
    apply_jacob: bool = True
    unconfined: bool = True
    fit_start_sec: float | None = None
    fit_end_sec: float | None = None
    projection_hours: float = 12.0
    show_projection: bool = True
    pump_pipe_radius_m: float = 0.0
    well_id: str | None = None
    fill_to_12h_confirmed: bool = False
    scenario: AquiferScenario = AquiferScenario.UNCONFINED_GRANULAR
    well: WellMetadata | None = None
    pump: PumpMetadata | None = None
    granular_thickness_m: float | None = None
    fractured_thickness_m: float | None = None
    known_granular_conductivity_m_day: float | None = None

    @property
    def profile(self) -> ScenarioProfile:
        """Regras de interpretação do cenário declarado."""
        return profile_for(self.scenario)

    @property
    def has_geological_profile(self) -> bool:
        """Se o usuário informou a partição granular/fraturado do perfil."""
        return (
            self.granular_thickness_m is not None
            and self.fractured_thickness_m is not None
        )

    @property
    def effective_thickness(self) -> float | None:
        """``b`` apenas quando a correção de Jacob está de fato ativa."""
        return self.saturated_thickness_m if self.apply_jacob else None

    @property
    def Q_m3_h(self) -> float:
        return self.Q_m3_s * 3600.0

    @property
    def Q_l_s(self) -> float:
        return self.Q_m3_s * 1000.0


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Parâmetros da fase de recuperação.

    Attributes
    ----------
    pumping_duration_min : Duração total do bombeamento t_p [min].
    ratio_min / ratio_max : Janela de ajuste em t/t' [-].
    """

    pumping_duration_min: float
    ratio_min: float | None = None
    ratio_max: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Saída completa de uma análise, pronta para UI, gráfico e relatório.

    Attributes
    ----------
    config : Configuração que gerou o resultado.
    cooper_jacob : Parâmetros do ajuste de rebaixamento.
    df_unified : Medições + pontos sintéticos, com coluna ``tipo_dado``.
    df_projection : Curva densa de projeção, ou ``None``.
    diagnostics : Alertas emitidos, em ordem de gravidade decrescente.
    max_drawdown_m : Maior rebaixamento **medido** [m].
    last_time_min : Instante da última medição [min].
    drawdown_12h_m : Rebaixamento em 12 h (medido ou extrapolado) [m].
    specific_capacity_m3_h_m : Q/s [m³/h/m].
    wellbore_storage_end_min : t_wbs [min], ou ``None``.
    recovery : Resultado da recuperação, quando analisada.
    """

    config: AnalysisConfig
    cooper_jacob: CooperJacobResult
    df_unified: pd.DataFrame
    df_projection: pd.DataFrame | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    max_drawdown_m: float = 0.0
    last_time_min: float = 0.0
    drawdown_12h_m: float | None = None
    specific_capacity_m3_h_m: float | None = None
    wellbore_storage_end_min: float | None = None
    recovery: TheisRecoveryResult | None = None
    step_test: StepDrawdownResult | None = None
    dual_porosity: DualPorosityResult | None = None
    late_time_slope_ratio: float | None = None

    @property
    def has_critical(self) -> bool:
        """Há algum diagnóstico que desaconselha o uso do resultado em laudo."""
        return any(d.severity is Severity.CRITICAL for d in self.diagnostics)

    def diagnostics_by(self, severity: Severity) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is severity)
