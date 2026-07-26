"""Rotinas matemáticas puras para interpretação de testes de bombeamento.

Escopo: aquíferos **livres (freáticos) granulares**, poço único ou poço de
observação, regime transiente, vazão constante.

Convenção de unidades (SI, salvo indicação explícita no nome do argumento)
-------------------------------------------------------------------------
================  ==========================  =========================
Grandeza          Símbolo                     Unidade adotada
================  ==========================  =========================
Vazão             Q                           m³/s
Tempo             t, t', t_p                  s
Rebaixamento      s, s', s''                  m
Distância radial  r                           m
Espessura sat.    b                           m
Transmissividade  T                           m²/s (também exposto m²/dia)
Condutividade     K                           m/s  (também exposto m/dia)
Armazenamento     S                           adimensional
================  ==========================  =========================

Todas as funções deste módulo são **puras** (não mutam entradas, não fazem
I/O, não dependem de estado global) e **vetorizadas** — aceitam escalares,
listas ou ``numpy.ndarray`` e operam elemento a elemento sempre que a
física permite.

Referências
-----------
- Cooper, H.H. & Jacob, C.E. (1946). *A generalized graphical method for
  evaluating formation constants and summarizing well-field history*.
  Trans. Am. Geophys. Union, 27(4), 526-534.
- Jacob, C.E. (1944). *Notes on determining permeability by pumping tests
  under water-table conditions*. USGS open-file report.
- Theis, C.V. (1935). *The relation between the lowering of the
  piezometric surface and the rate and duration of discharge of a well
  using ground-water storage*. Trans. Am. Geophys. Union, 16, 519-524.
- Schafer, D.C. (1978). *Casing storage can affect pumping test data*.
  Johnson Driller's Journal, 50(1), 6-8.  [critério de t_wbs]
- Kruseman, G.P. & de Ridder, N.A. (1994). *Analysis and Evaluation of
  Pumping Test Data*, 2ª ed., ILRI Publication 47.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

__all__ = [
    "HydroMathError",
    "HydroValueError",
    "NonPositiveTimeError",
    "InvalidGeometryError",
    "RegressionError",
    "FlowRateResult",
    "CooperJacobResult",
    "TheisRecoveryResult",
    "FitWindowSuggestion",
    "calculate_flow_rate",
    "jacob_unconfined_correction",
    "inverse_jacob_correction",
    "cooper_jacob_analysis",
    "theis_recovery_analysis",
    "predict_extrapolated_drawdown",
    "generate_synthetic_field_data",
    "wellbore_storage_time",
    "suggest_fit_window",
    "storativity_is_plausible",
    "specific_capacity",
    "StepDrawdownResult",
    "step_drawdown_analysis",
    "DualPorosityResult",
    "detect_dual_porosity",
    "detect_late_time_flattening",
]

FloatArray = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike

# --------------------------------------------------------------------------
# Constantes físicas / de conversão
# --------------------------------------------------------------------------
LN10: Final[float] = float(np.log(10.0))          # 2.302585... [-]
SECONDS_PER_HOUR: Final[float] = 3600.0           # [s/h]
SECONDS_PER_DAY: Final[float] = 86400.0           # [s/dia]
LITERS_PER_M3: Final[float] = 1000.0              # [L/m³]

#: Coeficiente 2.25 da aproximação logarítmica de Cooper-Jacob (= 4/e^γ,
#: onde γ ≈ 0.5772 é a constante de Euler-Mascheroni).
COOPER_JACOB_COEFF: Final[float] = 2.25

#: Limiar clássico do parâmetro u = r²S/(4Tt) abaixo do qual o erro da
#: aproximação logarítmica de Theis é < 1 % (Kruseman & de Ridder, 1994).
U_VALIDITY_THRESHOLD: Final[float] = 0.01

#: Razão s/b acima da qual a correção de Jacob deixa de ser confiável
#: (dessaturação relevante; migrar para Neuman/Boulton).
JACOB_RATIO_WARNING: Final[float] = 0.25

#: Número de ciclos logarítmicos, além do início da projeção, considerado
#: prudente extrapolar sem novos dados de campo (Kruseman & de Ridder, 1994).
EXTRAPOLATION_WARNING_CYCLES: Final[float] = 1.0

#: Coeficiente do critério de Schafer (1978) para o fim do efeito de
#: armazenamento no poço: t_wbs = 25 (r_c² - r_p²) / T, com T em m²/dia e
#: t em minutos.
WELLBORE_STORAGE_COEFF: Final[float] = 25.0

#: Faixas de plausibilidade física do coeficiente de armazenamento.
#: Livre: produção específica S_y. Confinado: armazenamento elástico.
S_PLAUSIBLE_UNCONFINED: Final[tuple[float, float]] = (0.01, 0.35)
S_PLAUSIBLE_CONFINED: Final[tuple[float, float]] = (1.0e-5, 1.0e-3)

#: Tempos de leitura usuais de caderneta de campo (minutos desde o início do
#: bombeamento) na fase "lenta" (t >= 1 h).
TYPICAL_FIELD_READING_TIMES_MIN: Final[tuple[float, ...]] = (
    60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 360.0, 480.0, 600.0, 720.0,
)


# --------------------------------------------------------------------------
# Hierarquia de exceções
# --------------------------------------------------------------------------
class HydroMathError(Exception):
    """Exceção base do módulo ``domain.math``."""


class HydroValueError(HydroMathError, ValueError):
    """Entrada numericamente inválida (vazia, NaN, Inf, shapes incompatíveis)."""


class NonPositiveTimeError(HydroValueError):
    """Tempo nulo ou negativo — invalida ``log10(t)`` e/ou divide por zero."""


class InvalidGeometryError(HydroValueError):
    """Parâmetro geométrico/hidráulico fisicamente impossível (Q, r, b ≤ 0)."""


class RegressionError(HydroMathError):
    """A reta de ajuste não pôde ser obtida ou não tem sentido físico."""


# --------------------------------------------------------------------------
# Helpers internos de validação (privados, puros)
# --------------------------------------------------------------------------
def _as_float_array(values: ArrayLike, name: str) -> FloatArray:
    """Converte para ``float64``, validando finitude e não-vacuidade."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensivo
        raise HydroValueError(
            f"'{name}' não pôde ser convertido para array numérico: {exc}"
        ) from exc

    if arr.size == 0:
        raise HydroValueError(f"'{name}' está vazio; forneça ao menos um valor.")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise HydroValueError(
            f"'{name}' contém {n_bad} valor(es) não finito(s) (NaN/Inf). "
            "Filtre ou interpole os dados brutos antes da análise."
        )
    return arr


def _require_positive_scalar(value: float, name: str, unit: str) -> float:
    """Valida um escalar estritamente positivo (Q, r, b...)."""
    val = float(value)
    if not np.isfinite(val):
        raise InvalidGeometryError(f"'{name}' deve ser finito (recebido {value!r}).")
    if val <= 0.0:
        raise InvalidGeometryError(
            f"'{name}' deve ser estritamente positivo [{unit}]; recebido {val:g}."
        )
    return val


def _require_positive_time(arr: FloatArray, name: str) -> None:
    """Garante t > 0 — necessário para ``log10(t)`` e para razões t/t'."""
    if np.any(arr <= 0.0):
        n_bad = int(np.count_nonzero(arr <= 0.0))
        raise NonPositiveTimeError(
            f"'{name}' possui {n_bad} valor(es) <= 0. O método logarítmico "
            "exige t > 0 (o instante t = 0 não pertence à reta de Cooper-Jacob)."
        )


def _broadcast_pair(
    a: FloatArray, b: FloatArray, name_a: str, name_b: str
) -> tuple[FloatArray, FloatArray]:
    """Faz broadcast de dois arrays, com erro descritivo em caso de conflito."""
    try:
        return np.broadcast_arrays(a, b)  # type: ignore[return-value]
    except ValueError as exc:
        raise HydroValueError(
            f"Shapes incompatíveis entre '{name_a}' {a.shape} e "
            f"'{name_b}' {b.shape}: {exc}"
        ) from exc


def _build_selection(
    axis: FloatArray,
    mask: ArrayLike | None,
    lower: float | None,
    upper: float | None,
) -> npt.NDArray[np.bool_]:
    """Combina janela [lower, upper] e máscara booleana em um único seletor."""
    selection = np.ones(axis.shape, dtype=bool)

    if lower is not None and upper is not None and float(lower) > float(upper):
        raise HydroValueError(
            f"Janela inválida: limite inferior ({lower}) > superior ({upper})."
        )
    if lower is not None:
        low = float(lower)
        if not np.isfinite(low):
            raise HydroValueError("O limite inferior da janela deve ser finito.")
        selection &= axis >= low
    if upper is not None:
        up = float(upper)
        if not np.isfinite(up):
            raise HydroValueError("O limite superior da janela deve ser finito.")
        selection &= axis <= up

    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != axis.shape:
            raise HydroValueError(
                f"'mask' {user_mask.shape} deve ter o mesmo shape dos dados "
                f"{axis.shape}."
            )
        selection &= user_mask

    return selection


# --------------------------------------------------------------------------
# Estruturas de retorno (imutáveis)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FlowRateResult:
    """Vazão de bombeamento em múltiplas unidades usuais."""

    q_m3_s: FloatArray
    q_m3_h: FloatArray
    q_l_s: FloatArray

    @property
    def mean_q_m3_s(self) -> float:
        """Média aritmética das medições, em m³/s."""
        return float(np.mean(self.q_m3_s))

    @property
    def mean_q_m3_h(self) -> float:
        """Média aritmética das medições, em m³/h."""
        return float(np.mean(self.q_m3_h))

    @property
    def mean_q_l_s(self) -> float:
        """Média aritmética das medições, em L/s."""
        return float(np.mean(self.q_l_s))


@dataclass(frozen=True, slots=True)
class CooperJacobResult:
    """Parâmetros hidrodinâmicos obtidos pelo método de Cooper-Jacob.

    Attributes
    ----------
    delta_s : Inclinação da reta por ciclo logarítmico, Δs' [m/ciclo].
    intercept : Coeficiente linear do ajuste s' = a·log10(t) + b [m].
    t0_sec : Intercepto no eixo do tempo, onde s' = 0 [s].
    transmissivity_m2_s / _m2_day : Transmissividade T.
    storativity : Coeficiente de armazenamento S [-].
    conductivity_m_s / _m_day : K = T / b, se ``saturated_thickness`` dado.
    r_squared : Coeficiente de determinação do ajuste linear [-].
    slope_stderr : Erro padrão da inclinação [m/ciclo].
    transmissivity_stderr_m2_s : Propagação de 1ª ordem do erro em T.
    transmissivity_ci95_m2_day : Intervalo de confiança 95 % de T [m²/dia].
    n_points : Número de pontos efetivamente usados na regressão.
    t_fit_min_sec / t_fit_max_sec : Extremos da janela ajustada [s].
    u_max : Maior valor de u = r²S/(4Tt) no trecho ajustado [-].
    approximation_is_valid : ``True`` se ``u_max`` < 0.01.
    """

    delta_s: float
    intercept: float
    t0_sec: float
    transmissivity_m2_s: float
    transmissivity_m2_day: float
    storativity: float
    conductivity_m_s: float | None
    conductivity_m_day: float | None
    r_squared: float
    slope_stderr: float
    transmissivity_stderr_m2_s: float
    transmissivity_ci95_m2_day: tuple[float, float]
    n_points: int
    t_fit_min_sec: float
    t_fit_max_sec: float
    u_max: float
    approximation_is_valid: bool


@dataclass(frozen=True, slots=True)
class TheisRecoveryResult:
    """Transmissividade obtida pela análise de recuperação de Theis.

    Notes
    -----
    ``ratio_at_zero`` equivale a S/S' (armazenamento no bombeamento sobre
    o da recuperação). Idealmente ≈ 1; valores > 1 são comuns em aquíferos
    livres com drenagem retardada.
    """

    delta_s_residual: float
    intercept: float
    transmissivity_m2_s: float
    transmissivity_m2_day: float
    conductivity_m_s: float | None
    conductivity_m_day: float | None
    r_squared: float
    slope_stderr: float
    transmissivity_stderr_m2_s: float
    transmissivity_ci95_m2_day: tuple[float, float]
    n_points: int
    ratio_at_zero: float
    time_ratio_min: float
    time_ratio_max: float


@dataclass(frozen=True, slots=True)
class FitWindowSuggestion:
    """Janela de ajuste sugerida automaticamente por :func:`suggest_fit_window`.

    Attributes
    ----------
    t_start_sec / t_end_sec : Extremos da janela sugerida [s].
    n_points : Pontos contidos na janela.
    r_squared : R² da reta nessa janela [-].
    u_max : u máximo resultante [-].
    score : Métrica interna de ordenação (maior é melhor).
    """

    t_start_sec: float
    t_end_sec: float
    n_points: int
    r_squared: float
    u_max: float
    score: float


# --------------------------------------------------------------------------
# 1) Vazão
# --------------------------------------------------------------------------
def calculate_flow_rate(
    volume_liters: ArrayLike,
    time_seconds: ArrayLike,
) -> FlowRateResult:
    r"""Calcula a vazão :math:`Q` pelo método volumétrico, :math:`Q = V/t`.

    Entradas escalares ou vetoriais são aceitas (broadcasting mútuo): em
    campo é usual repetir a medição e trabalhar com a série toda.

    Parameters
    ----------
    volume_liters : ArrayLike
        Volume acumulado no recipiente aferido, em **litros [L]**, > 0.
    time_seconds : ArrayLike
        Tempo de enchimento correspondente, em **segundos [s]**, > 0.

    Returns
    -------
    FlowRateResult
        ``q_m3_s`` [m³/s], ``q_m3_h`` [m³/h] e ``q_l_s`` [L/s].

    Raises
    ------
    HydroValueError, NonPositiveTimeError, InvalidGeometryError

    Examples
    --------
    >>> res = calculate_flow_rate(200.0, 18.0)
    >>> round(float(res.q_m3_h), 2)
    40.0
    >>> calculate_flow_rate([200.0, 205.0], [18.0, 18.5]).q_l_s.shape
    (2,)
    """
    volume = _as_float_array(volume_liters, "volume_liters")
    time_s = _as_float_array(time_seconds, "time_seconds")

    if np.any(volume <= 0.0):
        raise InvalidGeometryError(
            "'volume_liters' deve ser estritamente positivo [L]; "
            f"{int(np.count_nonzero(volume <= 0.0))} valor(es) <= 0."
        )
    _require_positive_time(time_s, "time_seconds")

    volume, time_s = _broadcast_pair(volume, time_s, "volume_liters", "time_seconds")

    q_m3_s = (volume / LITERS_PER_M3) / time_s
    return FlowRateResult(
        q_m3_s=q_m3_s,
        q_m3_h=q_m3_s * SECONDS_PER_HOUR,
        q_l_s=volume / time_s,
    )


def specific_capacity(Q_m3_h: float, drawdown_m: float) -> float:
    r"""Capacidade específica :math:`Q/s` [m³/h/m].

    Grandeza mais estável que ``S`` em ensaio de poço único e diretamente
    usada em outorga e dimensionamento de bomba.

    Examples
    --------
    >>> round(specific_capacity(24.0, 6.0), 3)
    4.0
    """
    q = _require_positive_scalar(Q_m3_h, "Q_m3_h", "m³/h")
    s = _require_positive_scalar(drawdown_m, "drawdown_m", "m")
    return q / s


# --------------------------------------------------------------------------
# 2) Correção de Jacob para aquífero livre (direta e inversa)
# --------------------------------------------------------------------------
def jacob_unconfined_correction(
    drawdown: ArrayLike,
    saturated_thickness: float | None,
    *,
    strict: bool = True,
    apply_correction: bool = True,
) -> FloatArray:
    r"""Corrige o rebaixamento medido em aquífero livre (Jacob, 1944).

    .. math:: s' = s - \frac{s^{2}}{2\,b}

    Somente ``s'`` (e não ``s``) deve alimentar :func:`cooper_jacob_analysis`
    ou :func:`theis_recovery_analysis`.

    Parameters
    ----------
    drawdown : ArrayLike
        Rebaixamento observado :math:`s` [m], positivo para baixo.
    saturated_thickness : float | None
        Espessura saturada **inicial** :math:`b` [m], > 0. Se ``None``, a
        correção é ignorada e os dados originais retornam sem erro.
    strict : bool, opcional
        Se ``True`` (padrão), erro quando algum ``s`` < 0. ``s >= b`` é
        sempre bloqueado quando a correção é aplicada.
    apply_correction : bool, opcional
        Se ``False``, retorna ``s`` inalterado (aquífero confinado/semi-
        confinado, ou opção desmarcada na interface).

    Returns
    -------
    FloatArray
        Rebaixamento corrigido :math:`s'` [m], mesmo shape da entrada.

    Notes
    -----
    Confiável para :math:`s/b <` :data:`JACOB_RATIO_WARNING` (0.25). Acima
    disso, fluxo vertical e drenagem retardada dominam e a interpretação
    deve migrar para Neuman ou Boulton. Este módulo não bloqueia o caso —
    expõe o diagnóstico para o chamador.

    Examples
    --------
    >>> float(jacob_unconfined_correction(2.0, 20.0))
    1.9
    >>> jacob_unconfined_correction([1.0, 2.0, 3.0], 20.0).round(3)
    array([0.975, 1.9  , 2.775])
    >>> jacob_unconfined_correction([1.0, 2.0], None)
    array([1., 2.])
    """
    s = _as_float_array(drawdown, "drawdown")

    if strict and np.any(s < 0.0):
        raise HydroValueError(
            "'drawdown' possui valores negativos (elevação do nível). "
            "Verifique a referência de medição ou use strict=False."
        )

    if not apply_correction or saturated_thickness is None:
        return s

    b = _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")

    if np.any(s >= b):
        s_max = float(np.max(s))
        raise InvalidGeometryError(
            f"Rebaixamento máximo ({s_max:.3f} m) >= espessura saturada "
            f"({b:.3f} m): o aquífero estaria totalmente dessaturado no poço. "
            "A correção de Jacob não é aplicável."
        )

    return s - (np.square(s) / (2.0 * b))


def inverse_jacob_correction(
    drawdown_corr: ArrayLike,
    saturated_thickness: float,
    *,
    context: str = "",
) -> FloatArray:
    r"""Inverte a correção de Jacob: :math:`s'` corrigido → :math:`s` real.

    Resolvendo :math:`s' = s - s^2/2b` para :math:`s < b`:

    .. math:: s = b - \sqrt{b^{2} - 2\,b\,s'}

    Parameters
    ----------
    drawdown_corr : ArrayLike
        Rebaixamento corrigido :math:`s'` [m].
    saturated_thickness : float
        Espessura saturada :math:`b` [m], > 0.
    context : str, opcional
        Trecho anexado à mensagem de erro, para o chamador identificar a
        origem (ex.: ``"na projeção"``).

    Raises
    ------
    InvalidGeometryError
        Discriminante negativo em algum ponto — o rebaixamento projetado
        supera a dessaturação total do aquífero.

    Examples
    --------
    >>> s = jacob_unconfined_correction([1.0, 2.0, 3.0], 20.0)
    >>> inverse_jacob_correction(s, 20.0).round(9)
    array([1., 2., 3.])
    """
    s_corr = _as_float_array(drawdown_corr, "drawdown_corr")
    b = _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")

    discriminant = b**2 - 2.0 * b * s_corr
    if np.any(discriminant < 0.0):
        sufixo = f" {context}" if context else ""
        raise InvalidGeometryError(
            "Correção inversa de Jacob indefinida: o rebaixamento corrigido"
            f"{sufixo} excede a dessaturação total do aquífero "
            f"(b = {b:.3f} m). Reduza o horizonte ou revise "
            "'saturated_thickness'."
        )
    return b - np.sqrt(discriminant)


# --------------------------------------------------------------------------
# 3) Cooper-Jacob (regime de rebaixamento)
# --------------------------------------------------------------------------
def cooper_jacob_analysis(
    time_sec: ArrayLike,
    drawdown_corr: ArrayLike,
    Q: float,
    r: float,
    *,
    t_start_sec: float | None = None,
    t_end_sec: float | None = None,
    mask: ArrayLike | None = None,
    saturated_thickness: float | None = None,
) -> CooperJacobResult:
    r"""Ajusta a reta semilogarítmica de Cooper-Jacob e extrai :math:`T` e :math:`S`.

    Para :math:`u = r^2 S / (4 T t) < 0.01`, a solução de Theis reduz-se a

    .. math::
        s' = \frac{2.303\,Q}{4\pi T}\,
             \log_{10}\!\left(\frac{2.25\,T\,t}{r^{2} S}\right)

    Da regressão :math:`s' = \Delta s' \log_{10}(t) + c` obtém-se

    .. math::
        T = \frac{2.303\,Q}{4\pi\,\Delta s'}, \qquad
        S = \frac{2.25\,T\,t_0}{r^{2}}, \qquad t_0 = 10^{-c/\Delta s'}

    Parameters
    ----------
    time_sec : ArrayLike
        Tempo desde o início do bombeamento [s], > 0.
    drawdown_corr : ArrayLike
        Rebaixamento **já corrigido** :math:`s'` [m].
    Q : float
        Vazão constante [m³/s].
    r : float
        Distância radial [m]. Em poço único, o raio do poço — ciente de que
        ``S`` resultará contaminado por perdas de carga (skin effect).
    t_start_sec, t_end_sec : float | None, opcional
        Janela temporal do trecho retilíneo [s]. Descartar os tempos
        iniciais é a prática usual (armazenamento no poço, u grande).
    mask : ArrayLike | None, opcional
        Máscara booleana explícita, combinada por conjunção com a janela.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m]; se dada, retorna também K = T/b.

    Returns
    -------
    CooperJacobResult

    Raises
    ------
    NonPositiveTimeError, InvalidGeometryError, HydroValueError, RegressionError

    Notes
    -----
    ``u_max`` é avaliado a posteriori com o próprio par (T, S) ajustado, no
    menor tempo do trecho. **Em ensaio de poço único** (:math:`r = r_w`),
    ``u`` é numericamente minúsculo e ``approximation_is_valid`` tende a ser
    sempre ``True`` — nesse caso o critério útil é
    :func:`wellbore_storage_time`, não ``u``.
    """
    t = _as_float_array(time_sec, "time_sec")
    s = _as_float_array(drawdown_corr, "drawdown_corr")
    q = _require_positive_scalar(Q, "Q", "m³/s")
    radius = _require_positive_scalar(r, "r", "m")
    b = (
        _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")
        if saturated_thickness is not None
        else None
    )

    if t.shape != s.shape:
        raise HydroValueError(
            f"'time_sec' {t.shape} e 'drawdown_corr' {s.shape} devem ter o "
            "mesmo shape (pares tempo-rebaixamento)."
        )

    selection = _build_selection(t, mask, t_start_sec, t_end_sec)
    t_fit, s_fit = t[selection], s[selection]

    if t_fit.size < 3:
        raise RegressionError(
            f"Apenas {t_fit.size} ponto(s) no trecho selecionado; são "
            "necessários ao menos 3 para uma reta com significado estatístico."
        )
    _require_positive_time(t_fit, "time_sec (trecho selecionado)")

    log_t = np.log10(t_fit)
    if np.ptp(log_t) == 0.0:
        raise RegressionError(
            "Todos os tempos do trecho são iguais: variância nula em "
            "log10(t), regressão indeterminada."
        )

    fit = stats.linregress(log_t, s_fit)
    delta_s = float(fit.slope)
    intercept = float(fit.intercept)

    if delta_s <= 0.0:
        raise RegressionError(
            f"Inclinação Δs' = {delta_s:.4g} m/ciclo <= 0. O rebaixamento "
            "deveria crescer com log10(t); verifique o sinal dos dados, a "
            "janela escolhida ou a ocorrência de recarga/recuperação."
        )

    transmissivity = (LN10 * q) / (4.0 * np.pi * delta_s)

    exponent = -intercept / delta_s
    if abs(exponent) > 300.0:
        raise RegressionError(
            f"Intercepto temporal fora da faixa numérica (10^{exponent:.1f} s). "
            "O ajuste provavelmente não representa o trecho retilíneo."
        )
    t0 = float(10.0**exponent)

    storativity = (COOPER_JACOB_COEFF * transmissivity * t0) / (radius**2)

    t_min = float(np.min(t_fit))
    t_max = float(np.max(t_fit))
    u_max = (radius**2 * storativity) / (4.0 * transmissivity * t_min)

    slope_stderr = float(fit.stderr)
    # Propagação de 1ª ordem: T ∝ 1/Δs'  =>  σ_T/T = σ_Δs'/Δs'
    t_stderr = transmissivity * (slope_stderr / delta_s)
    ci95 = _transmissivity_ci95_day(transmissivity, delta_s, slope_stderr, t_fit.size)

    return CooperJacobResult(
        delta_s=delta_s,
        intercept=intercept,
        t0_sec=t0,
        transmissivity_m2_s=transmissivity,
        transmissivity_m2_day=transmissivity * SECONDS_PER_DAY,
        storativity=storativity,
        conductivity_m_s=(transmissivity / b) if b else None,
        conductivity_m_day=(transmissivity / b * SECONDS_PER_DAY) if b else None,
        r_squared=float(fit.rvalue) ** 2,
        slope_stderr=slope_stderr,
        transmissivity_stderr_m2_s=t_stderr,
        transmissivity_ci95_m2_day=ci95,
        n_points=int(t_fit.size),
        t_fit_min_sec=t_min,
        t_fit_max_sec=t_max,
        u_max=float(u_max),
        approximation_is_valid=bool(u_max < U_VALIDITY_THRESHOLD),
    )


def _transmissivity_ci95_day(
    transmissivity_m2_s: float,
    slope: float,
    slope_stderr: float,
    n_points: int,
) -> tuple[float, float]:
    """IC 95 % de T [m²/dia], propagado do IC da inclinação (t de Student).

    T ∝ 1/Δs', função monotônica decrescente: o limite superior de T vem do
    limite **inferior** de Δs'. Se o IC de Δs' cruzar zero, o IC de T é
    ilimitado e retorna-se ``inf``.
    """
    dof = max(n_points - 2, 1)
    t_crit = float(stats.t.ppf(0.975, dof))
    slope_low = slope - t_crit * slope_stderr
    slope_high = slope + t_crit * slope_stderr

    t_day = transmissivity_m2_s * SECONDS_PER_DAY
    if slope_low <= 0.0:
        return (t_day * slope / slope_high, float("inf"))
    return (t_day * slope / slope_high, t_day * slope / slope_low)


# --------------------------------------------------------------------------
# 4) Recuperação de Theis
# --------------------------------------------------------------------------
def theis_recovery_analysis(
    time_pump_sec: ArrayLike,
    time_recovery_sec: ArrayLike,
    residual_drawdown: ArrayLike,
    Q: float,
    *,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    mask: ArrayLike | None = None,
    saturated_thickness: float | None = None,
) -> TheisRecoveryResult:
    r"""Analisa a fase de recuperação pelo método do rebaixamento residual.

    .. math::
        s'' = \frac{2.303\,Q}{4\pi T}\,\log_{10}\!\left(\frac{t}{t'}\right)

    com :math:`t = t_p + t'`. A reta em papel semilog fornece
    :math:`T = 2.303\,Q / (4\pi\,\Delta s'')`.

    Parameters
    ----------
    time_pump_sec : ArrayLike
        Duração total do bombeamento :math:`t_p` [s].
    time_recovery_sec : ArrayLike
        Tempo desde a parada da bomba :math:`t'` [s], > 0.
    residual_drawdown : ArrayLike
        Rebaixamento residual :math:`s''` [m]. Em aquífero livre, aplique
        antes :func:`jacob_unconfined_correction`.
    Q : float
        Vazão constante mantida durante o bombeamento [m³/s].
    ratio_min, ratio_max : float | None, opcional
        Janela de seleção em :math:`t/t'` [-].
    mask : ArrayLike | None, opcional
        Máscara booleana explícita.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m] para reportar K' = T'/b.

    Returns
    -------
    TheisRecoveryResult

    Notes
    -----
    O método **não fornece S** isoladamente. Sua grande vantagem prática:
    a recuperação ocorre com a bomba desligada, sem fluxo através da parede
    do poço, portanto **T' é imune ao skin effect** — em ensaio de poço
    único, é a estimativa confiável de T. Em aquíferos livres com drenagem
    retardada é comum obter :math:`T' > T_{rebaixamento}`.
    """
    t_p = _as_float_array(time_pump_sec, "time_pump_sec")
    t_rec = _as_float_array(time_recovery_sec, "time_recovery_sec")
    s_res = _as_float_array(residual_drawdown, "residual_drawdown")
    q = _require_positive_scalar(Q, "Q", "m³/s")
    b = (
        _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")
        if saturated_thickness is not None
        else None
    )

    _require_positive_time(t_rec, "time_recovery_sec")
    if np.any(t_p < 0.0):
        raise NonPositiveTimeError(
            "'time_pump_sec' (duração do bombeamento) não pode ser negativo."
        )

    t_p_b, t_rec_b = _broadcast_pair(
        t_p, t_rec, "time_pump_sec", "time_recovery_sec"
    )
    t_total = t_p_b + t_rec_b

    if t_total.shape != s_res.shape:
        raise HydroValueError(
            f"Shape resultante de t/t' {t_total.shape} difere de "
            f"'residual_drawdown' {s_res.shape}."
        )

    ratio = t_total / t_rec_b  # t/t' >= 1 por construção
    selection = _build_selection(ratio, mask, ratio_min, ratio_max)
    ratio_fit, s_fit = ratio[selection], s_res[selection]

    if ratio_fit.size < 3:
        raise RegressionError(
            f"Apenas {ratio_fit.size} ponto(s) na janela de t/t'; são "
            "necessários ao menos 3 para o ajuste."
        )
    if np.any(ratio_fit <= 0.0):
        raise NonPositiveTimeError(
            "Razão t/t' <= 0 no trecho selecionado: log10 indefinido."
        )

    log_ratio = np.log10(ratio_fit)
    if np.ptp(log_ratio) == 0.0:
        raise RegressionError(
            "Todas as razões t/t' são iguais: variância nula em log10(t/t')."
        )

    fit = stats.linregress(log_ratio, s_fit)
    delta_s2 = float(fit.slope)
    intercept = float(fit.intercept)

    if delta_s2 <= 0.0:
        raise RegressionError(
            f"Inclinação Δs'' = {delta_s2:.4g} m/ciclo <= 0. O rebaixamento "
            "residual deve diminuir à medida que t/t' → 1; confira o sinal "
            "de 's_res' e a definição de t'."
        )

    transmissivity = (LN10 * q) / (4.0 * np.pi * delta_s2)

    exponent = -intercept / delta_s2
    ratio_at_zero = float(10.0**exponent) if abs(exponent) < 300.0 else float("inf")

    slope_stderr = float(fit.stderr)
    t_stderr = transmissivity * (slope_stderr / delta_s2)
    ci95 = _transmissivity_ci95_day(
        transmissivity, delta_s2, slope_stderr, ratio_fit.size
    )

    return TheisRecoveryResult(
        delta_s_residual=delta_s2,
        intercept=intercept,
        transmissivity_m2_s=transmissivity,
        transmissivity_m2_day=transmissivity * SECONDS_PER_DAY,
        conductivity_m_s=(transmissivity / b) if b else None,
        conductivity_m_day=(transmissivity / b * SECONDS_PER_DAY) if b else None,
        r_squared=float(fit.rvalue) ** 2,
        slope_stderr=slope_stderr,
        transmissivity_stderr_m2_s=t_stderr,
        transmissivity_ci95_m2_day=ci95,
        n_points=int(ratio_fit.size),
        ratio_at_zero=ratio_at_zero,
        time_ratio_min=float(np.min(ratio_fit)),
        time_ratio_max=float(np.max(ratio_fit)),
    )


# --------------------------------------------------------------------------
# 5) Diagnósticos de validade
# --------------------------------------------------------------------------
def wellbore_storage_time(
    transmissivity_m2_day: float,
    casing_radius_m: float,
    pump_pipe_radius_m: float = 0.0,
) -> float:
    r"""Instante em que cessa o efeito de armazenamento no poço (Schafer, 1978).

    .. math:: t_{wbs} = \frac{25\,(r_c^{2} - r_p^{2})}{T}

    com :math:`T` em m²/dia e :math:`t_{wbs}` em **minutos**. Antes desse
    instante, o rebaixamento medido reflete o esvaziamento da própria coluna
    d'água do poço, não a resposta do aquífero: **descarte esses pontos do
    ajuste de Cooper-Jacob**.

    Parameters
    ----------
    transmissivity_m2_day : float
        Estimativa preliminar de T [m²/dia] (pode vir de um primeiro ajuste
        sobre toda a série; o critério é pouco sensível a erro em T).
    casing_radius_m : float
        Raio interno do revestimento na seção onde o nível oscila [m].
    pump_pipe_radius_m : float, opcional
        Raio externo da tubulação de recalque dentro do poço [m]. Reduz o
        volume armazenado. Padrão 0 (conservador: t_wbs máximo).

    Returns
    -------
    float
        :math:`t_{wbs}` em **minutos** desde o início do bombeamento.

    Examples
    --------
    >>> round(wellbore_storage_time(50.0, 0.10), 3)
    0.005
    >>> round(wellbore_storage_time(2.0, 0.15, 0.025), 4)
    0.2734
    """
    T = _require_positive_scalar(transmissivity_m2_day, "transmissivity_m2_day", "m²/dia")
    r_c = _require_positive_scalar(casing_radius_m, "casing_radius_m", "m")
    r_p = float(pump_pipe_radius_m)
    if r_p < 0.0 or r_p >= r_c:
        raise InvalidGeometryError(
            f"'pump_pipe_radius_m' ({r_p:g} m) deve estar em [0, r_c) com "
            f"r_c = {r_c:g} m."
        )
    return WELLBORE_STORAGE_COEFF * (r_c**2 - r_p**2) / T


def storativity_is_plausible(storativity: float, *, unconfined: bool = True) -> bool:
    """Verifica se ``S`` cai na faixa fisicamente plausível.

    Livre (produção específica S_y): :data:`S_PLAUSIBLE_UNCONFINED`.
    Confinado (armazenamento elástico): :data:`S_PLAUSIBLE_CONFINED`.

    Examples
    --------
    >>> storativity_is_plausible(0.15)
    True
    >>> storativity_is_plausible(3.0)
    False
    >>> storativity_is_plausible(2e-4, unconfined=False)
    True
    """
    lo, hi = S_PLAUSIBLE_UNCONFINED if unconfined else S_PLAUSIBLE_CONFINED
    return bool(np.isfinite(storativity) and lo <= storativity <= hi)


def suggest_fit_window(
    time_sec: ArrayLike,
    drawdown_corr: ArrayLike,
    Q: float,
    r: float,
    *,
    min_points: int = 5,
    require_valid_u: bool = True,
) -> FitWindowSuggestion | None:
    r"""Sugere a janela do trecho retilíneo de Cooper-Jacob.

    Varre todas as janelas **contíguas** com pelo menos ``min_points``
    pontos, ajusta a reta em cada uma e escolhe a de maior pontuação:

    .. math:: \text{score} = R^{2} + 0.02\,\log_{10}(n_{\text{pontos}})

    O termo em :math:`n` é um desempate suave que evita a degenerescência
    "3 pontos colineares dão R² = 1". Janelas com :math:`\Delta s' \le 0`
    são descartadas; se ``require_valid_u=True``, também são descartadas as
    que resultam em :math:`u_{max} \ge` :data:`U_VALIDITY_THRESHOLD` (a
    menos que **nenhuma** janela satisfaça o critério, caso em que a
    restrição é relaxada e ``u_max`` é reportado para o chamador decidir).

    Returns
    -------
    FitWindowSuggestion | None
        ``None`` se houver menos de ``min_points`` pontos utilizáveis ou se
        nenhuma janela produzir um ajuste fisicamente válido.

    Notes
    -----
    Heurística de **apoio**, não substituto do julgamento do hidrogeólogo:
    o algoritmo não distingue um trecho retilíneo genuíno de um trecho
    retilíneo espúrio produzido por barreira ou recarga. Sempre confira a
    janela sugerida contra o gráfico semi-log.
    """
    t = _as_float_array(time_sec, "time_sec")
    s = _as_float_array(drawdown_corr, "drawdown_corr")
    if t.shape != s.shape:
        raise HydroValueError("'time_sec' e 'drawdown_corr' devem ter o mesmo shape.")

    order = np.argsort(t)
    t, s = t[order], s[order]
    keep = t > 0.0
    t, s = t[keep], s[keep]

    n = t.size
    if n < max(min_points, 3):
        return None

    best: FitWindowSuggestion | None = None
    best_relaxed: FitWindowSuggestion | None = None

    log_t = np.log10(t)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            lt, ls = log_t[i:j], s[i:j]
            if np.ptp(lt) == 0.0:
                continue
            fit = stats.linregress(lt, ls)
            slope = float(fit.slope)
            if slope <= 0.0:
                continue
            transmissivity = (LN10 * Q) / (4.0 * np.pi * slope)
            exponent = -float(fit.intercept) / slope
            if abs(exponent) > 300.0:
                continue
            t0 = 10.0**exponent
            storativity = (COOPER_JACOB_COEFF * transmissivity * t0) / (r**2)
            u_max = (r**2 * storativity) / (4.0 * transmissivity * float(t[i]))

            r2 = float(fit.rvalue) ** 2
            score = r2 + 0.02 * np.log10(j - i)
            cand = FitWindowSuggestion(
                t_start_sec=float(t[i]),
                t_end_sec=float(t[j - 1]),
                n_points=int(j - i),
                r_squared=r2,
                u_max=float(u_max),
                score=float(score),
            )
            if best_relaxed is None or cand.score > best_relaxed.score:
                best_relaxed = cand
            if u_max < U_VALIDITY_THRESHOLD and (best is None or cand.score > best.score):
                best = cand

    if not require_valid_u:
        return best_relaxed
    return best if best is not None else best_relaxed


# --------------------------------------------------------------------------
# 6) Extrapolação temporal da reta de Cooper-Jacob
# --------------------------------------------------------------------------
def predict_extrapolated_drawdown(
    delta_s: float,
    t0_sec: float,
    time_start_sec: float,
    *,
    target_hours: float = 12.0,
    step_minutes: float = 5.0,
    saturated_thickness: float | None = None,
) -> pd.DataFrame:
    r"""Extrapola no tempo a reta semilogarítmica de Cooper-Jacob.

    .. math::
        s'_{\text{proj}}(t) = \Delta s' \cdot
            \log_{10}\!\left(\frac{t}{t_0}\right)

    Se ``saturated_thickness`` for informada, o resultado é convertido de
    volta ao rebaixamento real via :func:`inverse_jacob_correction`.

    Parameters
    ----------
    delta_s : float
        Inclinação :math:`\Delta s'` [m/ciclo], > 0.
    t0_sec : float
        Intercepto temporal :math:`t_0` [s], > 0.
    time_start_sec : float
        Instante em que a projeção começa [s], > 0. Tipicamente o maior
        tempo efetivamente medido.
    target_hours : float, opcional
        Horizonte final [h] (padrão: 12 h), > ``time_start_sec``.
    step_minutes : float, opcional
        Passo de discretização [min] (padrão: 5 min).
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m].

    Returns
    -------
    pandas.DataFrame
        Colunas ``tempo_min``, ``rebaixamento_projetado_m``, ``tipo_dado``.

    Notes
    -----
    **Limites de validade** — leia antes de usar em decisão de projeto:

    - A extrapolação não é uma nova medição: é a continuação da mesma reta,
      sob a premissa de que nada muda (vazão constante, sem recarga, sem
      limites físicos do aquífero, sem interferência de outros poços).
    - É aceito extrapolar com razoável confiança até cerca de
      :data:`EXTRAPOLATION_WARNING_CYCLES` ciclo(s) logarítmico(s) além de
      ``time_start_sec``. Além disso, ``UserWarning`` é emitido.
    - :math:`u` decresce com :math:`t`, então a aproximação logarítmica
      permanece numericamente válida — o que **não** garante que o aquífero
      real continue infinito e homogêneo por todo o horizonte.
    - Em aquífero livre, a correção inversa amplia incertezas conforme
      :math:`s/b` se aproxima de :data:`JACOB_RATIO_WARNING`.

    Examples
    --------
    >>> df = predict_extrapolated_drawdown(
    ...     delta_s=0.5, t0_sec=10.0, time_start_sec=3600.0,
    ...     target_hours=2.0, step_minutes=30.0,
    ... )
    >>> list(df.columns)
    ['tempo_min', 'rebaixamento_projetado_m', 'tipo_dado']
    """
    delta = _require_positive_scalar(delta_s, "delta_s", "m/ciclo")
    t0 = _require_positive_scalar(t0_sec, "t0_sec", "s")
    t_start = _require_positive_scalar(time_start_sec, "time_start_sec", "s")
    step_s = _require_positive_scalar(step_minutes, "step_minutes", "min") * 60.0
    b = (
        _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")
        if saturated_thickness is not None
        else None
    )

    target_h = float(target_hours)
    if not np.isfinite(target_h) or target_h <= 0.0:
        raise InvalidGeometryError(
            f"'target_hours' deve ser finito e estritamente positivo; "
            f"recebido {target_hours!r}."
        )
    t_end = target_h * SECONDS_PER_HOUR

    if t_end <= t_start:
        raise NonPositiveTimeError(
            f"Horizonte final ({t_end:.1f} s = {target_h:.3g} h) deve ser "
            f"estritamente maior que 'time_start_sec' ({t_start:.1f} s)."
        )

    t = np.arange(t_start, t_end, step_s, dtype=np.float64)
    if t.size == 0 or not np.isclose(t[-1], t_end):
        t = np.append(t, t_end)

    _require_positive_time(t, "t (vetor de tempo gerado)")

    s_proj_corr = delta * np.log10(t / t0)
    s_proj = (
        inverse_jacob_correction(s_proj_corr, b, context="projetado")
        if b is not None
        else s_proj_corr
    )

    if t_end / t_start > 10.0**EXTRAPOLATION_WARNING_CYCLES:
        warnings.warn(
            f"Horizonte de extrapolação (target_hours={target_h:.3g} h) "
            f"ultrapassa {EXTRAPOLATION_WARNING_CYCLES:.0f} ciclo(s) "
            "logarítmico(s) além de 'time_start_sec'. Resultados distantes "
            "do início da projeção são progressivamente menos confiáveis.",
            stacklevel=2,
        )

    return pd.DataFrame(
        {
            "tempo_min": t / 60.0,
            "rebaixamento_projetado_m": s_proj,
            "tipo_dado": "Projeção",
        }
    )


# --------------------------------------------------------------------------
# 7) Dados sintéticos de campo (discretos, grade de caderneta)
# --------------------------------------------------------------------------
def generate_synthetic_field_data(
    delta_s: float,
    t0_sec: float,
    Q_mean: float,
    last_time_min: float,
    *,
    target_hours: float = 12.0,
    static_level_m: float | None = None,
    saturated_thickness: float | None = None,
    reading_times_min: ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Gera pontos sintéticos de rebaixamento na grade discreta de campo.

    Diferente de :func:`predict_extrapolated_drawdown` (vetor denso, para
    traçar a curva contínua), esta função gera **pontos discretos** nos
    instantes em que uma leitura de caderneta seria registrada, com a mesma
    estrutura de colunas dos dados reais, para concatenação direta.

    Parameters
    ----------
    delta_s, t0_sec
        Parâmetros da reta de Cooper-Jacob (ver :class:`CooperJacobResult`).
    Q_mean : float
        Vazão média [m³/s]. Preenche a coluna ``vazao_m3_h``; não entra na
        equação do rebaixamento (já embutida em ``delta_s``).
    last_time_min : float
        Tempo do último ponto **real** [min]. Só leituras posteriores entram.
    target_hours : float, opcional
        Horizonte final [h] (padrão: 12 h).
    static_level_m : float | None, opcional
        Nível estático [m]; se dado, ``nd_m = static_level_m + rebaixamento``.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m].
    reading_times_min : ArrayLike | None, opcional
        Grade de leitura; padrão :data:`TYPICAL_FIELD_READING_TIMES_MIN`.

    Returns
    -------
    pandas.DataFrame
        Colunas ``tempo_min``, ``nd_m``, ``rebaixamento_m``, ``vazao_m3_h``,
        ``tipo_dado`` (= ``'Sintético'``).

    Notes
    -----
    Herda **todos** os limites de validade de
    :func:`predict_extrapolated_drawdown`. Os pontos gerados não são
    medições. Trate como dado de preenchimento/planejamento, nunca como
    substituto de leitura de campo.

    Examples
    --------
    >>> df = generate_synthetic_field_data(
    ...     delta_s=0.5, t0_sec=10.0, Q_mean=0.01, last_time_min=45.0,
    ...     target_hours=2.0,
    ... )
    >>> float(df["tempo_min"].iloc[-1])
    120.0
    """
    delta = _require_positive_scalar(delta_s, "delta_s", "m/ciclo")
    t0 = _require_positive_scalar(t0_sec, "t0_sec", "s")
    q = _require_positive_scalar(Q_mean, "Q_mean", "m³/s")
    b = (
        _require_positive_scalar(saturated_thickness, "saturated_thickness", "m")
        if saturated_thickness is not None
        else None
    )

    last_t = float(last_time_min)
    if not np.isfinite(last_t) or last_t < 0.0:
        raise NonPositiveTimeError(
            f"'last_time_min' deve ser finito e >= 0; recebido {last_time_min!r}."
        )

    target_h = float(target_hours)
    if not np.isfinite(target_h) or target_h <= 0.0:
        raise InvalidGeometryError(
            f"'target_hours' deve ser finito e estritamente positivo; "
            f"recebido {target_hours!r}."
        )
    target_min = target_h * 60.0

    if target_min <= last_t:
        raise NonPositiveTimeError(
            f"Horizonte final ({target_min:.1f} min = {target_h:.3g} h) deve "
            f"ser estritamente maior que 'last_time_min' ({last_t:.1f} min)."
        )

    grid = _as_float_array(
        reading_times_min
        if reading_times_min is not None
        else TYPICAL_FIELD_READING_TIMES_MIN,
        "reading_times_min",
    )

    selected = np.sort(np.unique(grid[(grid > last_t) & (grid <= target_min)]))
    if selected.size == 0 or not np.isclose(selected[-1], target_min):
        selected = np.append(selected, target_min)

    t_sec = selected * 60.0
    _require_positive_time(t_sec, "t (grade de leitura gerada)")

    s_corr = delta * np.log10(t_sec / t0)
    s_real = (
        inverse_jacob_correction(s_corr, b, context="sintético")
        if b is not None
        else s_corr
    )

    nd_m = (
        static_level_m + s_real
        if static_level_m is not None
        else np.full_like(s_real, np.nan)
    )

    if target_min / max(last_t, t0 / 60.0) > 10.0**EXTRAPOLATION_WARNING_CYCLES:
        warnings.warn(
            f"Horizonte do dataset sintético (target_hours={target_h:.3g} h) "
            f"ultrapassa {EXTRAPOLATION_WARNING_CYCLES:.0f} ciclo(s) "
            "logarítmico(s) além de 'last_time_min'.",
            stacklevel=2,
        )

    return pd.DataFrame(
        {
            "tempo_min": selected,
            "nd_m": nd_m,
            "rebaixamento_m": s_real,
            "vazao_m3_h": q * SECONDS_PER_HOUR,
            "tipo_dado": "Sintético",
        }
    )


# --------------------------------------------------------------------------
# 8) Teste de degraus (Jacob, 1947) — perdas no aquífero vs. perdas no poço
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StepDrawdownResult:
    """Decomposição do rebaixamento em perda de aquífero e perda de poço.

    Attributes
    ----------
    B : Coeficiente de perda no aquífero [s/m²] — parcela linear em Q.
    C : Coeficiente de perda no poço [s²/m⁵] — parcela quadrática em Q.
    r_squared : Qualidade do ajuste de s/Q = B + C·Q [-].
    B_stderr / C_stderr : Erros padrão dos coeficientes.
    n_steps : Número de degraus usados.
    Q_m3_s / drawdown_m : Dados de entrada, ordenados por Q.
    specific_drawdown : s/Q de cada degrau [s/m²].
    """

    B: float
    C: float
    r_squared: float
    B_stderr: float
    C_stderr: float
    n_steps: int
    Q_m3_s: FloatArray
    drawdown_m: FloatArray
    specific_drawdown: FloatArray

    def well_efficiency(self, Q_m3_s: float) -> float:
        r"""Eficiência do poço na vazão dada [-], :math:`BQ/(BQ + CQ^2)`.

        Valor entre 0 e 1. Abaixo de ~0,6 costuma indicar poço mal
        desenvolvido, colmatado ou com pré-filtro inadequado.
        """
        q = _require_positive_scalar(Q_m3_s, "Q_m3_s", "m³/s")
        total = self.B * q + self.C * q**2
        if total <= 0.0:
            return float("nan")
        return (self.B * q) / total

    def predicted_drawdown(self, Q_m3_s: ArrayLike) -> FloatArray:
        r"""Rebaixamento previsto :math:`s = BQ + CQ^2` [m]."""
        q = _as_float_array(Q_m3_s, "Q_m3_s")
        return self.B * q + self.C * np.square(q)


def step_drawdown_analysis(
    Q_m3_s: ArrayLike,
    drawdown_m: ArrayLike,
) -> StepDrawdownResult:
    r"""Analisa um teste de degraus pelo método de Jacob (1947).

    O rebaixamento no poço bombeado decompõe-se em duas parcelas:

    .. math:: s_w = B\,Q + C\,Q^{2}

    onde :math:`BQ` é a perda de carga no aquífero (laminar, proporcional a
    Q) e :math:`CQ^{2}` é a perda no próprio poço (turbulenta: entrada pelo
    filtro, pré-filtro colmatado, escoamento no interior do revestimento).
    Dividindo por Q, obtém-se uma reta:

    .. math:: \frac{s_w}{Q} = B + C\,Q

    de modo que o **rebaixamento específico** :math:`s/Q` plotado contra
    :math:`Q` fornece B (intercepto) e C (inclinação) por regressão linear
    simples.

    Parameters
    ----------
    Q_m3_s : ArrayLike
        Vazão estabilizada de cada degrau [m³/s], estritamente positiva.
    drawdown_m : ArrayLike
        Rebaixamento estabilizado correspondente [m], estritamente positivo.

    Returns
    -------
    StepDrawdownResult

    Raises
    ------
    RegressionError
        Menos de 3 degraus, ou todas as vazões iguais (reta indeterminada).
    InvalidGeometryError
        Alguma vazão ou rebaixamento <= 0.

    Notes
    -----
    - São necessários **ao menos 3 degraus** para a reta ter significado;
      4 a 5 é a prática usual de campo.
    - Cada degrau deve durar o suficiente para o nível estabilizar; se os
      degraus forem curtos demais, o rebaixamento ainda em queda infla o
      s/Q dos degraus finais e superestima C.
    - Um ``C`` negativo não tem sentido físico (implicaria perda de poço
      negativa) e em geral indica degraus não estabilizados ou erro de
      leitura. A função não bloqueia esse caso — reporta o valor para que
      o chamador possa diagnosticá-lo.
    - B obtido aqui é comparável a :math:`2.303/(4\pi T)\log_{10}(2.25Tt/r^2S)`
      do modelo de Theis, mas depende do tempo de duração do degrau; use-o
      para eficiência do poço, não como estimativa independente de T.

    Examples
    --------
    >>> # s = 800 Q + 20000 Q^2  (B=800, C=20000)
    >>> Q = np.array([0.005, 0.010, 0.015, 0.020])
    >>> s = 800.0 * Q + 20000.0 * Q**2
    >>> res = step_drawdown_analysis(Q, s)
    >>> round(res.B), round(res.C)
    (800, 20000)
    >>> round(res.well_efficiency(0.010), 3)
    0.8
    """
    q = _as_float_array(Q_m3_s, "Q_m3_s")
    s = _as_float_array(drawdown_m, "drawdown_m")

    if q.shape != s.shape:
        raise HydroValueError(
            f"'Q_m3_s' {q.shape} e 'drawdown_m' {s.shape} devem ter o mesmo shape."
        )
    if np.any(q <= 0.0):
        raise InvalidGeometryError("Todas as vazões devem ser estritamente positivas.")
    if np.any(s <= 0.0):
        raise InvalidGeometryError(
            "Todos os rebaixamentos devem ser estritamente positivos."
        )
    if q.size < 3:
        raise RegressionError(
            f"Apenas {q.size} degrau(s); são necessários ao menos 3 para "
            "ajustar a reta s/Q = B + C·Q."
        )

    order = np.argsort(q)
    q, s = q[order], s[order]

    if np.ptp(q) == 0.0:
        raise RegressionError(
            "Todas as vazões são iguais: variância nula em Q, regressão "
            "indeterminada. Um teste de degraus exige vazões distintas."
        )

    specific = s / q
    fit = stats.linregress(q, specific)

    # Erro padrão do intercepto: sigma_B = sigma_slope * sqrt(mean(Q^2))
    b_stderr = float(fit.stderr) * float(np.sqrt(np.mean(np.square(q))))

    return StepDrawdownResult(
        B=float(fit.intercept),
        C=float(fit.slope),
        r_squared=float(fit.rvalue) ** 2,
        B_stderr=b_stderr,
        C_stderr=float(fit.stderr),
        n_steps=int(q.size),
        Q_m3_s=q,
        drawdown_m=s,
        specific_drawdown=specific,
    )


# --------------------------------------------------------------------------
# 9) Assinaturas anômalas: dupla porosidade e achatamento tardio
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DualPorosityResult:
    """Assinatura de dupla porosidade detectada na curva semilog.

    Attributes
    ----------
    detected : Se o padrão foi identificado.
    split_index_early / split_index_late : Índices que delimitam o patamar
        de transição entre os dois trechos retilíneos.
    slope_early / slope_late : Inclinações [m/ciclo] dos trechos inicial
        (fraturas) e tardio (matriz/granular).
    transmissivity_early_m2_day / _late_m2_day : T de cada trecho.
    storativity_early / storativity_late : S de cada trecho — o inicial
        reflete o armazenamento das fraturas; o tardio, o do sistema todo.
    slope_ratio : ``slope_late / slope_early``; próximo de 1 no padrão
        clássico (retas paralelas).
    transition_slope_ratio : Inclinação do patamar sobre a média das duas
        retas; bem abaixo de 1 no padrão clássico.
    r_squared_early / r_squared_late : Qualidade de cada ajuste.
    """

    detected: bool
    split_index_early: int
    split_index_late: int
    slope_early: float
    slope_late: float
    transmissivity_early_m2_day: float
    transmissivity_late_m2_day: float
    storativity_early: float
    storativity_late: float
    slope_ratio: float
    transition_slope_ratio: float
    r_squared_early: float
    r_squared_late: float


def detect_dual_porosity(
    time_sec: ArrayLike,
    drawdown_corr: ArrayLike,
    Q: float,
    r: float,
    *,
    min_segment_points: int = 4,
    slope_ratio_tolerance: float = 0.35,
    max_transition_slope_ratio: float = 0.65,
) -> DualPorosityResult | None:
    r"""Procura a assinatura de dupla porosidade em papel semilog.

    Em um sistema de dupla porosidade (fraturas condutivas alimentadas por
    uma matriz porosa), a curva :math:`s' \times \log_{10} t` exibe um
    padrão característico: **dois trechos retilíneos aproximadamente
    paralelos**, separados por um **patamar de inclinação menor**. O trecho
    inicial reflete o esvaziamento das fraturas (S baixo); o patamar, a
    realimentação da matriz; o trecho tardio, o comportamento do sistema
    conjugado (S total).

    Esta função **não estima parâmetros de dupla porosidade** (ω, λ do
    modelo de Warren-Root/Moench), o que exigiria ajuste de curvas-tipo.
    Ela apenas identifica o padrão e ajusta cada trecho por Cooper-Jacob,
    reportando os dois pares (T, S) resultantes.

    Parameters
    ----------
    time_sec, drawdown_corr : ArrayLike
        Série do ensaio (rebaixamento já corrigido, se for o caso).
    Q, r : float
        Vazão [m³/s] e distância radial [m], para converter as inclinações
        em T e S.
    min_segment_points : int
        Mínimo de pontos em cada trecho retilíneo.
    slope_ratio_tolerance : float
        Tolerância no paralelismo: aceita ``slope_late/slope_early`` dentro
        de ``1 ± tolerance``.
    max_transition_slope_ratio : float
        A inclinação do patamar deve ser no máximo esta fração da média das
        inclinações dos dois trechos, senão não há patamar distinguível.

    Returns
    -------
    DualPorosityResult | None
        ``None`` se não houver pontos suficientes para sequer testar o
        padrão. Caso contrário, o melhor candidato encontrado — cheque
        ``.detected`` para saber se ele satisfaz os critérios.

    Notes
    -----
    Esta é uma heurística de **triagem**, não um diagnóstico definitivo. O
    mesmo padrão de duas retas pode ser produzido por uma barreira
    hidráulica, por variação de vazão durante o ensaio ou por recarga
    tardia. Confronte sempre com o perfil geológico e com o gráfico.
    """
    t = _as_float_array(time_sec, "time_sec")
    s = _as_float_array(drawdown_corr, "drawdown_corr")
    if t.shape != s.shape:
        raise HydroValueError("'time_sec' e 'drawdown_corr' devem ter o mesmo shape.")

    order = np.argsort(t)
    t, s = t[order], s[order]
    keep = t > 0.0
    t, s = t[keep], s[keep]

    n = t.size
    # early(min) + transicao(min 2) + late(min) pontos
    if n < 2 * min_segment_points + 2:
        return None

    log_t = np.log10(t)
    q = _require_positive_scalar(Q, "Q", "m³/s")
    radius = _require_positive_scalar(r, "r", "m")

    best: DualPorosityResult | None = None
    best_score = -np.inf

    for i in range(min_segment_points, n - min_segment_points - 1):
        for j in range(i + 2, n - min_segment_points + 1):
            early_t, early_s = log_t[:i], s[:i]
            trans_t, trans_s = log_t[i - 1 : j], s[i - 1 : j]
            late_t, late_s = log_t[j - 1 :], s[j - 1 :]

            if np.ptp(early_t) == 0 or np.ptp(late_t) == 0 or np.ptp(trans_t) == 0:
                continue

            fit_e = stats.linregress(early_t, early_s)
            fit_l = stats.linregress(late_t, late_s)
            fit_t = stats.linregress(trans_t, trans_s)

            slope_e, slope_l, slope_t = (
                float(fit_e.slope),
                float(fit_l.slope),
                float(fit_t.slope),
            )
            if slope_e <= 0.0 or slope_l <= 0.0:
                continue

            ratio = slope_l / slope_e
            mean_slope = 0.5 * (slope_e + slope_l)
            trans_ratio = slope_t / mean_slope if mean_slope > 0 else np.inf

            parallel = abs(ratio - 1.0) <= slope_ratio_tolerance
            has_plateau = trans_ratio <= max_transition_slope_ratio
            detected = bool(parallel and has_plateau)

            r2_e = float(fit_e.rvalue) ** 2
            r2_l = float(fit_l.rvalue) ** 2
            # Prioriza candidatos que satisfazem o padrão, depois qualidade
            score = (2.0 if detected else 0.0) + r2_e + r2_l - abs(ratio - 1.0)

            if score <= best_score:
                continue

            t_e = (LN10 * q) / (4.0 * np.pi * slope_e)
            t_l = (LN10 * q) / (4.0 * np.pi * slope_l)

            def _storativity(fit, transmissivity: float) -> float:
                exponent = -float(fit.intercept) / float(fit.slope)
                if abs(exponent) > 300.0:
                    return float("nan")
                t0 = 10.0**exponent
                return (COOPER_JACOB_COEFF * transmissivity * t0) / (radius**2)

            best_score = score
            best = DualPorosityResult(
                detected=detected,
                split_index_early=int(i),
                split_index_late=int(j),
                slope_early=slope_e,
                slope_late=slope_l,
                transmissivity_early_m2_day=t_e * SECONDS_PER_DAY,
                transmissivity_late_m2_day=t_l * SECONDS_PER_DAY,
                storativity_early=_storativity(fit_e, t_e),
                storativity_late=_storativity(fit_l, t_l),
                slope_ratio=ratio,
                transition_slope_ratio=float(trans_ratio),
                r_squared_early=r2_e,
                r_squared_late=r2_l,
            )

    return best


def detect_late_time_flattening(
    time_sec: ArrayLike,
    drawdown_corr: ArrayLike,
    *,
    tail_fraction: float = 0.4,
    flattening_threshold: float = 0.6,
) -> float | None:
    r"""Mede o achatamento do trecho final da curva semilog.

    Um trecho tardio nitidamente menos inclinado que o inicial indica que
    o sistema está recebendo água de uma fonte não prevista pelo modelo de
    Theis: **drenança** de camada adjacente (aquífero semiconfinado, ou base
    apenas semipermeável), **recarga** por corpo d'água próximo, ou
    realimentação de matriz em meio fraturado. Nesses casos Cooper-Jacob
    aplicado ao trecho tardio **superestima T**.

    Returns
    -------
    float | None
        Razão ``inclinação_tardia / inclinação_inicial``. Valores abaixo de
        ``flattening_threshold`` indicam achatamento relevante. ``None`` se
        não houver pontos suficientes.

    Examples
    --------
    >>> t = np.logspace(1, 4, 30)
    >>> s = 0.5 * np.log10(t)          # reta perfeita, sem achatamento
    >>> round(detect_late_time_flattening(t, s), 2)
    1.0
    """
    t = _as_float_array(time_sec, "time_sec")
    s = _as_float_array(drawdown_corr, "drawdown_corr")
    order = np.argsort(t)
    t, s = t[order], s[order]
    keep = t > 0.0
    t, s = t[keep], s[keep]

    n = t.size
    n_tail = int(round(n * tail_fraction))
    if n < 8 or n_tail < 3 or (n - n_tail) < 3:
        return None

    log_t = np.log10(t)
    head = stats.linregress(log_t[: n - n_tail], s[: n - n_tail])
    tail = stats.linregress(log_t[n - n_tail :], s[n - n_tail :])

    if float(head.slope) <= 0.0:
        return None
    return float(tail.slope) / float(head.slope)
