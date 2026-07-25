"""Rotinas matemáticas puras para interpretação de testes de bombeamento.

Escopo: aquíferos **livres (freáticos) granulares**, poço único, regime
transiente, vazão constante.

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
    "calculate_flow_rate",
    "jacob_unconfined_correction",
    "cooper_jacob_analysis",
    "theis_recovery_analysis",
    "predict_extrapolated_drawdown",
    "generate_synthetic_field_data",
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
#: (dessaturação relevante da porção captada do aquífero).
JACOB_RATIO_WARNING: Final[float] = 0.25

#: Número de ciclos logarítmicos, além do início da projeção
#: (``time_start_sec``), considerado prudente extrapolar sem novos dados de
#: campo (heurística de engenharia; Kruseman & de Ridder, 1994).
EXTRAPOLATION_WARNING_CYCLES: Final[float] = 1.0

#: Tempos de leitura usuais de caderneta de campo em ensaios de
#: bombeamento, em **minutos** desde o início do bombeamento, para a fase
#: de rebaixamento "lenta" (t >= 1 h). Valores típicos de manuais de campo
#: (ex. Kruseman & de Ridder, 1994); não inclui as leituras rápidas dos
#: primeiros minutos, que exigem cronometragem contínua e não são objeto
#: de geração sintética.
TYPICAL_FIELD_READING_TIMES_MIN: Final[tuple[float, ...]] = (
    60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 360.0, 480.0, 600.0, 720.0,
)


# --------------------------------------------------------------------------
# Hierarquia de exceções
# --------------------------------------------------------------------------
class HydroMathError(Exception):
    """Exceção base do módulo ``hydro_math``."""


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
    """Converte para ``float64`` contíguo, validando finitude e não-vacuidade."""
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


# --------------------------------------------------------------------------
# Estruturas de retorno (imutáveis)
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FlowRateResult:
    """Vazão de bombeamento em múltiplas unidades usuais.

    Attributes
    ----------
    q_m3_s : Vazão em m³/s (unidade de trabalho interna do módulo).
    q_m3_h : Vazão em m³/h (usual em relatórios de poço).
    q_l_s  : Vazão em L/s (usual em campo, método volumétrico).
    """

    q_m3_s: FloatArray
    q_m3_h: FloatArray
    q_l_s: FloatArray

    @property
    def mean_q_m3_s(self) -> float:
        """Média aritmética das medições, em m³/s."""
        return float(np.mean(self.q_m3_s))


@dataclass(frozen=True, slots=True)
class CooperJacobResult:
    """Parâmetros hidrodinâmicos obtidos pelo método de Cooper-Jacob.

    Attributes
    ----------
    delta_s : Inclinação da reta por ciclo logarítmico, Δs' [m/ciclo].
    intercept : Coeficiente linear do ajuste s' = a·log10(t) + b [m].
    t0_sec : Intercepto no eixo do tempo, onde s' = 0 [s].
    transmissivity_m2_s / _m2_day : Transmissividade T.
    storativity : Coeficiente de armazenamento S [-] (para aquífero livre,
        aproxima a produção específica S_y).
    conductivity_m_s / _m_day : K = T / b, apenas se ``saturated_thickness``
        for informado; caso contrário ``None``.
    r_squared : Coeficiente de determinação do ajuste linear [-].
    slope_stderr : Erro padrão da inclinação [m/ciclo].
    transmissivity_stderr_m2_s : Propagação de 1ª ordem do erro em T.
    n_points : Número de pontos efetivamente usados na regressão.
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
    n_points: int
    u_max: float
    approximation_is_valid: bool


@dataclass(frozen=True, slots=True)
class TheisRecoveryResult:
    """Transmissividade obtida pela análise de recuperação de Theis.

    Attributes
    ----------
    delta_s_residual : Inclinação Δs'' por ciclo de log10(t/t') [m/ciclo].
    intercept : Coeficiente linear do ajuste [m].
    transmissivity_m2_s / _m2_day : Transmissividade de recuperação T'.
    conductivity_m_s / _m_day : K' = T'/b, se ``saturated_thickness`` dado.
    r_squared : Coeficiente de determinação [-].
    slope_stderr : Erro padrão da inclinação [m/ciclo].
    transmissivity_stderr_m2_s : Propagação de 1ª ordem do erro em T'.
    n_points : Pontos usados na regressão.
    ratio_at_zero : Valor de (t/t') extrapolado para s'' = 0 [-]. Equivale a
        S/S' (armazenamento no bombeamento / na recuperação). Idealmente ≈ 1;
        valores > 1 são comuns em aquíferos livres com drenagem retardada.
    time_ratio_min / _max : Faixa de t/t' coberta pelo ajuste [-].
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
    n_points: int
    ratio_at_zero: float
    time_ratio_min: float
    time_ratio_max: float


# --------------------------------------------------------------------------
# 1) Vazão
# --------------------------------------------------------------------------
def calculate_flow_rate(
    volume_liters: ArrayLike,
    time_seconds: ArrayLike,
) -> FlowRateResult:
    r"""Calcula a vazão :math:`Q` pelo método volumétrico.

    .. math:: Q = \frac{V}{t}

    Entradas escalares ou vetoriais são aceitas: em campo é usual repetir a
    medição (enchimento de recipiente aferido) e trabalhar com a série toda.
    Os arrays sofrem *broadcasting* mútuo, de modo que um volume fixo pode ser
    combinado com um vetor de tempos e vice-versa.

    Parameters
    ----------
    volume_liters : ArrayLike
        Volume acumulado no recipiente aferido, em **litros [L]**.
        Deve ser estritamente positivo.
    time_seconds : ArrayLike
        Tempo de enchimento correspondente, em **segundos [s]**.
        Deve ser estritamente positivo (t = 0 implicaria vazão infinita).

    Returns
    -------
    FlowRateResult
        Estrutura imutável com ``q_m3_s`` [m³/s], ``q_m3_h`` [m³/h] e
        ``q_l_s`` [L/s], todos com o shape resultante do broadcasting.

    Raises
    ------
    HydroValueError
        Entradas vazias, com NaN/Inf ou de shapes incompatíveis.
    NonPositiveTimeError
        Algum tempo <= 0 (divisão por zero).
    InvalidGeometryError
        Algum volume <= 0.

    Examples
    --------
    >>> res = calculate_flow_rate(200.0, 18.0)
    >>> round(float(res.q_m3_h), 2)
    40.0
    >>> res_vec = calculate_flow_rate([200.0, 205.0, 198.0], [18.0, 18.5, 17.9])
    >>> res_vec.q_l_s.shape
    (3,)
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

    # Divisão segura: a positividade já foi garantida acima.
    q_m3_s = (volume / LITERS_PER_M3) / time_s
    return FlowRateResult(
        q_m3_s=q_m3_s,
        q_m3_h=q_m3_s * SECONDS_PER_HOUR,
        q_l_s=volume / time_s,
    )


# --------------------------------------------------------------------------
# 2) Correção de Jacob para aquífero livre
# --------------------------------------------------------------------------
def jacob_unconfined_correction(
    drawdown: ArrayLike,
    saturated_thickness: float | None,
    *,
    strict: bool = True,
    apply_correction: bool = True,
) -> FloatArray:
    r"""Corrige o rebaixamento medido em aquífero livre (Jacob, 1944).

    Em aquífero livre, o rebaixamento reduz a própria espessura saturada,
    tornando a transmissividade dependente de :math:`s`. A correção remove
    esse efeito de 1ª ordem, produzindo um rebaixamento equivalente ao que
    seria observado em um aquífero confinado de espessura :math:`b`:

    .. math:: s' = s - \frac{s^{2}}{2\,b}

    Somente ``s'`` (e não ``s``) deve alimentar
    :func:`cooper_jacob_analysis` ou :func:`theis_recovery_analysis`.

    Parameters
    ----------
    drawdown : ArrayLike
        Rebaixamento observado :math:`s`, em **metros [m]**, positivo para
        baixo (nível dinâmico abaixo do estático).
    saturated_thickness : float | None
        Espessura saturada **inicial** :math:`b` do aquífero, em metros [m],
        medida da base do aquífero ao nível estático. Deve ser > 0 quando
        informada. Se ``None``, a correção é ignorada (equivalente a
        ``apply_correction=False``) e os dados originais são retornados sem
        erro — útil quando ``b`` não é conhecida ou o usuário optou por não
        aplicar a correção de aquífero livre.
    strict : bool, opcional
        Se ``True`` (padrão), levanta erro quando algum ``s`` for negativo ou
        quando ``s >= b`` (dessaturação total — a correção perde sentido).
        Se ``False``, apenas ``s >= b`` é bloqueado e valores negativos são
        propagados (útil para séries com ruído em torno do zero inicial).
        Quando a correção não é aplicada (``apply_correction=False`` ou
        ``saturated_thickness is None``), a checagem ``s >= b`` não se aplica
        (não há ``b`` para comparar); a checagem de ``s`` negativo em modo
        estrito continua valendo.
    apply_correction : bool, opcional
        Se ``False`` (padrão ``True``), a correção **não é aplicada**: a
        função retorna ``s`` inalterado (:math:`s' = s`), independentemente
        de ``saturated_thickness``. Use para aquíferos confinados/semi-
        confinados, ou quando o usuário desmarcar a opção de correção na
        interface.

    Returns
    -------
    FloatArray
        Rebaixamento corrigido :math:`s'` [m], mesmo shape da entrada. Se a
        correção não for aplicada, é o próprio ``s`` (cópia em array de
        ponto flutuante).

    Raises
    ------
    InvalidGeometryError
        ``saturated_thickness`` <= 0, ou ``s >= b`` em algum ponto (somente
        quando a correção é de fato aplicada).
    HydroValueError
        Entrada vazia ou com valores não finitos; ``s`` negativo em modo
        estrito.

    Notes
    -----
    A correção é considerada confiável para :math:`s/b < 0.25`
    (Kruseman & de Ridder, 1994). Acima desse limite o fluxo vertical e a
    drenagem retardada tendem a dominar e a interpretação deve migrar para
    Neuman ou Boulton. Este módulo não bloqueia esse caso — expõe o
    diagnóstico via :data:`JACOB_RATIO_WARNING` para uso do chamador.

    Examples
    --------
    >>> float(jacob_unconfined_correction(2.0, 20.0))
    1.9
    >>> jacob_unconfined_correction([1.0, 2.0, 3.0], 20.0).round(3)
    array([0.975, 1.9  , 2.775])
    >>> jacob_unconfined_correction([1.0, 2.0, 3.0], None)
    array([1., 2., 3.])
    >>> jacob_unconfined_correction([1.0, 2.0, 3.0], 20.0, apply_correction=False)
    array([1., 2., 3.])
    """
    s = _as_float_array(drawdown, "drawdown")

    if strict and np.any(s < 0.0):
        raise HydroValueError(
            "'drawdown' possui valores negativos (elevação do nível). "
            "Verifique a referência de medição ou use strict=False."
        )

    if not apply_correction or saturated_thickness is None:
        # Sem correção de Jacob: s' = s, retornado sem exigir/validar b.
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

    ou seja, :math:`s'` é linear em :math:`\log_{10}(t)`. Da regressão
    :math:`s' = \Delta s' \log_{10}(t) + c` obtém-se:

    .. math::
        T = \frac{2.303\,Q}{4\pi\,\Delta s'}
        \qquad
        S = \frac{2.25\,T\,t_0}{r^{2}},
        \qquad t_0 = 10^{-c/\Delta s'}

    Parameters
    ----------
    time_sec : ArrayLike
        Tempo decorrido desde o início do bombeamento, em **segundos [s]**,
        estritamente positivo.
    drawdown_corr : ArrayLike
        Rebaixamento **já corrigido** :math:`s'` [m], obtido de
        :func:`jacob_unconfined_correction`. Mesmo comprimento de ``time_sec``.
    Q : float
        Vazão constante de bombeamento, em **m³/s**. Use
        ``calculate_flow_rate(...).mean_q_m3_s`` para convertê-la.
    r : float
        Distância radial do poço de observação ao poço bombeado, em **metros
        [m]**. Para análise no próprio poço bombeado use o raio do poço,
        ciente de que ``S`` resultará contaminado por perdas de carga.
    t_start_sec, t_end_sec : float | None, opcional
        Janela temporal do trecho retilíneo a ajustar [s]. ``None`` = sem
        limite. Descartar os tempos iniciais é a prática usual, pois neles
        ``u`` ainda é grande e o armazenamento no poço domina.
    mask : ArrayLike | None, opcional
        Máscara booleana explícita de seleção de pontos, combinada por
        conjunção lógica com a janela temporal.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m]; se informada, retorna também
        :math:`K = T/b`.

    Returns
    -------
    CooperJacobResult
        Parâmetros ajustados, métricas de qualidade e o diagnóstico de
        validade da aproximação logarítmica (``u_max``).

    Raises
    ------
    NonPositiveTimeError
        Algum tempo <= 0 no trecho selecionado (``log10`` indefinido).
    InvalidGeometryError
        ``Q`` <= 0, ``r`` <= 0 ou ``saturated_thickness`` <= 0.
    HydroValueError
        Shapes incompatíveis, dados não finitos ou máscara inválida.
    RegressionError
        Menos de 3 pontos no trecho, tempos todos iguais (variância nula em
        ``log10 t``) ou inclinação :math:`\Delta s' \le 0` — que implicaria
        transmissividade negativa/infinita.

    Notes
    -----
    ``u_max`` é avaliado a posteriori com o próprio par (T, S) ajustado, no
    menor tempo do trecho. Se ``approximation_is_valid`` for ``False``,
    reajuste descartando os primeiros pontos.
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

    # T = 2.303 Q / (4 pi Δs')
    transmissivity = (LN10 * q) / (4.0 * np.pi * delta_s)

    # t0: raiz da reta (s' = 0). Expoente limitado para evitar overflow.
    exponent = -intercept / delta_s
    if abs(exponent) > 300.0:
        raise RegressionError(
            f"Intercepto temporal fora da faixa numérica (10^{exponent:.1f} s). "
            "O ajuste provavelmente não representa o trecho retilíneo."
        )
    t0 = float(10.0**exponent)

    # S = 2.25 T t0 / r²
    storativity = (COOPER_JACOB_COEFF * transmissivity * t0) / (radius**2)

    # Diagnóstico: u = r² S / (4 T t), máximo no menor tempo do trecho.
    t_min = float(np.min(t_fit))
    u_max = (radius**2 * storativity) / (4.0 * transmissivity * t_min)

    slope_stderr = float(fit.stderr)
    # Propagação de 1ª ordem: T ∝ 1/Δs'  =>  σ_T/T = σ_Δs'/Δs'
    t_stderr = transmissivity * (slope_stderr / delta_s)

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
        n_points=int(t_fit.size),
        u_max=float(u_max),
        approximation_is_valid=bool(u_max < U_VALIDITY_THRESHOLD),
    )


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

    Após o desligamento da bomba, sobrepõe-se ao poço real um poço de
    recarga imaginário de mesma vazão. O rebaixamento residual resulta:

    .. math::
        s'' = \frac{2.303\,Q}{4\pi T}\,\log_{10}\!\left(\frac{t}{t'}\right)

    com :math:`t` = tempo desde o **início do bombeamento** e :math:`t'` =
    tempo desde a **parada da bomba**, de modo que :math:`t = t_p + t'`.
    A reta em papel semilog fornece diretamente

    .. math:: T = \frac{2.303\,Q}{4\pi\,\Delta s''}

    Parameters
    ----------
    time_pump_sec : ArrayLike
        Duração total do bombeamento :math:`t_p`, em **segundos [s]**.
        Normalmente escalar; um array é aceito (sofre broadcasting com
        ``time_recovery_sec``) para séries com paradas distintas.
    time_recovery_sec : ArrayLike
        Tempo desde a parada da bomba :math:`t'` [s], estritamente positivo
        (em :math:`t' = 0` a razão :math:`t/t'` diverge).
    residual_drawdown : ArrayLike
        Rebaixamento residual :math:`s''` [m] — diferença entre o nível
        medido durante a recuperação e o nível estático original. Em aquífero
        livre, aplique antes :func:`jacob_unconfined_correction`.
    Q : float
        Vazão constante mantida durante o bombeamento, em **m³/s**.
    ratio_min, ratio_max : float | None, opcional
        Janela de seleção em :math:`t/t'` [-]. Descartar razões muito altas
        (início da recuperação) é usual, pois nelas o armazenamento no poço
        ainda domina.
    mask : ArrayLike | None, opcional
        Máscara booleana explícita, combinada por conjunção com a janela.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m] para reportar :math:`K' = T'/b`.

    Returns
    -------
    TheisRecoveryResult
        Transmissividade de recuperação e métricas do ajuste.

    Raises
    ------
    NonPositiveTimeError
        ``t' <= 0`` (divisão por zero) ou ``t = t_p + t' <= 0``.
    InvalidGeometryError
        ``Q`` <= 0 ou ``saturated_thickness`` <= 0.
    HydroValueError
        Shapes incompatíveis ou dados não finitos.
    RegressionError
        Menos de 3 pontos, razões :math:`t/t'` todas iguais, ou inclinação
        :math:`\Delta s'' \le 0`.

    Notes
    -----
    O método **não fornece S** isoladamente: apenas a razão entre o
    armazenamento no bombeamento e na recuperação, dada pelo valor de
    :math:`t/t'` extrapolado para :math:`s'' = 0` (``ratio_at_zero``).
    Em aquíferos livres com drenagem retardada é comum obter
    :math:`T_{recuperação} > T_{rebaixamento}`; a comparação entre as duas
    estimativas é um bom indicador da qualidade do ensaio.
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
    t_total = t_p_b + t_rec_b  # t = t_p + t'

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
        n_points=int(ratio_fit.size),
        ratio_at_zero=ratio_at_zero,
        time_ratio_min=float(np.min(ratio_fit)),
        time_ratio_max=float(np.max(ratio_fit)),
    )


# --------------------------------------------------------------------------
# 5) Extrapolação temporal da reta de Cooper-Jacob
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

    Projeta a reta já ajustada por :func:`cooper_jacob_analysis` para além
    do período efetivamente medido, assumindo que as condições do ensaio
    (vazão constante, aquífero de extensão infinita, sem recarga ou efeitos
    de contorno) permanecem inalteradas até o instante alvo:

    .. math::
        s'_{\text{proj}}(t) = \Delta s' \cdot
            \log_{10}\!\left(\frac{t}{t_0}\right)

    onde :math:`\Delta s'` (``delta_s``) e :math:`t_0` (``t0_sec``) são,
    respectivamente, a inclinação por ciclo logarítmico e o intercepto
    temporal (:math:`s' = 0`) da regressão de Cooper-Jacob — ver
    :class:`CooperJacobResult`.

    Se ``saturated_thickness`` (espessura saturada :math:`b`) for informada,
    o rebaixamento corrigido projetado :math:`s'_{\text{proj}}` é convertido
    de volta ao rebaixamento real no poço, invertendo a correção de Jacob
    (:func:`jacob_unconfined_correction`, :math:`s' = s - s^2/2b`):

    .. math::
        s_{\text{proj}} = b - \sqrt{b^{2} - 2\,b\,s'_{\text{proj}}}

    (raiz negativa descartada, pois fisicamente :math:`s < b`).

    Parameters
    ----------
    delta_s : float
        Inclinação :math:`\Delta s'` da reta de Cooper-Jacob, em **metros
        por ciclo logarítmico [m]** (atributo ``delta_s`` de
        :class:`CooperJacobResult`). Deve ser estritamente positivo.
    t0_sec : float
        Intercepto temporal :math:`t_0` da reta, em **segundos [s]**
        (atributo ``t0_sec`` de :class:`CooperJacobResult`). Deve ser
        estritamente positivo.
    time_start_sec : float
        Instante em que a projeção **começa**, em **segundos [s]**, contado
        no mesmo referencial de ``t0_sec`` (i.e., desde o início do
        bombeamento). Tipicamente o maior tempo efetivamente medido no
        ensaio (ou o próprio ``t0_sec``), para que a projeção continue de
        onde os dados reais terminam. Deve ser estritamente positivo.
    target_hours : float, opcional
        Horizonte final da extrapolação, em **horas**, contado no mesmo
        referencial de ``t0_sec`` (padrão: 12 h). Deve corresponder a um
        tempo, em segundos, estritamente maior que ``time_start_sec``.
    step_minutes : float, opcional
        Passo de discretização do vetor de tempo, em **minutos** (padrão:
        5 min). Deve ser estritamente positivo.
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m]. Se informada, aplica a correção
        inversa de Jacob e a coluna retornada passa a representar o
        rebaixamento real no poço; caso contrário, representa o
        rebaixamento corrigido :math:`s'`.

    Returns
    -------
    pandas.DataFrame
        Uma linha por instante do vetor de tempo gerado, com as colunas:

        - ``tempo_min`` : tempo desde o início do bombeamento [min].
        - ``rebaixamento_projetado_m`` : rebaixamento projetado [m] — real,
          se ``saturated_thickness`` foi informado; corrigido, caso
          contrário.
        - ``tipo_dado`` : constante de texto ``'Projeção'``, para
          diferenciar estes pontos de dados observados ao concatenar com
          séries medidas (ex.: ``pd.concat([df_medido, df_projetado])``).

    Raises
    ------
    InvalidGeometryError
        ``delta_s`` <= 0; ``saturated_thickness`` <= 0; ``target_hours``
        não finito ou <= 0; ou correção inversa de Jacob indefinida
        (:math:`b^{2} < 2\,b\,s'_{\text{proj}}` em algum instante — a
        projeção supera a dessaturação total do aquífero).
    NonPositiveTimeError
        ``t0_sec`` <= 0; ``time_start_sec`` <= 0; ``step_minutes`` <= 0; ou
        o horizonte final (``target_hours`` convertido em segundos) não é
        estritamente maior que ``time_start_sec``.

    Notes
    -----
    **Limites de validade da extrapolação** — leia antes de usar os
    resultados para decisões de projeto:

    - A extrapolação **não constitui uma nova medição**: é a continuação da
      mesma reta ajustada aos dados observados, sob a premissa de que nada
      muda no sistema (vazão constante, sem recarga, sem limites físicos do
      aquífero, sem interferência de outros poços) durante todo o horizonte
      de ``target_hours``.
    - É prática aceita extrapolar com razoável confiança até
      aproximadamente :data:`EXTRAPOLATION_WARNING_CYCLES` ciclo(s)
      logarítmico(s) além de ``time_start_sec`` (Kruseman & de Ridder,
      1994). Além disso, o erro cresce rapidamente e de forma não
      quantificável por este método puramente extrapolativo. Um
      ``UserWarning`` é emitido quando esse limite é ultrapassado.
    - A reta de Cooper-Jacob pressupõe :math:`u = r^{2}S/(4Tt) < 0.01`;
      como :math:`u` decresce com :math:`t`, a aproximação logarítmica
      tende a permanecer numericamente válida na extrapolação — mas isso
      **não** garante que o aquífero real continue se comportando como
      infinito e homogêneo por todo o horizonte projetado.
    - Em aquíferos livres, a correção inversa de Jacob amplia incertezas
      pequenas em :math:`s'` conforme :math:`s/b` se aproxima de
      :data:`JACOB_RATIO_WARNING` (0.25); acima desse limite, tanto a
      correção direta quanto a inversa perdem confiabilidade física.
    - Trate o resultado como **estimativa indicativa**, não como previsão
      determinística, e reajuste a projeção assim que novos dados de campo
      estiverem disponíveis.

    Examples
    --------
    >>> df = predict_extrapolated_drawdown(
    ...     delta_s=0.5, t0_sec=10.0, time_start_sec=3600.0,
    ...     target_hours=2.0, step_minutes=30.0,
    ... )
    >>> list(df.columns)
    ['tempo_min', 'rebaixamento_projetado_m', 'tipo_dado']
    >>> bool((df["tipo_dado"] == "Projeção").all())
    True
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

    # Vetor de tempo: de t_start a t_end, passo step_s, garantindo que o
    # instante final (t_end) esteja sempre incluído mesmo que não caia
    # exatamente sobre a grade do passo.
    t = np.arange(t_start, t_end, step_s, dtype=np.float64)
    if t.size == 0 or not np.isclose(t[-1], t_end):
        t = np.append(t, t_end)

    _require_positive_time(t, "t (vetor de tempo gerado)")

    s_proj_corr = delta * np.log10(t / t0)

    if b is not None:
        discriminant = b**2 - 2.0 * b * s_proj_corr
        if np.any(discriminant < 0.0):
            raise InvalidGeometryError(
                "Correção inversa de Jacob indefinida: o rebaixamento "
                "corrigido projetado excede a dessaturação total do "
                f"aquífero (b={b:.3f} m) em algum instante do horizonte "
                "solicitado. Reduza 'target_hours' ou revise "
                "'saturated_thickness'."
            )
        s_proj = b - np.sqrt(discriminant)
    else:
        s_proj = s_proj_corr

    # Aviso: extrapolação além do limiar de ciclos logarítmicos aceitos
    # sem novos dados de campo, contado a partir de time_start_sec.
    if t_end / t_start > 10.0**EXTRAPOLATION_WARNING_CYCLES:
        warnings.warn(
            f"Horizonte de extrapolação (target_hours={target_h:.3g} h) "
            f"ultrapassa {EXTRAPOLATION_WARNING_CYCLES:.0f} ciclo(s) "
            "logarítmico(s) além de 'time_start_sec'. Resultados distantes "
            "do início da projeção são progressivamente menos confiáveis "
            "(ver seção Notes da docstring).",
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
# 6) Dados sintéticos de campo (discretos, grade de caderneta)
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

    Diferente de :func:`predict_extrapolated_drawdown` (que gera um vetor
    denso, em passo fixo, próprio para traçar a curva contínua projetada),
    esta função gera **pontos discretos** nos instantes em que uma leitura
    seria de fato registrada em uma caderneta de campo — permitindo montar
    um dataset sintético com a **mesma estrutura de colunas** dos dados
    reais, para concatenação direta (``pd.concat``) e produção de um
    dataset único e contínuo (real + sintético).

    O rebaixamento de cada ponto usa a mesma reta de Cooper-Jacob de
    :func:`predict_extrapolated_drawdown`:

    .. math::
        s'(t) = \Delta s' \cdot \log_{10}\!\left(\frac{t}{t_0}\right)

    com conversão opcional para o rebaixamento real via correção inversa de
    Jacob (ver :func:`predict_extrapolated_drawdown`, seção *Notes*, para os
    mesmos limites de validade — eles se aplicam integralmente aqui).

    Parameters
    ----------
    delta_s : float
        Inclinação :math:`\Delta s'` da reta de Cooper-Jacob, em **metros
        por ciclo logarítmico [m]** (``CooperJacobResult.delta_s``). Deve
        ser estritamente positivo.
    t0_sec : float
        Intercepto temporal :math:`t_0` da reta, em **segundos [s]**
        (``CooperJacobResult.t0_sec``). Deve ser estritamente positivo.
    Q_mean : float
        Vazão média do ensaio, em **m³/s** (mesma unidade de trabalho de
        :func:`calculate_flow_rate`). Usada apenas para preencher a coluna
        ``vazao_m3_h`` do dataset sintético (constante em todas as linhas);
        não entra na equação do rebaixamento, já embutida em ``delta_s``.
        Deve ser estritamente positivo.
    last_time_min : float
        Tempo do último ponto **real** registrado, em **minutos** desde o
        início do bombeamento (mesmo referencial de ``t0_sec``). Somente
        leituras padrão de caderneta posteriores a este instante entram no
        dataset sintético. Deve ser >= 0.
    target_hours : float, opcional
        Horizonte final do dataset sintético, em **horas** desde o início
        do bombeamento (padrão: 12 h). Se a grade de leituras não incluir
        esse instante exatamente, ele é adicionado como último ponto.
    static_level_m : float | None, opcional
        Nível estático (profundidade do nível d'água em repouso, antes do
        bombeamento), em **metros [m]**. Se informado, a coluna ``nd_m``
        (nível dinâmico) é calculada como
        ``nd_m = static_level_m + rebaixamento_m``. Se omitido, ``nd_m`` é
        retornada como ``NaN`` (o chamador preenche externamente, se
        necessário).
    saturated_thickness : float | None, opcional
        Espessura saturada :math:`b` [m]. Se informada, aplica a correção
        inversa de Jacob e ``rebaixamento_m`` passa a representar o
        rebaixamento real no poço; caso contrário, representa o
        rebaixamento corrigido :math:`s'`.
    reading_times_min : ArrayLike | None, opcional
        Grade de tempos de leitura a considerar, em minutos. Por padrão,
        usa :data:`TYPICAL_FIELD_READING_TIMES_MIN`. Informe para adequar
        à rotina de campo específica do ensaio.

    Returns
    -------
    pandas.DataFrame
        Uma linha por leitura sintética, com as colunas:

        - ``tempo_min`` : tempo desde o início do bombeamento [min].
        - ``nd_m`` : nível dinâmico [m], ou ``NaN`` se ``static_level_m``
          não foi informado.
        - ``rebaixamento_m`` : rebaixamento [m] — real, se
          ``saturated_thickness`` foi informado; corrigido, caso contrário.
        - ``vazao_m3_h`` : vazão constante do ensaio, em m³/h.
        - ``tipo_dado`` : constante de texto ``'Sintético'``, para
          diferenciar estes pontos dos dados efetivamente medidos ao
          concatenar (ex.: ``pd.concat([df_real, df_sintetico])``).

    Raises
    ------
    InvalidGeometryError
        ``delta_s`` <= 0; ``Q_mean`` <= 0; ``saturated_thickness`` <= 0;
        ``target_hours`` não finito ou <= 0; ou correção inversa de Jacob
        indefinida (rebaixamento projetado excede a dessaturação total do
        aquífero).
    NonPositiveTimeError
        ``t0_sec`` <= 0; ``last_time_min`` < 0; ou o horizonte final
        (``target_hours`` convertido) não é estritamente maior que
        ``last_time_min``.
    HydroValueError
        ``reading_times_min`` vazio, não finito, ou sem nenhum ponto
        estritamente positivo após conversão para segundos.

    Notes
    -----
    Esta função herda **todos** os limites de validade descritos em
    :func:`predict_extrapolated_drawdown` (seção *Notes*): os pontos
    gerados não são medições, mas a continuação teórica da mesma reta
    ajustada, sob a premissa de vazão constante e ausência de recarga,
    limites de aquífero ou interferências externas até ``target_hours``.
    Trate o resultado como dado de **preenchimento/planejamento** (por
    exemplo, para compor um gráfico contínuo até 12h em um relatório
    preliminar), nunca como substituto de leituras de campo reais.

    Examples
    --------
    >>> df = generate_synthetic_field_data(
    ...     delta_s=0.5, t0_sec=10.0, Q_mean=0.01, last_time_min=45.0,
    ...     target_hours=2.0,
    ... )
    >>> list(df.columns)
    ['tempo_min', 'nd_m', 'rebaixamento_m', 'vazao_m3_h', 'tipo_dado']
    >>> bool((df["tempo_min"] > 45.0).all())
    True
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

    if b is not None:
        discriminant = b**2 - 2.0 * b * s_corr
        if np.any(discriminant < 0.0):
            raise InvalidGeometryError(
                "Correção inversa de Jacob indefinida: o rebaixamento "
                "corrigido projetado excede a dessaturação total do "
                f"aquífero (b={b:.3f} m) em algum instante do horizonte "
                "solicitado. Reduza 'target_hours' ou revise "
                "'saturated_thickness'."
            )
        s_real = b - np.sqrt(discriminant)
    else:
        s_real = s_corr

    nd_m = (
        static_level_m + s_real
        if static_level_m is not None
        else np.full_like(s_real, np.nan)
    )

    if target_min / max(last_t, t0 / 60.0) > 10.0**EXTRAPOLATION_WARNING_CYCLES:
        warnings.warn(
            f"Horizonte do dataset sintético (target_hours={target_h:.3g} h) "
            f"ultrapassa {EXTRAPOLATION_WARNING_CYCLES:.0f} ciclo(s) "
            "logarítmico(s) além de 'last_time_min'. Os pontos mais "
            "distantes do último dado real são progressivamente menos "
            "confiáveis (ver seção Notes da docstring).",
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
# Seleção de trecho (helper compartilhado pelos itens 3 e 4)
# --------------------------------------------------------------------------
def _build_selection(
    axis: FloatArray,
    mask: ArrayLike | None,
    lower: float | None,
    upper: float | None,
) -> npt.NDArray[np.bool_]:
    """Combina janela [lower, upper] e máscara booleana em um único seletor."""
    selection = np.ones(axis.shape, dtype=bool)

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
    if lower is not None and upper is not None and float(lower) > float(upper):
        raise HydroValueError(
            f"Janela inválida: limite inferior ({lower}) > superior ({upper})."
        )

    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != axis.shape:
            raise HydroValueError(
                f"'mask' {user_mask.shape} deve ter o mesmo shape dos dados "
                f"{axis.shape}."
            )
        selection &= user_mask

    return selection
