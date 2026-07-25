"""
plotting.py
===========

Funções de plotagem para ensaios de aquífero (testes de bombeamento) em
hidrogeologia, com qualidade adequada para publicação científica.

Métodos contemplados
--------------------
- ``plot_hydro_analysis``  : gráfico principal do ensaio, separando de forma
  inequívoca as **medições de campo** da **projeção teórica** (Cooper-Jacob),
  com sombreamento do intervalo de extrapolação e painel de resumo
  hidrodinâmico (Q, T, S, s_12h).
- ``plot_cooper_jacob``    : ajuste semi-log da fase de rebaixamento
  (aproximação logarítmica de Theis), com T e S.
- ``plot_recovery_theis``  : análise da fase de recuperação (rebaixamento
  residual vs. razão de tempos t/t').

Convenções de unidades (SI hidrogeológico usual)
------------------------------------------------
    t   : minutos (eixo X, escala log10)
    s   : metros (eixo Y, invertido — zero na superfície)
    Q   : m³/h  (rótulo); convertido internamente quando necessário
    T   : m²/dia
    S   : adimensional
    r   : metros (distância radial ao poço de bombeamento)

Rigor científico
----------------
A projeção teórica **nunca** é desenhada com marcadores preenchidos, para não
sugerir medição real; usa traço distinto, cor dessaturada e uma faixa sombreada
que delimita explicitamente a região extrapolada (fora do suporte amostral).

Dependências:
    matplotlib, seaborn, numpy, pandas
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ArrayLike = Union[Sequence[float], np.ndarray]

__all__ = [
    "plot_hydro_analysis",
    "plot_cooper_jacob",
    "plot_recovery_theis",
    "cooper_jacob_drawdown",
    "build_cooper_jacob_projection",
    "apply_scientific_style",
    "PALETTE",
]

# --------------------------------------------------------------------------- #
# Paleta e estilo                                                              #
# --------------------------------------------------------------------------- #

#: Paleta única do módulo — garante coerência entre todas as figuras.
PALETTE = {
    "observed": "#1f5fa8",      # azul petróleo — dados reais
    "observed_edge": "#0d2b4d",
    "projection": "#4a4a4a",    # cinza escuro — modelo teórico
    "fit": "#c0392b",           # vermelho tijolo — reta de ajuste
    "recovery": "#2e8b57",      # verde — recuperação
    "extrapolation": "#7f8c9a",  # faixa de extrapolação
    "annotation": "#2c3e50",
}

_STYLE_APPLIED = False


def apply_scientific_style() -> None:
    """
    Aplica o tema visual do módulo (grid suave, tipografia de artigo,
    300 DPI na exportação). Idempotente: chamadas repetidas não acumulam.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
        },
    )
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.axisbelow": True,
            "legend.fontsize": 9.5,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.4",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.default": "regular",
        }
    )
    _STYLE_APPLIED = True


apply_scientific_style()


# --------------------------------------------------------------------------- #
# Utilitários internos                                                         #
# --------------------------------------------------------------------------- #

def _ensure_dir(filepath: str) -> None:
    """Garante que o diretório de destino do arquivo exista."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _fmt_sci(value: float, digits: int = 3) -> str:
    """Formata em notação científica LaTeX: 2.30 × 10⁻⁴."""
    if value is None or not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    expoente = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** expoente
    if -2 <= expoente <= 3:  # faixa legível em notação decimal
        return f"{value:,.{digits}g}".replace(",", " ")
    return rf"{mantissa:.{digits - 1}f}\times 10^{{{expoente}}}"


def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    """Formata número em ponto fixo, com travessão para ausência de valor."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _resolve_columns(
    df: pd.DataFrame,
    col_t: Optional[str],
    col_s: Optional[str],
    origem: str,
) -> Tuple[str, str]:
    """
    Descobre as colunas de tempo e rebaixamento de um DataFrame.

    Aceita nomes explícitos; caso contrário procura aliases usuais e, em
    último recurso, usa as duas primeiras colunas numéricas.
    """
    if col_t is not None and col_s is not None:
        faltantes = [c for c in (col_t, col_s) if c not in df.columns]
        if faltantes:
            raise KeyError(
                f"Colunas {faltantes} ausentes em `{origem}`. "
                f"Disponíveis: {list(df.columns)}"
            )
        return col_t, col_s

    alias_t = ("tempo_min", "t_min", "tempo", "time_min", "t", "time")
    alias_s = (
        "rebaixamento_m", "rebaixamento", "s", "s_m",
        "drawdown", "drawdown_m", "nivel_dinamico",
    )
    lower = {c.lower(): c for c in df.columns}

    achado_t = col_t or next((lower[a] for a in alias_t if a in lower), None)
    achado_s = col_s or next((lower[a] for a in alias_s if a in lower), None)

    if achado_t is None or achado_s is None:
        numericas = df.select_dtypes(include="number").columns.tolist()
        if len(numericas) < 2:
            raise ValueError(
                f"`{origem}` precisa de ao menos duas colunas numéricas "
                f"(tempo e rebaixamento). Encontrado: {list(df.columns)}"
            )
        achado_t = achado_t or numericas[0]
        achado_s = achado_s or numericas[1]

    return achado_t, achado_s


def _decade_formatter(value: float, _pos: int = 0) -> str:
    """
    Rótulo de década legível: 1, 10, 100, 1000 na faixa usual de ensaios;
    notação científica fora dela (convenção de hidrogeologia aplicada).
    """
    if value <= 0:
        return ""
    expoente = int(np.round(np.log10(value)))
    if -2 <= expoente <= 4:
        return f"{value:,.0f}".replace(",", " ") if value >= 1 else f"{value:g}"
    return rf"$10^{{{expoente}}}$"


def _normaliza_rotulo(texto: object) -> str:
    """Normaliza rótulo de origem: minúsculas, sem acentos e sem espaços."""
    import unicodedata

    bruto = str(texto).strip().lower()
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", bruto)
        if not unicodedata.combining(c)
    )
    return sem_acento.replace(" ", "").replace("-", "").replace("_", "")


def _setup_semilog_axis(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Configura eixo X log10 com décadas marcadas e subdivisões suaves."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_decade_formatter))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.22)


def _invert_y_once(ax: Axes) -> None:
    """Inverte o eixo Y (zero no topo = superfície) sem duplo flip."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _textbox(ax: Axes, x: float, y: float, texto: str, va: str = "bottom",
             ha: str = "left", fontsize: float = 10.5) -> None:
    """Caixa de texto padronizada em coordenadas de eixo."""
    ax.text(
        x, y, texto,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        linespacing=1.6,
        color=PALETTE["annotation"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="0.35", alpha=0.92, linewidth=0.8),
        zorder=6,
    )


# --------------------------------------------------------------------------- #
# Modelo analítico — Cooper-Jacob                                              #
# --------------------------------------------------------------------------- #

def cooper_jacob_drawdown(
    t_min: Union[float, ArrayLike],
    Q_m3h: float,
    T_m2dia: float,
    S: float,
    r_m: float,
) -> Union[float, np.ndarray]:
    r"""
    Rebaixamento teórico pela aproximação logarítmica de Cooper-Jacob:

    .. math::
        s = \frac{2{,}30\,Q}{4\pi T}\,
            \log_{10}\!\left(\frac{2{,}25\,T\,t}{r^{2} S}\right)

    Parâmetros
    ----------
    t_min : float ou array_like
        Tempo desde o início do bombeamento (minutos).
    Q_m3h : float
        Vazão de bombeamento (m³/h).
    T_m2dia : float
        Transmissividade (m²/dia).
    S : float
        Coeficiente de armazenamento (adimensional).
    r_m : float
        Distância radial do ponto de observação ao poço (m).

    Retorna
    -------
    float ou np.ndarray
        Rebaixamento (m). Valores no ramo inválido do modelo
        (argumento do log ≤ 1) retornam ``nan``.

    Notas
    -----
    A aproximação exige :math:`u = r^2 S / (4 T t) < 0{,}01`; para tempos
    muito curtos o resultado é fisicamente inválido e a função devolve
    ``nan`` em vez de valores negativos espúrios.
    """
    t_dias = np.asarray(t_min, dtype=float) / 1440.0
    Q_dia = Q_m3h * 24.0

    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 2.25 * T_m2dia * t_dias / (r_m ** 2 * S)
        s = (2.30 * Q_dia / (4.0 * np.pi * T_m2dia)) * np.log10(arg)
        s = np.where(arg > 1.0, s, np.nan)

    return float(s) if np.ndim(s) == 0 else s


def build_cooper_jacob_projection(
    t_inicio_min: float,
    t_fim_min: float,
    Q_m3h: float,
    T_m2dia: float,
    S: float,
    r_m: float,
    n_pontos: int = 120,
    col_t: str = "tempo_min",
    col_s: str = "rebaixamento_m",
) -> pd.DataFrame:
    """
    Gera o DataFrame da curva teórica de Cooper-Jacob em espaçamento
    logarítmico entre ``t_inicio_min`` e ``t_fim_min``.
    """
    if t_inicio_min <= 0:
        raise ValueError("`t_inicio_min` deve ser > 0 (escala logarítmica).")

    t = np.logspace(np.log10(t_inicio_min), np.log10(t_fim_min), n_pontos)
    s = cooper_jacob_drawdown(t, Q_m3h, T_m2dia, S, r_m)
    return pd.DataFrame({col_t: t, col_s: s}).dropna()


# --------------------------------------------------------------------------- #
# 1. Análise integrada do ensaio (função principal)                            #
# --------------------------------------------------------------------------- #

def plot_hydro_analysis(
    df_real: pd.DataFrame,
    df_projection: Optional[pd.DataFrame] = None,
    show_projection: bool = True,
    title: str = "Análise do Teste de Bombeamento",
    *,
    # --- parâmetros hidrodinâmicos exibidos no painel de resumo ---
    Q: Optional[float] = None,
    T: Optional[float] = None,
    S: Optional[float] = None,
    s_12h: Optional[float] = None,
    r: Optional[float] = None,
    poco: Optional[str] = None,
    # --- controle de colunas ---
    col_t: Optional[str] = None,
    col_s: Optional[str] = None,
    col_origem: Optional[str] = None,
    origens_sinteticas: Sequence[str] = (
        "sintetico", "sintético", "sintetica", "sintética",
        "modelado", "simulado", "projetado", "interpolado", "estimado",
    ),
    col_t_proj: Optional[str] = None,
    col_s_proj: Optional[str] = None,
    # --- estética / saída ---
    summary: str = "outside",     # "outside" | "panel" | "box" | "none"
    legend_loc: str = "outside",  # "outside" | qualquer loc do Matplotlib
    xlabel: str = "Tempo desde o início do bombeamento, $t$ (min)",
    ylabel: str = "Rebaixamento, $s$ (m)",
    mark_12h: bool = True,
    figsize: Tuple[float, float] = (10.5, 6.0),
    save_path: Optional[str] = "outputs/hydro_analysis.png",
    dpi: int = 300,
) -> Figure:
    r"""
    Gráfico principal do ensaio de bombeamento, com separação inequívoca
    entre **medições de campo** e **projeção teórica**.

    Elementos gráficos
    ------------------
    - Dados reais: círculos preenchidos com borda escura, unidos por linha
      contínua (interpolação visual entre medições consecutivas).
    - Projeção: linha tracejada em cor dessaturada, **sem marcadores
      preenchidos**, de modo a não simular observação de campo.
    - Faixa sombreada delimitando o intervalo de extrapolação (todo instante
      posterior à última medição disponível).
    - Eixo X em :math:`\log_{10}(t)` com grid nas décadas; eixo Y invertido
      (zero no topo, representando a superfície / nível estático).
    - Painel lateral de resumo com Q, T, S e :math:`s_{12h}`.

    Parâmetros
    ----------
    df_real : pandas.DataFrame
        Medições de campo. Colunas de tempo (min) e rebaixamento (m)
        são detectadas automaticamente (``tempo_min``/``rebaixamento_m`` e
        aliases usuais) ou informadas via ``col_t``/``col_s``.
    df_projection : pandas.DataFrame, opcional
        Curva teórica já calculada. Se ``None`` e ``show_projection=True``,
        a curva é gerada internamente por Cooper-Jacob desde que ``Q``,
        ``T``, ``S`` e ``r`` sejam fornecidos.
    show_projection : bool, padrão True
        Liga/desliga a camada teórica e a faixa de extrapolação.
    title : str
        Título do gráfico.
    Q : float, opcional
        Vazão de bombeamento (m³/h).
    T : float, opcional
        Transmissividade (m²/dia).
    S : float, opcional
        Coeficiente de armazenamento (adimensional).
    s_12h : float, opcional
        Rebaixamento estimado em 12 h (m). Se omitido e houver
        ``Q``, ``T``, ``S`` e ``r``, é calculado por Cooper-Jacob; se ainda
        assim indisponível, é interpolado da projeção quando esta cobre
        720 min.
    r : float, opcional
        Distância radial ao poço de bombeamento (m). Necessária para o
        cálculo analítico.
    poco : str, opcional
        Identificação do poço/piezômetro, exibida como subtítulo.
    col_origem : str, opcional
        Coluna do **dataset unificado** que identifica a procedência de cada
        ponto (ex.: ``"origem"`` com valores ``"medido"``/``"sintetico"``).
        Quando presente, todos os pontos são desenhados na **mesma série**,
        com o mesmo marcador circular, o mesmo tamanho e a mesma linha
        contínua ligando-os — a única diferença é o preenchimento: sólido
        para medição de campo, vazado para ponto sintético/interpolado,
        com entrada própria na legenda. Isso preserva a leitura de curva
        única sem apresentar valor gerado como dado observado.
    origens_sinteticas : sequência de str
        Rótulos (sem acento/caixa consideradas) tratados como sintéticos.
    summary : {"outside", "panel", "box", "none"}
        Resumo hidrodinâmico. ``"outside"`` (padrão) ancora a caixa **fora
        da área de plotagem**, no canto superior direito, via
        ``bbox_to_anchor``; nunca sobrepõe legenda nem pontos.
    legend_loc : str
        ``"outside"`` (padrão) posiciona a legenda abaixo do eixo, em
        colunas; qualquer valor aceito por ``Axes.legend(loc=...)`` também
        funciona.
    mark_12h : bool
        Marca o instante de 12 h (720 min) com linha vertical e anotação.
    save_path : str ou None
        Caminho do PNG de saída. ``None`` desativa a gravação.

    Retorna
    -------
    matplotlib.figure.Figure

    Exemplo
    -------
    >>> fig = plot_hydro_analysis(
    ...     df_real=df_campo,
    ...     show_projection=True,
    ...     Q=32.0, T=145.0, S=3.2e-4, r=0.15,
    ...     poco="PZ-01",
    ... )
    """
    if summary not in {"outside", "panel", "box", "none"}:
        raise ValueError(
            "`summary` deve ser 'outside', 'panel', 'box' ou 'none'."
        )
    if df_real is None or len(df_real) == 0:
        raise ValueError("`df_real` está vazio: não há medições para plotar.")

    ct, cs = _resolve_columns(df_real, col_t, col_s, "df_real")

    colunas = [ct, cs]
    if col_origem is not None:
        if col_origem not in df_real.columns:
            raise KeyError(
                f"Coluna de origem '{col_origem}' ausente em `df_real`. "
                f"Disponíveis: {list(df_real.columns)}"
            )
        colunas.append(col_origem)

    real = df_real[colunas].copy()
    real[[ct, cs]] = real[[ct, cs]].apply(pd.to_numeric, errors="coerce")
    real = real.dropna(subset=[ct, cs]).sort_values(ct)
    real = real[real[ct] > 0]  # escala log exige t > 0
    if real.empty:
        raise ValueError(
            "Nenhuma medição válida com t > 0 em `df_real` "
            "(a escala logarítmica não admite t ≤ 0)."
        )

    t_real = real[ct].to_numpy()
    s_real = real[cs].to_numpy()

    # Máscara de procedência: True = ponto sintético/interpolado.
    if col_origem is not None:
        alvos = {_normaliza_rotulo(x) for x in origens_sinteticas}
        mask_sint = (
            real[col_origem].astype(str).map(_normaliza_rotulo).isin(alvos)
        ).to_numpy()
    else:
        mask_sint = np.zeros(t_real.size, dtype=bool)

    # Último instante com medição de campo — limite do suporte amostral.
    t_medidos = t_real[~mask_sint]
    t_ultimo = float(t_medidos[-1]) if t_medidos.size else float(t_real[-1])

    # ---------------------------------------------------------------- #
    # Camada teórica                                                    #
    # ---------------------------------------------------------------- #
    proj: Optional[pd.DataFrame] = None
    if show_projection:
        if df_projection is not None and len(df_projection) > 0:
            cpt, cps = _resolve_columns(
                df_projection, col_t_proj, col_s_proj, "df_projection"
            )
            proj = (
                df_projection[[cpt, cps]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .sort_values(cpt)
                .rename(columns={cpt: "t", cps: "s"})
            )
            proj = proj[proj["t"] > 0]
        elif None not in (Q, T, S, r):
            t_fim = max(t_ultimo * 3.0, 720.0 * 1.5)
            proj = build_cooper_jacob_projection(
                t_inicio_min=max(t_real[0] * 0.5, 1e-2),
                t_fim_min=t_fim,
                Q_m3h=Q, T_m2dia=T, S=S, r_m=r,
                col_t="t", col_s="s",
            )
        else:
            warnings.warn(
                "show_projection=True mas não há `df_projection` nem os "
                "parâmetros (Q, T, S, r) para gerar a curva teórica; "
                "a projeção foi omitida.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---------------------------------------------------------------- #
    # s_12h — valor exibido no resumo                                   #
    # ---------------------------------------------------------------- #
    T_12H_MIN = 720.0
    if s_12h is None:
        if None not in (Q, T, S, r):
            valor = cooper_jacob_drawdown(T_12H_MIN, Q, T, S, r)
            s_12h = None if np.isnan(valor) else float(valor)
        elif proj is not None and proj["t"].min() <= T_12H_MIN <= proj["t"].max():
            s_12h = float(np.interp(T_12H_MIN, proj["t"], proj["s"]))

    # ---------------------------------------------------------------- #
    # Figura                                                            #
    # ---------------------------------------------------------------- #
    if summary == "panel":
        fig, (ax, ax_info) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [3.3, 1.0], "wspace": 0.04},
        )
        ax_info.axis("off")
    else:
        largura = figsize[0] if summary == "outside" else figsize[0] * 0.8
        fig, ax = plt.subplots(figsize=(largura, figsize[1]))
        ax_info = None

    # --- faixa de extrapolação (desenhada primeiro, ao fundo) ---
    if proj is not None and proj["t"].max() > t_ultimo:
        ax.axvspan(
            t_ultimo, proj["t"].max(),
            facecolor=PALETTE["extrapolation"], alpha=0.13,
            edgecolor="none", zorder=0,
            label="Intervalo de extrapolação (sem medição)",
        )
        ax.axvline(t_ultimo, color=PALETTE["extrapolation"],
                   linewidth=1.0, linestyle=":", alpha=0.9, zorder=1)

    # --- projeção teórica: tracejada, sem marcadores preenchidos ---
    if proj is not None:
        ax.plot(
            proj["t"], proj["s"],
            color=PALETTE["projection"],
            linewidth=1.7, linestyle="--", dashes=(6, 3),
            marker="", zorder=2,
            label="Projeção Teórica (Modelo Cooper-Jacob)",
        )

    # --- série medida: linha contínua única + círculos de mesmo tamanho ---
    # A linha percorre todos os pontos do dataset unificado, sem quebra;
    # o preenchimento do marcador é o único elemento que distingue a
    # procedência (sólido = campo, vazado = sintético).
    ax.plot(
        t_real, s_real,
        color=PALETTE["observed"], linewidth=1.4, linestyle="-",
        marker="", zorder=3,
    )
    ax.plot(
        t_real[~mask_sint], s_real[~mask_sint],
        color=PALETTE["observed"], linestyle="none",
        marker="o", markersize=6.5,
        markerfacecolor=PALETTE["observed"],
        markeredgecolor=PALETTE["observed_edge"],
        markeredgewidth=0.8, zorder=4,
        label="Medições de Campo (Dados Reais)",
    )
    if mask_sint.any():
        ax.plot(
            t_real[mask_sint], s_real[mask_sint],
            color=PALETTE["observed"], linestyle="none",
            marker="o", markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=PALETTE["observed"],
            markeredgewidth=1.3, zorder=4,
            label="Pontos Sintéticos / Interpolados (Modelo)",
        )

    # --- marcação do horizonte de 12 h ---
    if mark_12h and s_12h is not None:
        x_lim_dir = proj["t"].max() if proj is not None else t_ultimo
        if T_12H_MIN <= x_lim_dir * 1.05:
            ax.axvline(T_12H_MIN, color=PALETTE["fit"], linewidth=1.1,
                       linestyle="-.", alpha=0.75, zorder=3)
            ax.plot(
                [T_12H_MIN], [s_12h],
                marker="X", markersize=9,
                markerfacecolor=PALETTE["fit"],
                markeredgecolor="white", markeredgewidth=0.9,
                linestyle="none", zorder=5,
                label=r"Rebaixamento previsto em 12 h ($s_{12h}$)",
            )

    # --- eixos ---
    _setup_semilog_axis(ax, xlabel, ylabel)
    _invert_y_once(ax)

    subtitulo = f"Poço {poco}" if poco else None
    if subtitulo:
        ax.set_title(f"{title}\n{subtitulo}", pad=12)
        ax.title.set_multialignment("center")
    else:
        ax.set_title(title, pad=12)

    if legend_loc == "outside":
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.135),
            ncol=2, frameon=True, fancybox=False,
            borderpad=0.7, labelspacing=0.5, columnspacing=1.6,
        )
    else:
        ax.legend(
            loc=legend_loc, frameon=True, fancybox=False,
            borderpad=0.7, labelspacing=0.6,
        )

    # ---------------------------------------------------------------- #
    # Resumo hidrodinâmico                                              #
    # ---------------------------------------------------------------- #
    linhas = [
        (r"$Q$", f"{_fmt_num(Q, 1)} m³/h" if Q is not None else "—"),
        (r"$T$", rf"${_fmt_sci(T)}$ m²/dia" if T is not None else "—"),
        (r"$S$", rf"${_fmt_sci(S)}$" if S is not None else "—"),
        (r"$s_{12h}$", f"{_fmt_num(s_12h, 2)} m" if s_12h is not None else "—"),
    ]

    if summary == "panel" and ax_info is not None:
        ax_info.text(
            0.06, 0.97, "PARÂMETROS\nHIDRODINÂMICOS",
            transform=ax_info.transAxes, fontsize=9.0, fontweight="bold",
            color=PALETTE["annotation"], va="top", ha="left", linespacing=1.4,
        )
        ax_info.plot([0.06, 0.94], [0.895, 0.895], transform=ax_info.transAxes,
                     color="0.55", linewidth=0.9)

        y = 0.83
        for simbolo, valor in linhas:
            ax_info.text(0.06, y, simbolo, transform=ax_info.transAxes,
                         fontsize=12, va="top", ha="left",
                         color=PALETTE["annotation"])
            ax_info.text(0.30, y, f"= {valor}", transform=ax_info.transAxes,
                         fontsize=10.5, va="top", ha="left",
                         color=PALETTE["annotation"])
            y -= 0.085

        rodape = [r"$r$ = " + f"{_fmt_num(r, 2)} m" if r is not None else "",
                  f"n = {len(t_real)} medições",
                  f"t final medido = {_fmt_num(t_ultimo, 0)} min"]
        ax_info.text(
            0.06, y - 0.03, "\n".join(x for x in rodape if x),
            transform=ax_info.transAxes, fontsize=8.8, va="top", ha="left",
            color="0.35", linespacing=1.7,
        )
        if proj is not None:
            ax_info.text(
                0.06, 0.06,
                "Curva teórica válida apenas\nsob as hipóteses de\nCooper-Jacob "
                "(u < 0,01;\naquífero homogêneo,\nconfinado e infinito).",
                transform=ax_info.transAxes, fontsize=8.0, va="bottom",
                ha="left", color="0.40", linespacing=1.6, style="italic",
            )
        # moldura discreta do painel
        ax_info.add_patch(plt.Rectangle(
            (0.01, 0.02), 0.99, 0.97, transform=ax_info.transAxes,
            facecolor="#f7f8fa", edgecolor="0.75", linewidth=0.8, zorder=-1,
        ))

    elif summary == "outside":
        # Caixa ancorada FORA da área de plotagem (x > 1 em coordenadas de
        # eixo): não há como sobrepor legenda, pontos ou curva teórica.
        texto = "\n".join(f"{sim} = {val}" for sim, val in linhas)
        rodape = [x for x in (
            rf"$r$ = {_fmt_num(r, 2)} m" if r is not None else "",
            f"n = {int((~mask_sint).sum())} medições de campo",
            f"n = {int(mask_sint.sum())} pontos sintéticos"
            if mask_sint.any() else "",
        ) if x]
        ax.text(
            1.025, 1.0, "PARÂMETROS\nHIDRODINÂMICOS\n\n" + texto,
            transform=ax.transAxes, fontsize=10.5, va="top", ha="left",
            linespacing=1.7, color=PALETTE["annotation"],
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f8fa",
                      edgecolor="0.6", linewidth=0.9),
            zorder=6, clip_on=False,
        )
        if rodape:
            ax.text(
                1.025, 0.995 - 0.055 * (len(linhas) + 3.6),
                "\n".join(rodape),
                transform=ax.transAxes, fontsize=8.8, va="top", ha="left",
                linespacing=1.7, color="0.35", clip_on=False,
            )

    elif summary == "box":
        texto = "\n".join(f"{sim} = {val}" for sim, val in linhas)
        _textbox(ax, 0.035, 0.06, texto, va="bottom", ha="left", fontsize=10)

    # `tight_layout` não lida bem com artistas ancorados fora dos eixos:
    # nesses modos o espaçamento é reservado manualmente.
    if summary == "panel":
        fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.11)
    elif summary == "outside":
        fig.subplots_adjust(
            left=0.085, right=0.755, top=0.89,
            bottom=0.26 if legend_loc == "outside" else 0.12,
        )
    else:
        fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=dpi)

    return fig


# --------------------------------------------------------------------------- #
# 2. Cooper-Jacob (ajuste da fase de rebaixamento)                             #
# --------------------------------------------------------------------------- #

def plot_cooper_jacob(
    time_min: ArrayLike,
    drawdown_corr: ArrayLike,
    fit_line: ArrayLike,
    T: float,
    S: float,
    title: str = "Método de Cooper-Jacob",
    save_path: Optional[str] = "outputs/cooper_jacob.png",
    xlabel: str = "Tempo, $t$ (min)",
    ylabel: str = "Rebaixamento corrigido, $s'$ (m)",
) -> Figure:
    """
    Gráfico semi-logarítmico do ajuste de Cooper-Jacob, com a caixa de
    parâmetros hidráulicos (T e S) estimados pela reta.

    Ver o módulo para convenção de unidades. ``fit_line`` deve ter o mesmo
    comprimento de ``time_min``.
    """
    time_min = np.asarray(time_min, dtype=float)
    drawdown_corr = np.asarray(drawdown_corr, dtype=float)
    fit_line = np.asarray(fit_line, dtype=float)

    if not (time_min.size == drawdown_corr.size == fit_line.size):
        raise ValueError(
            "`time_min`, `drawdown_corr` e `fit_line` devem ter o mesmo "
            f"comprimento (recebidos {time_min.size}, {drawdown_corr.size}, "
            f"{fit_line.size})."
        )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        time_min, drawdown_corr,
        s=45, facecolors=PALETTE["observed"], edgecolors="black",
        linewidths=0.6, zorder=3, label="Rebaixamento observado",
    )

    ordem = np.argsort(time_min)
    ax.plot(
        time_min[ordem], fit_line[ordem],
        color=PALETTE["fit"], linewidth=1.8, zorder=2,
        label="Reta ajustada (Cooper-Jacob)",
    )

    _setup_semilog_axis(ax, xlabel, ylabel)
    _invert_y_once(ax)
    ax.set_title(title, pad=12)

    _textbox(
        ax, 0.03, 0.05,
        rf"$T = {_fmt_sci(T)}$ m²/dia" + "\n" + rf"$S = {_fmt_sci(S)}$",
    )

    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)

    return fig


# --------------------------------------------------------------------------- #
# 3. Recuperação de Theis (fase de recuperação)                                #
# --------------------------------------------------------------------------- #

def plot_recovery_theis(
    t_ratio: ArrayLike,
    residual_drawdown: ArrayLike,
    fit_line: ArrayLike,
    T_rec: float,
    title: str = "Método de Recuperação de Theis",
    save_path: Optional[str] = "outputs/recovery_theis.png",
    xlabel: str = "Razão de tempo, $t/t'$",
    ylabel: str = "Rebaixamento residual, $s'$ (m)",
) -> Figure:
    """
    Gráfico semi-logarítmico da fase de recuperação, relacionando o
    rebaixamento residual com a razão de tempos t/t' e exibindo a
    transmissividade estimada nessa fase (T_rec).
    """
    t_ratio = np.asarray(t_ratio, dtype=float)
    residual_drawdown = np.asarray(residual_drawdown, dtype=float)
    fit_line = np.asarray(fit_line, dtype=float)

    if not (t_ratio.size == residual_drawdown.size == fit_line.size):
        raise ValueError(
            "`t_ratio`, `residual_drawdown` e `fit_line` devem ter o mesmo "
            f"comprimento (recebidos {t_ratio.size}, "
            f"{residual_drawdown.size}, {fit_line.size})."
        )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        t_ratio, residual_drawdown,
        s=45, facecolors=PALETTE["recovery"], edgecolors="black",
        linewidths=0.6, zorder=3, label="Rebaixamento residual observado",
    )

    ordem = np.argsort(t_ratio)
    ax.plot(
        t_ratio[ordem], fit_line[ordem],
        color=PALETTE["fit"], linewidth=1.8, zorder=2,
        label="Reta ajustada (Theis — recuperação)",
    )

    _setup_semilog_axis(ax, xlabel, ylabel)
    ax.set_title(title, pad=12)

    _textbox(ax, 0.03, 0.95, rf"$T_{{rec}} = {_fmt_sci(T_rec)}$ m²/dia",
             va="top")

    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)

    return fig


# --------------------------------------------------------------------------- #
# Exemplo de uso                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Parâmetros sintéticos do ensaio ---
    Q_ENSAIO, T_ENSAIO, S_ENSAIO, R_ENSAIO = 32.0, 145.0, 3.2e-4, 0.15

    # --- 1. Dataset unificado: medições de campo + pontos sintéticos ---
    t_campo = np.array([1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240],
                       dtype=float)
    s_campo = cooper_jacob_drawdown(
        t_campo, Q_ENSAIO, T_ENSAIO, S_ENSAIO, R_ENSAIO
    ) + rng.normal(0, 0.03, size=t_campo.size)

    # Pontos gerados pelo modelo para completar a série até 12 h.
    t_sint = np.array([320, 420, 540, 720], dtype=float)
    s_sint = cooper_jacob_drawdown(t_sint, Q_ENSAIO, T_ENSAIO, S_ENSAIO,
                                   R_ENSAIO)

    df_unificado = pd.DataFrame({
        "tempo_min": np.concatenate([t_campo, t_sint]),
        "rebaixamento_m": np.concatenate([s_campo, s_sint]),
        "origem": ["medido"] * t_campo.size + ["sintetico"] * t_sint.size,
    })

    plot_hydro_analysis(
        df_real=df_unificado,
        col_origem="origem",
        show_projection=True,
        title="Análise do Teste de Bombeamento",
        Q=Q_ENSAIO, T=T_ENSAIO, S=S_ENSAIO, r=R_ENSAIO,
        poco="PZ-01",
        save_path="outputs/hydro_analysis.png",
    )

    # Variante sem projeção teórica (apenas a série unificada)
    plot_hydro_analysis(
        df_real=df_unificado,
        col_origem="origem",
        show_projection=False,
        Q=Q_ENSAIO, T=T_ENSAIO, S=S_ENSAIO, r=R_ENSAIO,
        poco="PZ-01",
        save_path="outputs/hydro_analysis_sem_projecao.png",
    )

    # --- 2. Cooper-Jacob ---
    t = np.logspace(0, 3, 30)
    s_true = 0.6 + 0.9 * np.log10(t)
    s_obs = s_true + rng.normal(0, 0.05, size=t.size)
    plot_cooper_jacob(
        time_min=t, drawdown_corr=s_obs, fit_line=s_true,
        T=850.0, S=2.3e-4,
        title="Ensaio de Bombeamento — Poço PZ-01 (Cooper-Jacob)",
    )

    # --- 3. Recuperação de Theis ---
    ratio = np.logspace(0, 2.5, 25)
    s_res_true = 1.2 * np.log10(ratio)
    s_res_obs = s_res_true + rng.normal(0, 0.04, size=ratio.size)
    plot_recovery_theis(
        t_ratio=ratio, residual_drawdown=s_res_obs, fit_line=s_res_true,
        T_rec=910.0,
        title="Ensaio de Recuperação — Poço PZ-01 (Método de Theis)",
    )

    print("Gráficos gerados com sucesso em 'outputs/'.")
