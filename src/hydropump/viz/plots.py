"""
Funções de plotagem para ensaios de aquífero (testes de bombeamento) em
hidrogeologia, com qualidade adequada para publicação científica.

Métodos contemplados
--------------------
- ``plot_hydro_analysis``  : gráfico principal do ensaio, separando de forma
  inequívoca as **medições de campo** da **projeção teórica** (Cooper-Jacob).
- ``plot_cooper_jacob``    : ajuste semi-log da fase de rebaixamento.
- ``plot_recovery_theis``  : análise da fase de recuperação.

Diferenças em relação à versão original
----------------------------------------
- Nenhuma função grava em disco por padrão (``save_path=None``); gravar é
  responsabilidade explícita do chamador. O default antigo apontava para um
  caminho relativo (``outputs/...``) que falha com ``PermissionError`` num
  executável instalado em ``Program Files``.
- Todas as funções que criam uma ``Figure`` **não a fecham** — quem chama
  deve usar o gerenciador de contexto :func:`managed_figure` ou, no
  Streamlit, ``plt.close(fig)`` após ``st.pyplot(fig)``. Sem isso, cada
  rerun do script (a cada interação de widget) retém uma nova figura no
  estado global do pyplot.
- Rótulos de procedência (medido/sintético) são normalizados por
  :func:`hydropump.text.normalize_label`, compartilhada com a camada de I/O.

Convenções de unidades (SI hidrogeológico usual)
------------------------------------------------
    t   : minutos (eixo X, escala log10)
    s   : metros (eixo Y, invertido — zero na superfície)
    Q   : m³/h  (rótulo)
    T   : m²/dia
    S   : adimensional
    r   : metros

Dependências: matplotlib, seaborn, numpy, pandas
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hydropump.text import normalize_label
from hydropump.viz.theme import PALETTE, apply_scientific_style

apply_scientific_style()

ArrayLike = Sequence[float] | np.ndarray

__all__ = [
    "plot_hydro_analysis",
    "plot_cooper_jacob",
    "plot_recovery_theis",
    "plot_step_drawdown",
    "cooper_jacob_drawdown",
    "build_cooper_jacob_projection",
    "managed_figure",
    "PALETTE",
]


@contextmanager
def managed_figure(fig: Figure) -> Iterator[Figure]:
    """Garante ``plt.close(fig)`` mesmo se o bloco levantar exceção.

    Uso típico no Streamlit::

        with managed_figure(pl.plot_hydro_analysis(...)) as fig:
            st.pyplot(fig)
    """
    try:
        yield fig
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Utilitários internos                                                         #
# --------------------------------------------------------------------------- #

def _ensure_dir(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _fmt_sci(value: float | None, digits: int = 3) -> str:
    """Formata em notação científica LaTeX: 2.30 × 10⁻⁴."""
    if value is None or not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    expoente = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**expoente
    if -2 <= expoente <= 3:
        return f"{value:,.{digits}g}".replace(",", " ")
    return rf"{mantissa:.{digits - 1}f}\times 10^{{{expoente}}}"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _resolve_columns(
    df: pd.DataFrame,
    col_t: str | None,
    col_s: str | None,
    origem: str,
) -> tuple[str, str]:
    """Descobre as colunas de tempo e rebaixamento de um DataFrame."""
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
    if value <= 0:
        return ""
    expoente = int(np.round(np.log10(value)))
    if -2 <= expoente <= 4:
        return f"{value:,.0f}".replace(",", " ") if value >= 1 else f"{value:g}"
    return rf"$10^{{{expoente}}}$"


def _setup_semilog_axis(ax: Axes, xlabel: str, ylabel: str) -> None:
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
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _textbox(ax: Axes, x: float, y: float, texto: str, va: str = "bottom",
             ha: str = "left", fontsize: float = 10.5) -> None:
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
    t_min: float | ArrayLike,
    Q_m3h: float,
    T_m2dia: float,
    S: float,
    r_m: float,
) -> float | np.ndarray:
    r"""Rebaixamento teórico pela aproximação logarítmica de Cooper-Jacob.

    Valores no ramo inválido do modelo (argumento do log ≤ 1) retornam
    ``nan`` em vez de números negativos espúrios.
    """
    t_dias = np.asarray(t_min, dtype=float) / 1440.0
    Q_dia = Q_m3h * 24.0

    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 2.25 * T_m2dia * t_dias / (r_m**2 * S)
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
    """Curva teórica de Cooper-Jacob em espaçamento log entre os dois tempos."""
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
    df_projection: pd.DataFrame | None = None,
    show_projection: bool = True,
    title: str = "Análise do Teste de Bombeamento",
    *,
    Q: float | None = None,
    T: float | None = None,
    S: float | None = None,
    T_ci95: tuple[float, float] | None = None,
    s_12h: float | None = None,
    r: float | None = None,
    poco: str | None = None,
    col_t: str | None = None,
    col_s: str | None = None,
    col_origem: str | None = None,
    origens_sinteticas: Sequence[str] = (
        "sintetico", "sintético", "sintetica", "sintética",
        "modelado", "simulado", "projetado", "interpolado", "estimado",
    ),
    col_t_proj: str | None = None,
    col_s_proj: str | None = None,
    summary: str = "outside",
    legend_loc: str = "outside",
    xlabel: str = "Tempo desde o início do bombeamento, $t$ (min)",
    ylabel: str = "Rebaixamento, $s$ (m)",
    mark_12h: bool = True,
    figsize: tuple[float, float] = (10.5, 6.0),
    save_path: str | None = None,
    dpi: int = 300,
) -> Figure:
    r"""Gráfico principal do ensaio, separando medições de campo e projeção.

    Ver o docstring completo na versão anterior do módulo — a API é
    intencionalmente estável. Mudança de contrato: ``save_path`` agora vale
    ``None`` por padrão (nada é gravado a menos que peça explicitamente); a
    função **não fecha** a figura retornada — use :func:`managed_figure`.
    """
    if summary not in {"outside", "panel", "box", "none"}:
        raise ValueError("`summary` deve ser 'outside', 'panel', 'box' ou 'none'.")
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
    real = real[real[ct] > 0]
    if real.empty:
        raise ValueError(
            "Nenhuma medição válida com t > 0 em `df_real` "
            "(a escala logarítmica não admite t ≤ 0)."
        )

    t_real = real[ct].to_numpy()
    s_real = real[cs].to_numpy()

    if col_origem is not None:
        alvos = {normalize_label(x) for x in origens_sinteticas}
        mask_sint = (
            real[col_origem].astype(str).map(normalize_label).isin(alvos)
        ).to_numpy()
    else:
        mask_sint = np.zeros(t_real.size, dtype=bool)

    t_medidos = t_real[~mask_sint]
    t_ultimo = float(t_medidos[-1]) if t_medidos.size else float(t_real[-1])

    # --- Camada teórica ---
    proj: pd.DataFrame | None = None
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

    T_12H_MIN = 720.0
    if s_12h is None:
        if None not in (Q, T, S, r):
            valor = cooper_jacob_drawdown(T_12H_MIN, Q, T, S, r)
            s_12h = None if np.isnan(valor) else float(valor)
        elif proj is not None and proj["t"].min() <= T_12H_MIN <= proj["t"].max():
            s_12h = float(np.interp(T_12H_MIN, proj["t"], proj["s"]))

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

    if proj is not None and proj["t"].max() > t_ultimo:
        ax.axvspan(
            t_ultimo, proj["t"].max(),
            facecolor=PALETTE["extrapolation"], alpha=0.13,
            edgecolor="none", zorder=0,
            label="Intervalo de extrapolação (sem medição)",
        )
        ax.axvline(t_ultimo, color=PALETTE["extrapolation"],
                   linewidth=1.0, linestyle=":", alpha=0.9, zorder=1)

    if proj is not None:
        ax.plot(
            proj["t"], proj["s"],
            color=PALETTE["projection"],
            linewidth=1.7, linestyle="--", dashes=(6, 3),
            marker="", zorder=2,
            label="Projeção Teórica (Modelo Cooper-Jacob)",
        )

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

    if mark_12h and s_12h is not None:
        x_lim_dir = proj["t"].max() if proj is not None else t_ultimo
        if x_lim_dir * 1.05 >= T_12H_MIN:
            ax.axvline(T_12H_MIN, color=PALETTE["fit"], linewidth=1.1,
                       linestyle="-.", alpha=0.75, zorder=3)
            ax.plot(
                [T_12H_MIN], [s_12h],
                marker="X", markersize=9,
                markerfacecolor=PALETTE["fit"],
                markeredgecolor="white", markeredgewidth=0.9,
                linestyle="none", zorder=5,
                label=r"Rebaixamento em 12 h ($s_{12h}$)",
            )

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

    t_label = rf"${_fmt_sci(T)}$ m²/dia" if T is not None else "—"
    if T_ci95 is not None and np.isfinite(T_ci95[0]) and np.isfinite(T_ci95[1]):
        t_label += rf" (IC95%: {_fmt_sci(T_ci95[0])}–{_fmt_sci(T_ci95[1])})"

    linhas = [
        (r"$Q$", f"{_fmt_num(Q, 1)} m³/h" if Q is not None else "—"),
        (r"$T$", t_label),
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
        ax_info.add_patch(plt.Rectangle(
            (0.01, 0.02), 0.99, 0.97, transform=ax_info.transAxes,
            facecolor="#f7f8fa", edgecolor="0.75", linewidth=0.8, zorder=-1,
        ))

    elif summary == "outside":
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
    save_path: str | None = None,
    xlabel: str = "Tempo, $t$ (min)",
    ylabel: str = "Rebaixamento corrigido, $s'$ (m)",
) -> Figure:
    """Gráfico semi-log do ajuste de Cooper-Jacob, com a caixa de T e S."""
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
    save_path: str | None = None,
    xlabel: str = "Razão de tempo, $t/t'$",
    ylabel: str = "Rebaixamento residual, $s'$ (m)",
) -> Figure:
    """Gráfico semi-log da fase de recuperação (rebaixamento residual x t/t')."""
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

    _textbox(ax, 0.03, 0.95, rf"$T_{{rec}} = {_fmt_sci(T_rec)}$ m²/dia", va="top")

    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)

    return fig


# --------------------------------------------------------------------------- #
# 4. Teste de degraus — rebaixamento específico (s/Q) x vazão (Q)              #
# --------------------------------------------------------------------------- #

def plot_step_drawdown(
    Q_m3_h: ArrayLike,
    specific_drawdown: ArrayLike,
    B: float,
    C: float,
    *,
    r_squared: float | None = None,
    efficiency_at: tuple[float, float] | None = None,
    title: str = "Teste de Degraus — Rebaixamento Específico x Vazão",
    xlabel: str = "Vazão, $Q$ (m³/h)",
    ylabel: str = "Rebaixamento específico, $s/Q$ (m por m³/h)",
    figsize: tuple[float, float] = (7.5, 5.5),
    save_path: str | None = None,
) -> Figure:
    r"""Gráfico de Jacob (1947) para teste de degraus.

    Plota :math:`s/Q` contra :math:`Q`. Se a decomposição
    :math:`s = BQ + CQ^2` é válida, os pontos caem sobre a reta
    :math:`s/Q = B + CQ`, cujo intercepto é a perda no aquífero e cuja
    inclinação é a perda no poço.

    Parameters
    ----------
    Q_m3_h : ArrayLike
        Vazão de cada degrau [m³/h] — unidade de campo, não SI.
    specific_drawdown : ArrayLike
        s/Q de cada degrau, na unidade correspondente (m por m³/h).
    B, C : float
        Coeficientes já ajustados, **nas mesmas unidades do gráfico**.
    r_squared : float, opcional
        Exibido na caixa de resumo.
    efficiency_at : (Q, eficiência), opcional
        Marca a eficiência do poço numa vazão de interesse.
    save_path : str | None
        ``None`` (padrão) não grava nada. A figura NÃO é fechada — use
        :func:`managed_figure` ou ``plt.close``.

    Notes
    -----
    Um ``C`` negativo (reta descendente) não tem sentido físico: implicaria
    perda de poço negativa. Na prática indica degraus que não estabilizaram
    ou erro de leitura. O gráfico desenha o que for passado — a crítica
    fica a cargo do chamador.
    """
    q = np.asarray(Q_m3_h, dtype=float)
    sq = np.asarray(specific_drawdown, dtype=float)
    if q.size != sq.size:
        raise ValueError(
            f"`Q_m3_h` ({q.size}) e `specific_drawdown` ({sq.size}) devem ter "
            "o mesmo comprimento."
        )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        q, sq,
        s=60, facecolors=PALETTE["observed"], edgecolors=PALETTE["observed_edge"],
        linewidths=0.9, zorder=4, label="Degraus medidos",
    )

    q_line = np.linspace(0.0, float(np.max(q)) * 1.12, 100)
    ax.plot(
        q_line, B + C * q_line,
        color=PALETTE["fit"], linewidth=1.8, zorder=3,
        label=r"Ajuste $s/Q = B + C\,Q$",
    )

    # Intercepto B: fronteira entre perda de aquífero e perda de poço
    ax.axhline(B, color=PALETTE["extrapolation"], linewidth=1.0,
               linestyle=":", alpha=0.9, zorder=2)
    ax.annotate(
        f"B = {B:.4g}", xy=(0.0, B), xytext=(6, 6), textcoords="offset points",
        fontsize=9, color=PALETTE["annotation"], va="bottom", ha="left",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.set_xlim(left=0.0)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.55)

    linhas = [
        rf"$B$ = {B:.4g}   (perda no aquífero)",
        rf"$C$ = {C:.4g}   (perda no poço)",
    ]
    if r_squared is not None:
        linhas.append(rf"$R^2$ = {r_squared:.4f}")
    if efficiency_at is not None:
        q_ref, eff = efficiency_at
        linhas.append(rf"Eficiência @ {q_ref:.1f} m³/h = {eff * 100:.1f}%")
    _textbox(ax, 0.035, 0.95, "\n".join(linhas), va="top", ha="left", fontsize=9.5)

    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300)

    return fig
