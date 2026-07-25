"""
src/schematics.py
==================

Diagrama esquemático vetorial (corte 2D) do poço de bombeamento, gerado com
Matplotlib, para exibição no Streamlit (ex.: dentro de um
``st.expander("Ver Esquema do Poço")``).

O desenho é puramente instrutivo (fora de escala real) e mostra:
    - Terreno / coluna de material do aquífero e base impermeável;
    - Nível Estático (NE) — linha d'água em azul claro;
    - Nível Dinâmico (ND) — linha d'água em azul escuro, com o cone de
      rebaixamento conectando o poço bombeado ao nível estático regional;
    - Tubo do poço com a seção filtrante hachurada, defronte ao aquífero;
    - (opcional) poço de observação a uma distância radial "r";
    - (opcional) cota vertical lateral indicando a espessura saturada "b".

Uso típico:
    from src import schematics as sq
    fig = sq.draw_well_schematic(type="observation", NE=2.0, ND=5.0, b=10.0, r=15.0)
    st.pyplot(fig)
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Paleta de cores (pensada para fundo transparente sobre tema claro/escuro)
# ---------------------------------------------------------------------------
COR_TERRENO = "#c9b28a"
COR_TERRENO_BORDA = "#8a7048"
COR_AQUIFERO = "#e4d9bf"
COR_AQUIFERO_BORDA = "#8a7048"
COR_BASE_IMPERMEAVEL = "#5b4a36"
COR_SATURADO = "#bfe0f2"
COR_TUBO = "#8a95a1"
COR_TUBO_BORDA = "#4d5560"
COR_FILTRO = "#e8ecef"
COR_FILTRO_BORDA = "#4d5560"
COR_NE = "#5fb8e8"          # azul claro — Nível Estático
COR_ND = "#0b4f8a"          # azul escuro — Nível Dinâmico
COR_CONE = "#0b4f8a"
COR_TEXTO = "#25313d"
COR_SETA = "#25313d"


def _nivel_agua_marker(ax, x: float, y: float, cor: str, tamanho: float = 0.28) -> None:
    """Desenha o símbolo triangular convencional de nível d'água."""
    ax.plot(x, y, marker="v", markersize=tamanho * 34, color=cor,
             markeredgecolor=COR_TEXTO, markeredgewidth=0.6, zorder=6)


def _cota_vertical(ax, x: float, y_top: float, y_bottom: float, texto: str,
                    cor: str = COR_SETA, deslocamento: float = 0.0) -> None:
    """Desenha uma cota vertical (seta dupla) com rótulo central."""
    seta = FancyArrowPatch(
        (x + deslocamento, y_top), (x + deslocamento, y_bottom),
        arrowstyle="<->", mutation_scale=12, color=cor, linewidth=1.2, zorder=7,
    )
    ax.add_patch(seta)
    ax.text(x + deslocamento + 0.15, (y_top + y_bottom) / 2, texto, color=COR_TEXTO,
             fontsize=9, va="center", ha="left", rotation=90, zorder=7)


def _cota_horizontal(ax, y: float, x_left: float, x_right: float, texto: str,
                      cor: str = COR_SETA) -> None:
    """Desenha uma cota horizontal (seta dupla) com rótulo central, acima da seta."""
    seta = FancyArrowPatch(
        (x_left, y), (x_right, y), arrowstyle="<->", mutation_scale=12,
        color=cor, linewidth=1.2, zorder=7,
    )
    ax.add_patch(seta)
    ax.text((x_left + x_right) / 2, y - 0.22, texto, color=COR_TEXTO, fontsize=9,
             va="bottom", ha="center", zorder=7)


def draw_well_schematic(
    type: Literal["single", "observation"] = "single",
    NE: float = 2.0,
    ND: float = 5.0,
    b: float = 10.0,
    r: float = 0.076,
    show_b: bool = True,
):
    """
    Gera um diagrama esquemático vetorial (corte 2D) do poço de bombeamento.

    Parâmetros
    ----------
    type : {"single", "observation"}
        "single"      -> mostra apenas o poço bombeado, com indicação de r_w.
        "observation" -> mostra o poço bombeado e um poço de observação a uma
                         distância radial "r", com o cone de rebaixamento.
    NE : float
        Nível Estático (m), profundidade do nível d'água em repouso.
    ND : float
        Nível Dinâmico (m), profundidade do nível d'água durante o bombeamento.
        Deve ser >= NE.
    b : float
        Espessura saturada do aquífero (m), medida a partir do NE.
    r : float
        Se type == "single": raio do poço, r_w (m).
        Se type == "observation": distância radial ao poço de observação (m).
    show_b : bool
        Se True, desenha a cota vertical lateral indicando a espessura
        saturada "b".

    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura com fundo transparente, sem eixos numéricos.
    """
    # --- Sanitização básica -------------------------------------------------
    NE = float(NE)
    ND = float(max(ND, NE + 0.05))
    b = float(max(b, 0.5))
    r = float(max(r, 0.001))
    is_obs = type == "observation"

    # --- Geometria vertical (unidades arbitrárias, não em escala real) -----
    y_terreno_topo = -0.9
    y_solo = 0.0
    y_base_aquifero = NE + b
    y_base_impermeavel = y_base_aquifero + 0.9
    y_fundo_poco = y_base_aquifero + 0.5

    # --- Geometria horizontal ------------------------------------------------
    raio_tubo = 0.38
    if is_obs:
        x_obs = 6.5
        x_max = x_obs + 2.2
    else:
        x_obs = None
        x_max = 4.5
    x_min = -x_max

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # ------------------------------------------------------------------
    # 1) Terreno, matriz do aquífero e base impermeável
    # ------------------------------------------------------------------
    ax.add_patch(Rectangle(
        (x_min, y_terreno_topo), x_max - x_min, y_solo - y_terreno_topo,
        facecolor=COR_TERRENO, edgecolor=COR_TERRENO_BORDA, hatch="///", linewidth=0.8, zorder=1,
    ))
    ax.add_patch(Rectangle(
        (x_min, y_solo), x_max - x_min, y_base_aquifero - y_solo,
        facecolor=COR_AQUIFERO, edgecolor=COR_AQUIFERO_BORDA, hatch="...", linewidth=0.8, zorder=1,
    ))
    ax.add_patch(Rectangle(
        (x_min, y_base_aquifero), x_max - x_min, y_base_impermeavel - y_base_aquifero,
        facecolor=COR_BASE_IMPERMEAVEL, edgecolor=COR_BASE_IMPERMEAVEL, hatch="\\\\\\\\",
        linewidth=0.8, zorder=1,
    ))

    # ------------------------------------------------------------------
    # 2) Cone de rebaixamento — curva conectando ND (no poço) ao NE regional
    # ------------------------------------------------------------------
    x_curva = np.linspace(0.05, x_max, 200)
    decaimento = (x_max - raio_tubo) / 2.6
    y_cone = NE + (ND - NE) * np.exp(-(x_curva - raio_tubo) / decaimento)
    y_cone = np.clip(y_cone, NE, ND)

    # zona saturada (abaixo do cone / abaixo do NE) tingida de azul
    x_pos = np.concatenate(([raio_tubo], x_curva))
    y_pos = np.concatenate(([ND], y_cone))
    ax.fill_between(x_pos, y_pos, y_base_aquifero, color=COR_SATURADO, alpha=0.65, zorder=2)
    ax.fill_between([-x_max, -raio_tubo], [NE, NE], y_base_aquifero, color=COR_SATURADO,
                     alpha=0.65, zorder=2)

    # linha do cone (lado direito, onde fica o poço de observação, se houver)
    ax.plot(x_pos, y_pos, color=COR_CONE, linewidth=1.6, zorder=5)
    # espelha simetricamente para a esquerda apenas como referência visual do cone
    ax.plot(-x_pos, y_pos, color=COR_CONE, linewidth=1.0, linestyle=(0, (4, 3)),
            alpha=0.55, zorder=4)

    # linha de referência do NE regional (tracejada, azul claro)
    ax.axhline(NE, color=COR_NE, linewidth=1.3, linestyle="--", zorder=4,
               xmin=(raio_tubo + 0.3 - x_min) / (x_max - x_min) if not is_obs else 0.02)
    ax.text(x_min + 0.15, NE - 0.18, f"NE = {NE:g} m", color=COR_NE, fontsize=9.5,
             fontweight="bold", va="bottom", ha="left", zorder=6)

    # ------------------------------------------------------------------
    # 3) Poço bombeado — tubo, coluna d'água rebaixada e filtro
    # ------------------------------------------------------------------
    ax.add_patch(Rectangle(
        (-raio_tubo, y_terreno_topo), 2 * raio_tubo, y_fundo_poco - y_terreno_topo,
        facecolor=COR_TUBO, edgecolor=COR_TUBO_BORDA, linewidth=1.2, zorder=8,
    ))
    # coluna d'água dentro do poço, já rebaixada até o ND
    ax.add_patch(Rectangle(
        (-raio_tubo * 0.72, ND), raio_tubo * 1.44, y_fundo_poco - ND,
        facecolor=COR_ND, edgecolor="none", zorder=9,
    ))
    # seção filtrante (hachurada), defronte ao aquífero
    y_filtro_topo = NE + 0.15 * b
    y_filtro_base = y_base_aquifero - 0.25
    ax.add_patch(Rectangle(
        (-raio_tubo * 0.72, y_filtro_topo), raio_tubo * 1.44, y_filtro_base - y_filtro_topo,
        facecolor=COR_FILTRO, edgecolor=COR_FILTRO_BORDA, hatch="---", linewidth=0.8, zorder=10,
    ))

    _nivel_agua_marker(ax, 0.0, ND, COR_ND)
    ax.text(raio_tubo + 0.25, ND, f"ND = {ND:g} m", color=COR_ND, fontsize=9.5,
             fontweight="bold", va="center", ha="left", zorder=6)

    # rótulo do poço bombeado
    ax.text(0.0, y_terreno_topo - 0.25, "Poço Bombeado", color=COR_TEXTO, fontsize=9,
             va="top", ha="center", fontstyle="italic", zorder=6)

    # ------------------------------------------------------------------
    # 4) Raio do poço (r_w) ou poço de observação + distância r
    # ------------------------------------------------------------------
    if not is_obs:
        y_cota_r = y_terreno_topo - 0.7
        _cota_horizontal(ax, y_cota_r, 0.0, raio_tubo, f"$r_w$ = {r:g} m")
    else:
        y_nivel_obs = float(np.interp(x_obs, x_curva, y_cone))
        raio_obs = 0.14
        y_fundo_obs = y_nivel_obs + 1.2

        ax.add_patch(Rectangle(
            (x_obs - raio_obs, y_terreno_topo), 2 * raio_obs, y_fundo_obs - y_terreno_topo,
            facecolor=COR_TUBO, edgecolor=COR_TUBO_BORDA, linewidth=1.0, zorder=8,
        ))
        ax.add_patch(Rectangle(
            (x_obs - raio_obs * 0.7, y_nivel_obs), raio_obs * 1.4, y_fundo_obs - y_nivel_obs,
            facecolor=COR_NE, edgecolor="none", zorder=9,
        ))
        # pequena ponta filtrante do piezômetro
        ax.add_patch(Rectangle(
            (x_obs - raio_obs * 0.7, y_fundo_obs - 0.35), raio_obs * 1.4, 0.35,
            facecolor=COR_FILTRO, edgecolor=COR_FILTRO_BORDA, hatch="---", linewidth=0.6, zorder=10,
        ))
        _nivel_agua_marker(ax, x_obs, y_nivel_obs, COR_NE, tamanho=0.22)
        ax.text(x_obs, y_terreno_topo - 0.25, "Poço de\nObservação", color=COR_TEXTO,
                 fontsize=8.5, va="top", ha="center", fontstyle="italic", zorder=6)

        y_cota_r = y_terreno_topo - 0.7
        _cota_horizontal(ax, y_cota_r, 0.0, x_obs, f"r = {r:g} m")

    # ------------------------------------------------------------------
    # 5) Cota vertical lateral da espessura saturada (b)
    # ------------------------------------------------------------------
    if show_b:
        x_cota_b = x_max - 0.55
        _cota_vertical(ax, x_cota_b, NE, y_base_aquifero, f"b = {b:g} m")

    # rótulo da base impermeável
    ax.text(x_max - 0.15, (y_base_aquifero + y_base_impermeavel) / 2, "Base impermeável",
             color="white", fontsize=8, va="center", ha="right", zorder=6)

    # ------------------------------------------------------------------
    # 6) Ajustes finais dos eixos
    # ------------------------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_base_impermeavel + 0.3, y_terreno_topo - 1.1)  # invertido (profundidade)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0.4)

    return fig
