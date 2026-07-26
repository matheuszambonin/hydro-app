"""
Diagrama esquemático vetorial (corte 2D) do poço de bombeamento, gerado com
Matplotlib, para exibição no Streamlit (ex.: dentro de um
``st.expander("Ver Esquema do Poço")``).

O desenho é puramente instrutivo (fora de escala real) e mostra:
    - Terreno / coluna de material do aquífero e base impermeável;
    - Nível Estático (NE) e Nível Dinâmico (ND), com o cone de rebaixamento;
    - Tubo do poço com a seção filtrante hachurada;
    - (opcional) poço de observação a uma distância radial "r";
    - (opcional) cota vertical lateral indicando a espessura saturada "b".

Uso típico::

    from hydropump.viz import schematics as sq
    fig = sq.draw_well_schematic(well_type="observation", NE=2.0, ND=5.0, b=10.0, r=15.0)
    st.pyplot(fig)
    plt.close(fig)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from hydropump.viz.theme import apply_scientific_style

apply_scientific_style()

import matplotlib.pyplot as plt  # noqa: E402  (após garantir backend Agg)

__all__ = ["draw_well_schematic"]

WellType = Literal["single", "observation"]

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
COR_NE = "#5fb8e8"
COR_ND = "#0b4f8a"
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
    well_type: WellType = "single",
    NE: float = 2.0,
    ND: float = 5.0,
    b: float = 10.0,
    r: float = 0.076,
    show_b: bool = True,
    penetration_fraction: float = 1.0,
):
    """
    Gera um diagrama esquemático vetorial (corte 2D) do poço de bombeamento.

    Parameters
    ----------
    well_type : {"single", "observation"}
        "single"      -> mostra apenas o poço bombeado, com indicação de r_w.
        "observation" -> mostra o poço bombeado e um poço de observação a
                         uma distância radial "r", com o cone de rebaixamento.
    NE : float
        Nível Estático (m).
    ND : float
        Nível Dinâmico (m). Deve ser >= NE.
    b : float
        Espessura saturada do aquífero (m), medida a partir do NE.
    r : float
        Se ``well_type == "single"``: raio do poço, r_w (m).
        Se ``well_type == "observation"``: distância radial ao poço de
        observação (m).
    show_b : bool
        Se True, desenha a cota vertical lateral da espessura saturada.
    penetration_fraction : float, opcional
        Fração da espessura saturada que o poço efetivamente penetra,
        entre 0.20 e 1.0 (padrão: 1.0 — poço **totalmente penetrante**,
        cujo filtro alcança a base do aquífero). Valores menores que 1.0
        desenham um poço **parcialmente penetrante**: o revestimento e o
        filtro terminam no meio da coluna saturada, com a zona saturada
        (azul) e a base impermeável continuando visivelmente abaixo do
        fundo do poço — a distinção física é importante porque poços
        parcialmente penetrantes introduzem componentes de fluxo vertical
        não capturadas pelo modelo radial de Cooper-Jacob/Theis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura com fundo transparente, sem eixos numéricos. O chamador é
        responsável por ``plt.close(fig)`` após exibi-la.
    """
    NE = float(NE)
    ND = float(max(ND, NE + 0.05))
    b = float(max(b, 0.5))
    r = float(max(r, 0.001))
    penetration_fraction = float(np.clip(penetration_fraction, 0.20, 1.0))
    is_partial = penetration_fraction < 0.999
    is_obs = well_type == "observation"

    y_terreno_topo = -0.9
    y_solo = 0.0
    y_base_aquifero = NE + b
    y_base_impermeavel = y_base_aquifero + 0.9

    # Profundidade até onde o revestimento/filtro do poço efetivamente
    # penetra. Totalmente penetrante: alcança quase a base do aquífero
    # (comportamento original). Parcialmente penetrante: termina no meio
    # da coluna saturada, com a zona azul (saturada) e a base impermeável
    # seguindo visivelmente abaixo do fundo do poço.
    if is_partial:
        y_fundo_poco = NE + penetration_fraction * b
        y_filtro_topo = NE + 0.15 * b
        y_filtro_base = NE + max(penetration_fraction * b - 0.10 * b, 0.20 * b)
    else:
        y_fundo_poco = y_base_aquifero + 0.5
        y_filtro_topo = NE + 0.15 * b
        y_filtro_base = y_base_aquifero - 0.25

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

    # 1) Terreno, matriz do aquífero e base impermeável
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

    # 2) Cone de rebaixamento
    x_curva = np.linspace(0.05, x_max, 200)
    decaimento = (x_max - raio_tubo) / 2.6
    y_cone = NE + (ND - NE) * np.exp(-(x_curva - raio_tubo) / decaimento)
    y_cone = np.clip(y_cone, NE, ND)

    x_pos = np.concatenate(([raio_tubo], x_curva))
    y_pos = np.concatenate(([ND], y_cone))
    ax.fill_between(x_pos, y_pos, y_base_aquifero, color=COR_SATURADO, alpha=0.65, zorder=2)
    ax.fill_between([-x_max, -raio_tubo], [NE, NE], y_base_aquifero, color=COR_SATURADO,
                     alpha=0.65, zorder=2)

    ax.plot(x_pos, y_pos, color=COR_CONE, linewidth=1.6, zorder=5)
    ax.plot(-x_pos, y_pos, color=COR_CONE, linewidth=1.0, linestyle=(0, (4, 3)),
            alpha=0.55, zorder=4)

    ax.axhline(NE, color=COR_NE, linewidth=1.3, linestyle="--", zorder=4,
               xmin=(raio_tubo + 0.3 - x_min) / (x_max - x_min) if not is_obs else 0.02)
    ax.text(x_min + 0.15, NE - 0.18, f"NE = {NE:g} m", color=COR_NE, fontsize=9.5,
             fontweight="bold", va="bottom", ha="left", zorder=6)

    # 3) Poço bombeado
    ax.add_patch(Rectangle(
        (-raio_tubo, y_terreno_topo), 2 * raio_tubo, y_fundo_poco - y_terreno_topo,
        facecolor=COR_TUBO, edgecolor=COR_TUBO_BORDA, linewidth=1.2, zorder=8,
    ))
    ax.add_patch(Rectangle(
        (-raio_tubo * 0.72, ND), raio_tubo * 1.44, y_fundo_poco - ND,
        facecolor=COR_ND, edgecolor="none", zorder=9,
    ))
    ax.add_patch(Rectangle(
        (-raio_tubo * 0.72, y_filtro_topo), raio_tubo * 1.44, y_filtro_base - y_filtro_topo,
        facecolor=COR_FILTRO, edgecolor=COR_FILTRO_BORDA, hatch="---", linewidth=0.8, zorder=10,
    ))

    _nivel_agua_marker(ax, 0.0, ND, COR_ND)
    ax.text(raio_tubo + 0.25, ND, f"ND = {ND:g} m", color=COR_ND, fontsize=9.5,
             fontweight="bold", va="center", ha="left", zorder=6)
    ax.text(0.0, y_terreno_topo - 0.25, "Poço Bombeado", color=COR_TEXTO, fontsize=9,
             va="top", ha="center", fontstyle="italic", zorder=6)

    if is_partial:
        # Linha tracejada logo abaixo do fundo do poço, sinalizando que a
        # zona saturada (e a base impermeável) continuam além de onde o
        # poço efetivamente penetra — a diferença física central entre um
        # poço parcial e um totalmente penetrante.
        y_linha_continuacao = y_fundo_poco + 0.12
        ax.plot(
            [-raio_tubo * 2.2, x_max * 0.55], [y_linha_continuacao, y_linha_continuacao],
            color=COR_AQUIFERO_BORDA, linewidth=0.9, linestyle=(0, (2, 2)),
            alpha=0.85, zorder=5,
        )

        # O rótulo "ND = ... m" fica em x = raio_tubo + 0.25, y = ND. Quando
        # o fundo do poço cai perto do nível dinâmico (acontece quando
        # penetration_fraction * b ≈ rebaixamento), os dois textos colidem.
        #
        # A folga precisa ser expressa em unidades de DADO, e a conversão
        # depende da escala do eixo. Como o aspecto é "equal", a escala é
        # ditada pelo eixo mais restritivo — por isso o max() entre as duas
        # razões. Escalar a folga apenas com b falha quando NE é grande e b
        # é pequeno, pois nesse caso a extensão vertical do eixo (NE+b+...)
        # é muito maior que b.
        _fig_w, _fig_h = 7.2, 7.6  # deve acompanhar o figsize usado abaixo
        y_range = (y_base_impermeavel + 0.3) - (y_terreno_topo - 1.1)
        x_range = x_max - x_min
        data_per_inch = max(x_range / _fig_w, y_range / _fig_h)
        line_height_data = (8.0 / 72.0) * data_per_inch  # uma linha de 8 pt
        folga = 3.4 * line_height_data

        y_anno = y_fundo_poco + 0.10
        deslocado = abs(y_anno - ND) < folga
        if deslocado:
            y_anno = ND + folga if y_fundo_poco >= ND else ND - folga
            # Linha-guia ligando o rótulo deslocado ao fundo real do poço,
            # para não perder a referência do que ele está anotando.
            ax.plot(
                [raio_tubo + 0.18, raio_tubo + 0.18],
                [y_fundo_poco, y_anno],
                color=COR_TUBO_BORDA, linewidth=0.7, linestyle="-",
                alpha=0.6, zorder=5,
            )

        ax.text(
            raio_tubo + 0.25, y_anno,
            "Poço parcialmente penetrante\n(aquífero continua abaixo)",
            color=COR_TUBO_BORDA, fontsize=8, va="top", ha="left",
            fontstyle="italic", zorder=6,
        )

    # 4) Raio do poço (r_w) ou poço de observação + distância r
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
        ax.add_patch(Rectangle(
            (x_obs - raio_obs * 0.7, y_fundo_obs - 0.35), raio_obs * 1.4, 0.35,
            facecolor=COR_FILTRO, edgecolor=COR_FILTRO_BORDA, hatch="---", linewidth=0.6, zorder=10,
        ))
        _nivel_agua_marker(ax, x_obs, y_nivel_obs, COR_NE, tamanho=0.22)
        ax.text(x_obs, y_terreno_topo - 0.25, "Poço de\nObservação", color=COR_TEXTO,
                 fontsize=8.5, va="top", ha="center", fontstyle="italic", zorder=6)

        y_cota_r = y_terreno_topo - 0.7
        _cota_horizontal(ax, y_cota_r, 0.0, x_obs, f"r = {r:g} m")

    # 5) Cota vertical lateral da espessura saturada (b)
    if show_b:
        x_cota_b = x_max - 0.55
        _cota_vertical(ax, x_cota_b, NE, y_base_aquifero, f"b = {b:g} m")

    ax.text(x_max - 0.15, (y_base_aquifero + y_base_impermeavel) / 2, "Base impermeável",
             color="white", fontsize=8, va="center", ha="right", zorder=6)

    # 6) Ajustes finais dos eixos
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_base_impermeavel + 0.3, y_terreno_topo - 1.1)  # invertido (profundidade)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0.4)

    return fig
