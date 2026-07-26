"""Tema visual compartilhado por todos os gráficos do HydroPump.

Centraliza a paleta e o estilo Matplotlib/Seaborn para que nenhum outro
módulo precise redefinir cores ou chamar ``sns.set_theme`` de novo.
"""

from __future__ import annotations

import matplotlib

# Backend não interativo: obrigatório antes de qualquer outro import de
# matplotlib/seaborn em um processo --windowed (sem terminal). Sem isso, o
# Matplotlib pode tentar resolver um backend Tk/Qt inexistente no ambiente
# empacotado e derrubar o processo sem nenhuma mensagem visível ao usuário.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

__all__ = ["PALETTE", "apply_scientific_style"]

#: Paleta única do módulo — garante coerência entre todas as figuras.
PALETTE = {
    "observed": "#1f5fa8",       # azul petróleo — dados reais
    "observed_edge": "#0d2b4d",
    "projection": "#4a4a4a",     # cinza escuro — modelo teórico
    "fit": "#c0392b",            # vermelho tijolo — reta de ajuste
    "recovery": "#2e8b57",       # verde — recuperação
    "extrapolation": "#7f8c9a",  # faixa de extrapolação
    "annotation": "#2c3e50",
}

_STYLE_APPLIED = False


def apply_scientific_style() -> None:
    """Aplica o tema visual do módulo. Idempotente: chamadas repetidas não acumulam."""
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
