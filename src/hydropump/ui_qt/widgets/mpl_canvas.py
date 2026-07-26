"""Canvas Matplotlib embutido em um widget Qt.

Substitui o botão "Baixar gráfico PNG" do Streamlit pela barra de
ferramentas nativa do Matplotlib (zoom, pan, salvar em qualquer formato),
que é o que um usuário de desktop espera encontrar sob um gráfico.
"""

from __future__ import annotations

import os

# Fixa explicitamente o binding Qt usado pelo Matplotlib. Sem isso, se mais
# de um binding (PyQt5/6, PySide2/6) estiver instalado no ambiente do
# usuário, o matplotlib pode escolher um diferente do que a aplicação usa,
# causando erro de "duas cópias do Qt carregadas" em runtime.
os.environ.setdefault("QT_API", "pyside6")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

__all__ = ["MplCanvas"]


class MplCanvas(QWidget):
    """Canvas + barra de ferramentas, com troca segura de figura.

    Cada chamada a :meth:`set_figure` fecha a figura anterior via
    ``plt.close()`` — necessário porque :mod:`hydropump.viz.plots` cria
    figuras via ``plt.subplots()`` (registradas no gerenciador global do
    pyplot); sem o `close`, cada recálculo do usuário vazaria uma figura.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self.set_figure(Figure(figsize=(6, 4)))

    def set_figure(self, figure: Figure) -> None:
        """Substitui a figura exibida, liberando a anterior."""
        old_figure = self._figure
        if self._canvas is not None:
            self._layout.removeWidget(self._toolbar)
            self._layout.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()

        self._figure = figure
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

        if old_figure is not None and old_figure is not figure:
            plt.close(old_figure)

    def figure(self) -> Figure:
        """Figura atualmente exibida."""
        assert self._figure is not None
        return self._figure

    def save_png(self, path: str, dpi: int = 300) -> None:
        """Salva a figura atual em PNG, no dpi pedido (padrão: qualidade de impressão)."""
        self.figure().savefig(path, dpi=dpi, bbox_inches="tight")

    def closeEvent(self, event) -> None:  # noqa: N802 — nome exigido pelo Qt
        if self._figure is not None:
            plt.close(self._figure)
        super().closeEvent(event)
