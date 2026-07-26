"""Diálogo do esquema do poço, com controle interativo de penetração parcial.

Mantém o esquema numa janela própria (em vez de inline na aba de visão
geral) e permite alternar, ao vivo, entre poço totalmente e parcialmente
penetrante — a distinção física que motivou este widget: um poço que não
atinge a base do aquífero introduz componentes de fluxo vertical que o
modelo radial de Cooper-Jacob/Theis não captura, e o desenho deve deixar
isso visualmente óbvio, não apenas assumir penetração total.
"""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QVBoxLayout, QWidget

from hydropump.ui_qt.widgets.mpl_canvas import MplCanvas
from hydropump.viz import schematics as sq

__all__ = ["SchematicDialog"]


class SchematicDialog(QDialog):
    """Mostra o esquema do poço, com opção interativa de penetração parcial."""

    def __init__(
        self,
        *,
        well_type: str,
        NE: float,
        ND: float,
        b: float | None,
        r: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Esquema do poço")
        self.resize(600, 720)

        self._well_type = well_type
        self._NE = NE
        self._ND = ND
        self._b = b if b is not None else 20.0
        self._r = r
        self._show_b = b is not None

        self.check_partial = QCheckBox(
            "Poço parcialmente penetrante (não atinge a base do aquífero)"
        )
        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(20.0, 100.0)
        self.spin_fraction.setDecimals(0)
        self.spin_fraction.setSuffix(" %")
        self.spin_fraction.setValue(60.0)
        self.spin_fraction.setEnabled(False)
        self.spin_fraction.setToolTip(
            "Fração da espessura saturada (b) que o filtro do poço efetivamente\n"
            "penetra. Poços parcialmente penetrantes introduzem componentes de\n"
            "fluxo vertical não capturadas pelo modelo radial de Cooper-Jacob/Theis."
        )

        self.check_partial.toggled.connect(self.spin_fraction.setEnabled)
        self.check_partial.toggled.connect(self._redraw)
        self.spin_fraction.valueChanged.connect(self._redraw)

        form = QFormLayout()
        form.addRow(self.check_partial)
        form.addRow("Fração da espessura saturada penetrada:", self.spin_fraction)

        self.canvas = MplCanvas()

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.canvas, 1)

        self._redraw()

    def update_geometry(
        self,
        *,
        well_type: str,
        NE: float,
        ND: float,
        b: float | None,
        r: float,
    ) -> None:
        """Atualiza a geometria exibida sem recriar o diálogo.

        Preserva o estado do checkbox/fração de penetração parcial já
        escolhido pelo usuário — só a geometria (NE, ND, b, r, tipo de
        poço) é atualizada. Usado para manter o esquema em sincronia com
        os valores da barra lateral enquanto o diálogo permanece aberto.
        """
        self._well_type = well_type
        self._NE = NE
        self._ND = ND
        self._b = b if b is not None else 20.0
        self._r = r
        self._show_b = b is not None
        self._redraw()

    def _redraw(self, *_args) -> None:
        fraction = self.spin_fraction.value() / 100.0 if self.check_partial.isChecked() else 1.0
        fig = sq.draw_well_schematic(
            well_type=self._well_type,
            NE=self._NE,
            ND=self._ND,
            b=self._b,
            r=self._r,
            show_b=self._show_b,
            penetration_fraction=fraction,
        )
        self.canvas.set_figure(fig)
