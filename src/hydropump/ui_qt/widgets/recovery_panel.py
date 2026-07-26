"""Painel da fase de recuperação (Theis) — aba secundária da janela principal.

Este widget só cuida de UI (formulário + canvas + rótulo de resumo). A
extração da série, o mapeamento de colunas e a chamada ao serviço ficam na
janela principal, que já tem acesso ao :class:`AnalysisConfig` corrente.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydropump.ui_qt.widgets.mpl_canvas import MplCanvas

__all__ = ["RecoveryPanel"]


class RecoveryPanel(QWidget):
    """Carrega dados de recuperação, parametriza e dispara o cálculo de Theis."""

    load_requested = Signal()
    compute_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.button_load = QPushButton("Carregar dados de recuperação...")
        self.button_load.setToolTip(
            "Carrega um CSV/Excel com o tempo desde o desligamento da bomba (t')\n"
            "e o nível dinâmico (ou rebaixamento residual) medido durante a\n"
            "recuperação."
        )
        self.label_file = QLabel("Nenhum arquivo carregado.")
        self.label_file.setStyleSheet("color: #7f8c9a; font-size: 11px;")

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 100_000.0)
        self.spin_duration.setDecimals(1)
        self.spin_duration.setSuffix(" min")
        self.spin_duration.setValue(120.0)
        self.spin_duration.setToolTip(
            "Duração total do bombeamento (t_p): quanto tempo a bomba ficou\n"
            "ligada antes de ser desligada para iniciar a recuperação."
        )

        self.check_window = QCheckBox("Restringir janela de ajuste em t/t'")
        self.check_window.setToolTip(
            "Descarta razões t/t' muito altas (início da recuperação), onde o\n"
            "armazenamento no poço ainda domina e a reta não é confiável."
        )
        self.spin_ratio_min = QDoubleSpinBox()
        self.spin_ratio_min.setRange(1.0, 1_000_000.0)
        self.spin_ratio_min.setDecimals(2)
        self.spin_ratio_min.setValue(1.0)
        self.spin_ratio_min.setEnabled(False)
        self.spin_ratio_min.setToolTip("Menor razão t/t' incluída no ajuste da reta.")
        self.spin_ratio_max = QDoubleSpinBox()
        self.spin_ratio_max.setRange(1.0, 1_000_000.0)
        self.spin_ratio_max.setDecimals(2)
        self.spin_ratio_max.setValue(1000.0)
        self.spin_ratio_max.setEnabled(False)
        self.spin_ratio_max.setToolTip("Maior razão t/t' incluída no ajuste da reta.")
        self.check_window.toggled.connect(self.spin_ratio_min.setEnabled)
        self.check_window.toggled.connect(self.spin_ratio_max.setEnabled)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.addRow("Duração do bombeamento:", self.spin_duration)
        form.addRow(self.check_window)
        form.addRow("t/t' mínimo:", self.spin_ratio_min)
        form.addRow("t/t' máximo:", self.spin_ratio_max)
        box = QGroupBox("Parâmetros da recuperação")
        box.setLayout(form)

        self.button_compute = QPushButton("Calcular recuperação de Theis")
        self.button_compute.setEnabled(False)
        self.button_compute.setToolTip(
            "Ajusta a reta s'' × log10(t/t') e compara T' (recuperação, imune\n"
            "ao skin effect) com T obtido do rebaixamento."
        )

        self.label_summary = QLabel("")
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet(
            "background: #eef4fb; border: 1px solid #c7d9ec; border-radius: 6px; "
            "padding: 8px 12px; color: #0d2b4d; font-size: 12px;"
        )
        self.label_summary.setVisible(False)

        self.canvas = MplCanvas()

        top_row = QHBoxLayout()
        top_row.addWidget(self.button_load)
        top_row.addWidget(self.label_file, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(box)
        layout.addWidget(self.button_compute)
        layout.addWidget(self.label_summary)
        layout.addWidget(self.canvas, 1)

        self.button_load.clicked.connect(lambda *_: self.load_requested.emit())
        self.button_compute.clicked.connect(lambda *_: self.compute_requested.emit())

    def ratio_min(self) -> float | None:
        """Limite inferior de t/t' para o ajuste, ou ``None`` (sem restrição)."""
        return self.spin_ratio_min.value() if self.check_window.isChecked() else None

    def ratio_max(self) -> float | None:
        """Limite superior de t/t' para o ajuste, ou ``None`` (sem restrição)."""
        return self.spin_ratio_max.value() if self.check_window.isChecked() else None

    def set_file_label(self, text: str) -> None:
        """Atualiza o rótulo do arquivo carregado e habilita o botão de cálculo."""
        self.label_file.setText(text)
        self.button_compute.setEnabled(True)

    def set_summary(self, text: str) -> None:
        """Exibe o resumo comparativo T (rebaixamento) x T' (recuperação)."""
        self.label_summary.setText(text)
        self.label_summary.setVisible(True)
