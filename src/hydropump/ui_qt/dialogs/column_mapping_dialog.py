"""Diálogo de mapeamento de colunas da caderneta de campo.

Reproduz, em janela modal, o expander "Mapeamento de Colunas" da versão
Streamlit — com detecção automática de colunas por alias e possibilidade de
correção manual. Reutilizável tanto para a fase de rebaixamento quanto para
a de recuperação, bastando trocar os conjuntos de aliases e o rótulo do
campo de valor.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from hydropump.io.mapping import detect_column

__all__ = ["ColumnMappingDialog"]


class ColumnMappingDialog(QDialog):
    """Confirma ou corrige a detecção automática de colunas de uma tabela.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabela cujas colunas serão mapeadas.
    time_aliases : sequência de str
        Aliases usados para detectar a coluna de tempo.
    level_aliases : sequência de str
        Aliases da coluna de nível (ND ou ND de recuperação).
    drawdown_aliases : sequência de str
        Aliases da coluna de rebaixamento (direto ou residual).
    value_label : str
        Rótulo do campo de valor quando **não** é nível dinâmico (ex.:
        "rebaixamento (m)" ou "rebaixamento residual (m)").
    level_checkbox_label : str
        Texto do checkbox que alterna entre nível dinâmico e valor direto.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        time_aliases: tuple[str, ...],
        level_aliases: tuple[str, ...],
        drawdown_aliases: tuple[str, ...],
        value_label: str = "rebaixamento (m)",
        level_checkbox_label: str = (
            "Arquivo traz Nível Dinâmico (ND) em vez de rebaixamento direto"
        ),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mapeamento de colunas")
        self.setMinimumWidth(440)

        columns = [str(c) for c in df.columns]
        detected_t = detect_column(df.columns, time_aliases)
        detected_nd = detect_column(df.columns, level_aliases)
        detected_s = detect_column(df.columns, drawdown_aliases)

        self.combo_time = QComboBox()
        self.combo_time.addItems(columns)
        if detected_t:
            self.combo_time.setCurrentText(str(detected_t))

        self.radio_min = QRadioButton("minutos")
        self.radio_sec = QRadioButton("segundos")
        self.radio_min.setChecked(True)
        unit_row = QHBoxLayout()
        unit_row.addWidget(self.radio_min)
        unit_row.addWidget(self.radio_sec)
        unit_row.addStretch(1)
        unit_widget = QWidget()
        unit_widget.setLayout(unit_row)

        self.check_level = QCheckBox(level_checkbox_label)
        self.check_level.setChecked(detected_nd is not None and detected_s is None)

        self.combo_value = QComboBox()
        self.combo_value.addItems(columns)
        preferred = detected_nd if self.check_level.isChecked() else detected_s
        if preferred:
            self.combo_value.setCurrentText(str(preferred))

        self._value_label_widget = QLabel(f"Coluna de {value_label}:")
        self._level_label_text = "Coluna de nível dinâmico (m):"
        self._value_label_text = f"Coluna de {value_label}:"
        self.check_level.toggled.connect(self._on_toggle_level)

        form = QFormLayout()
        form.addRow("Coluna de tempo:", self.combo_time)
        form.addRow("Unidade do tempo:", unit_widget)
        form.addRow(self.check_level)
        form.addRow(self._value_label_widget, self.combo_value)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self._on_toggle_level(self.check_level.isChecked())

    def _on_toggle_level(self, checked: bool) -> None:
        self._value_label_widget.setText(
            self._level_label_text if checked else self._value_label_text
        )

    def result_mapping(self) -> dict:
        """Estado escolhido, pronto para :func:`hydropump.io.mapping.extract_series`."""
        return {
            "time_column": self.combo_time.currentText(),
            "value_column": self.combo_value.currentText(),
            "time_unit": "segundos" if self.radio_sec.isChecked() else "minutos",
            "values_are_levels": self.check_level.isChecked(),
        }

    @staticmethod
    def detection_is_confident(
        df: pd.DataFrame,
        time_aliases: tuple[str, ...],
        level_aliases: tuple[str, ...],
        drawdown_aliases: tuple[str, ...],
    ) -> bool:
        """``True`` se tempo e (nível ou rebaixamento) foram detectados sem ambiguidade."""
        return detect_column(df.columns, time_aliases) is not None and (
            detect_column(df.columns, level_aliases) is not None
            or detect_column(df.columns, drawdown_aliases) is not None
        )
