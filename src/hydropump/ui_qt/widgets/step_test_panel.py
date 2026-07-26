"""Painel do teste de degraus — entrada tabelada e gráfico s/Q × Q.

O usuário informa, por degrau: volume do recipiente, tempo de enchimento e
rebaixamento estabilizado. A vazão de cada degrau é calculada aqui
(Q = V/t) em vez de exigida pronta, porque em campo o que se anota é
justamente o par balde/cronômetro.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydropump.ui_qt.widgets.mpl_canvas import MplCanvas

__all__ = ["StepTestPanel"]

_COLUMNS = (
    "Volume (L)",
    "Tempo de enchimento (s)",
    "Vazão Q (m³/h)",
    "Rebaixamento s (m)",
    "s/Q (m por m³/h)",
)
_COL_VOLUME, _COL_TIME, _COL_Q, _COL_DRAWDOWN, _COL_SPECIFIC = range(5)

#: Colunas calculadas pelo software — o usuário não digita nelas.
_READONLY_COLUMNS = (_COL_Q, _COL_SPECIFIC)

_DEFAULT_ROWS = 4


class StepTestPanel(QWidget):
    """Tabela de degraus + gráfico de rebaixamento específico."""

    compute_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        intro = QLabel(
            "Informe um degrau por linha: volume do recipiente, tempo de "
            "enchimento e rebaixamento <b>estabilizado</b> naquele degrau. "
            "A vazão e o rebaixamento específico são calculados "
            "automaticamente. São necessários ao menos 3 degraus com vazões "
            "distintas."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #5a6472; font-size: 11px;")

        self.table = QTableWidget(_DEFAULT_ROWS, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        header = self.table.horizontalHeader()
        for col in range(len(_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(220)
        for row in range(_DEFAULT_ROWS):
            self._prepare_row(row)
        self.table.itemChanged.connect(self._on_item_changed)

        self.button_add = QPushButton("+ Adicionar degrau")
        self.button_remove = QPushButton("− Remover último")
        self.button_clear = QPushButton("Limpar")
        self.button_compute = QPushButton("Calcular teste de degraus")
        self.button_compute.setToolTip(
            "Ajusta s/Q = B + C·Q (Jacob, 1947), separando a perda de carga\n"
            "no aquífero (B) da perda no próprio poço (C), e calcula a\n"
            "eficiência do poço."
        )

        self.button_add.clicked.connect(lambda *_: self._add_row())
        self.button_remove.clicked.connect(lambda *_: self._remove_row())
        self.button_clear.clicked.connect(lambda *_: self._clear())
        self.button_compute.clicked.connect(lambda *_: self.compute_requested.emit())

        buttons = QHBoxLayout()
        buttons.addWidget(self.button_add)
        buttons.addWidget(self.button_remove)
        buttons.addWidget(self.button_clear)
        buttons.addStretch(1)
        buttons.addWidget(self.button_compute)

        self.label_summary = QLabel("")
        self.label_summary.setWordWrap(True)
        self.label_summary.setStyleSheet(
            "background: #eef4fb; border: 1px solid #c7d9ec; border-radius: 6px; "
            "padding: 8px 12px; color: #0d2b4d; font-size: 12px;"
        )
        self.label_summary.setVisible(False)

        self.canvas = MplCanvas()

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.label_summary)
        layout.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------
    # Tabela
    # ------------------------------------------------------------------
    def _prepare_row(self, row: int) -> None:
        """Cria as células da linha, travando as colunas calculadas."""
        for col in range(len(_COLUMNS)):
            item = QTableWidgetItem("")
            if col in _READONLY_COLUMNS:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item.setBackground(Qt.GlobalColor.transparent)
                item.setForeground(Qt.GlobalColor.darkGray)
            item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            )
            self.table.setItem(row, col, item)

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._prepare_row(row)

    def _remove_row(self) -> None:
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)

    def _clear(self) -> None:
        self.table.setRowCount(0)
        for _ in range(_DEFAULT_ROWS):
            self._add_row()
        self.label_summary.setVisible(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Recalcula Q e s/Q da linha quando volume, tempo ou s mudam."""
        if item.column() in _READONLY_COLUMNS:
            return
        self.table.blockSignals(True)
        try:
            self._recalculate_row(item.row())
        finally:
            self.table.blockSignals(False)

    def _cell_value(self, row: int, col: int) -> float | None:
        item = self.table.item(row, col)
        if item is None:
            return None
        text = item.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _set_cell(self, row: int, col: int, text: str) -> None:
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, col, item)
        item.setText(text)

    def _recalculate_row(self, row: int) -> None:
        volume = self._cell_value(row, _COL_VOLUME)
        seconds = self._cell_value(row, _COL_TIME)
        drawdown = self._cell_value(row, _COL_DRAWDOWN)

        if volume is None or seconds is None or seconds <= 0 or volume <= 0:
            self._set_cell(row, _COL_Q, "")
            self._set_cell(row, _COL_SPECIFIC, "")
            return

        q_m3_h = (volume / 1000.0) / seconds * 3600.0
        self._set_cell(row, _COL_Q, f"{q_m3_h:.3f}")

        if drawdown is not None and drawdown > 0 and q_m3_h > 0:
            self._set_cell(row, _COL_SPECIFIC, f"{drawdown / q_m3_h:.4f}")
        else:
            self._set_cell(row, _COL_SPECIFIC, "")

    # ------------------------------------------------------------------
    # Leitura dos dados
    # ------------------------------------------------------------------
    def steps(self) -> tuple[np.ndarray, np.ndarray]:
        """Degraus válidos como ``(Q [m³/s], rebaixamento [m])``.

        Linhas incompletas ou não numéricas são ignoradas silenciosamente —
        é normal deixar linhas em branco no fim da tabela.
        """
        q_list: list[float] = []
        s_list: list[float] = []
        for row in range(self.table.rowCount()):
            volume = self._cell_value(row, _COL_VOLUME)
            seconds = self._cell_value(row, _COL_TIME)
            drawdown = self._cell_value(row, _COL_DRAWDOWN)
            if None in (volume, seconds, drawdown):
                continue
            if volume <= 0 or seconds <= 0 or drawdown <= 0:
                continue
            q_list.append((volume / 1000.0) / seconds)  # m³/s
            s_list.append(drawdown)
        return np.asarray(q_list, dtype=float), np.asarray(s_list, dtype=float)

    def set_summary(self, text: str) -> None:
        self.label_summary.setText(text)
        self.label_summary.setVisible(True)
