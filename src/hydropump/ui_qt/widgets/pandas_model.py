"""Modelo Qt (QAbstractTableModel) sobre um pandas.DataFrame.

Evita copiar os dados para uma estrutura intermediária: o ``QTableView``
lê diretamente do DataFrame por índice. Colore a linha conforme a coluna de
origem (``tipo_dado``), reproduzindo o destaque visual que a tabela do
Streamlit tinha (azul para medido, cinza para sintético/projeção).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

__all__ = ["DataFrameModel"]

_MEASURED_BG = QColor("#e7f0fb")
_MEASURED_FG = QColor("#0d2b4d")
_SYNTHETIC_BG = QColor("#f2f2f2")
_SYNTHETIC_FG = QColor("#666666")

#: Índice inválido reutilizado como default de rowCount/columnCount — Qt
#: exige exatamente esse valor para reconhecer a assinatura como override
#: do método virtual da classe base. Definido uma única vez em nível de
#: módulo em vez de chamar QModelIndex() no próprio default do argumento.
_ROOT_INDEX = QModelIndex()


class DataFrameModel(QAbstractTableModel):
    """Exibe um DataFrame em um QTableView.

    Parameters
    ----------
    df : pandas.DataFrame
        Dados a exibir.
    origin_column : str | None
        Nome da coluna que identifica a procedência da linha (valores
        ``"Medido"`` vs. qualquer outro). Se ``None``, nenhuma cor é
        aplicada.
    """

    def __init__(
        self, df: pd.DataFrame, origin_column: str | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self._df = df.reset_index(drop=True)
        self._origin_column = origin_column

    def rowCount(self, parent: QModelIndex = _ROOT_INDEX) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = _ROOT_INDEX) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        if self._origin_column is not None and role in (
            Qt.ItemDataRole.BackgroundRole,
            Qt.ItemDataRole.ForegroundRole,
        ):
            origin = str(self._df.at[index.row(), self._origin_column]).strip().lower()
            is_measured = origin == "medido"
            if role == Qt.ItemDataRole.BackgroundRole:
                return _MEASURED_BG if is_measured else _SYNTHETIC_BG
            return _MEASURED_FG if is_measured else _SYNTHETIC_FG

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def dataframe(self) -> pd.DataFrame:
        """Devolve o DataFrame subjacente (para exportação, por exemplo)."""
        return self._df
