"""Cartão de métrica hidrodinâmica — equivalente Qt ao card HTML do Streamlit."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

__all__ = ["MetricCard"]

_VARIANT_BORDER = {
    "default": "#1f5fa8",
    "alt": "#c0392b",
    "alt2": "#2e8b57",
}


class MetricCard(QFrame):
    """Cartão com rótulo, valor em destaque e subtexto opcional.

    Parameters
    ----------
    label : str
        Rótulo curto (ex.: "Transmissividade, T").
    value : str
        Valor formatado em destaque.
    sub : str
        Subtexto (ex.: "R² = 0.998").
    variant : {"default", "alt", "alt2"}
        Cor da barra lateral esquerda: azul, vermelho tijolo ou verde.
    """

    def __init__(
        self,
        label: str,
        value: str = "—",
        sub: str = "",
        variant: str = "default",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(96)

        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            "color: #5a6472; font-size: 11px; letter-spacing: 0.5px; background: transparent;"
        )

        self._value = QLabel(value)
        self._value.setStyleSheet(
            "color: #0d2b4d; font-size: 20px; font-weight: 700; background: transparent;"
        )

        self._sub = QLabel(sub)
        self._sub.setStyleSheet("color: #7f8c9a; font-size: 11px; background: transparent;")
        self._sub.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addWidget(self._value)
        layout.addWidget(self._sub)
        layout.addStretch(1)

        self.set_variant(variant)

    def set_value(self, value: str, sub: str = "") -> None:
        """Atualiza o valor exibido e o subtexto."""
        self._value.setText(value)
        self._sub.setText(sub)

    def set_variant(self, variant: str) -> None:
        """Troca a cor da barra lateral (ex.: para sinalizar valor implausível)."""
        color = _VARIANT_BORDER.get(variant, _VARIANT_BORDER["default"])
        self.setStyleSheet(
            f"QFrame#metricCard {{ background: #f7f9fc; border: 1px solid #d7dee6; "
            f"border-left: 5px solid {color}; border-radius: 8px; }}"
        )
