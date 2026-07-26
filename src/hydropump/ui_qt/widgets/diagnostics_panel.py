"""Painel de diagnósticos tipados (:class:`hydropump.domain.models.Diagnostic`).

Renderiza cada :class:`Diagnostic` como um cartão colorido por gravidade —
o equivalente Qt de ``st.error/st.warning/st.info`` no Streamlit, mas
persistente numa aba própria em vez de aparecer inline no fluxo de widgets.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from hydropump.domain.models import Diagnostic, Severity

__all__ = ["DiagnosticsPanel"]

# (cor da borda/título, cor de fundo, ícone)
_SEVERITY_STYLE: dict[Severity, tuple[str, str, str]] = {
    Severity.CRITICAL: ("#c0392b", "#fdf1ef", "🛑"),
    Severity.WARNING: ("#9a6b00", "#fff8e6", "⚠️"),
    Severity.INFO: ("#1f5fa8", "#eef4fb", "ℹ️"),
}

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
}


class _DiagnosticCard(QFrame):
    def __init__(self, diagnostic: Diagnostic, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        border, bg, icon = _SEVERITY_STYLE[diagnostic.severity]
        self.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 1px solid {border}; "
            f"border-left: 4px solid {border}; border-radius: 6px; }}"
        )

        title = QLabel(f"{icon}  {diagnostic.title}")
        title.setStyleSheet(
            f"color: {border}; font-weight: 700; font-size: 13px; background: transparent; border: none;"
        )
        title.setWordWrap(True)

        detail = QLabel(diagnostic.detail)
        detail.setStyleSheet("color: #2c3e50; font-size: 12px; background: transparent; border: none;")
        detail.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addWidget(detail)


class DiagnosticsPanel(QScrollArea):
    """Lista rolável de diagnósticos, ordenada por gravidade decrescente."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._empty_label = QLabel(
            "Nenhum alerta — resultado dentro dos parâmetros esperados. ✅"
        )
        self._empty_label.setStyleSheet("color: #2e8b57; font-size: 13px; padding: 12px;")
        self._layout.insertWidget(0, self._empty_label)

    def set_diagnostics(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        """Substitui a lista exibida pelos diagnósticos fornecidos."""
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not diagnostics:
            self._layout.insertWidget(0, self._empty_label)
            return

        ordered = sorted(diagnostics, key=lambda d: _SEVERITY_ORDER[d.severity])
        for i, d in enumerate(ordered):
            self._layout.insertWidget(i, _DiagnosticCard(d))
