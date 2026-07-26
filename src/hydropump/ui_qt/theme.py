"""Tema visual único da aplicação Qt — claro, forçado independente do SO.

Por padrão, o Qt herda a paleta de cores do sistema operacional. Num SO com
modo escuro ativado, isso mistura o fundo escuro nativo (nas fontes de
menus, campos, abas, tabelas) com os widgets customizados desta aplicação
que assumem fundo claro (:class:`~hydropump.ui_qt.widgets.metric_card.MetricCard`,
:class:`~hydropump.ui_qt.widgets.diagnostics_panel.DiagnosticsPanel`), e o
resultado são textos de baixo contraste ou lisos ilegíveis — exatamente o
sintoma relatado.

A correção não é ajustar cor por cor: é parar de depender do tema do SO.
Este módulo aplica explicitamente uma paleta (:class:`QPalette`) e uma
folha de estilo (QSS) fixas sobre o estilo ``Fusion`` do Qt — a mesma
abordagem usada por aplicações profissionais que não quer herdar
comportamento imprevisível do ambiente do usuário. Chame :func:`apply_theme`
uma única vez, logo após criar o ``QApplication``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = [
    "apply_theme",
    "BACKGROUND",
    "SURFACE",
    "SURFACE_ALT",
    "BORDER",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "ACCENT",
    "ACCENT_TEXT",
]

# Paleta base — também reaproveitada pelos widgets customizados (cards,
# diagnósticos) para garantir que tudo fale a mesma linguagem visual.
BACKGROUND = "#f4f6f9"
SURFACE = "#ffffff"
SURFACE_ALT = "#eef2f7"
BORDER = "#c9d3de"
#: Borda de campos editáveis e do indicador de caixa de seleção: mais
#: escura que BORDER de propósito, para que a caixa vazia seja claramente
#: visível sobre fundo claro (era o problema das caixas "invisíveis").
INPUT_BORDER = "#9aa8b6"
INDICATOR_BORDER = "#7a8794"
TEXT_PRIMARY = "#1c2733"
TEXT_SECONDARY = "#5a6472"
ACCENT = "#1f5fa8"
ACCENT_TEXT = "#ffffff"
DISABLED_TEXT = "#9aa5b1"


def _build_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#c0392b"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(ACCENT_TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_SECONDARY))

    # Papéis "3D". O estilo Fusion deriva DESTES as bordas de caixas de
    # seleção, botões de rádio, campos e molduras. Deixá-los sem definir é
    # o que fazia as caixas de seleção quase sumirem: o Qt caía em valores
    # calculados a partir de Window, resultando em cinza-claro sobre
    # cinza-claro. Defini-los explicitamente é a correção de raiz — não
    # adianta só pintar a borda via folha de estilo.
    palette.setColor(QPalette.ColorRole.Light, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#e3e9f0"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#b3c0cd"))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#7a8794"))
    palette.setColor(QPalette.ColorRole.Shadow, QColor("#5a6472"))

    disabled = QColor(DISABLED_TEXT)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)

    return palette


_STYLESHEET_BASE = f"""
QWidget {{
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QMainWindow, QDialog {{
    background: {BACKGROUND};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BACKGROUND};
    border: none;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    background: {SURFACE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_PRIMARY};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {SURFACE};
}}
QTabBar::tab {{
    background: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    padding: 6px 14px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {SURFACE};
    font-weight: 600;
}}
QHeaderView::section {{
    background: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    padding: 4px;
    border: 1px solid {BORDER};
}}
QTableView {{
    background: {SURFACE};
    color: {TEXT_PRIMARY};
    gridline-color: {BORDER};
}}
QToolTip {{
    background: {TEXT_PRIMARY};
    color: {SURFACE};
    border: none;
    padding: 4px 6px;
}}
QPushButton {{
    background: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background: #e2e9f2;
}}
QPushButton:disabled {{
    color: {DISABLED_TEXT};
}}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QDateTimeEdit, QPlainTextEdit {{
    background: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {INPUT_BORDER};
    border-radius: 4px;
    padding: 3px 6px;
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus,
QDateTimeEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QMenuBar {{
    background: {SURFACE};
    color: {TEXT_PRIMARY};
}}
QMenuBar::item:selected {{
    background: {SURFACE_ALT};
}}
QMenu {{
    background: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
}}
QStatusBar {{
    background: {SURFACE_ALT};
    color: {TEXT_SECONDARY};
}}
QCheckBox, QRadioButton, QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
"""

# Folha de estilo do indicador de caixa de seleção / botão de rádio.
# Só é aplicada se conseguirmos gravar os ícones de marcação em disco:
# ao estilizar ``::indicator`` via QSS, o Qt deixa de desenhar o "✓"
# nativo, então precisamos fornecer a imagem nós mesmos.
_INDICATOR_QSS_TEMPLATE = """
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    background: {surface};
    border: 1px solid {indicator_border};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border: 1px solid {accent};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border: 1px solid {accent};
    image: url("{check_icon}");
}}
QRadioButton::indicator:checked {{
    background: {accent};
    border: 4px solid {surface};
    outline: 1px solid {accent};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {disabled_bg};
    border: 1px solid {disabled_border};
}}
QCheckBox::indicator:checked:disabled {{
    background: {disabled_border};
    border: 1px solid {disabled_border};
}}
"""

_CHECK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" '
    'viewBox="0 0 15 15">'
    '<path d="M3.2 7.8 L6.1 10.7 L11.8 4.6" fill="none" stroke="#ffffff" '
    'stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)


def _write_indicator_icons() -> str | None:
    """Grava o ícone de marcação em disco e devolve o caminho (ou ``None``).

    O QSS do Qt não aceita data-URI em ``image:``, só caminho de arquivo ou
    recurso compilado. Como não queremos depender de ``.qrc`` compilado,
    gravamos o SVG no diretório de dados do aplicativo na primeira execução.
    Se isso falhar (permissão, disco cheio), devolvemos ``None`` e o tema
    simplesmente não estiliza o indicador — a correção de paleta acima já
    garante contraste aceitável por si só.
    """
    try:
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        target_dir = Path(base) / "icons" if base else Path(tempfile.gettempdir()) / "hydropump-icons"
        target_dir.mkdir(parents=True, exist_ok=True)
        icon_path = target_dir / "check.svg"
        if not icon_path.exists() or icon_path.read_text(encoding="utf-8") != _CHECK_SVG:
            icon_path.write_text(_CHECK_SVG, encoding="utf-8")
        # QSS exige barras normais, inclusive no Windows.
        return str(icon_path).replace("\\", "/")
    except Exception:  # noqa: BLE001 — degradação silenciosa é intencional
        return None


def _build_stylesheet() -> str:
    qss = _STYLESHEET_BASE
    check_icon = _write_indicator_icons()
    if check_icon:
        qss += _INDICATOR_QSS_TEMPLATE.format(
            surface=SURFACE,
            indicator_border=INDICATOR_BORDER,
            accent=ACCENT,
            check_icon=check_icon,
            disabled_bg="#eceff3",
            disabled_border="#b8c2cc",
        )
    return qss


def apply_theme(app: QApplication) -> None:
    """Aplica o tema claro fixo. Chame uma vez, logo após criar o QApplication."""
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(_build_stylesheet())
