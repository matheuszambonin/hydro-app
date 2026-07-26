"""
main.py
-------
Entry point da aplicação desktop HydroPump (PySide6/Qt).

Comparado ao antigo ``run_app.py`` (Streamlit), este arquivo é
propositalmente simples: não há servidor local, não há porta a escolher,
não há navegador a abrir, não há trava de instância baseada em socket — o
próprio sistema operacional já impede duas janelas do mesmo processo GUI
de conflitarem entre si. O app abre como qualquer programa de desktop.

Referenciado no ``app.spec`` como ``Analysis(["main.py"])``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

APP_NAME = "HydroPump"

# Backend Matplotlib de renderização (ver hydropump.viz.theme); harmless
# aqui porque os gráficos são embutidos via FigureCanvasQTAgg diretamente,
# sem depender do backend global do pyplot.
os.environ.setdefault("QT_API", "pyside6")


def _log_path() -> str:
    log_dir = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), APP_NAME)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "hydropump.log")


def _install_crash_logger() -> None:
    """Em build --windowed não há console para ver um traceback não tratado.

    Registra qualquer exceção não capturada da thread principal em arquivo,
    em vez de deixar o processo simplesmente sumir sem explicação.
    """
    log_path = _log_path()

    def _excepthook(exc_type, exc_value, exc_tb):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n--- Erro não tratado ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        # Também tenta mostrar um diálogo, se o Qt já estiver de pé.
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "HydroPump — erro inesperado",
                    f"Ocorreu um erro inesperado.\n\nDetalhes em:\n{log_path}",
                )
        except Exception:
            pass

    sys.excepthook = _excepthook


def main() -> int:
    _install_crash_logger()

    # Permite `python main.py` sem instalar o pacote (modo dev).
    src_dir = Path(__file__).resolve().parent / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from PySide6.QtWidgets import QApplication

    from hydropump.ui_qt.main_window import MainWindow
    from hydropump.ui_qt.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
