"""Smoke test da interface Qt.

Não testa lógica de negócio (isso já é coberto pelos outros arquivos) —
só garante que a janela principal e todos os widgets se constroem sem
lançar exceção. Roda em modo "offscreen" (sem precisar de um display real),
o que o ``pytest-qt`` + Qt já suportam nativamente via a plataforma
``offscreen`` do próprio Qt.

Pulado automaticamente se PySide6 não estiver instalado no ambiente (por
exemplo, num CI que só roda os testes de domínio/serviço).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def main_window(qtbot):
    from hydropump.ui_qt.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_main_window_monta_sem_erro(main_window):
    assert main_window.windowTitle().startswith("HydroPump")


def test_abas_principais_existem(main_window):
    titles = [main_window.tabs.tabText(i) for i in range(main_window.tabs.count())]
    assert any("Visão Geral" in t for t in titles)
    assert any("Tabela" in t for t in titles)
    assert any("Diagnósticos" in t for t in titles)
    assert any("Recuperação" in t for t in titles)


def test_botoes_de_exportacao_comecam_desabilitados(main_window):
    assert not main_window.action_export_chart.isEnabled()
    assert not main_window.action_export_table.isEnabled()
    assert not main_window.action_export_report.isEnabled()


def test_parameters_panel_constroi_config_valido(main_window):
    cfg = main_window.params.build_config(Q_m3_s=0.01, well_id="PZ-01")
    assert cfg.static_level_m == pytest.approx(10.0)
    assert cfg.radius_m == pytest.approx(0.076)
    assert cfg.well_id == "PZ-01"
