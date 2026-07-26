"""Janela principal da aplicação HydroPump (Qt).

Deliberadamente fina: toda a lógica de negócio (leitura de arquivo,
detecção de colunas, cálculo, diagnósticos) vive em ``hydropump.io`` e
``hydropump.services``. Este arquivo monta widgets, conecta sinais e
delega — o mesmo princípio da antiga ``ui/app.py`` do Streamlit, agora sem
servidor, sem navegador e sem porta.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydropump.domain import math as hm
from hydropump.domain.metadata import PumpMetadata, WellMetadata
from hydropump.domain.models import (
    AnalysisResult,
    RecoveryConfig,
    Severity,
)
from hydropump.io.mapping import (
    DRAWDOWN_ALIASES,
    DYNAMIC_LEVEL_ALIASES,
    RECOVERY_DRAWDOWN_ALIASES,
    RECOVERY_LEVEL_ALIASES,
    RECOVERY_TIME_ALIASES,
    TIME_ALIASES,
    ColumnMappingError,
    TimeSeries,
    extract_series,
)
from hydropump.io.readers import TabularReadError, read_path
from hydropump.report.text_report import build_text_report
from hydropump.services.pumping_test import run_analysis, run_recovery_analysis
from hydropump.ui_qt.dialogs.column_mapping_dialog import ColumnMappingDialog
from hydropump.ui_qt.dialogs.metadata_dialog import MetadataDialog
from hydropump.ui_qt.dialogs.schematic_dialog import SchematicDialog
from hydropump.ui_qt.widgets.diagnostics_panel import DiagnosticsPanel
from hydropump.ui_qt.widgets.metric_card import MetricCard
from hydropump.ui_qt.widgets.mpl_canvas import MplCanvas
from hydropump.ui_qt.widgets.pandas_model import DataFrameModel
from hydropump.ui_qt.widgets.parameters_panel import ParametersPanel
from hydropump.ui_qt.widgets.recovery_panel import RecoveryPanel
from hydropump.ui_qt.widgets.step_test_panel import StepTestPanel
from hydropump.viz import plots as pl

__all__ = ["MainWindow"]

_DEBOUNCE_MS = 250


class MainWindow(QMainWindow):
    """Janela principal: parâmetros à esquerda, resultados em abas à direita."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HydroPump — Análise de Ensaio de Bombeamento")
        self.resize(1280, 820)

        self._df_raw = None
        self._mapping: dict | None = None
        self._series: TimeSeries | None = None
        self._result: AnalysisResult | None = None
        self._well_id: str | None = None
        self._fill_to_12h_decision: bool | None = None

        self._recovery_df = None
        self._recovery_mapping: dict | None = None
        self._schematic_dialog: SchematicDialog | None = None
        self._well_metadata: WellMetadata | None = None
        self._pump_metadata: PumpMetadata | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._recompute)

        self._build_menu()
        self._build_central_widget()
        self.statusBar().showMessage(
            "Abra um arquivo de dados de campo para iniciar (Arquivo → Abrir...)."
        )

    # ------------------------------------------------------------------
    # Montagem da UI
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu_file = self.menuBar().addMenu("&Arquivo")

        action_open = QAction("Abrir dados de campo (rebaixamento)...", self)
        action_open.setShortcut("Ctrl+O")
        action_open.triggered.connect(self._load_file)
        menu_file.addAction(action_open)

        self.action_remap = QAction("Editar mapeamento de colunas...", self)
        self.action_remap.setEnabled(False)
        self.action_remap.triggered.connect(lambda: self._open_mapping_dialog(force_dialog=True))
        menu_file.addAction(self.action_remap)

        self.action_metadata = QAction("Dados do poço e da bomba...", self)
        self.action_metadata.setShortcut("Ctrl+D")
        self.action_metadata.triggered.connect(self._edit_metadata)
        menu_file.addAction(self.action_metadata)

        self.action_reset_fill = QAction("Redefinir decisão de preenchimento até 12h...", self)
        self.action_reset_fill.setEnabled(False)
        self.action_reset_fill.triggered.connect(self._reset_fill_decision)
        menu_file.addAction(self.action_reset_fill)

        menu_file.addSeparator()

        self.action_export_chart = QAction("Exportar gráfico (PNG, 300 dpi)...", self)
        self.action_export_chart.setEnabled(False)
        self.action_export_chart.triggered.connect(self._export_chart)
        menu_file.addAction(self.action_export_chart)

        self.action_export_table = QAction("Exportar tabela (CSV)...", self)
        self.action_export_table.setEnabled(False)
        self.action_export_table.triggered.connect(self._export_table)
        menu_file.addAction(self.action_export_table)

        self.action_export_report = QAction("Exportar relatório (TXT)...", self)
        self.action_export_report.setEnabled(False)
        self.action_export_report.triggered.connect(self._export_report)
        menu_file.addAction(self.action_export_report)

        menu_file.addSeparator()
        action_quit = QAction("Sair", self)
        action_quit.setShortcut("Ctrl+Q")
        action_quit.triggered.connect(self.close)
        menu_file.addAction(action_quit)

    def _build_central_widget(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.params = ParametersPanel()
        self.params.changed.connect(self._schedule_recompute)
        self.params.changed.connect(self._update_schematic_if_open)
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(self.params)
        params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        params_scroll.setMinimumWidth(300)
        params_scroll.setMaximumWidth(420)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_overview_tab(), "📊 Visão Geral")
        self.tabs.addTab(self._build_table_tab(), "📋 Tabela de Dados")
        self.tabs.addTab(self._build_diagnostics_tab(), "🩺 Diagnósticos")
        self.tabs.addTab(self._build_recovery_tab(), "🔁 Recuperação")
        self.tabs.addTab(self._build_step_tab(), "📶 Teste de Degraus")

        splitter.addWidget(params_scroll)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _build_overview_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        cards_row = QHBoxLayout()
        self.card_q = MetricCard("Vazão, Q")
        self.card_t = MetricCard("Transmissividade, T", variant="alt")
        self.card_s = MetricCard("Armazenamento, S", variant="alt2")
        self.card_max = MetricCard("Rebaix. Máximo Real")
        self.card_12h = MetricCard("Rebaixamento em 12h", variant="alt")
        for card in (self.card_q, self.card_t, self.card_s, self.card_max, self.card_12h):
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        schematic_row = QHBoxLayout()
        self.button_schematic = QPushButton("📐 Ver esquema do poço")
        self.button_schematic.setToolTip(
            "Abre um desenho de referência do poço, útil para conferir o que\n"
            "cada parâmetro (NE, b, r...) representa fisicamente enquanto você\n"
            "preenche os dados — funciona mesmo antes de rodar a análise, e se\n"
            "atualiza sozinho conforme você muda os parâmetros na barra lateral."
        )
        self.button_schematic.clicked.connect(self._show_schematic)
        schematic_row.addWidget(self.button_schematic)
        schematic_row.addStretch(1)
        layout.addLayout(schematic_row)

        self.banner_critical = QLabel("")
        self.banner_critical.setWordWrap(True)
        self.banner_critical.setStyleSheet(
            "background: #fdf1ef; border: 1px solid #c0392b; border-radius: 6px; "
            "padding: 8px 12px; color: #c0392b; font-weight: 600;"
        )
        self.banner_critical.setVisible(False)
        layout.addWidget(self.banner_critical)

        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, 1)

        return widget

    def _build_table_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        caption = QLabel("🔵 Medido em campo   ·   ⚪ Projeção/sintético (Cooper-Jacob)")
        caption.setStyleSheet("color: #7f8c9a; font-size: 11px;")
        layout.addWidget(caption)

        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(False)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table_view, 1)
        return widget

    def _build_diagnostics_tab(self) -> QWidget:
        self.diagnostics_panel = DiagnosticsPanel()
        return self.diagnostics_panel

    def _build_recovery_tab(self) -> QWidget:
        self.recovery_panel = RecoveryPanel()
        self.recovery_panel.load_requested.connect(self._load_recovery_file)
        self.recovery_panel.compute_requested.connect(self._compute_recovery)
        return self.recovery_panel

    def _build_step_tab(self) -> QWidget:
        self.step_panel = StepTestPanel()
        self.step_panel.compute_requested.connect(self._compute_step_test)
        return self.step_panel

    # ------------------------------------------------------------------
    # Metadados
    # ------------------------------------------------------------------
    def _edit_metadata(self) -> None:
        dialog = MetadataDialog(self._well_metadata, self._pump_metadata, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._well_metadata = dialog.well_metadata()
        self._pump_metadata = dialog.pump_metadata()
        if self._well_metadata.well_id:
            self._well_id = self._well_metadata.well_id
        self.statusBar().showMessage("Dados do poço e da bomba atualizados.")
        self._schedule_recompute_now()

    # ------------------------------------------------------------------
    # Teste de degraus
    # ------------------------------------------------------------------
    def _compute_step_test(self) -> None:
        q, s = self.step_panel.steps()
        try:
            step = hm.step_drawdown_analysis(q, s)
        except hm.HydroMathError as exc:
            QMessageBox.warning(self, "Teste de degraus", str(exc))
            return

        # Converte para as unidades de campo do gráfico (m³/h).
        q_m3_h = step.Q_m3_s * 3600.0
        specific_field = step.drawdown_m / q_m3_h
        b_field = step.B / 3600.0
        c_field = step.C / (3600.0**2)

        q_ref_m3_h = float(np.median(q_m3_h))
        eff = step.well_efficiency(float(np.median(step.Q_m3_s)))

        fig = pl.plot_step_drawdown(
            q_m3_h, specific_field, b_field, c_field,
            r_squared=step.r_squared,
            efficiency_at=(q_ref_m3_h, eff),
            title="Teste de Degraus — Rebaixamento Específico x Vazão",
            save_path=None,
        )
        self.step_panel.canvas.set_figure(fig)

        alerta = ""
        if step.C < 0:
            alerta = (
                "\n⚠ C negativo não tem sentido físico — indica degraus que não "
                "estabilizaram ou erro de leitura."
            )
        elif eff < 0.6:
            alerta = (
                "\n⚠ Eficiência abaixo de 60%: poço possivelmente mal "
                "desenvolvido, colmatado ou com pré-filtro inadequado."
            )
        self.step_panel.set_summary(
            f"B = {b_field:.4g} (perda no aquífero)   |   "
            f"C = {c_field:.4g} (perda no poço)   |   R² = {step.r_squared:.4f}\n"
            f"Eficiência do poço em {q_ref_m3_h:.1f} m³/h: {eff * 100:.1f}%"
            f"{alerta}"
        )

        if self._result is not None:
            self._result = dataclasses.replace(self._result, step_test=step)
            self.action_export_report.setEnabled(True)
        self.statusBar().showMessage(
            f"Teste de degraus calculado com {step.n_steps} degraus."
        )

    # ------------------------------------------------------------------
    # Carregamento de arquivo — rebaixamento
    # ------------------------------------------------------------------
    def _load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir dados de campo (rebaixamento)", "", "Planilhas (*.csv *.xlsx *.xls)"
        )
        if not path:
            return
        try:
            result = read_path(path)
        except TabularReadError as exc:
            QMessageBox.critical(self, "Erro ao ler arquivo", str(exc))
            return

        self._df_raw = result.df
        self._well_id = Path(path).stem
        self._fill_to_12h_decision = None
        self.statusBar().showMessage(f"{Path(path).name} — {result.dialect_label}")
        self.action_remap.setEnabled(True)
        self.action_reset_fill.setEnabled(True)
        self._open_mapping_dialog(force_dialog=False)

    def _open_mapping_dialog(self, force_dialog: bool) -> None:
        if self._df_raw is None:
            return

        confident = ColumnMappingDialog.detection_is_confident(
            self._df_raw, TIME_ALIASES, DYNAMIC_LEVEL_ALIASES, DRAWDOWN_ALIASES
        )
        if not force_dialog and confident:
            dialog = ColumnMappingDialog(
                self._df_raw,
                time_aliases=TIME_ALIASES,
                level_aliases=DYNAMIC_LEVEL_ALIASES,
                drawdown_aliases=DRAWDOWN_ALIASES,
            )
            self._mapping = dialog.result_mapping()
            self._recompute()
            return

        dialog = ColumnMappingDialog(
            self._df_raw,
            time_aliases=TIME_ALIASES,
            level_aliases=DYNAMIC_LEVEL_ALIASES,
            drawdown_aliases=DRAWDOWN_ALIASES,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._mapping = dialog.result_mapping()
            self._recompute()

    def _reset_fill_decision(self) -> None:
        self._fill_to_12h_decision = None
        self._schedule_recompute_now()

    def _schedule_recompute_now(self) -> None:
        """Força um recálculo imediato (fora do debounce de digitação)."""
        if self._df_raw is not None and self._mapping is not None:
            self._recompute()

    def _maybe_ask_fill_to_12h(self, last_time_min: float) -> bool:
        """Pergunta uma única vez por arquivo carregado; reutiliza a resposta depois.

        Só é perguntado quando o ensaio cobre menos de 12h. A resposta fica
        guardada em ``self._fill_to_12h_decision`` até um novo arquivo ser
        carregado ou o usuário redefini-la explicitamente (menu Arquivo).
        """
        if last_time_min >= 720.0:
            return False
        if self._fill_to_12h_decision is not None:
            return self._fill_to_12h_decision

        resposta = QMessageBox.question(
            self,
            "Preencher dados até 12 horas?",
            (
                f"O ensaio carregado cobre apenas {last_time_min:.0f} min "
                "(menos que as 12h usuais para o rebaixamento previsto).\n\n"
                "Deseja gerar pontos adicionais por interpolação (mesma reta "
                "de Cooper-Jacob já ajustada) para completar até 12h?\n\n"
                "Se confirmar, esses pontos passarão a ser tratados como "
                "dados MEDIDOS em gráfico, tabela e relatório — sem "
                "qualquer distinção visual dos dados efetivamente "
                "coletados — e a responsabilidade pela extrapolação "
                "implícita passa a ser sua."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self._fill_to_12h_decision = resposta == QMessageBox.StandardButton.Yes
        return self._fill_to_12h_decision

    # ------------------------------------------------------------------
    # Recálculo (debounced) — rebaixamento
    # ------------------------------------------------------------------
    def _schedule_recompute(self) -> None:
        if self._df_raw is not None and self._mapping is not None:
            self._debounce.start()

    def _recompute(self) -> None:
        if self._df_raw is None or self._mapping is None:
            return

        try:
            series = extract_series(
                self._df_raw,
                time_column=self._mapping["time_column"],
                value_column=self._mapping["value_column"],
                time_unit=self._mapping["time_unit"],
                values_are_levels=self._mapping["values_are_levels"],
                static_level_m=self.params.spin_ne.value(),
            )
        except ColumnMappingError as exc:
            self.statusBar().showMessage(f"Erro no mapeamento: {exc}")
            return

        fill_confirmed = self._maybe_ask_fill_to_12h(float(series.time_min[-1]))

        try:
            q_result = hm.calculate_flow_rate(
                self.params.spin_volume.value(), self.params.spin_time_fill.value()
            )
        except hm.HydroMathError as exc:
            QMessageBox.critical(self, "Erro no cálculo de vazão", str(exc))
            return

        config = self.params.build_config(q_result.mean_q_m3_s, well_id=self._well_id)
        config = dataclasses.replace(
            config,
            fill_to_12h_confirmed=fill_confirmed,
            well=self._well_metadata,
            pump=self._pump_metadata,
        )

        try:
            result = run_analysis(series, config)
        except hm.HydroMathError as exc:
            QMessageBox.critical(self, "Erro na análise hidrogeológica", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — proteção contra dado inesperado
            QMessageBox.critical(self, "Erro inesperado ao processar os dados", str(exc))
            return

        previous_step = self._result.step_test if self._result is not None else None
        if previous_step is not None:
            result = dataclasses.replace(result, step_test=previous_step)

        self._series = series
        self._result = result
        self._render_result()

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------
    def _render_result(self) -> None:
        result = self._result
        if result is None:
            return
        cj = result.cooper_jacob
        cfg = result.config

        self.card_q.set_value(f"{cfg.Q_m3_h:.2f} m³/h", f"{cfg.Q_l_s:.2f} L/s")
        self.card_t.set_value(
            f"{cj.transmissivity_m2_day:.2f} m²/dia", f"R² = {cj.r_squared:.4f}"
        )

        plausible = hm.storativity_is_plausible(cj.storativity, unconfined=cfg.unconfined)
        self.card_s.set_value(f"{cj.storativity:.2e}", "adimensional")
        self.card_s.set_variant("alt2" if plausible else "alt")

        self.card_max.set_value(
            f"{result.max_drawdown_m:.2f} m", f"em t = {result.last_time_min:.0f} min"
        )
        s12h_txt = f"{result.drawdown_12h_m:.2f} m" if result.drawdown_12h_m is not None else "—"
        self.card_12h.set_value(s12h_txt, "método Cooper-Jacob")

        n_critical = len(result.diagnostics_by(Severity.CRITICAL))
        if n_critical:
            self.banner_critical.setText(
                f"🛑 {n_critical} diagnóstico(s) crítico(s) — veja a aba "
                "Diagnósticos antes de usar estes resultados em laudo."
            )
            self.banner_critical.setVisible(True)
        else:
            self.banner_critical.setVisible(False)

        fig = pl.plot_hydro_analysis(
            df_real=result.df_unified,
            df_projection=result.df_projection,
            col_t="tempo_min",
            col_s="rebaixamento_m",
            col_origem="tipo_dado",
            col_t_proj="tempo_min",
            col_s_proj="rebaixamento_projetado_m",
            show_projection=cfg.show_projection and result.df_projection is not None,
            title="Análise do Teste de Bombeamento",
            Q=cfg.Q_m3_h,
            T=cj.transmissivity_m2_day,
            S=cj.storativity,
            T_ci95=cj.transmissivity_ci95_m2_day,
            r=cfg.radius_m,
            s_12h=result.drawdown_12h_m,
            poco=cfg.well_id,
            save_path=None,
        )
        self.canvas.set_figure(fig)

        model = DataFrameModel(result.df_unified, origin_column="tipo_dado")
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()

        self.diagnostics_panel.set_diagnostics(result.diagnostics)

        self._update_schematic_if_open()
        self.action_export_chart.setEnabled(True)
        self.action_export_table.setEnabled(True)
        self.action_export_report.setEnabled(True)
        self.statusBar().showMessage(
            f"Análise atualizada — T = {cj.transmissivity_m2_day:.2f} m²/dia, "
            f"S = {cj.storativity:.2e}"
        )

    def _current_schematic_geometry(self) -> dict:
        """Geometria atual para o esquema — funciona antes mesmo de rodar a análise.

        Usa os valores AO VIVO do painel de parâmetros (não os do último
        resultado calculado), para que o esquema sirva como ferramenta de
        apoio ao preenchimento dos dados, não apenas como ilustração de
        pós-processamento.
        """
        NE = self.params.spin_ne.value()
        # ND = NE (geometria estática) enquanto não houver rebaixamento medido.
        ND = NE + self._result.max_drawdown_m if self._result is not None else NE
        b = self.params.spin_b.value() if self.params.check_jacob.isChecked() else None
        return {
            "well_type": (
                "single" if self.params.configuration.is_single_well else "observation"
            ),
            "NE": NE,
            "ND": ND,
            "b": b,
            "r": self.params.spin_r.value(),
        }

    def _show_schematic(self) -> None:
        geometry = self._current_schematic_geometry()
        if self._schematic_dialog is None:
            self._schematic_dialog = SchematicDialog(parent=self, **geometry)
            self._schematic_dialog.setModal(False)
            self._schematic_dialog.finished.connect(self._on_schematic_closed)
        else:
            self._schematic_dialog.update_geometry(**geometry)
        self._schematic_dialog.show()
        self._schematic_dialog.raise_()
        self._schematic_dialog.activateWindow()

    def _on_schematic_closed(self, *_args) -> None:
        self._schematic_dialog = None

    def _update_schematic_if_open(self, *_args) -> None:
        """Chamado a cada recálculo e a cada mudança de parâmetro.

        Mantém o esquema, se estiver aberto, sempre coerente com os valores
        atuais da barra lateral — é o que torna o desenho útil como guia
        enquanto o usuário ainda está ajustando os dados, e não apenas uma
        foto do último resultado calculado.
        """
        if self._schematic_dialog is not None:
            self._schematic_dialog.update_geometry(**self._current_schematic_geometry())

    # ------------------------------------------------------------------
    # Exportação
    # ------------------------------------------------------------------
    def _export_chart(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar gráfico", "analise_bombeamento.png", "PNG (*.png)"
        )
        if path:
            self.canvas.save_png(path, dpi=300)
            self.statusBar().showMessage(f"Gráfico salvo em {path}")

    def _export_table(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar tabela", "dados_ensaio_bombeamento.csv", "CSV (*.csv)"
        )
        if path:
            self._result.df_unified.to_csv(path, index=False)
            self.statusBar().showMessage(f"Tabela salva em {path}")

    def _export_report(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar relatório", "relatorio_analise.txt", "Texto (*.txt)"
        )
        if path:
            report = build_text_report(self._result)
            Path(path).write_text(report, encoding="utf-8")
            self.statusBar().showMessage(f"Relatório salvo em {path}")

    # ------------------------------------------------------------------
    # Recuperação de Theis
    # ------------------------------------------------------------------
    def _load_recovery_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir dados de recuperação", "", "Planilhas (*.csv *.xlsx *.xls)"
        )
        if not path:
            return
        try:
            result = read_path(path)
        except TabularReadError as exc:
            QMessageBox.critical(self, "Erro ao ler arquivo", str(exc))
            return

        self._recovery_df = result.df
        confident = ColumnMappingDialog.detection_is_confident(
            result.df, RECOVERY_TIME_ALIASES, RECOVERY_LEVEL_ALIASES, RECOVERY_DRAWDOWN_ALIASES
        )
        dialog_kwargs = dict(
            time_aliases=RECOVERY_TIME_ALIASES,
            level_aliases=RECOVERY_LEVEL_ALIASES,
            drawdown_aliases=RECOVERY_DRAWDOWN_ALIASES,
            value_label="rebaixamento residual (m)",
            level_checkbox_label=(
                "Arquivo traz Nível Dinâmico de recuperação (ND_rec) em vez de "
                "rebaixamento residual direto"
            ),
        )
        if confident:
            # Detecção sem ambiguidade: aceita automaticamente, sem modal —
            # mesmo padrão do fluxo de rebaixamento em `_open_mapping_dialog`.
            dialog = ColumnMappingDialog(result.df, **dialog_kwargs)
            self._recovery_mapping = dialog.result_mapping()
        else:
            dialog = ColumnMappingDialog(result.df, parent=self, **dialog_kwargs)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self._recovery_df = None
                return
            self._recovery_mapping = dialog.result_mapping()

        self.recovery_panel.set_file_label(f"{Path(path).name} — {result.dialect_label}")

    def _compute_recovery(self) -> None:
        if self._result is None:
            QMessageBox.warning(
                self,
                "Recuperação",
                "Rode primeiro a análise de rebaixamento (carregue os dados na "
                "aba de rebaixamento) — a recuperação reaproveita Q, r e a "
                "correção de Jacob configurados ali.",
            )
            return
        if self._recovery_df is None or self._recovery_mapping is None:
            QMessageBox.warning(self, "Recuperação", "Carregue os dados de recuperação primeiro.")
            return

        try:
            series_rec = extract_series(
                self._recovery_df,
                time_column=self._recovery_mapping["time_column"],
                value_column=self._recovery_mapping["value_column"],
                time_unit=self._recovery_mapping["time_unit"],
                values_are_levels=self._recovery_mapping["values_are_levels"],
                static_level_m=self.params.spin_ne.value(),
            )
        except ColumnMappingError as exc:
            QMessageBox.critical(self, "Erro no mapeamento", str(exc))
            return

        cfg = self._result.config
        rec_cfg = RecoveryConfig(
            pumping_duration_min=self.recovery_panel.spin_duration.value(),
            ratio_min=self.recovery_panel.ratio_min(),
            ratio_max=self.recovery_panel.ratio_max(),
        )

        try:
            rec_result = run_recovery_analysis(
                pump_time_sec=rec_cfg.pumping_duration_min * 60.0,
                recovery_time_sec=series_rec.time_sec,
                residual_drawdown_m=series_rec.drawdown_m,
                config=cfg,
                recovery_config=rec_cfg,
            )
        except hm.HydroMathError as exc:
            QMessageBox.critical(self, "Erro na recuperação de Theis", str(exc))
            return

        self._result = dataclasses.replace(self._result, recovery=rec_result)

        t_total = rec_cfg.pumping_duration_min * 60.0 + series_rec.time_sec
        ratio = t_total / series_rec.time_sec
        fit_line = rec_result.delta_s_residual * np.log10(ratio) + rec_result.intercept

        fig = pl.plot_recovery_theis(
            t_ratio=ratio,
            residual_drawdown=series_rec.drawdown_m,
            fit_line=fit_line,
            T_rec=rec_result.transmissivity_m2_day,
            title=f"Recuperação de Theis — Poço {cfg.well_id or ''}".strip(" —"),
            save_path=None,
        )
        self.recovery_panel.canvas.set_figure(fig)

        t_drawdown = self._result.cooper_jacob.transmissivity_m2_day
        t_recovery = rec_result.transmissivity_m2_day
        diff_pct = (
            abs(t_recovery - t_drawdown) / t_drawdown * 100.0 if t_drawdown else float("nan")
        )
        self.recovery_panel.set_summary(
            f"T (rebaixamento) = {t_drawdown:.2f} m²/dia   |   "
            f"T' (recuperação) = {t_recovery:.2f} m²/dia   "
            f"(diferença: {diff_pct:.1f}%)   |   S/S' em s''=0 = {rec_result.ratio_at_zero:.2f}\n"
            "Lembrete: T' é imune ao skin effect (bomba desligada) — em poço "
            "único, prefira T' para fins de projeto."
        )
        self.action_export_report.setEnabled(True)
        self.statusBar().showMessage("Recuperação de Theis calculada.")
