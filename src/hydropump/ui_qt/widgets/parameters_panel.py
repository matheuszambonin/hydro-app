"""Painel de parâmetros do ensaio de rebaixamento.

Equivalente Qt à sidebar do Streamlit original. Emite ``changed`` sempre
que qualquer controle muda; quem decide *quando* recalcular (tipicamente
com debounce) é a janela principal — este widget só expõe estado.

Todos os ``QFormLayout`` usam ``WrapAllRows``: o rótulo fica numa linha e o
campo na linha de baixo, em vez de lado a lado. Isso evita que rótulos
longos (ex.: "Distância ao poço de observação, r:") empurrem a largura
total do formulário além do espaço disponível na barra lateral, o que
causava uma barra de rolagem horizontal incômoda no rodapé do painel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hydropump.domain.models import AnalysisConfig, WellConfiguration
from hydropump.domain.scenarios import AquiferScenario, profile_for

__all__ = ["ParametersPanel"]

_TOOLTIP_R_SINGLE = (
    "Raio interno do poço ou do filtro. Usado tanto na equação de\n"
    "Cooper-Jacob quanto no critério de armazenamento no poço (t_wbs)."
)
_TOOLTIP_R_OBSERVATION = (
    "Distância radial do poço de observação (piezômetro) ao poço\n"
    "bombeado. Quanto maior a distância, menor o efeito do skin do\n"
    "poço bombeado sobre o armazenamento (S) estimado."
)


def _wrap_rows(form: QFormLayout) -> QFormLayout:
    """Empilha rótulo acima do campo em vez de lado a lado (evita overflow horizontal)."""
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    return form


class ParametersPanel(QWidget):
    """Reúne os parâmetros de entrada e sabe construir um :class:`AnalysisConfig`."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Cenário hidrogeológico ---
        self.combo_scenario = QComboBox()
        for scenario in AquiferScenario:
            self.combo_scenario.addItem(scenario.label, scenario)
        self.combo_scenario.setToolTip(
            "Cenário hidrogeológico declarado. Define os padrões da correção\n"
            "de Jacob, a faixa plausível de S, quais anomalias procurar na\n"
            "curva (dupla porosidade, drenança) e as ressalvas do relatório."
        )
        self.label_scenario_note = QLabel()
        self.label_scenario_note.setWordWrap(True)
        self.label_scenario_note.setStyleSheet(
            "color: #5a6472; font-size: 10px; background: transparent;"
        )

        self.spin_granular = QDoubleSpinBox()
        self.spin_granular.setRange(0.0, 2000.0)
        self.spin_granular.setDecimals(2)
        self.spin_granular.setSuffix(" m")
        self.spin_granular.setSpecialValueText("não informado")
        self.spin_granular.setToolTip(
            "Espessura da porção granular (topo), se o perfil geológico do\n"
            "poço for conhecido. Opcional: sem ele, T continua sendo o valor\n"
            "global do sistema."
        )
        self.spin_fractured = QDoubleSpinBox()
        self.spin_fractured.setRange(0.0, 2000.0)
        self.spin_fractured.setDecimals(2)
        self.spin_fractured.setSuffix(" m")
        self.spin_fractured.setSpecialValueText("não informado")
        self.spin_fractured.setToolTip(
            "Espessura da porção fraturada (base), se conhecida pelo perfil\n"
            "geológico. Opcional."
        )
        self._label_granular = QLabel("Espessura granular (perfil):")
        self._label_fractured = QLabel("Espessura fraturada (perfil):")

        scenario_form = _wrap_rows(QFormLayout())
        scenario_form.addRow("Cenário:", self.combo_scenario)
        scenario_form.addRow(self.label_scenario_note)
        scenario_form.addRow(self._label_granular, self.spin_granular)
        scenario_form.addRow(self._label_fractured, self.spin_fractured)
        scenario_box = QGroupBox("Cenário hidrogeológico")
        scenario_box.setLayout(scenario_form)

        # --- Geometria e nível ---
        self.spin_ne = QDoubleSpinBox()
        self.spin_ne.setRange(0.0, 1000.0)
        self.spin_ne.setDecimals(2)
        self.spin_ne.setSuffix(" m")
        self.spin_ne.setValue(10.0)
        self.spin_ne.setToolTip(
            "Nível Estático (NE): profundidade do nível d'água em repouso,\n"
            "medida a partir da boca do poço, antes do início do bombeamento."
        )

        self.check_jacob = QCheckBox("Aplicar correção de Jacob (aquífero livre)")
        self.check_jacob.setChecked(True)
        self.check_jacob.setToolTip(
            "Corrige o rebaixamento medido (s → s') para compensar a redução da\n"
            "própria espessura saturada em aquíferos livres (Jacob, 1944).\n"
            "Poços parcialmente penetrantes ou aquíferos confinados/semi-confinados\n"
            "geralmente dispensam essa correção."
        )

        self.spin_b = QDoubleSpinBox()
        self.spin_b.setRange(0.01, 5000.0)
        self.spin_b.setDecimals(2)
        self.spin_b.setSuffix(" m")
        self.spin_b.setValue(20.0)
        self.spin_b.setToolTip(
            "Espessura saturada (b): espessura saturada inicial do aquífero,\n"
            "da base ao nível estático. Se não souber a espessura total do\n"
            "aquífero, use a profundidade da coluna saturada perfurada no poço."
        )

        self.check_unconfined = QCheckBox("Aquífero livre (freático)")
        self.check_unconfined.setChecked(True)
        self.check_unconfined.setToolTip(
            "Define a faixa de plausibilidade usada para validar o coeficiente\n"
            "de armazenamento (S) no resultado: produção específica (S_y,\n"
            "tipicamente 0,01–0,35) para aquífero livre, ou armazenamento\n"
            "elástico (tipicamente 1e-5–1e-3) para aquífero confinado."
        )

        geo_form = _wrap_rows(QFormLayout())
        geo_form.addRow("Nível estático, NE:", self.spin_ne)
        geo_form.addRow(self.check_jacob)
        geo_form.addRow("Espessura saturada, b:", self.spin_b)
        geo_form.addRow(self.check_unconfined)
        geo_box = QGroupBox("Geometria e nível")
        geo_box.setLayout(geo_form)

        # --- Configuração do poço ---
        self.combo_config = QComboBox()
        self.combo_config.addItems(
            [
                "Poço único (medição no poço bombeado)",
                "Poço de observação (medição em piezômetro vizinho)",
            ]
        )
        self.combo_config.setToolTip(
            "Define se o rebaixamento foi medido no próprio poço bombeado\n"
            "(Poço Único) ou em um piezômetro/poço de observação vizinho.\n"
            "Em poço único, o armazenamento (S) estimado é sensível a\n"
            "perdas de carga na parede do poço (skin effect)."
        )

        self.spin_r = QDoubleSpinBox()
        self.spin_r.setRange(0.001, 10000.0)
        self.spin_r.setDecimals(3)
        self.spin_r.setSuffix(" m")
        self.spin_r.setValue(0.076)
        self.spin_r.setToolTip(_TOOLTIP_R_SINGLE)

        self.spin_pump_pipe = QDoubleSpinBox()
        self.spin_pump_pipe.setRange(0.0, 5.0)
        self.spin_pump_pipe.setDecimals(3)
        self.spin_pump_pipe.setSuffix(" m")
        self.spin_pump_pipe.setValue(0.0)
        self.spin_pump_pipe.setToolTip(
            "Raio externo do tubo de recalque dentro do poço. Reduz o volume\n"
            "armazenado; deixe 0 se não souber (estimativa conservadora de t_wbs,\n"
            "o instante em que cessa o efeito de armazenamento no poço)."
        )

        self._r_label = QLabel("Raio do poço, r_w:")
        self._pump_pipe_label = QLabel("Raio do tubo de recalque:")

        well_form = _wrap_rows(QFormLayout())
        well_form.addRow("Configuração:", self.combo_config)
        well_form.addRow(self._r_label, self.spin_r)
        well_form.addRow(self._pump_pipe_label, self.spin_pump_pipe)
        well_box = QGroupBox("Configuração do poço")
        well_box.setLayout(well_form)

        # --- Vazão volumétrica ---
        self.spin_volume = QDoubleSpinBox()
        self.spin_volume.setRange(0.01, 10000.0)
        self.spin_volume.setDecimals(2)
        self.spin_volume.setSuffix(" L")
        self.spin_volume.setValue(20.0)
        self.spin_volume.setToolTip(
            "Volume coletado no recipiente aferido (balde/proveta) durante\n"
            "o ensaio volumétrico de vazão."
        )

        self.spin_time_fill = QDoubleSpinBox()
        self.spin_time_fill.setRange(0.01, 10000.0)
        self.spin_time_fill.setDecimals(2)
        self.spin_time_fill.setSuffix(" s")
        self.spin_time_fill.setValue(12.0)
        self.spin_time_fill.setToolTip(
            "Tempo cronometrado para encher o volume acima. Q = Volume / Tempo."
        )

        flow_form = _wrap_rows(QFormLayout())
        flow_form.addRow("Volume do balde:", self.spin_volume)
        flow_form.addRow("Tempo de enchimento:", self.spin_time_fill)
        flow_box = QGroupBox("Vazão (método volumétrico)")
        flow_box.setLayout(flow_form)

        # --- Projeção teórica ---
        self.check_projection = QCheckBox("Exibir projeção teórica no gráfico")
        self.check_projection.setChecked(True)
        self.check_projection.setToolTip(
            "Desenha, além dos dados medidos, a continuação teórica da reta\n"
            "de Cooper-Jacob (linha tracejada) até o horizonte abaixo. Não é\n"
            "uma nova medição — é a mesma reta ajustada, extrapolada sob a\n"
            "premissa de vazão constante e aquífero infinito."
        )

        self.spin_hours = QDoubleSpinBox()
        self.spin_hours.setRange(0.5, 720.0)
        self.spin_hours.setDecimals(1)
        self.spin_hours.setSuffix(" h")
        self.spin_hours.setValue(12.0)
        self.spin_hours.setToolTip(
            "Horizonte final da extrapolação de Cooper-Jacob exibida no\n"
            "gráfico (ex.: 12h, 24h). Independente do preenchimento de\n"
            "dados até 12h (que é perguntado separadamente, se necessário)."
        )

        proj_form = _wrap_rows(QFormLayout())
        proj_form.addRow(self.check_projection)
        proj_form.addRow("Horizonte da projeção:", self.spin_hours)
        proj_box = QGroupBox("Projeção")
        proj_box.setLayout(proj_form)

        layout = QVBoxLayout(self)
        layout.addWidget(scenario_box)
        layout.addWidget(geo_box)
        layout.addWidget(well_box)
        layout.addWidget(flow_box)
        layout.addWidget(proj_box)
        layout.addStretch(1)

        # --- sinais ---
        self.combo_config.currentIndexChanged.connect(self._on_config_changed)
        self.check_jacob.toggled.connect(self.spin_b.setEnabled)

        # IMPORTANTE: nunca conecte um sinal com argumento (valueChanged(float),
        # toggled(bool), currentIndexChanged(int)) diretamente a `self.changed.emit`.
        # Qt/PySide6 trunca argumentos automaticamente quando o alvo é um método
        # Python comum, mas `Signal.emit` é um callable especial que exige
        # correspondência EXATA de assinatura — conectar direto causa
        # `TypeError: changed() only accepts 0 argument(s), 1 given!` toda vez
        # que o widget muda. Por isso usamos `_emit_changed`, que aceita
        # qualquer quantidade de argumentos e os descarta antes de emitir.
        for spin in (
            self.spin_ne,
            self.spin_b,
            self.spin_r,
            self.spin_pump_pipe,
            self.spin_volume,
            self.spin_time_fill,
            self.spin_hours,
        ):
            spin.valueChanged.connect(self._emit_changed)
        for check in (self.check_jacob, self.check_unconfined, self.check_projection):
            check.toggled.connect(self._emit_changed)
        self.combo_config.currentIndexChanged.connect(self._emit_changed)
        self.combo_scenario.currentIndexChanged.connect(self._on_scenario_changed)
        self.combo_scenario.currentIndexChanged.connect(self._emit_changed)
        for spin in (self.spin_granular, self.spin_fractured):
            spin.valueChanged.connect(self._emit_changed)

        self._on_config_changed(0)
        self._on_scenario_changed(0)

    def _emit_changed(self, *_args) -> None:
        """Repassa qualquer sinal de mudança para `changed`, descartando argumentos."""
        self.changed.emit()

    def _on_scenario_changed(self, _index: int) -> None:
        """Ajusta defaults e visibilidade conforme o cenário declarado."""
        scenario = self.scenario
        profile = profile_for(scenario)

        self.label_scenario_note.setText(profile.storativity_note)

        is_mixed = scenario is AquiferScenario.MIXED_GRANULAR_FRACTURED
        for widget in (
            self._label_granular,
            self.spin_granular,
            self._label_fractured,
            self.spin_fractured,
        ):
            widget.setVisible(is_mixed)

        # Aplica os defaults do cenário sem disparar recálculo em cascata:
        # o sinal de mudança do próprio combo já cuida disso.
        for widget, value in (
            (self.check_jacob, profile.apply_jacob_by_default),
            (self.check_unconfined, profile.unconfined),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self.spin_b.setEnabled(self.check_jacob.isChecked())

    @property
    def scenario(self) -> AquiferScenario:
        return self.combo_scenario.currentData()

    def _on_config_changed(self, index: int) -> None:
        is_single = index == 0
        self._r_label.setText(
            "Raio do poço, r_w:" if is_single else "Distância ao poço de observação, r:"
        )
        self.spin_r.setToolTip(_TOOLTIP_R_SINGLE if is_single else _TOOLTIP_R_OBSERVATION)
        self._pump_pipe_label.setVisible(is_single)
        self.spin_pump_pipe.setVisible(is_single)

    @property
    def configuration(self) -> WellConfiguration:
        return (
            WellConfiguration.SINGLE_WELL
            if self.combo_config.currentIndex() == 0
            else WellConfiguration.OBSERVATION_WELL
        )

    def build_config(self, Q_m3_s: float, well_id: str | None = None) -> AnalysisConfig:
        """Monta um :class:`AnalysisConfig` a partir do estado atual dos widgets."""
        cfg_type = self.configuration
        return AnalysisConfig(
            static_level_m=self.spin_ne.value(),
            radius_m=self.spin_r.value(),
            Q_m3_s=Q_m3_s,
            configuration=cfg_type,
            saturated_thickness_m=self.spin_b.value() if self.check_jacob.isChecked() else None,
            apply_jacob=self.check_jacob.isChecked(),
            unconfined=self.check_unconfined.isChecked(),
            projection_hours=self.spin_hours.value(),
            show_projection=self.check_projection.isChecked(),
            pump_pipe_radius_m=self.spin_pump_pipe.value() if cfg_type.is_single_well else 0.0,
            well_id=well_id,
            scenario=self.scenario,
            granular_thickness_m=(
                self.spin_granular.value() if self.spin_granular.value() > 0 else None
            ),
            fractured_thickness_m=(
                self.spin_fractured.value() if self.spin_fractured.value() > 0 else None
            ),
        )
