"""Diálogo de cadastro do poço e da bomba.

Concentra tudo que identifica o ensaio mas não entra nas equações. O único
campo com consequência de cálculo é a altura da boca do poço combinada com
o datum de medição — e mesmo essa afeta apenas as profundidades reportadas,
não T/S/capacidade específica (ver ``WellMetadata.level_note``).

A conversão de coordenadas é **bidirecional e ao vivo**: digitar em um dos
sistemas preenche o outro automaticamente, com uma trava de reentrância
para os dois blocos não ficarem se atualizando em laço.
"""

from __future__ import annotations

from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydropump.domain.geo import (
    CoordinateError,
    GeographicCoordinate,
    UTMCoordinate,
    geographic_to_utm,
    utm_to_geographic,
    utm_zone_for_longitude,
)
from hydropump.domain.metadata import (
    MeasurementDatum,
    MetadataError,
    PumpMetadata,
    WellMetadata,
)

__all__ = ["MetadataDialog"]

_PURPOSES = [
    "",
    "Abastecimento humano",
    "Abastecimento industrial",
    "Irrigação",
    "Dessedentação animal",
    "Monitoramento",
    "Rebaixamento / obra",
    "Outro",
]

_PUMP_TYPES = ["", "Submersa", "Injetora", "Centrífuga", "Compressor (air-lift)", "Outro"]


def _wrap_form() -> QFormLayout:
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    return form


class MetadataDialog(QDialog):
    """Formulário de metadados do poço e da bomba."""

    def __init__(
        self,
        well: WellMetadata | None = None,
        pump: PumpMetadata | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dados do poço e da bomba")
        self._updating = False  # trava de reentrância da conversão

        # Duas colunas lado a lado (poço à esquerda; coordenadas e bomba à
        # direita) em vez de tudo empilhado verticalmente. Um formulário
        # deste tamanho empilhado numa coluna só facilmente ultrapassa a
        # altura da tela, e quando a janela abre menor que o conteúdo o Qt
        # comprime cada linha em vez de simplesmente rolar — daí os campos
        # "achatados". A correção estrutural é: mais largo, menos alto, e
        # com rolagem própria como garantia final se ainda assim não couber.
        left_column = QVBoxLayout()
        left_column.addWidget(self._build_well_box())
        left_column.addStretch(1)

        right_column = QVBoxLayout()
        right_column.addWidget(self._build_coordinates_box())
        right_column.addWidget(self._build_pump_box())
        right_column.addStretch(1)

        columns = QHBoxLayout()
        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)

        content = QWidget()
        content.setLayout(columns)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)

        # Tamanho padrão generoso e mais horizontal que vertical — o
        # usuário não deveria precisar redimensionar manualmente para ver
        # o formulário por inteiro. Ainda assim redimensionável livremente
        # (não é setFixedSize), e com a QScrollArea acima como garantia
        # final em telas pequenas.
        self.resize(880, 620)
        self.setMinimumSize(640, 420)

        if well is not None:
            self._load_well(well)
        if pump is not None:
            self._load_pump(pump)

    # ------------------------------------------------------------------
    # Blocos do formulário
    # ------------------------------------------------------------------
    def _build_well_box(self) -> QGroupBox:
        self.edit_well_id = QLineEdit()
        self.edit_well_id.setPlaceholderText("ex.: PT-07, PZ-01")

        self.edit_datetime = QDateTimeEdit(QDateTime.currentDateTime())
        self.edit_datetime.setCalendarPopup(True)
        self.edit_datetime.setDisplayFormat("dd/MM/yyyy HH:mm")

        self.edit_location = QLineEdit()
        self.edit_location.setPlaceholderText("Propriedade, município/UF")

        self.combo_purpose = QComboBox()
        self.combo_purpose.addItems(_PURPOSES)
        self.combo_purpose.setEditable(True)

        self.spin_well_depth = QDoubleSpinBox()
        self.spin_well_depth.setRange(0.0, 5000.0)
        self.spin_well_depth.setDecimals(2)
        self.spin_well_depth.setSuffix(" m")
        self.spin_well_depth.setSpecialValueText("não informado")
        self.spin_well_depth.setValue(0.0)
        self.spin_well_depth.setToolTip(
            "Profundidade total perfurada (opcional). Usada apenas para\n"
            "registro no relatório."
        )

        self.spin_wellhead = QDoubleSpinBox()
        self.spin_wellhead.setRange(0.0, 20.0)
        self.spin_wellhead.setDecimals(3)
        self.spin_wellhead.setSuffix(" m")
        self.spin_wellhead.setToolTip(
            "Altura da boca do poço (topo do revestimento) acima do terreno."
        )

        self.combo_datum = QComboBox()
        self.combo_datum.addItems(
            [
                "A partir da boca do poço (topo do revestimento)",
                "A partir da superfície do terreno",
            ]
        )
        self.combo_datum.setToolTip(
            "Referência usada em campo para medir NE e ND.\n\n"
            "Importante: como o rebaixamento é a DIFERENÇA entre dois níveis\n"
            "medidos na mesma referência (s = ND − NE), a altura da boca se\n"
            "cancela. T, S e a capacidade específica NÃO mudam com esta\n"
            "escolha — apenas as profundidades absolutas reportadas."
        )

        self.edit_weather = QLineEdit()
        self.edit_weather.setPlaceholderText("ex.: ensolarado, 28 °C (opcional)")

        self.edit_notes = QPlainTextEdit()
        self.edit_notes.setPlaceholderText("Observações de campo (opcional)")
        self.edit_notes.setMaximumHeight(60)

        form = _wrap_form()
        form.addRow("Identificação do poço:", self.edit_well_id)
        form.addRow("Data/hora do ensaio:", self.edit_datetime)
        form.addRow("Local:", self.edit_location)
        form.addRow("Finalidade:", self.combo_purpose)
        form.addRow("Profundidade do poço (opcional):", self.spin_well_depth)
        form.addRow("Altura da boca do poço:", self.spin_wellhead)
        form.addRow("As leituras de nível foram tomadas:", self.combo_datum)
        form.addRow("Clima (opcional):", self.edit_weather)
        form.addRow("Observações:", self.edit_notes)

        box = QGroupBox("Poço")
        box.setLayout(form)
        return box

    def _build_coordinates_box(self) -> QGroupBox:
        self.check_coords = QCheckBox("Informar coordenadas")
        self.check_coords.setChecked(True)

        # --- geográficas ---
        self.spin_lat = QDoubleSpinBox()
        self.spin_lat.setRange(-90.0, 90.0)
        self.spin_lat.setDecimals(6)
        self.spin_lat.setSuffix(" °")
        self.spin_lat.setValue(-22.906800)

        self.spin_lon = QDoubleSpinBox()
        self.spin_lon.setRange(-180.0, 180.0)
        self.spin_lon.setDecimals(6)
        self.spin_lon.setSuffix(" °")
        self.spin_lon.setValue(-43.172900)

        geo_form = _wrap_form()
        geo_form.addRow("Latitude:", self.spin_lat)
        geo_form.addRow("Longitude:", self.spin_lon)
        geo_box = QGroupBox("Geográfica (WGS84 / SIRGAS2000)")
        geo_box.setLayout(geo_form)

        # --- UTM ---
        self.spin_easting = QDoubleSpinBox()
        self.spin_easting.setRange(0.0, 1_000_000.0)
        self.spin_easting.setDecimals(2)
        self.spin_easting.setSuffix(" m")

        self.spin_northing = QDoubleSpinBox()
        self.spin_northing.setRange(0.0, 10_000_000.0)
        self.spin_northing.setDecimals(2)
        self.spin_northing.setSuffix(" m")

        self.spin_zone = QSpinBox()
        self.spin_zone.setRange(1, 60)

        self.combo_hemisphere = QComboBox()
        self.combo_hemisphere.addItems(["S", "N"])

        utm_form = _wrap_form()
        utm_form.addRow("Easting (E):", self.spin_easting)
        utm_form.addRow("Northing (N):", self.spin_northing)
        utm_form.addRow("Fuso:", self.spin_zone)
        utm_form.addRow("Hemisfério:", self.combo_hemisphere)
        utm_box = QGroupBox("UTM")
        utm_box.setLayout(utm_form)

        self.label_coords = QLabel()
        self.label_coords.setWordWrap(True)
        self.label_coords.setStyleSheet(
            "background: #eef4fb; border: 1px solid #c7d9ec; border-radius: 5px; "
            "padding: 6px 10px; color: #0d2b4d; font-size: 11px;"
        )

        columns = QHBoxLayout()
        columns.addWidget(geo_box, 1)
        columns.addWidget(utm_box, 1)

        inner = QVBoxLayout()
        inner.addWidget(self.check_coords)
        inner.addLayout(columns)
        inner.addWidget(self.label_coords)

        box = QGroupBox("Coordenadas")
        box.setLayout(inner)

        # Conversão bidirecional ao vivo
        for spin in (self.spin_lat, self.spin_lon):
            spin.valueChanged.connect(self._geographic_changed)
        for widget in (self.spin_easting, self.spin_northing, self.spin_zone):
            widget.valueChanged.connect(self._utm_changed)
        self.combo_hemisphere.currentIndexChanged.connect(self._utm_changed)

        self.check_coords.toggled.connect(geo_box.setEnabled)
        self.check_coords.toggled.connect(utm_box.setEnabled)
        self.check_coords.toggled.connect(self.label_coords.setVisible)

        self._geographic_changed()
        return box

    def _build_pump_box(self) -> QGroupBox:
        self.combo_pump_type = QComboBox()
        self.combo_pump_type.addItems(_PUMP_TYPES)
        self.combo_pump_type.setEditable(True)

        self.edit_pump_brand = QLineEdit()
        self.edit_pump_brand.setPlaceholderText("Marca / modelo (opcional)")

        self.spin_intake = QDoubleSpinBox()
        self.spin_intake.setRange(0.01, 2000.0)
        self.spin_intake.setDecimals(2)
        self.spin_intake.setSuffix(" m")
        self.spin_intake.setValue(30.0)
        self.spin_intake.setToolTip(
            "Profundidade do crivo (sucção) da bomba. OBRIGATÓRIO.\n"
            "Define o rebaixamento máximo admissível antes de a bomba\n"
            "começar a aspirar ar."
        )

        self.spin_max_flow = QDoubleSpinBox()
        self.spin_max_flow.setRange(0.01, 10_000.0)
        self.spin_max_flow.setDecimals(2)
        self.spin_max_flow.setSuffix(" m³/h")
        self.spin_max_flow.setValue(20.0)
        self.spin_max_flow.setToolTip(
            "Vazão máxima nominal da bomba. OBRIGATÓRIO.\n"
            "Contextualiza a vazão do ensaio e limita a vazão de exploração."
        )

        form = _wrap_form()
        form.addRow("Tipo (opcional):", self.combo_pump_type)
        form.addRow("Marca/modelo (opcional):", self.edit_pump_brand)
        form.addRow("Profundidade do crivo (obrigatório):", self.spin_intake)
        form.addRow("Vazão máxima (obrigatório):", self.spin_max_flow)

        box = QGroupBox("Bomba instalada")
        box.setLayout(form)
        return box

    # ------------------------------------------------------------------
    # Conversão de coordenadas
    # ------------------------------------------------------------------
    def _geographic_changed(self, *_args) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            geo = GeographicCoordinate(self.spin_lat.value(), self.spin_lon.value())
            utm = geographic_to_utm(geo)
            self.spin_easting.setValue(utm.easting)
            self.spin_northing.setValue(utm.northing)
            self.spin_zone.setValue(utm.zone)
            self.combo_hemisphere.setCurrentText(utm.hemisphere)
            self._refresh_coordinate_label(geo, utm)
        except CoordinateError as exc:
            self.label_coords.setText(f"Coordenada inválida: {exc}")
        finally:
            self._updating = False

    def _utm_changed(self, *_args) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            utm = UTMCoordinate(
                self.spin_easting.value(),
                self.spin_northing.value(),
                self.spin_zone.value(),
                self.combo_hemisphere.currentText(),
            )
            geo = utm_to_geographic(utm)
            self.spin_lat.setValue(geo.latitude)
            self.spin_lon.setValue(geo.longitude)
            self._refresh_coordinate_label(geo, utm)
        except CoordinateError as exc:
            self.label_coords.setText(f"Coordenada inválida: {exc}")
        finally:
            self._updating = False

    def _refresh_coordinate_label(
        self, geo: GeographicCoordinate, utm: UTMCoordinate
    ) -> None:
        aviso = ""
        natural = utm_zone_for_longitude(geo.longitude)
        if natural != utm.zone:
            aviso = (
                f"  ⚠ O fuso natural desta longitude é {natural}; "
                f"você está usando {utm.zone}."
            )
        self.label_coords.setText(
            f"{geo.format_dms()}   |   {utm.format_compact()}{aviso}"
        )

    # ------------------------------------------------------------------
    # Carga / leitura
    # ------------------------------------------------------------------
    def _load_well(self, well: WellMetadata) -> None:
        self.edit_well_id.setText(well.well_id)
        if well.datetime_utc:
            self.edit_datetime.setDateTime(QDateTime(well.datetime_utc))
        self.edit_location.setText(well.location)
        self.combo_purpose.setCurrentText(well.purpose)
        self.spin_well_depth.setValue(well.well_depth_m or 0.0)
        self.spin_wellhead.setValue(well.wellhead_height_m)
        self.combo_datum.setCurrentIndex(
            0 if well.measurement_datum is MeasurementDatum.WELLHEAD else 1
        )
        self.edit_weather.setText(well.weather)
        self.edit_notes.setPlainText(well.notes)
        if well.coordinates is not None:
            self.check_coords.setChecked(True)
            self.spin_lat.setValue(well.coordinates.latitude)
            self.spin_lon.setValue(well.coordinates.longitude)
        else:
            self.check_coords.setChecked(False)

    def _load_pump(self, pump: PumpMetadata) -> None:
        self.combo_pump_type.setCurrentText(pump.pump_type)
        self.edit_pump_brand.setText(pump.brand)
        self.spin_intake.setValue(pump.intake_depth_m)
        self.spin_max_flow.setValue(pump.max_flow_m3_h)

    def _on_accept(self) -> None:
        try:
            self.pump_metadata()
        except MetadataError as exc:
            QMessageBox.warning(self, "Dado obrigatório da bomba", str(exc))
            return
        self.accept()

    def well_metadata(self) -> WellMetadata:
        """Monta o :class:`WellMetadata` a partir do formulário."""
        datum = (
            MeasurementDatum.WELLHEAD
            if self.combo_datum.currentIndex() == 0
            else MeasurementDatum.GROUND
        )
        common = dict(
            well_id=self.edit_well_id.text().strip(),
            datetime_utc=self.edit_datetime.dateTime().toPython(),
            location=self.edit_location.text().strip(),
            purpose=self.combo_purpose.currentText().strip(),
            well_depth_m=(
                self.spin_well_depth.value() if self.spin_well_depth.value() > 0 else None
            ),
            wellhead_height_m=self.spin_wellhead.value(),
            measurement_datum=datum,
            weather=self.edit_weather.text().strip(),
            notes=self.edit_notes.toPlainText().strip(),
        )
        if not self.check_coords.isChecked():
            return WellMetadata(**common)
        return WellMetadata.from_geographic(
            self.spin_lat.value(), self.spin_lon.value(), **common
        )

    def pump_metadata(self) -> PumpMetadata:
        """Monta o :class:`PumpMetadata`. Levanta ``MetadataError`` se faltar dado."""
        return PumpMetadata(
            intake_depth_m=self.spin_intake.value(),
            max_flow_m3_h=self.spin_max_flow.value(),
            pump_type=self.combo_pump_type.currentText().strip(),
            brand=self.edit_pump_brand.text().strip(),
        )
