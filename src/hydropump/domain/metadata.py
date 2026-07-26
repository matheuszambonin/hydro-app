"""Metadados de identificação do poço e da bomba instalada.

Estes dados não entram em nenhuma equação — existem para que o resultado
seja rastreável e o relatório seja um documento de laudo, não apenas uma
lista de números. A única exceção é :attr:`WellMetadata.wellhead_height_m`
combinada com :attr:`WellMetadata.measurement_datum`, que afeta como os
níveis são **reportados** (ver a nota em :meth:`WellMetadata.level_note`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from hydropump.domain.geo import (
    GeographicCoordinate,
    UTMCoordinate,
    geographic_to_utm,
    utm_to_geographic,
)

__all__ = [
    "MeasurementDatum",
    "WellMetadata",
    "PumpMetadata",
    "MetadataError",
]


class MetadataError(ValueError):
    """Metadado obrigatório ausente ou inconsistente."""


class MeasurementDatum(StrEnum):
    """Referência a partir da qual as profundidades foram medidas em campo."""

    WELLHEAD = "boca"
    GROUND = "solo"

    @property
    def label(self) -> str:
        return (
            "Boca do poço (topo do revestimento)"
            if self is MeasurementDatum.WELLHEAD
            else "Superfície do terreno"
        )


@dataclass(frozen=True, slots=True)
class WellMetadata:
    """Identificação e contexto do poço ensaiado.

    Attributes
    ----------
    well_id : Identificação do poço (ex.: "PZ-01", "PT-03").
    datetime_utc : Data e hora do ensaio.
    location : Local / propriedade / município.
    purpose : Finalidade do poço (abastecimento, monitoramento, irrigação...).
    coordinates : Coordenada geográfica, se informada.
    utm : Coordenada UTM correspondente (derivada, quando há geográfica).
    well_depth_m : Profundidade total perfurada [m] (opcional).
    wellhead_height_m : Altura da boca do poço em relação ao terreno [m].
    measurement_datum : Se as leituras de nível foram tomadas a partir da
        boca do poço ou da superfície do terreno.
    weather : Condição climática durante o ensaio (opcional).
    notes : Observações livres.
    """

    well_id: str = ""
    datetime_utc: datetime | None = None
    location: str = ""
    purpose: str = ""
    coordinates: GeographicCoordinate | None = None
    utm: UTMCoordinate | None = None
    well_depth_m: float | None = None
    wellhead_height_m: float = 0.0
    measurement_datum: MeasurementDatum = MeasurementDatum.WELLHEAD
    weather: str = ""
    notes: str = ""

    @classmethod
    def from_geographic(
        cls, latitude: float, longitude: float, **kwargs
    ) -> WellMetadata:
        """Cria a partir de lat/lon, derivando a UTM automaticamente."""
        geo = GeographicCoordinate(latitude, longitude)
        return cls(coordinates=geo, utm=geographic_to_utm(geo), **kwargs)

    @classmethod
    def from_utm(
        cls,
        easting: float,
        northing: float,
        zone: int,
        hemisphere: str = "S",
        **kwargs,
    ) -> WellMetadata:
        """Cria a partir de UTM, derivando a geográfica automaticamente."""
        utm = UTMCoordinate(easting, northing, zone, hemisphere)
        return cls(coordinates=utm_to_geographic(utm), utm=utm, **kwargs)

    def static_level_from_ground(self, static_level_measured_m: float) -> float:
        """Converte o NE medido para profundidade **a partir do terreno**.

        Se a medição foi feita a partir da boca do poço, desconta a altura
        da boca; se já foi feita a partir do terreno, devolve inalterado.
        """
        if self.measurement_datum is MeasurementDatum.WELLHEAD:
            return static_level_measured_m - self.wellhead_height_m
        return static_level_measured_m

    def static_level_from_wellhead(self, static_level_measured_m: float) -> float:
        """Converte o NE medido para profundidade **a partir da boca do poço**."""
        if self.measurement_datum is MeasurementDatum.GROUND:
            return static_level_measured_m + self.wellhead_height_m
        return static_level_measured_m

    @property
    def level_note(self) -> str:
        """Nota explicando o efeito (e o não-efeito) do datum de medição.

        Ponto importante e contraintuitivo: como o rebaixamento é uma
        **diferença** entre dois níveis medidos com a mesma referência
        (s = ND − NE), a altura da boca do poço se cancela. Portanto T, S
        e a capacidade específica **não** são afetados pela escolha do
        datum — apenas as profundidades absolutas reportadas mudam.
        """
        if self.wellhead_height_m == 0.0:
            return (
                "Altura da boca do poço não informada (ou igual a zero): "
                "níveis reportados na mesma referência em que foram medidos."
            )
        return (
            f"Leituras tomadas a partir de: {self.measurement_datum.label}. "
            f"Altura da boca do poço: {self.wellhead_height_m:.3f} m. "
            "Observação: como o rebaixamento é a diferença entre dois níveis "
            "medidos na mesma referência (s = ND − NE), a altura da boca se "
            "cancela — T, S e a capacidade específica NÃO são afetados pela "
            "escolha do datum. Apenas as profundidades absolutas mudam."
        )

    def coordinate_summary(self) -> str:
        """Resumo das coordenadas nos dois sistemas, ou aviso de ausência."""
        if self.coordinates is None or self.utm is None:
            return "Coordenadas não informadas."
        return (
            f"Geográfica (WGS84/SIRGAS2000): {self.coordinates.format_decimal()}\n"
            f"                                {self.coordinates.format_dms()}\n"
            f"UTM:                            {self.utm.format_compact()}"
        )


@dataclass(frozen=True, slots=True)
class PumpMetadata:
    """Bomba instalada durante o ensaio.

    Apenas ``intake_depth_m`` e ``max_flow_m3_h`` são obrigatórios: são os
    dois que têm consequência prática na interpretação — a profundidade do
    crivo limita o rebaixamento admissível, e a vazão máxima contextualiza
    a vazão do ensaio.

    Attributes
    ----------
    intake_depth_m : Profundidade do crivo (sucção) da bomba [m]. Obrigatório.
    max_flow_m3_h : Vazão máxima nominal da bomba [m³/h]. Obrigatório.
    pump_type : Tipo (submersa, injetora, centrífuga...). Opcional.
    brand : Marca/modelo. Opcional.
    """

    intake_depth_m: float
    max_flow_m3_h: float
    pump_type: str = ""
    brand: str = ""

    def __post_init__(self) -> None:
        if not (self.intake_depth_m > 0):
            raise MetadataError(
                "Profundidade do crivo da bomba é obrigatória e deve ser "
                f"maior que zero (recebido {self.intake_depth_m!r})."
            )
        if not (self.max_flow_m3_h > 0):
            raise MetadataError(
                "Vazão máxima da bomba é obrigatória e deve ser maior que "
                f"zero (recebido {self.max_flow_m3_h!r})."
            )

    def available_drawdown_m(self, static_level_m: float) -> float:
        """Rebaixamento máximo antes de o nível atingir o crivo [m].

        Margem operacional: abaixo disso a bomba começa a aspirar ar e
        cavita. Serve para checar se a vazão de exploração projetada é
        sustentável com a instalação atual.
        """
        return self.intake_depth_m - static_level_m

    def flow_utilization(self, Q_m3_h: float) -> float:
        """Fração da vazão nominal da bomba usada no ensaio [-]."""
        return Q_m3_h / self.max_flow_m3_h

    def describe(self) -> str:
        parts = []
        if self.pump_type:
            parts.append(f"Tipo: {self.pump_type}")
        if self.brand:
            parts.append(f"Marca/modelo: {self.brand}")
        parts.append(f"Profundidade do crivo: {self.intake_depth_m:.2f} m")
        parts.append(f"Vazão máxima nominal: {self.max_flow_m3_h:.2f} m³/h")
        return " | ".join(parts)
