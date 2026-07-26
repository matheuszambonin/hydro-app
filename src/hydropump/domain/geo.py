"""Conversão entre coordenadas geográficas e UTM, sem dependências externas.

Implementa as séries de Snyder (USGS Professional Paper 1395, 1987) para a
projeção Transversa de Mercator, exatas a nível milimétrico dentro de um
fuso UTM — precisão muito além do necessário para locação de poço.

Por que não ``pyproj``
----------------------
O ``pyproj`` resolveria isto em duas linhas, mas arrasta a biblioteca PROJ
e sua base de dados de grades (dezenas de MB) para dentro do executável, e
é uma fonte conhecida de dor no PyInstaller. Para a conversão UTM↔geográfica
pura (sem transformação de datum), as fórmulas fechadas abaixo são o
caminho certo.

Datum
-----
Usa o elipsoide **WGS84**. Para fins práticos no Brasil, SIRGAS2000 e WGS84
são equivalentes (diferença da ordem de centímetros, muito abaixo da
incerteza de um GPS de navegação usado em campo). Coordenadas em datums
antigos (Córrego Alegre, SAD69) exigem transformação de datum, que este
módulo **não** faz — a diferença chega a dezenas de metros e o usuário deve
converter antes de inserir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "WGS84_A",
    "WGS84_F",
    "CoordinateError",
    "UTMCoordinate",
    "GeographicCoordinate",
    "geographic_to_utm",
    "utm_to_geographic",
    "utm_zone_for_longitude",
    "central_meridian",
    "latitude_band",
    "format_dms",
]

# Elipsoide WGS84
WGS84_A: float = 6378137.0
WGS84_F: float = 1.0 / 298.257223563

_E2 = 2.0 * WGS84_F - WGS84_F**2          # primeira excentricidade ao quadrado
_EP2 = _E2 / (1.0 - _E2)                   # segunda excentricidade ao quadrado
_K0 = 0.9996                               # fator de escala no meridiano central
_FALSE_EASTING = 500_000.0
_FALSE_NORTHING_SOUTH = 10_000_000.0

#: Faixas de latitude MGRS (C a X, omitindo I e O). Cada faixa cobre 8°,
#: exceto X, que cobre 12°.
_BANDS = "CDEFGHJKLMNPQRSTUVWX"


class CoordinateError(ValueError):
    """Coordenada fora de faixa válida ou inconsistente."""


@dataclass(frozen=True, slots=True)
class GeographicCoordinate:
    """Latitude/longitude em graus decimais, datum WGS84/SIRGAS2000."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise CoordinateError(
                f"Latitude fora de faixa: {self.latitude}° (esperado -90 a 90)."
            )
        if not -180.0 <= self.longitude <= 180.0:
            raise CoordinateError(
                f"Longitude fora de faixa: {self.longitude}° (esperado -180 a 180)."
            )

    def format_decimal(self, digits: int = 6) -> str:
        return f"{self.latitude:.{digits}f}, {self.longitude:.{digits}f}"

    def format_dms(self) -> str:
        return f"{format_dms(self.latitude, 'lat')}  {format_dms(self.longitude, 'lon')}"


@dataclass(frozen=True, slots=True)
class UTMCoordinate:
    """Coordenada UTM.

    Attributes
    ----------
    easting : E [m], nominalmente 100.000–900.000 dentro do fuso.
    northing : N [m]. No hemisfério sul usa-se o falso norte de 10.000.000.
    zone : Fuso UTM (1–60).
    hemisphere : ``"N"`` ou ``"S"``.
    band : Letra da faixa de latitude MGRS (informativa).
    """

    easting: float
    northing: float
    zone: int
    hemisphere: str
    band: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.zone <= 60:
            raise CoordinateError(f"Fuso UTM inválido: {self.zone} (esperado 1 a 60).")
        if self.hemisphere not in ("N", "S"):
            raise CoordinateError(
                f"Hemisfério inválido: {self.hemisphere!r} (esperado 'N' ou 'S')."
            )
        if not math.isfinite(self.easting) or not math.isfinite(self.northing):
            raise CoordinateError("Easting/Northing devem ser finitos.")

    @property
    def zone_label(self) -> str:
        """Rótulo usual do fuso, ex.: ``"23K"`` (ou ``"23S"`` se a faixa é desconhecida)."""
        return f"{self.zone}{self.band or self.hemisphere}"

    def format_compact(self) -> str:
        return f"{self.zone_label}  E {self.easting:.2f}  N {self.northing:.2f}"


def utm_zone_for_longitude(longitude: float) -> int:
    """Fuso UTM padrão para uma longitude (ignora as exceções da Noruega/Svalbard)."""
    if longitude == 180.0:
        return 60
    return int(math.floor((longitude + 180.0) / 6.0)) + 1


def central_meridian(zone: int) -> float:
    """Longitude do meridiano central do fuso, em graus."""
    return (zone - 1) * 6.0 - 180.0 + 3.0


def latitude_band(latitude: float) -> str:
    """Letra da faixa de latitude MGRS. Vazio fora da cobertura (-80° a 84°)."""
    if not -80.0 <= latitude <= 84.0:
        return ""
    if latitude >= 72.0:  # faixa X cobre 12°
        return "X"
    index = int((latitude + 80.0) // 8.0)
    return _BANDS[min(index, len(_BANDS) - 1)]


def _meridional_arc(lat_rad: float) -> float:
    """Distância M ao longo do meridiano, do equador até a latitude dada."""
    return WGS84_A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0) * lat_rad
        - (3.0 * _E2 / 8.0 + 3.0 * _E2**2 / 32.0 + 45.0 * _E2**3 / 1024.0)
        * math.sin(2.0 * lat_rad)
        + (15.0 * _E2**2 / 256.0 + 45.0 * _E2**3 / 1024.0) * math.sin(4.0 * lat_rad)
        - (35.0 * _E2**3 / 3072.0) * math.sin(6.0 * lat_rad)
    )


def geographic_to_utm(
    coord: GeographicCoordinate, zone: int | None = None
) -> UTMCoordinate:
    """Converte latitude/longitude para UTM.

    Parameters
    ----------
    coord : GeographicCoordinate
    zone : int | None
        Força um fuso específico (útil para manter a coerência quando um
        levantamento cruza a divisa entre fusos). Por padrão usa o fuso
        natural da longitude.

    Examples
    --------
    >>> c = GeographicCoordinate(-22.9068, -43.1729)  # Rio de Janeiro
    >>> u = geographic_to_utm(c)
    >>> u.zone, u.hemisphere, u.band
    (23, 'S', 'K')
    >>> round(u.easting, 2), round(u.northing, 2)
    (687394.59, 7465634.13)
    """
    lat = math.radians(coord.latitude)
    lon = math.radians(coord.longitude)

    z = zone if zone is not None else utm_zone_for_longitude(coord.longitude)
    lon0 = math.radians(central_meridian(z))

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n = WGS84_A / math.sqrt(1.0 - _E2 * sin_lat**2)
    t = tan_lat**2
    c = _EP2 * cos_lat**2
    a_ = (lon - lon0) * cos_lat
    m = _meridional_arc(lat)

    easting = _K0 * n * (
        a_
        + (1.0 - t + c) * a_**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * _EP2) * a_**5 / 120.0
    ) + _FALSE_EASTING

    northing = _K0 * (
        m
        + n
        * tan_lat
        * (
            a_**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * a_**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * _EP2) * a_**6 / 720.0
        )
    )

    hemisphere = "N" if coord.latitude >= 0 else "S"
    if hemisphere == "S":
        northing += _FALSE_NORTHING_SOUTH

    return UTMCoordinate(
        easting=easting,
        northing=northing,
        zone=z,
        hemisphere=hemisphere,
        band=latitude_band(coord.latitude),
    )


def utm_to_geographic(coord: UTMCoordinate) -> GeographicCoordinate:
    """Converte UTM para latitude/longitude.

    Examples
    --------
    >>> u = UTMCoordinate(687394.59, 7465634.13, 23, "S")
    >>> g = utm_to_geographic(u)
    >>> round(g.latitude, 4), round(g.longitude, 4)
    (-22.9068, -43.1729)
    """
    x = coord.easting - _FALSE_EASTING
    y = coord.northing
    if coord.hemisphere == "S":
        y -= _FALSE_NORTHING_SOUTH

    lon0 = math.radians(central_meridian(coord.zone))

    m = y / _K0
    mu = m / (
        WGS84_A * (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0)
    )

    e1 = (1.0 - math.sqrt(1.0 - _E2)) / (1.0 + math.sqrt(1.0 - _E2))
    lat1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    tan_lat1 = math.tan(lat1)

    c1 = _EP2 * cos_lat1**2
    t1 = tan_lat1**2
    n1 = WGS84_A / math.sqrt(1.0 - _E2 * sin_lat1**2)
    r1 = WGS84_A * (1.0 - _E2) / (1.0 - _E2 * sin_lat1**2) ** 1.5
    d = x / (n1 * _K0)

    lat = lat1 - (n1 * tan_lat1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * _EP2) * d**4 / 24.0
        + (
            61.0
            + 90.0 * t1
            + 298.0 * c1
            + 45.0 * t1**2
            - 252.0 * _EP2
            - 3.0 * c1**2
        )
        * d**6
        / 720.0
    )

    lon = lon0 + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * _EP2 + 24.0 * t1**2)
        * d**5
        / 120.0
    ) / cos_lat1

    return GeographicCoordinate(math.degrees(lat), math.degrees(lon))


def format_dms(value: float, kind: str = "lat") -> str:
    """Formata graus decimais como graus/minutos/segundos com hemisfério.

    Examples
    --------
    >>> print(format_dms(-22.9068, "lat"))
    22°54'24.48" S
    >>> print(format_dms(-43.1729, "lon"))
    43°10'22.44" W
    """
    suffix = ("N" if value >= 0 else "S") if kind == "lat" else ("E" if value >= 0 else "W")

    magnitude = abs(value)
    degrees = int(magnitude)
    minutes_full = (magnitude - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0

    # Corrige arredondamento em 60.00" / 60'
    if round(seconds, 2) >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        degrees += 1

    return f"{degrees}°{minutes:02d}'{seconds:05.2f}\" {suffix}"
