"""Testes dos módulos adicionados na v3.1: geo, cenários, metadados, degraus."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydropump.domain import math as hm
from hydropump.domain.geo import (
    CoordinateError,
    GeographicCoordinate,
    UTMCoordinate,
    central_meridian,
    geographic_to_utm,
    latitude_band,
    utm_to_geographic,
    utm_zone_for_longitude,
)
from hydropump.domain.metadata import (
    MeasurementDatum,
    MetadataError,
    PumpMetadata,
    WellMetadata,
)
from hydropump.domain.scenarios import AquiferScenario, profile_for


# --------------------------------------------------------------------------
# Conversão de coordenadas
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lat,lon",
    [
        (-22.9068, -43.1729),   # Rio de Janeiro
        (-23.5505, -46.6333),   # São Paulo
        (-15.7939, -47.8828),   # Brasília
        (-3.1190, -60.0217),    # Manaus
        (-30.0346, -51.2177),   # Porto Alegre
        (0.0349, -51.0694),     # Macapá (sobre o equador)
        (51.5074, -0.1278),     # Londres (hemisfério norte)
    ],
)
def test_geo_utm_round_trip_submilimetrico(lat, lon):
    geo = GeographicCoordinate(lat, lon)
    back = utm_to_geographic(geographic_to_utm(geo))

    erro_lat_m = abs(back.latitude - lat) * 111_320.0
    erro_lon_m = abs(back.longitude - lon) * 111_320.0 * math.cos(math.radians(lat))
    assert math.hypot(erro_lat_m, erro_lon_m) < 1e-3  # < 1 mm


@pytest.mark.parametrize("lat", [-60.0, -23.0, 0.0, 45.0])
def test_easting_e_500000_no_meridiano_central(lat):
    """Invariante da projeção: no MC o easting é exatamente o falso leste."""
    zona = 23
    coord = GeographicCoordinate(lat, central_meridian(zona))
    assert geographic_to_utm(coord).easting == pytest.approx(500_000.0, abs=1e-6)


def test_zona_e_faixa_conhecidas():
    utm = geographic_to_utm(GeographicCoordinate(-22.9068, -43.1729))
    assert (utm.zone, utm.hemisphere, utm.band) == (23, "S", "K")
    assert utm.zone_label == "23K"


def test_utm_zone_for_longitude():
    assert utm_zone_for_longitude(-43.1729) == 23
    assert utm_zone_for_longitude(-60.0217) == 20
    assert utm_zone_for_longitude(-177.0) == 1
    assert utm_zone_for_longitude(180.0) == 60


def test_latitude_band_fora_de_cobertura_e_vazia():
    assert latitude_band(-85.0) == ""
    assert latitude_band(88.0) == ""
    assert latitude_band(-22.9) == "K"


def test_coordenada_invalida_e_rejeitada():
    with pytest.raises(CoordinateError):
        GeographicCoordinate(95.0, 0.0)
    with pytest.raises(CoordinateError):
        UTMCoordinate(500_000.0, 0.0, 99, "S")
    with pytest.raises(CoordinateError):
        UTMCoordinate(500_000.0, 0.0, 23, "X")


# --------------------------------------------------------------------------
# Cenários
# --------------------------------------------------------------------------
def test_perfis_de_cenario_tem_regras_coerentes():
    confinado = profile_for(AquiferScenario.CONFINED)
    assert not confinado.apply_jacob_by_default
    assert not confinado.unconfined
    assert confinado.storativity_range[1] <= 1e-3

    laterita = profile_for(AquiferScenario.UNCONFINED_LATERITIC_BASE)
    assert laterita.b_is_well_constrained
    assert laterita.apply_jacob_by_default

    misto = profile_for(AquiferScenario.MIXED_GRANULAR_FRACTURED)
    assert misto.check_dual_porosity
    # faixa alargada: cobre desde porosidade de fratura até produção específica
    assert misto.storativity_range[0] < profile_for(
        AquiferScenario.UNCONFINED_GRANULAR
    ).storativity_range[0]


def test_todo_cenario_tem_metodologia_e_ressalvas():
    for scenario in AquiferScenario:
        profile = profile_for(scenario)
        assert profile.methodology.strip()
        assert profile.caveats


# --------------------------------------------------------------------------
# Metadados
# --------------------------------------------------------------------------
def test_bomba_exige_crivo_e_vazao_maxima():
    with pytest.raises(MetadataError):
        PumpMetadata(intake_depth_m=0.0, max_flow_m3_h=10.0)
    with pytest.raises(MetadataError):
        PumpMetadata(intake_depth_m=30.0, max_flow_m3_h=0.0)


def test_bomba_calcula_rebaixamento_disponivel_e_uso():
    pump = PumpMetadata(intake_depth_m=36.0, max_flow_m3_h=25.0)
    assert pump.available_drawdown_m(8.5) == pytest.approx(27.5)
    assert pump.flow_utilization(12.5) == pytest.approx(0.5)


def test_conversao_entre_datum_boca_e_solo():
    medido_da_boca = WellMetadata(
        wellhead_height_m=0.65, measurement_datum=MeasurementDatum.WELLHEAD
    )
    assert medido_da_boca.static_level_from_ground(10.0) == pytest.approx(9.35)
    assert medido_da_boca.static_level_from_wellhead(10.0) == pytest.approx(10.0)

    medido_do_solo = WellMetadata(
        wellhead_height_m=0.65, measurement_datum=MeasurementDatum.GROUND
    )
    assert medido_do_solo.static_level_from_wellhead(10.0) == pytest.approx(10.65)
    assert medido_do_solo.static_level_from_ground(10.0) == pytest.approx(10.0)


def test_metadata_deriva_utm_da_geografica_e_vice_versa():
    por_geo = WellMetadata.from_geographic(-22.9068, -43.1729)
    assert por_geo.utm is not None and por_geo.utm.zone == 23

    por_utm = WellMetadata.from_utm(687394.59, 7465634.13, 23, "S")
    assert por_utm.coordinates is not None
    assert por_utm.coordinates.latitude == pytest.approx(-22.9068, abs=1e-4)


# --------------------------------------------------------------------------
# Teste de degraus
# --------------------------------------------------------------------------
def test_step_drawdown_recupera_B_e_C_conhecidos():
    Q = np.array([0.005, 0.010, 0.015, 0.020])
    B_true, C_true = 800.0, 20000.0
    s = B_true * Q + C_true * Q**2

    res = hm.step_drawdown_analysis(Q, s)
    assert pytest.approx(B_true) == res.B
    assert pytest.approx(C_true) == res.C
    assert res.r_squared > 0.9999


def test_eficiencia_do_poco_bate_com_a_definicao():
    Q = np.array([0.005, 0.010, 0.015, 0.020])
    B, C = 800.0, 20000.0
    res = hm.step_drawdown_analysis(Q, B * Q + C * Q**2)

    q = 0.010
    esperado = (B * q) / (B * q + C * q**2)
    assert res.well_efficiency(q) == pytest.approx(esperado)


def test_step_drawdown_rejeita_dados_insuficientes_ou_invalidos():
    with pytest.raises(hm.RegressionError):
        hm.step_drawdown_analysis([0.01, 0.02], [1.0, 2.0])
    with pytest.raises(hm.RegressionError):  # vazões todas iguais
        hm.step_drawdown_analysis([0.01, 0.01, 0.01], [1.0, 2.0, 3.0])
    with pytest.raises(hm.InvalidGeometryError):
        hm.step_drawdown_analysis([0.01, 0.02, -0.01], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# Dupla porosidade e achatamento tardio
# --------------------------------------------------------------------------
def _dual_porosity_curve():
    """Duas retas de mesma inclinação separadas por um patamar mais suave."""
    t1 = np.logspace(1, np.log10(300), 8)
    t2 = np.logspace(np.log10(320), np.log10(3000), 5)
    t3 = np.logspace(np.log10(3200), 5, 8)
    s1 = 0.5 * np.log10(t1 / 5.0)
    fim1 = 0.5 * np.log10(300 / 5.0)
    s2 = fim1 + 0.12 * np.log10(t2 / 300.0)
    fim2 = fim1 + 0.12 * np.log10(3000 / 300.0)
    s3 = fim2 + 0.5 * np.log10(t3 / 3000.0)
    return np.concatenate([t1, t2, t3]), np.concatenate([s1, s2, s3])


def test_detecta_assinatura_de_dupla_porosidade():
    t, s = _dual_porosity_curve()
    res = hm.detect_dual_porosity(t, s, Q=0.01, r=10.0)

    assert res is not None and res.detected
    assert res.slope_ratio == pytest.approx(1.0, abs=0.2)
    # o armazenamento das fraturas tem de ser MENOR que o do sistema todo
    assert res.storativity_early < res.storativity_late


def test_nao_detecta_dupla_porosidade_em_reta_pura():
    t = np.logspace(1, 5, 25)
    res = hm.detect_dual_porosity(t, 0.5 * np.log10(t / 5.0), Q=0.01, r=10.0)
    assert res is None or not res.detected


def test_detecta_achatamento_tardio():
    t = np.logspace(1, 5, 30)
    s = np.where(
        t < 3000,
        0.5 * np.log10(t / 5.0),
        0.5 * np.log10(3000 / 5.0) + 0.1 * np.log10(t / 3000.0),
    )
    assert hm.detect_late_time_flattening(t, s) < 0.6


def test_reta_pura_nao_acusa_achatamento():
    t = np.logspace(1, 5, 30)
    assert hm.detect_late_time_flattening(t, 0.5 * np.log10(t / 5.0)) == pytest.approx(
        1.0, abs=0.05
    )
