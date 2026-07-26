"""Testes de ``hydropump.io``: leitura tabular e mapeamento de colunas."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from hydropump.io.mapping import (
    TIME_ALIASES,
    ColumnMappingError,
    detect_column,
    extract_series,
)
from hydropump.io.readers import TabularReadError, read_tabular


def _as_upload(text: str, name: str) -> io.BytesIO:
    buf = io.BytesIO(text.encode("utf-8"))
    buf.name = name  # type: ignore[attr-defined]
    return buf


CSV_BRASILEIRO = """tempo_min;nd_m;vol_litros;tempo_balde_s
1;12,10;18,0;10,2
5;12,45;17,5;10,5
10;12,60;17,2;10,6
30;12,90;16,9;10,8
60;13,05;16,8;10,9
120;13,10;16,8;10,9
"""

CSV_INTERNACIONAL = """tempo_min,nd_m,vol_litros,tempo_balde_s
1,12.10,18.0,10.2
5,12.45,17.5,10.5
10,12.60,17.2,10.6
"""


def test_le_csv_brasileiro_ponto_e_virgula_com_decimal_virgula():
    result = read_tabular(_as_upload(CSV_BRASILEIRO, "ensaio.csv"))
    assert result.separator == ";"
    assert result.decimal == ","
    assert result.df["nd_m"].dtype.kind == "f"
    assert result.df["nd_m"].iloc[0] == pytest.approx(12.10)


def test_le_csv_internacional_virgula_com_decimal_ponto():
    result = read_tabular(_as_upload(CSV_INTERNACIONAL, "ensaio.csv"))
    assert result.separator == ","
    assert result.decimal == "."
    assert result.df["nd_m"].iloc[0] == pytest.approx(12.10)


def test_rejeita_arquivo_vazio():
    with pytest.raises(TabularReadError):
        read_tabular(_as_upload("", "vazio.csv"))


def test_rejeita_arquivo_sem_coluna_numerica():
    texto = "a;b;c\nx;y;z\n"
    with pytest.raises(TabularReadError):
        read_tabular(_as_upload(texto, "ruim.csv"))


def test_detect_column_encontra_alias():
    assert detect_column(["Tempo (min)", "ND (m)"], TIME_ALIASES) == "Tempo (min)"


def test_detect_column_retorna_none_sem_match():
    assert detect_column(["a", "b"], TIME_ALIASES) is None


def test_extract_series_a_partir_de_nivel_dinamico():
    df = pd.DataFrame({"t": [1, 5, 10, 30], "nd": [10.1, 10.4, 10.6, 10.9]})
    series = extract_series(
        df, time_column="t", value_column="nd",
        values_are_levels=True, static_level_m=10.0,
    )
    np.testing.assert_allclose(series.drawdown_m, [0.1, 0.4, 0.6, 0.9])


def test_extract_series_descarta_invalidos_e_ordena():
    df = pd.DataFrame({"t": [10, 1, -5, np.nan, 30], "s": [1.0, 0.1, 0.2, 0.3, 0.9]})
    series = extract_series(df, time_column="t", value_column="s")
    assert list(series.time_min) == [1.0, 10.0, 30.0]
    assert series.n_discarded == 2


def test_extract_series_exige_ao_menos_3_pontos():
    df = pd.DataFrame({"t": [1, 2], "s": [0.1, 0.2]})
    with pytest.raises(ColumnMappingError):
        extract_series(df, time_column="t", value_column="s")


def test_extract_series_converte_segundos_para_minutos():
    df = pd.DataFrame({"t": [60, 300, 600], "s": [0.1, 0.2, 0.3]})
    series = extract_series(df, time_column="t", value_column="s", time_unit="segundos")
    np.testing.assert_allclose(series.time_min, [1.0, 5.0, 10.0])
