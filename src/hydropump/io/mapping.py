"""Mapeamento das colunas da caderneta para a série (tempo, rebaixamento).

Separado da interface para que a mesma detecção sirva à GUI, a um CLI e aos
testes — antes esta lógica vivia inline no script do Streamlit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hydropump.text import normalize_label

__all__ = [
    "TIME_ALIASES",
    "DYNAMIC_LEVEL_ALIASES",
    "DRAWDOWN_ALIASES",
    "RECOVERY_TIME_ALIASES",
    "RECOVERY_LEVEL_ALIASES",
    "RECOVERY_DRAWDOWN_ALIASES",
    "ColumnMappingError",
    "TimeSeries",
    "detect_column",
    "extract_series",
]

TIME_ALIASES: tuple[str, ...] = (
    "tempomin", "tempo", "tmin", "timemin", "t", "time", "tempo(min)",
)
DYNAMIC_LEVEL_ALIASES: tuple[str, ...] = (
    "ndm", "niveldinamico", "nd", "nivel", "nd(m)", "niveldinamico(m)",
)
DRAWDOWN_ALIASES: tuple[str, ...] = (
    "rebaixamentom", "rebaixamento", "s", "sm", "drawdown", "drawdownm",
    "rebaixamento(m)",
)
RECOVERY_TIME_ALIASES: tuple[str, ...] = (
    "temporecmin", "temporec", "tlinha", "trec", "tempoderecuperacao",
    "temporec(min)",
)
RECOVERY_LEVEL_ALIASES: tuple[str, ...] = (
    "ndrecm", "nivelderecuperacao", "ndrec", "nrec", "ndrec(m)",
)
RECOVERY_DRAWDOWN_ALIASES: tuple[str, ...] = (
    "s2m", "rebaixamentoresidualm", "rebaixamentoresidual", "sduasm", "s''m",
)


class ColumnMappingError(Exception):
    """As colunas escolhidas não produzem uma série utilizável."""


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """Série limpa, ordenada e pronta para o ajuste.

    Attributes
    ----------
    time_min : Tempos [min], estritamente positivos e crescentes.
    drawdown_m : Rebaixamentos [m] correspondentes.
    n_discarded : Linhas descartadas (NaN, texto ou t <= 0).
    """

    time_min: np.ndarray
    drawdown_m: np.ndarray
    n_discarded: int

    @property
    def time_sec(self) -> np.ndarray:
        return self.time_min * 60.0

    def __len__(self) -> int:
        return int(self.time_min.size)


def detect_column(columns: Iterable[object], aliases: Sequence[str]) -> str | None:
    """Encontra a primeira coluna cujo nome normalizado bate com um alias.

    Examples
    --------
    >>> detect_column(["Tempo (min)", "ND (m)"], TIME_ALIASES)
    'Tempo (min)'
    >>> detect_column(["a", "b"], TIME_ALIASES) is None
    True
    """
    normalized = {normalize_label(c): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def extract_series(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    time_unit: str = "minutos",
    values_are_levels: bool = False,
    static_level_m: float = 0.0,
) -> TimeSeries:
    """Converte duas colunas da planilha em uma :class:`TimeSeries` válida.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabela lida da caderneta.
    time_column, value_column : str
        Colunas escolhidas (pela detecção automática ou pelo usuário).
    time_unit : {"minutos", "segundos", "horas"}
        Unidade da coluna de tempo na planilha.
    values_are_levels : bool
        ``True`` se ``value_column`` traz o nível dinâmico ND; nesse caso o
        rebaixamento é calculado como ``s = ND - NE``.
    static_level_m : float
        Nível estático NE [m], usado apenas quando ``values_are_levels``.

    Raises
    ------
    ColumnMappingError
        Coluna ausente, ou menos de 3 pontos válidos com t > 0.
    """
    for column in (time_column, value_column):
        if column not in df.columns:
            raise ColumnMappingError(
                f"Coluna «{column}» não existe na planilha. "
                f"Disponíveis: {list(df.columns)}"
            )

    divisor = {"minutos": 1.0, "segundos": 60.0, "horas": 1.0 / 60.0}
    if time_unit not in divisor:
        raise ColumnMappingError(
            f"Unidade de tempo desconhecida: {time_unit!r}. "
            f"Use uma de {sorted(divisor)}."
        )

    raw_time = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
    time_min = raw_time / divisor[time_unit]

    raw_value = pd.to_numeric(df[value_column], errors="coerce").to_numpy(dtype=float)
    drawdown = raw_value - static_level_m if values_are_levels else raw_value

    valid = np.isfinite(time_min) & np.isfinite(drawdown) & (time_min > 0.0)
    n_discarded = int(valid.size - np.count_nonzero(valid))

    time_min, drawdown = time_min[valid], drawdown[valid]
    order = np.argsort(time_min)

    series = TimeSeries(time_min[order], drawdown[order], n_discarded)

    if len(series) < 3:
        raise ColumnMappingError(
            "São necessários ao menos 3 pontos com tempo > 0 e rebaixamento "
            f"numérico; foram encontrados {len(series)}. Confira o mapeamento "
            "de colunas e o separador decimal do arquivo."
        )
    return series
