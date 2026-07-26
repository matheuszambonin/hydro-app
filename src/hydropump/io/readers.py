"""Leitura de caderneta de campo em CSV ou Excel.

O problema que este módulo resolve
----------------------------------
``pd.read_csv(file)`` assume ``sep=","`` e ``decimal="."``. Uma planilha
exportada do Excel em português do Brasil sai com ``sep=";"`` e vírgula
decimal — e o pandas devolve, sem erro algum, uma única coluna de strings.
O usuário só via "são necessários ao menos 3 pontos válidos", sem pista da
causa. Aqui a detecção é explícita e o resultado informa o que foi assumido.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["TabularReadError", "ReadResult", "read_tabular", "read_path"]

_CANDIDATE_SEPARATORS = (";", ",", "\t", "|")
_SAMPLE_BYTES = 64 * 1024


class TabularReadError(Exception):
    """O arquivo não pôde ser lido como tabela."""


@dataclass(frozen=True, slots=True)
class ReadResult:
    """Tabela lida mais o dialeto que foi efetivamente assumido.

    Attributes
    ----------
    df : Tabela.
    separator : Separador de campo, ou ``None`` para Excel.
    decimal : Separador decimal, ou ``None`` para Excel.
    source_name : Nome do arquivo de origem.
    """

    df: pd.DataFrame
    separator: str | None
    decimal: str | None
    source_name: str

    @property
    def dialect_label(self) -> str:
        if self.separator is None:
            return "Excel"
        sep = {"\t": "TAB"}.get(self.separator, self.separator)
        return f"CSV (separador «{sep}», decimal «{self.decimal}»)"


def _decode(raw: bytes) -> str:
    """Decodifica tentando UTF-8 (com e sem BOM) e depois Latin-1."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise TabularReadError(
        "Não foi possível decodificar o arquivo. Salve-o novamente em UTF-8."
    )


def _sniff_separator(sample: str) -> str:
    """Descobre o separador de campo do CSV.

    Tenta o ``csv.Sniffer`` e, se ele falhar (comum em arquivos com poucas
    linhas), decide pela contagem de ocorrências na primeira linha.
    """
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_SEPARATORS))
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {sep: first_line.count(sep) for sep in _CANDIDATE_SEPARATORS}
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else ","


def _numeric_column_count(df: pd.DataFrame) -> int:
    return int(df.select_dtypes(include="number").shape[1])


def read_tabular(file: Any, name: str | None = None) -> ReadResult:
    """Lê um CSV ou Excel de caderneta de campo.

    Parameters
    ----------
    file : file-like ou bytes
        Objeto retornado por ``st.file_uploader`` ou um handle binário.
    name : str | None
        Nome do arquivo, usado para escolher o leitor pela extensão. Se
        omitido, tenta ``file.name``.

    Returns
    -------
    ReadResult

    Raises
    ------
    TabularReadError
        Arquivo vazio, ilegível ou sem nenhuma coluna numérica.

    Notes
    -----
    Para CSV a estratégia é: detectar o separador, ler com ponto decimal e,
    se **nenhuma** coluna numérica aparecer, reler com vírgula decimal. Uma
    planilha brasileira típica cai no segundo caso.
    """
    filename = name or getattr(file, "name", "") or ""
    suffix = Path(filename).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            df = pd.read_excel(file)
        except Exception as exc:  # noqa: BLE001 — a lib lança tipos variados
            raise TabularReadError(f"Falha ao ler a planilha Excel: {exc}") from exc
        if df.empty:
            raise TabularReadError("A planilha não contém linhas de dados.")
        return ReadResult(df, None, None, filename)

    raw = file.read() if hasattr(file, "read") else file
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw:
        raise TabularReadError("O arquivo está vazio.")

    text = _decode(raw)
    separator = _sniff_separator(text[:_SAMPLE_BYTES])

    best: ReadResult | None = None
    for decimal in (".", ","):
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                decimal=decimal,
                skipinitialspace=True,
            )
        except Exception:  # noqa: BLE001 — tentativa seguinte pode funcionar
            continue
        if df.empty:
            continue
        candidate = ReadResult(df, separator, decimal, filename)
        if best is None or _numeric_column_count(df) > _numeric_column_count(best.df):
            best = candidate
        if _numeric_column_count(df) >= 2:
            break

    if best is None:
        raise TabularReadError(
            "Não foi possível interpretar o arquivo como CSV. Verifique se "
            "há uma linha de cabeçalho e ao menos duas colunas."
        )
    if _numeric_column_count(best.df) == 0:
        raise TabularReadError(
            f"Nenhuma coluna numérica encontrada com {best.dialect_label}. "
            "Confira se o arquivo usa outro separador ou se os números estão "
            "misturados com texto (ex.: «12,5 m»)."
        )
    return best


def read_path(path: str | Path) -> ReadResult:
    """Conveniência para uso em CLI e testes: lê a partir de um caminho."""
    p = Path(path)
    if not p.exists():
        raise TabularReadError(f"Arquivo não encontrado: {p}")
    with p.open("rb") as fh:
        return read_tabular(fh, name=p.name)
