"""Normalização de rótulos, compartilhada por leitura de dados e plotagem.

Antes do refactor esta função existia em duas cópias quase idênticas
(``app._normaliza`` e ``plotting._normaliza_rotulo``), com risco de divergir.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize_label"]


def normalize_label(value: object) -> str:
    """Reduz um rótulo a uma chave comparável: minúsculas, sem acento nem separadores.

    Examples
    --------
    >>> normalize_label(" Tempo_Min ")
    'tempomin'
    >>> normalize_label("Nível Dinâmico (ND)")
    'niveldinamico(nd)'
    >>> normalize_label("Sintético") == normalize_label("sintetico")
    True
    """
    raw = str(value).strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(c)
    )
    for sep in (" ", "_", "-"):
        unaccented = unaccented.replace(sep, "")
    return unaccented
