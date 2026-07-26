"""Fixtures compartilhadas. Com `src-layout` + `pyproject.toml` (ver
`[tool.pytest.ini_options] pythonpath = ["src"]`), nenhum sys.path manual é
necessário aqui — mantido apenas como fallback para quem rodar os testes
sem instalar o pacote em modo editável.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
