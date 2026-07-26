# HydroPump — Análise de Ensaio de Bombeamento (Cooper-Jacob / Theis)

Aplicação de **desktop nativa** (PySide6/Qt) para interpretação de ensaios
de bombeamento em aquíferos livres: vazão volumétrica → correção de Jacob →
Cooper-Jacob (rebaixamento) → recuperação de Theis → diagnósticos
hidrogeológicos → gráfico científico → relatório.

Distribuída para o usuário final como instalador Windows (PyInstaller +
Inno Setup), gerado automaticamente via GitHub Actions a cada tag `v*`.
Diferente de uma solução baseada em navegador (Streamlit, Flask+HTML), o
HydroPump abre como qualquer programa comum: sem servidor local, sem porta,
sem navegador — apenas uma janela.

## Estrutura

```
.
├── main.py                     # entry point do executável (PyInstaller)
├── app.spec                    # especificação do PyInstaller
├── installer.iss               # script do Inno Setup
├── pyproject.toml
├── src/
│   └── hydropump/
│       ├── domain/
│       │   ├── math.py          # rotinas matemáticas puras (Jacob, Cooper-Jacob, Theis)
│       │   └── models.py        # AnalysisConfig, AnalysisResult, Diagnostic
│       ├── io/
│       │   ├── readers.py       # leitura robusta de CSV/Excel (BR e internacional)
│       │   └── mapping.py       # detecção de colunas e extração da série
│       ├── services/
│       │   └── pumping_test.py  # orquestra o domínio matemático + diagnósticos
│       ├── viz/
│       │   ├── theme.py         # paleta e backend Matplotlib
│       │   ├── plots.py         # gráfico principal, Cooper-Jacob, recuperação
│       │   └── schematics.py    # esquema vetorial do poço
│       ├── report/
│       │   └── text_report.py   # relatório .txt para exportação
│       ├── text.py              # normalização de rótulos (compartilhada)
│       └── ui_qt/               # interface Qt (única camada dependente de framework de UI)
│           ├── main_window.py   # janela principal — monta tudo, chama o serviço
│           ├── widgets/         # ParametersPanel, MetricCard, MplCanvas, DataFrameModel, ...
│           └── dialogs/         # ColumnMappingDialog
├── tests/                       # pytest — ver "Testes" abaixo
└── .github/workflows/release.yml
```

A camada `domain/`, `io/`, `services/`, `viz/` e `report/` **não depende de
nenhum framework de UI** — é isso que torna a aplicação inteira testável
sem abrir uma janela, e foi o que permitiu trocar a interface (de Streamlit
para Qt) sem tocar em uma linha do núcleo científico.

## Por que Qt em vez de Streamlit

Streamlit é ótimo para prototipagem, mas tecnicamente é "um site rodando
num servidor local que abre no navegador": precisa escolher porta livre,
evitar instâncias duplicadas, esperar o servidor responder antes de abrir
o navegador, e depende de uma API de bootstrap que muda entre versões. A
versão Qt elimina essa camada inteira — é uma janela nativa do sistema
operacional, com menu, barra de status, diálogos de arquivo nativos e
gráficos Matplotlib embutidos diretamente (com zoom/pan/exportação via a
barra de ferramentas nativa do próprio Matplotlib).

## Instalação para desenvolvimento (Manjaro Linux)

Usando [`uv`](https://docs.astral.sh/uv/) (recomendado):

```bash
sudo pacman -S --needed uv     # ou: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

Com `venv` + `pip` tradicional:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Execução (desenvolvimento)

```bash
uv run python main.py
```

## Testes e qualidade

```bash
uv run pytest --cov=hydropump --cov-report=term-missing
uv run ruff check .
uv run mypy src/
```

O teste mais importante do repositório é `tests/test_theis_roundtrip.py`:
gera rebaixamento pela solução **exata** de Theis (W(u), via
`scipy.special.exp1`) com T e S conhecidos, e verifica que
`cooper_jacob_analysis` os recupera dentro de tolerância. Qualquer alteração
na cadeia `linregress → T → t0 → S` que quebre esse teste é regressão.

`tests/test_ui_smoke.py` verifica que a janela Qt monta sem lançar exceção
(via `pytest-qt`, em modo `offscreen` — não precisa de display). Ele é
pulado automaticamente se `PySide6` não estiver instalado no ambiente.

## Build do instalador Windows

Localmente (requer Windows ou uma VM/CI Windows — PyInstaller não faz
cross-compile a partir do Linux):

```powershell
uv sync --extra build
uv run pyinstaller app.spec --noconfirm --clean
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

O pipeline em `.github/workflows/release.yml` faz isso automaticamente a
cada tag `v*` empurrada para o repositório, publicando o `.exe` resultante
como GitHub Release.

## Dados próprios

A leitura de CSV detecta automaticamente separador (`,` `;` `\t` `|`) e
separador decimal (`.` ou `,`), cobrindo tanto planilhas exportadas em
português (Excel BR) quanto em formato internacional. Ao carregar um
arquivo, a aplicação detecta as colunas automaticamente por alias (ver
`hydropump.io.mapping`); um diálogo de confirmação/correção aparece sempre
que a detecção for ambígua, e pode ser reaberto a qualquer momento em
**Arquivo → Editar mapeamento de colunas...**.

Colunas esperadas (fase de rebaixamento):
- Tempo: `tempo_min`, `tempo`, `t`, `time`, ...
- Rebaixamento: `rebaixamento_m`, `s`, `drawdown`, ...
- Nível dinâmico (alternativa ao rebaixamento direto): `nd_m`, `nd`, ...

Colunas esperadas (fase de recuperação, aba **🔁 Recuperação**):
- Tempo de recuperação: `tempo_rec_min`, `t_rec`, ...
- Nível dinâmico de recuperação ou rebaixamento residual: `nd_rec_m`, `s2_m`, ...

## Ressalvas hidrogeológicas importantes

- **Poço único**: `S` (armazenamento) é altamente sensível a perdas de
  carga na parede do poço (*skin effect*) e deve ser lido apenas como
  ordem de grandeza. A aplicação sinaliza isso automaticamente e, quando
  disponível, prefira `T'` da recuperação de Theis — imune ao skin effect
  (é exatamente para isso que existe a aba **Recuperação**).
- **Correção de Jacob**: confiável apenas para `s_max/b < 25%`. Acima
  disso, a aplicação emite um diagnóstico **crítico** — considere migrar
  para Neuman ou Boulton.
- **Armazenamento no poço (*wellbore storage*)**: nos primeiros instantes
  do ensaio, o rebaixamento medido reflete o esvaziamento da própria
  coluna d'água do poço, não a resposta do aquífero. A aplicação estima
  `t_wbs` (critério de Schafer, 1978) e alerta se o ajuste inclui pontos
  anteriores a esse instante.
- **Extrapolação**: a projeção teórica além dos dados medidos não é uma
  nova medição — é a continuação da mesma reta sob a premissa de vazão
  constante, sem recarga e sem limites físicos do aquífero.

## Licença

Proprietário — uso interno.
