# Análise de Ensaio de Bombeamento (Cooper-Jacob / Theis)

Pipeline completo para interpretação de ensaios de aquífero livre:
vazão volumétrica → correção de Jacob → Cooper-Jacob (rebaixamento) →
Theis (recuperação) → resumo no terminal → gráficos em `output/`.

## Estrutura

```
.
├── main.py                     # script principal (execute este arquivo)
├── requirements.txt
├── src/
│   ├── data_loader.py           # leitura/organização da caderneta de campo
│   ├── hydro_math.py            # rotinas matemáticas (Jacob, Cooper-Jacob, Theis)
│   └── plotting.py              # geração dos gráficos semi-log
├── data/
│   ├── rebaixamento_PZ01.csv    # dados de exemplo — fase de rebaixamento
│   └── recuperacao_PZ01.csv     # dados de exemplo — fase de recuperação
└── output/                      # gráficos gerados (criado automaticamente)
```

## Instalação (Manjaro Linux)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

O script pede os parâmetros do ensaio (nível estático, espessura saturada
`b`, distância `r`, e uma medição avulsa de balde/tempo). Pressione ENTER
em cada campo para aceitar os valores padrão do poço de exemplo PZ-01 —
o script também funciona de forma totalmente não-interativa (ex.: `python
main.py < /dev/null`), usando os padrões automaticamente.

Ao final, o resumo (vazão, T, S, T_rec, R², validade da aproximação de
Theis) é impresso no terminal, e dois gráficos PNG (300 DPI) são salvos
em `output/`:

- `cooper_jacob_PZ01.png`
- `recovery_theis_PZ01.png`

## Dados próprios

Para usar dados reais, substitua os CSVs em `data/` mantendo as colunas:

- Rebaixamento: `tempo_min`, `nd_m`, `vol_litros`, `tempo_balde_s`
- Recuperação: `tempo_rec_min`, `nd_rec_m`

e ajuste os parâmetros solicitados pelo script (ou os valores padrão em
`coletar_parametros()` dentro de `main.py`).
