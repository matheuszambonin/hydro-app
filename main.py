#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======

Script principal de interpretação de um ensaio de bombeamento (aquífero
livre), integrando os três módulos do pacote:

    src/data_loader.py  -> leitura e organização da caderneta de campo
    src/hydro_math.py   -> rotinas matemáticas (Jacob, Cooper-Jacob, Theis)
    src/plotting.py     -> geração dos gráficos semi-log de publicação

Fluxo executado
----------------
1. Solicita (ou usa valores padrão) os parâmetros de entrada do ensaio:
   espessura saturada (b), distância do poço de observação (r), e uma
   medição avulsa de vazão pelo método volumétrico (volume do balde e
   tempo de enchimento).
2. Carrega os arquivos de exemplo da pasta ``data/`` (rebaixamento e
   recuperação).
3. Processa a caderneta bruta (data_loader), aplica a correção de Jacob
   para aquífero livre (hydro_math) e ajusta as retas de Cooper-Jacob
   (rebaixamento) e de Theis (recuperação).
4. Exibe um resumo organizado dos resultados no terminal.
5. Gera e salva os dois gráficos de análise em ``output/``.

Uso
---
    $ python main.py

Testado em Manjaro Linux (Python 3.11+). Dependências: numpy, pandas,
scipy, matplotlib, seaborn (ver requirements.txt).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Callable, TypeVar

import numpy as np

# --------------------------------------------------------------------------- #
# Torna os módulos de src/ importáveis independentemente do diretório em que
# o script é chamado (ex.: `python main.py` a partir de outra pasta).
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import (  # noqa: E402
    TesteBombeamento,
    DadosPoçoError,
    carregar_csv_rebaixamento,
    carregar_csv_recuperacao,
)
from hydro_math import (  # noqa: E402
    HydroMathError,
    RegressionError,
    calculate_flow_rate,
    jacob_unconfined_correction,
    cooper_jacob_analysis,
    theis_recovery_analysis,
)
from plotting import plot_cooper_jacob, plot_recovery_theis  # noqa: E402


# --------------------------------------------------------------------------- #
# Utilidades de terminal (entrada de dados e formatação do resumo)
# --------------------------------------------------------------------------- #
T = TypeVar("T")

LARGURA = 70


def _linha(char: str = "-") -> None:
    print(char * LARGURA)


def _titulo(texto: str) -> None:
    _linha("=")
    print(texto.center(LARGURA))
    _linha("=")


def pedir_valor(mensagem: str, padrao: T, conversor: Callable[[str], T] = float) -> T:
    """
    Solicita um valor numérico ao usuário via terminal, com valor padrão.

    Pressionar ENTER aceita o valor padrão. Caso o script seja executado
    de forma não-interativa (sem stdin disponível, ex.: em um pipeline
    automatizado), o valor padrão é usado silenciosamente.
    """
    prompt = f"{mensagem} [padrão: {padrao}]: "
    try:
        bruto = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(f"(entrada não disponível — usando padrão: {padrao})")
        return padrao

    if not bruto:
        return padrao

    try:
        return conversor(bruto)
    except ValueError:
        print(f"Valor inválido; usando padrão: {padrao}")
        return padrao


# --------------------------------------------------------------------------- #
# Etapa 1 — Parâmetros de entrada do ensaio
# --------------------------------------------------------------------------- #
def coletar_parametros() -> dict:
    _titulo("PARÂMETROS DE ENTRADA DO ENSAIO")
    print(
        "Pressione ENTER em qualquer campo para aceitar o valor padrão "
        "(dados fictícios do poço de exemplo PZ-01).\n"
    )

    nivel_estatico = pedir_valor("Nível estático NE (m)", 8.50)
    espessura_saturada = pedir_valor("Espessura saturada do aquífero, b (m)", 22.0)
    distancia_r = pedir_valor("Distância ao poço de observação, r (m)", 30.0)
    altura_boca_poco = pedir_valor("Altura da boca do poço em relação ao terreno (m)", 0.60)

    print("\n-- Medição volumétrica avulsa de vazão (balde/proveta) --")
    volume_balde_l = pedir_valor("Volume coletado no balde (L)", 20.0)
    tempo_enchimento_s = pedir_valor("Tempo de enchimento do balde (s)", 0.95)

    return {
        "nivel_estatico": nivel_estatico,
        "espessura_saturada": espessura_saturada,
        "distancia_r": distancia_r,
        "altura_boca_poco": altura_boca_poco,
        "volume_balde_l": volume_balde_l,
        "tempo_enchimento_s": tempo_enchimento_s,
    }


# --------------------------------------------------------------------------- #
# Etapa 2 — Carregamento dos dados de exemplo
# --------------------------------------------------------------------------- #
def carregar_dados_exemplo() -> tuple:
    caminho_reb = os.path.join(DATA_DIR, "rebaixamento_PZ01.csv")
    caminho_rec = os.path.join(DATA_DIR, "recuperacao_PZ01.csv")

    print(f"\nCarregando dados de rebaixamento: {caminho_reb}")
    df_reb_bruto = carregar_csv_rebaixamento(caminho_reb)

    print(f"Carregando dados de recuperação:   {caminho_rec}")
    df_rec_bruto = carregar_csv_recuperacao(caminho_rec)

    return df_reb_bruto, df_rec_bruto


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        params = coletar_parametros()

        # ------------------------------------------------------------- #
        # Vazão de referência (medição avulsa, apenas para conferência)
        # ------------------------------------------------------------- #
        flow_ref = calculate_flow_rate(
            params["volume_balde_l"], params["tempo_enchimento_s"]
        )

        # ------------------------------------------------------------- #
        # Instancia o ensaio e carrega a caderneta bruta (data_loader.py)
        # ------------------------------------------------------------- #
        teste = TesteBombeamento(
            nivel_estatico=params["nivel_estatico"],
            altura_boca_poco=params["altura_boca_poco"],
            espessura_saturada=params["espessura_saturada"],
            distancia_r=params["distancia_r"],
        )

        df_reb_bruto, df_rec_bruto = carregar_dados_exemplo()

        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            df_reb = teste.processar_rebaixamento(df_reb_bruto)
            tempo_total_bombeamento_min = float(df_reb["tempo_min"].max())
            df_rec = teste.processar_recuperacao(
                df_rec_bruto, tempo_total_bombeamento=tempo_total_bombeamento_min
            )
        for aviso in avisos:
            print(f"[AVISO] {aviso.message}")

        # Vazão de ensaio adotada: média das medições volumétricas da
        # própria caderneta de rebaixamento (mais robusta que uma única
        # leitura avulsa).
        Q_m3s = teste.vazao_media_m3s

        # ------------------------------------------------------------- #
        # Etapa 3 — Correção de Jacob para aquífero livre (hydro_math.py)
        # ------------------------------------------------------------- #
        b = params["espessura_saturada"]
        r = params["distancia_r"]

        s_corrigido = jacob_unconfined_correction(df_reb["s_m"].to_numpy(), b)

        # Na recuperação, os primeiros instantes de t' costumam apresentar
        # rebaixamento residual ainda maior que a espessura crítica; usamos
        # strict=False para tolerar eventuais valores levemente negativos
        # (poço já praticamente recuperado) sem interromper a análise.
        s2_corrigido = jacob_unconfined_correction(
            df_rec["s2_m"].to_numpy(), b, strict=False
        )

        # ------------------------------------------------------------- #
        # Etapa 4 — Cooper-Jacob (rebaixamento) e Theis (recuperação)
        # ------------------------------------------------------------- #
        tempo_reb_sec = df_reb["tempo_min"].to_numpy() * 60.0

        # Descarta os primeiros minutos do ensaio (armazenamento no poço
        # ainda domina e a aproximação logarítmica de Theis não é válida).
        t_start_sec = 10.0 * 60.0

        cj = cooper_jacob_analysis(
            tempo_reb_sec,
            s_corrigido,
            Q=Q_m3s,
            r=r,
            t_start_sec=t_start_sec,
            saturated_thickness=b,
        )

        tempo_total_bombeamento_sec = tempo_total_bombeamento_min * 60.0
        tempo_rec_sec = df_rec["tempo_rec_min"].to_numpy() * 60.0

        rec = theis_recovery_analysis(
            tempo_total_bombeamento_sec,
            tempo_rec_sec,
            s2_corrigido,
            Q=Q_m3s,
            saturated_thickness=b,
        )

        # ------------------------------------------------------------- #
        # Etapa 5 — Resumo no terminal
        # ------------------------------------------------------------- #
        imprimir_resumo(params, flow_ref, teste, cj, rec)

        # ------------------------------------------------------------- #
        # Etapa 6 — Gráficos (plotting.py)
        # ------------------------------------------------------------- #
        gerar_graficos(df_reb, tempo_reb_sec, s_corrigido, cj, tempo_total_bombeamento_sec,
                        tempo_rec_sec, s2_corrigido, rec)

        print("\nProcessamento concluído com sucesso.")
        return 0

    except (DadosPoçoError, HydroMathError) as exc:
        _titulo("ERRO NO PROCESSAMENTO DOS DADOS")
        print(f"{type(exc).__name__}: {exc}")
        return 1
    except FileNotFoundError as exc:
        _titulo("ARQUIVO NÃO ENCONTRADO")
        print(str(exc))
        return 1


# --------------------------------------------------------------------------- #
# Resumo formatado no terminal
# --------------------------------------------------------------------------- #
def imprimir_resumo(params, flow_ref, teste, cj, rec) -> None:
    _titulo("RESUMO DO ENSAIO DE BOMBEAMENTO — POÇO PZ-01 (EXEMPLO)")

    print("\n[Parâmetros de entrada]")
    _linha()
    print(f"  Nível estático (NE) ............ {params['nivel_estatico']:.2f} m")
    print(f"  Espessura saturada (b) ......... {params['espessura_saturada']:.2f} m")
    print(f"  Distância r (obs./bombeado) .... {params['distancia_r']:.2f} m")

    print("\n[Vazão]")
    _linha()
    print(f"  Medição avulsa (balde) ......... {flow_ref.q_m3_h:.3f} m³/h "
          f"({flow_ref.q_l_s:.3f} L/s)")
    print(f"  Vazão média adotada (ensaio) ... {teste.vazao_media_m3h:.3f} m³/h "
          f"({teste.vazao_media_m3s:.6f} m³/s)")

    print("\n[Fase de Rebaixamento — Método de Cooper-Jacob]")
    _linha()
    print(f"  Δs' por ciclo log ........ {cj.delta_s:.4f} m/ciclo")
    print(f"  Transmissividade T ....... {cj.transmissivity_m2_day:.3e} m²/dia "
          f"({cj.transmissivity_m2_s:.3e} m²/s)")
    print(f"  Armazenamento S .......... {cj.storativity:.3e}")
    if cj.conductivity_m_day is not None:
        print(f"  Condutividade K .......... {cj.conductivity_m_day:.3e} m/dia")
    print(f"  Coef. de determinação R² . {cj.r_squared:.4f}  ({cj.n_points} pontos)")
    print(f"  u_max .................... {cj.u_max:.3e} "
          f"({'aproximação válida' if cj.approximation_is_valid else 'ATENÇÃO: aproximação pode não ser válida'})")

    print("\n[Fase de Recuperação — Método de Theis]")
    _linha()
    print(f"  Δs'' por ciclo log ....... {rec.delta_s_residual:.4f} m/ciclo")
    print(f"  Transmissividade T_rec ... {rec.transmissivity_m2_day:.3e} m²/dia "
          f"({rec.transmissivity_m2_s:.3e} m²/s)")
    if rec.conductivity_m_day is not None:
        print(f"  Condutividade K_rec ...... {rec.conductivity_m_day:.3e} m/dia")
    print(f"  (t/t') extrapolado (S/S') . {rec.ratio_at_zero:.3f}")
    print(f"  Coef. de determinação R² . {rec.r_squared:.4f}  ({rec.n_points} pontos)")

    print("\n[Comparação entre as fases]")
    _linha()
    razao_T = rec.transmissivity_m2_day / cj.transmissivity_m2_day
    print(f"  T_recuperação / T_rebaixamento = {razao_T:.2f}")
    _linha("=")


# --------------------------------------------------------------------------- #
# Geração e salvamento dos gráficos
# --------------------------------------------------------------------------- #
def gerar_graficos(df_reb, tempo_reb_sec, s_corrigido, cj, tp_sec, tempo_rec_sec,
                    s2_corrigido, rec) -> None:
    # --- Gráfico 1: Cooper-Jacob (rebaixamento) ---
    fit_line_cj = cj.intercept + cj.delta_s * np.log10(tempo_reb_sec)

    caminho_cj = os.path.join(OUTPUT_DIR, "cooper_jacob_PZ01.png")
    plot_cooper_jacob(
        time_min=df_reb["tempo_min"].to_numpy(),
        drawdown_corr=s_corrigido,
        fit_line=fit_line_cj,
        T=cj.transmissivity_m2_day,
        S=cj.storativity,
        title="Ensaio de Bombeamento — Poço PZ-01 (Cooper-Jacob)",
        save_path=caminho_cj,
    )
    print(f"\nGráfico salvo: {caminho_cj}")

    # --- Gráfico 2: Recuperação de Theis ---
    razao_t_tl = (tp_sec + tempo_rec_sec) / tempo_rec_sec
    fit_line_rec = rec.intercept + rec.delta_s_residual * np.log10(razao_t_tl)

    caminho_rec = os.path.join(OUTPUT_DIR, "recovery_theis_PZ01.png")
    plot_recovery_theis(
        t_ratio=razao_t_tl,
        residual_drawdown=s2_corrigido,
        fit_line=fit_line_rec,
        T_rec=rec.transmissivity_m2_day,
        title="Ensaio de Recuperação — Poço PZ-01 (Método de Theis)",
        save_path=caminho_rec,
    )
    print(f"Gráfico salvo: {caminho_rec}")


if __name__ == "__main__":
    sys.exit(main())
