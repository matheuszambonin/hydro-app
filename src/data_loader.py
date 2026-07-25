"""
data_loader.py
==============

Módulo para carga e processamento de dados brutos de caderneta de campo
de testes de bombeamento (ensaios de aquífero), contemplando as fases de
REBAIXAMENTO (drawdown) e RECUPERAÇÃO (recovery).

Convenções hidrogeológicas adotadas
------------------------------------
- Todas as profundidades/níveis (NE, ND) são medidas a partir de uma
  referência fixa na boca do poço (ex.: topo do revestimento), com
  valores crescentes para baixo (quanto maior o valor, mais profundo
  está o nível d'água).
- NE (nível estático, m): nível d'água antes do início do bombeamento,
  em equilíbrio hidrostático com o aquífero.
- ND (nível dinâmico, m): nível d'água medido durante o bombeamento,
  em um instante t.
- s (rebaixamento, m): s = ND - NE. Fisicamente, durante o bombeamento,
  espera-se s >= 0 (o nível dinâmico só pode ser igual ou mais profundo
  que o estático).
- s'' (rebaixamento residual, m): mesma lógica aplicada à fase de
  recuperação, usando o nível medido após o desligamento da bomba.
- t' (tempo de recuperação, min): tempo decorrido desde o desligamento
  da bomba.
- t (tempo total acumulado, min): t = t_total_bombeamento + t'. Usado
  no método de Theis de recuperação, através da razão t/t'.
- Q (vazão, m³/s ou m³/h): calculada pelo método volumétrico
  (balde/proveta), a partir do volume coletado e do tempo de coleta.

Requisitos: pandas, numpy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Exceções customizadas
# --------------------------------------------------------------------------- #
class DadosPoçoError(Exception):
    """Erro genérico relacionado à consistência dos dados do poço/ensaio."""


class ColunasAusentesError(DadosPoçoError):
    """Levantada quando colunas obrigatórias não estão presentes no DataFrame."""


# --------------------------------------------------------------------------- #
# Classe principal
# --------------------------------------------------------------------------- #
@dataclass
class TesteBombeamento:
    """
    Encapsula os parâmetros hidrogeológicos de um poço/ensaio de
    bombeamento e fornece métodos para processar as fases de
    rebaixamento e recuperação a partir de dados brutos de campo.

    Parâmetros
    ----------
    nivel_estatico : float
        Nível estático (NE), em metros, medido a partir da boca do poço.
    altura_boca_poco : float
        Altura da boca (brocal) do poço em relação ao terreno natural, em
        metros. Utilizada para eventuais correções de cota/referência.
    espessura_saturada : float
        Espessura saturada do aquífero livre (b), em metros. Parâmetro
        de entrada para fórmulas de aquífero livre (ex.: correção de
        Jacob-Cooper para rebaixamentos elevados).
    distancia_r : float
        Distância radial (r) entre o poço de bombeamento e o ponto de
        observação (poço de observação ou o próprio poço bombeado, r ~
        raio efetivo do poço), em metros.

    Atributos calculados
    ---------------------
    vazao_media_m3s : Optional[float]
        Vazão média do ensaio (m³/s), calculada após `processar_rebaixamento`.
    vazao_media_m3h : Optional[float]
        Vazão média do ensaio (m³/h), calculada após `processar_rebaixamento`.
    """

    nivel_estatico: float
    altura_boca_poco: float
    espessura_saturada: float
    distancia_r: float

    vazao_media_m3s: Optional[float] = field(default=None, init=False)
    vazao_media_m3h: Optional[float] = field(default=None, init=False)

    df_rebaixamento: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    df_recuperacao: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------------- #
    # Validações de entrada
    # ----------------------------------------------------------------- #
    def __post_init__(self) -> None:
        if self.nivel_estatico < 0:
            raise DadosPoçoError(
                f"Nível estático (NE) inválido: {self.nivel_estatico} m. "
                "Deve ser um valor não negativo (medido a partir da boca do poço)."
            )
        if self.espessura_saturada <= 0:
            raise DadosPoçoError(
                f"Espessura saturada (b) inválida: {self.espessura_saturada} m. "
                "Deve ser estritamente positiva para um aquífero livre."
            )
        if self.distancia_r <= 0:
            raise DadosPoçoError(
                f"Distância ao ponto de observação (r) inválida: {self.distancia_r} m. "
                "Deve ser estritamente positiva."
            )

    # ----------------------------------------------------------------- #
    # Fase de REBAIXAMENTO
    # ----------------------------------------------------------------- #
    def processar_rebaixamento(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa os dados brutos da fase de rebaixamento.

        Parâmetros
        ----------
        df : pd.DataFrame
            Deve conter as colunas:
            - 'tempo_min'     : tempo decorrido desde o início do bombeamento (min)
            - 'nd_m'          : nível dinâmico medido (m)
            - 'vol_litros'    : volume coletado no ensaio volumétrico (L)
            - 'tempo_balde_s' : tempo de coleta do volume (s)

        Retorna
        -------
        pd.DataFrame
            Cópia do DataFrame de entrada acrescida das colunas:
            - 's_m'        : rebaixamento bruto, s = nd_m - NE (m)
            - 'Qi_m3s'     : vazão instantânea (m³/s)
            - 'Qi_m3h'     : vazão instantânea (m³/h)

        Efeitos colaterais
        ------------------
        Armazena em `self.vazao_media_m3s` / `self.vazao_media_m3h` a vazão
        média do ensaio (média aritmética das vazões instantâneas válidas),
        e em `self.df_rebaixamento` o DataFrame processado.
        """
        colunas_obrigatorias = {"tempo_min", "nd_m", "vol_litros", "tempo_balde_s"}
        self._validar_colunas(df, colunas_obrigatorias, fase="rebaixamento")

        dados = df.copy()

        # --- Validações de consistência física ---
        if (dados["tempo_balde_s"] <= 0).any():
            raise DadosPoçoError(
                "Foram encontrados valores de 'tempo_balde_s' <= 0. "
                "O tempo de coleta do volume deve ser estritamente positivo "
                "(divisão por zero na vazão)."
            )
        if (dados["vol_litros"] < 0).any():
            raise DadosPoçoError("Foram encontrados valores negativos em 'vol_litros'.")

        # Aviso: nível dinâmico menor que o estático não é fisicamente
        # esperado durante o bombeamento (o poço deveria rebaixar, não subir
        # acima do nível de repouso).
        nd_menor_que_ne = dados["nd_m"] < self.nivel_estatico
        if nd_menor_que_ne.any():
            linhas = dados.index[nd_menor_que_ne].tolist()
            warnings.warn(
                f"Atenção: nível dinâmico (nd_m) menor que o nível estático "
                f"(NE = {self.nivel_estatico} m) nas linhas {linhas}. Isso é "
                "fisicamente inconsistente para a fase de rebaixamento "
                "(esperado ND >= NE). Verifique erro de leitura/digitação.",
                stacklevel=2,
            )

        # s = ND - NE  →  rebaixamento bruto (m)
        dados["s_m"] = dados["nd_m"] - self.nivel_estatico

        # Conversão de vazão: L -> m³ (1 m³ = 1000 L) e s -> h (1 h = 3600 s)
        vol_m3 = dados["vol_litros"] / 1000.0
        dados["Qi_m3s"] = vol_m3 / dados["tempo_balde_s"]
        dados["Qi_m3h"] = dados["Qi_m3s"] * 3600.0

        # Vazão média do ensaio (ignora eventuais NaN gerados por linhas inválidas)
        self.vazao_media_m3s = float(np.nanmean(dados["Qi_m3s"]))
        self.vazao_media_m3h = float(np.nanmean(dados["Qi_m3h"]))

        self.df_rebaixamento = dados
        return dados

    # ----------------------------------------------------------------- #
    # Fase de RECUPERAÇÃO
    # ----------------------------------------------------------------- #
    def processar_recuperacao(
        self, df: pd.DataFrame, tempo_total_bombeamento: float
    ) -> pd.DataFrame:
        """
        Processa os dados brutos da fase de recuperação (método de Theis
        da recuperação residual).

        Parâmetros
        ----------
        df : pd.DataFrame
            Deve conter as colunas:
            - 'tempo_rec_min' : tempo decorrido desde o desligamento da
                                bomba, t' (min)
            - 'nd_rec_m'      : nível dinâmico medido durante a recuperação (m)
        tempo_total_bombeamento : float
            Tempo total (min) em que a bomba permaneceu ligada antes do
            desligamento (t_total_bomb).

        Retorna
        -------
        pd.DataFrame
            Cópia do DataFrame de entrada acrescida das colunas:
            - 't_total_min' : tempo total acumulado desde o início do
                              bombeamento, t = t_total_bomb + t' (min)
            - 'razao_t_tl'  : razão t/t' utilizada no método de Theis
                              (gráfico s'' x log(t/t'))
            - 's2_m'        : rebaixamento residual, s'' = nd_rec_m - NE (m)
        """
        colunas_obrigatorias = {"tempo_rec_min", "nd_rec_m"}
        self._validar_colunas(df, colunas_obrigatorias, fase="recuperação")

        if tempo_total_bombeamento <= 0:
            raise DadosPoçoError(
                f"Tempo total de bombeamento inválido: {tempo_total_bombeamento} min. "
                "Deve ser estritamente positivo."
            )

        dados = df.copy()

        if (dados["tempo_rec_min"] <= 0).any():
            raise DadosPoçoError(
                "Foram encontrados valores de 'tempo_rec_min' (t') <= 0. "
                "O tempo de recuperação deve ser estritamente positivo "
                "(t=0 geraria divisão por zero na razão t/t')."
            )

        # Aviso: nível de recuperação abaixo do estático (ainda em rebaixamento
        # residual acentuado) é esperado no início da recuperação, mas um
        # nível MENOR que o estático ao longo de toda a série pode indicar
        # erro de leitura ou NE mal calibrado.
        nd_menor_que_ne = dados["nd_rec_m"] < self.nivel_estatico
        if nd_menor_que_ne.any():
            linhas = dados.index[nd_menor_que_ne].tolist()
            warnings.warn(
                f"Atenção: nível dinâmico de recuperação (nd_rec_m) menor que "
                f"o nível estático (NE = {self.nivel_estatico} m) nas linhas "
                f"{linhas}. Verifique se o poço já recuperou completamente ou "
                "se há erro de leitura/digitação.",
                stacklevel=2,
            )

        # t = t_total_bomb + t'  (tempo total acumulado desde o início do bombeamento)
        dados["t_total_min"] = tempo_total_bombeamento + dados["tempo_rec_min"]

        # Razão t/t', utilizada no eixo semi-log do método de Theis (recuperação)
        dados["razao_t_tl"] = dados["t_total_min"] / dados["tempo_rec_min"]

        # s'' = ND_rec - NE  →  rebaixamento residual (m)
        dados["s2_m"] = dados["nd_rec_m"] - self.nivel_estatico

        self.df_recuperacao = dados
        return dados

    # ----------------------------------------------------------------- #
    # Utilitários internos
    # ----------------------------------------------------------------- #
    @staticmethod
    def _validar_colunas(df: pd.DataFrame, obrigatorias: set, fase: str) -> None:
        """Garante que todas as colunas obrigatórias estão presentes no DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado um pandas.DataFrame para a fase de {fase}.")

        faltantes = obrigatorias - set(df.columns)
        if faltantes:
            raise ColunasAusentesError(
                f"Colunas obrigatórias ausentes para a fase de {fase}: "
                f"{sorted(faltantes)}. Colunas esperadas: {sorted(obrigatorias)}."
            )

        if df.empty:
            raise DadosPoçoError(f"DataFrame de {fase} está vazio.")


# --------------------------------------------------------------------------- #
# Funções de conveniência (API funcional, para quem preferir não usar a classe)
# --------------------------------------------------------------------------- #
def carregar_csv_rebaixamento(caminho_csv: str, **kwargs) -> pd.DataFrame:
    """Lê um CSV bruto de campo (fase de rebaixamento) e retorna um DataFrame.

    Parâmetros adicionais (`**kwargs`) são repassados para `pandas.read_csv`
    (ex.: `sep=';'`, `decimal=','`, comum em cadernetas de campo brasileiras).
    """
    try:
        return pd.read_csv(caminho_csv, **kwargs)
    except FileNotFoundError as exc:
        raise DadosPoçoError(f"Arquivo não encontrado: {caminho_csv}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DadosPoçoError(f"Arquivo CSV vazio: {caminho_csv}") from exc


def carregar_csv_recuperacao(caminho_csv: str, **kwargs) -> pd.DataFrame:
    """Lê um CSV bruto de campo (fase de recuperação) e retorna um DataFrame."""
    try:
        return pd.read_csv(caminho_csv, **kwargs)
    except FileNotFoundError as exc:
        raise DadosPoçoError(f"Arquivo não encontrado: {caminho_csv}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DadosPoçoError(f"Arquivo CSV vazio: {caminho_csv}") from exc


# --------------------------------------------------------------------------- #
# Exemplo de uso (executado apenas se o módulo for rodado diretamente)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Dados fictícios de caderneta de campo — apenas para demonstração.
    df_rebaixamento_bruto = pd.DataFrame(
        {
            "tempo_min": [1, 5, 10, 30, 60, 120],
            "nd_m": [12.10, 12.45, 12.60, 12.90, 13.05, 13.10],
            "vol_litros": [18.0, 17.5, 17.2, 16.9, 16.8, 16.8],
            "tempo_balde_s": [10.2, 10.5, 10.6, 10.8, 10.9, 10.9],
        }
    )

    df_recuperacao_bruto = pd.DataFrame(
        {
            "tempo_rec_min": [1, 5, 10, 30, 60],
            "nd_rec_m": [12.95, 12.60, 12.40, 12.15, 12.02],
        }
    )

    teste = TesteBombeamento(
        nivel_estatico=12.00,
        altura_boca_poco=0.60,
        espessura_saturada=25.0,
        distancia_r=0.15,
    )

    reb = teste.processar_rebaixamento(df_rebaixamento_bruto)
    print("=== Fase de Rebaixamento ===")
    print(reb)
    print(f"\nVazão média: {teste.vazao_media_m3s:.6f} m³/s "
          f"({teste.vazao_media_m3h:.3f} m³/h)\n")

    rec = teste.processar_recuperacao(
        df_recuperacao_bruto, tempo_total_bombeamento=120
    )
    print("=== Fase de Recuperação ===")
    print(rec)
