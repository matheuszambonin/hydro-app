"""
app.py
======

Interface gráfica (Streamlit) para o software de interpretação de ensaios
de bombeamento em aquíferos livres (Cooper-Jacob).

Combina:
    - hydro_math.py : rotinas matemáticas puras (vazão, correção de Jacob,
      análise de Cooper-Jacob, projeção/extrapolação de rebaixamento).
    - plotting.py    : gráfico científico integrado (medições reais +
      pontos sintéticos + projeção teórica).

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import unicodedata
from pathlib import Path
from typing import Optional
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# hydro_math.py e plotting.py vivem em src/, ao lado deste arquivo.
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydro_math as hm
import plotting as pl
import schematics as sq

# --------------------------------------------------------------------------
# Configuração da página e estilo dos cards
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Análise de Ensaio de Bombeamento",
    page_icon="💧",
    layout="wide",
)

CARD_CSS = """
<style>
.metric-card {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef2f7 100%);
    border: 1px solid #d7dee6;
    border-left: 5px solid #1f5fa8;
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    margin-bottom: 6px;
    min-height: 108px;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #5a6472;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0d2b4d;
    line-height: 1.15;
}
.metric-card .sub {
    font-size: 0.76rem;
    color: #7f8c9a;
    margin-top: 3px;
}
.metric-card.alt { border-left-color: #c0392b; }
.metric-card.alt2 { border-left-color: #2e8b57; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", variant: str = "") -> str:
    """Monta o HTML de um card de métrica estilizado."""
    cls = f"metric-card {variant}".strip()
    return (
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


# --------------------------------------------------------------------------
# Helpers de leitura / detecção automática de colunas
# --------------------------------------------------------------------------
def _normaliza(txt: object) -> str:
    bruto = str(txt).strip().lower()
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", bruto) if not unicodedata.combining(c)
    )
    return sem_acento.replace(" ", "").replace("_", "").replace("-", "")


def _detect_column(columns, aliases) -> Optional[str]:
    norm_map = {_normaliza(c): c for c in columns}
    for alias in aliases:
        if alias in norm_map:
            return norm_map[alias]
    return None


TIME_ALIASES = ("tempomin", "tempo", "tmin", "timemin", "t", "time")
ND_ALIASES = ("ndm", "niveldinamico", "nd", "nivel")
DRAWDOWN_ALIASES = ("rebaixamentom", "rebaixamento", "s", "sm", "drawdown", "drawdownm")


def read_uploaded_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    # Leitura do CSV com sniffing de delimitador brasileiro
    content = file.read()
    file.seek(0)
    
    text_sample = content.decode("utf-8", errors="ignore")[:4096] if isinstance(content, bytes) else content[:4096]

    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=";,|\t")
        sep = dialect.delimiter
    except Exception:
        sep = ";" if ";" in text_sample else ","

    try:
        df = pd.read_csv(file, sep=sep, decimal=",")
        # Se nenhuma coluna numérica for detectada, tenta com decimal de ponto "."
        if len(df.select_dtypes(include="number").columns) == 0:
            file.seek(0)
            df = pd.read_csv(file, sep=sep, decimal=".")
        return df
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=",", decimal=".")


# --------------------------------------------------------------------------
# Sidebar — upload e parâmetros de entrada
# --------------------------------------------------------------------------
st.sidebar.title("💧 Parâmetros do Ensaio")

st.sidebar.subheader("📁 Cadernetas de Campo")
uploaded_file = st.sidebar.file_uploader(
    "1. Rebaixamento (Obrigatório)", type=["csv", "xlsx", "xls"]
)
uploaded_file_rec = st.sidebar.file_uploader(
    "2. Recuperação (Opcional)", type=["csv", "xlsx", "xls"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Geometria e Nível")
NE = st.sidebar.number_input(
    "Nível Estático, NE (m)", min_value=0.0, value=10.0, step=0.1, format="%.2f",
    help="Profundidade do nível d'água em repouso, antes do bombeamento.",
)

aplicar_correcao_jacob = st.sidebar.checkbox(
    "Aplicar Correção de Jacob para Aquífero Livre",
    value=True,
    help=(
        "Corrige o rebaixamento medido (s → s') para compensar a redução "
        "da própria espessura saturada em aquíferos livres (Jacob, 1944). "
        "Poços parcialmente penetrantes ou aquíferos confinados/semi-"
        "confinados geralmente dispensam essa correção."
    ),
)

if aplicar_correcao_jacob:
    b = st.sidebar.number_input(
        "Espessura Saturada, b (m)", min_value=0.01, value=20.0, step=0.5, format="%.2f",
        help=(
            "Espessura saturada inicial do aquífero (base do aquífero ao "
            "nível estático). Caso não saiba a espessura total do aquífero, "
            "utilize a profundidade da coluna saturada perfurada no poço."
        ),
    )
else:
    b = None

tipo_configuracao = st.sidebar.radio(
    "Tipo de Configuração do Teste",
    [
        "Poço Único (Medição no poço bombeado)",
        "Poço de Observação (Medição em piezômetro vizinho)",
    ],
    help=(
        "Define se o rebaixamento foi medido no próprio poço bombeado "
        "(Poço Único) ou em um piezômetro/poço de observação vizinho."
    ),
)
eh_poco_unico = tipo_configuracao.startswith("Poço Único")

if eh_poco_unico:
    r = st.sidebar.number_input(
        "Raio do Poço, r_w (m)", min_value=0.001, value=0.076, step=0.001, format="%.3f",
        help="Informe o raio interno do poço ou do filtro.",
    )
else:
    r = st.sidebar.number_input(
        "Distância ao Poço de Observação, r (m)", min_value=0.01, value=10.0, step=0.5, format="%.2f",
        help="Distância radial do poço de observação ao poço bombeado.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Vazão (Método Volumétrico)")
volume_balde = st.sidebar.number_input(
    "Volume do Balde (L)", min_value=0.01, value=20.0, step=1.0, format="%.2f"
)
tempo_enchimento = st.sidebar.number_input(
    "Tempo de Enchimento (s)", min_value=0.01, value=12.0, step=0.5, format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Controles de Projeção")
show_projection = st.sidebar.checkbox("Exibir Projeção Teórica no Gráfico", value=True)
target_hours = st.sidebar.number_input(
    "Tempo Limite da Projeção (h)", min_value=0.5, value=12.0, step=0.5,
    help="Horizonte final da extrapolação de Cooper-Jacob (ex.: 12h, 24h).",
)

# --------------------------------------------------------------------------
# Mapeamento de colunas do arquivo enviado
# --------------------------------------------------------------------------
df_raw = None
col_t_sel = col_s_sel = col_nd_sel = None
time_unit = "minutos"
usa_nd = False

if uploaded_file is not None:
    try:
        df_raw = read_uploaded_file(uploaded_file)
    except Exception as exc:  # arquivo malformado / extensão inesperada
        st.sidebar.error(f"Não foi possível ler o arquivo: {exc}")
        df_raw = None

    if df_raw is not None and len(df_raw.columns) > 0:
        with st.sidebar.expander("Mapeamento de Colunas", expanded=False):
            time_unit = st.radio(
                "Unidade do tempo na planilha", ["minutos", "segundos"], horizontal=True
            )
            detected_t = _detect_column(df_raw.columns, TIME_ALIASES)
            detected_nd = _detect_column(df_raw.columns, ND_ALIASES)
            detected_s = _detect_column(df_raw.columns, DRAWDOWN_ALIASES)

            cols_list = list(df_raw.columns)
            col_t_sel = st.selectbox(
                "Coluna de tempo", cols_list,
                index=cols_list.index(detected_t) if detected_t else 0,
            )
            usa_nd = st.checkbox(
                "Arquivo traz Nível Dinâmico (ND) em vez de Rebaixamento direto",
                value=detected_nd is not None,
            )
            if usa_nd:
                col_nd_sel = st.selectbox(
                    "Coluna de nível dinâmico (ND, m)", cols_list,
                    index=cols_list.index(detected_nd) if detected_nd else 0,
                )
            else:
                col_s_sel = st.selectbox(
                    "Coluna de rebaixamento (m)", cols_list,
                    index=cols_list.index(detected_s) if detected_s else 0,
                )

# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("💧 Análise de Ensaio de Bombeamento")
st.caption(
    "Interpretação de aquífero livre pelo método de Cooper-Jacob — "
    "cálculo de Q, T e S, com projeção teórica de rebaixamento."
)

if df_raw is None:
    st.info(
        "⬅️ Envie o arquivo de dados de campo (CSV ou Excel) na barra lateral "
        "para iniciar a análise."
    )
    st.stop()

# --------------------------------------------------------------------------
# Processamento — vazão, correção de Jacob, Cooper-Jacob, projeções
# --------------------------------------------------------------------------
try:
    df_campo = df_raw.copy()
    tempo_bruto = pd.to_numeric(df_campo[col_t_sel], errors="coerce").to_numpy(dtype=float)
    tempo_min_full = tempo_bruto / 60.0 if time_unit == "segundos" else tempo_bruto

    if usa_nd:
        nd_full = pd.to_numeric(df_campo[col_nd_sel], errors="coerce").to_numpy(dtype=float)
        rebaixamento_full = nd_full - NE
    else:
        rebaixamento_full = pd.to_numeric(df_campo[col_s_sel], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(tempo_min_full) & np.isfinite(rebaixamento_full) & (tempo_min_full > 0)
    tempo_min = tempo_min_full[valid]
    rebaixamento = rebaixamento_full[valid]

    if tempo_min.size < 3:
        st.error(
            "São necessários ao menos 3 pontos válidos (tempo > 0 e rebaixamento "
            "numérico) para a análise de Cooper-Jacob. Verifique o mapeamento "
            "de colunas na barra lateral."
        )
        st.stop()

    ordem = np.argsort(tempo_min)
    tempo_min = tempo_min[ordem]
    rebaixamento = rebaixamento[ordem]
    tempo_sec = tempo_min * 60.0

    # --- Vazão pelo método volumétrico ---
    q_result = hm.calculate_flow_rate(volume_balde, tempo_enchimento)
    Q_m3_s = q_result.mean_q_m3_s
    Q_m3_h = float(q_result.q_m3_h)
    Q_l_s = float(np.mean(q_result.q_l_s))

    # --- Correção de Jacob (aquífero livre) ---
    # Se o checkbox estiver desmarcado (ou b não informado), s' = s.
    drawdown_corr = hm.jacob_unconfined_correction(
        rebaixamento, b, strict=False, apply_correction=aplicar_correcao_jacob,
    )

    # --- Cooper-Jacob: ajuste da reta e obtenção de T e S ---
    cj = hm.cooper_jacob_analysis(
        tempo_sec, drawdown_corr, Q_m3_s, r, saturated_thickness=b,
    )

    ultimo_tempo_min = float(tempo_min[-1])
    rebaixamento_max_real = float(np.max(rebaixamento))

    # --- Diagnóstico: relevância da correção de Jacob (s_max / b) ---
    pct_s_max_sobre_b: Optional[float] = None
    if aplicar_correcao_jacob and b:
        pct_s_max_sobre_b = (rebaixamento_max_real / b) * 100.0

    # --- Rebaixamento previsto em 12h — métrica fixa do painel de resumo ---
    if ultimo_tempo_min < 720.0:
        df_proj_12h = hm.predict_extrapolated_drawdown(
            delta_s=cj.delta_s, t0_sec=cj.t0_sec,
            time_start_sec=ultimo_tempo_min * 60.0,
            target_hours=12.0, saturated_thickness=b,
        )
        s_12h: Optional[float] = float(df_proj_12h["rebaixamento_projetado_m"].iloc[-1])
    else:
        # 12h já está dentro do período efetivamente medido: interpola.
        s_12h = float(np.interp(720.0, tempo_min, rebaixamento))

    # --- Projeção teórica para o gráfico, no horizonte definido pelo usuário ---
    df_projection = None
    df_sintetico = None
    horizonte_ja_coberto = target_hours * 60.0 <= ultimo_tempo_min
    if show_projection and not horizonte_ja_coberto:
        df_projection = hm.predict_extrapolated_drawdown(
            delta_s=cj.delta_s, t0_sec=cj.t0_sec,
            time_start_sec=ultimo_tempo_min * 60.0,
            target_hours=target_hours, saturated_thickness=b,
        )
        df_sintetico = hm.generate_synthetic_field_data(
            delta_s=cj.delta_s, t0_sec=cj.t0_sec, Q_mean=Q_m3_s,
            last_time_min=ultimo_tempo_min, target_hours=target_hours,
            static_level_m=NE, saturated_thickness=b,
        )

## --- Análise da Fase de Recuperação (se o 2º arquivo foi enviado) ---
    res_rec = None
    t_ratio_plot = None
    s2_plot = None
    fit_line_rec = None

    if uploaded_file_rec is not None:
        try:
            df_rec_raw = read_uploaded_file(uploaded_file_rec)
            cols_rec = list(df_rec_raw.columns)
            
            col_t_rec = _detect_column(cols_rec, ("temporecmin", "temporec", "trec", "time", "tempo")) or cols_rec[0]
            col_s_rec = _detect_column(cols_rec, ("ndrecm", "s2m", "rebaixamentoresidual", "s2", "nd", "rebaixamento")) or cols_rec[1]
            
            t_rec = pd.to_numeric(df_rec_raw[col_t_rec], errors="coerce").to_numpy(dtype=float)
            s_rec_raw = pd.to_numeric(df_rec_raw[col_s_rec], errors="coerce").to_numpy(dtype=float)
            
            s2_m = s_rec_raw - NE if np.nanmean(s_rec_raw) > NE else s_rec_raw
            valid_rec = np.isfinite(t_rec) & np.isfinite(s2_m) & (t_rec > 0)
            
            if np.count_nonzero(valid_rec) >= 3:
                t_rec_valid = t_rec[valid_rec]
                s2_valid = s2_m[valid_rec]
                
                res_rec = hm.theis_recovery_analysis(
                    time_pump_sec=ultimo_tempo_min * 60.0,
                    time_recovery_sec=t_rec_valid * 60.0,
                    residual_drawdown=s2_valid,
                    Q=Q_m3_s,
                    saturated_thickness=b,
                )
                
                # Prepara os dados para o gráfico de Theis (s'' vs t/t')
                t_ratio_plot = (ultimo_tempo_min + t_rec_valid) / t_rec_valid
                fit_line_rec = res_rec.delta_s_residual * np.log10(t_ratio_plot) + res_rec.intercept
                s2_plot = s2_valid

        except Exception as exc_rec:
            st.sidebar.warning(f"Não foi possível processar a recuperação: {exc_rec}")

except hm.HydroMathError as exc:
    st.error(f"Erro na análise hidrogeológica: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 — proteção contra erros de dados inesperados
    st.error(f"Erro inesperado ao processar os dados: {exc}")
    st.stop()

# --------------------------------------------------------------------------
# Dataset unificado (medido + sintético) para gráfico e tabela
# --------------------------------------------------------------------------
df_medido = pd.DataFrame({
    "tempo_min": tempo_min,
    "nd_m": NE + rebaixamento,
    "rebaixamento_m": rebaixamento,
    "vazao_m3_h": Q_m3_h,
    "tipo_dado": "Medido",
})

if df_sintetico is not None and not df_sintetico.empty:
    df_unificado = pd.concat(
        [df_medido, df_sintetico[["tempo_min", "nd_m", "rebaixamento_m", "vazao_m3_h", "tipo_dado"]]],
        ignore_index=True,
    )
else:
    df_unificado = df_medido.copy()
df_unificado["tipo_dado"] = "Medido"

# --------------------------------------------------------------------------
# Abas do painel principal
# --------------------------------------------------------------------------
tab_graficos, tab_tabela = st.tabs(["📊 Visão Geral e Gráficos", "📋 Tabela de Dados"])

with tab_graficos:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        metric_card("Vazão, Q", f"{Q_m3_h:.2f} m³/h", f"{Q_l_s:.2f} L/s"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        metric_card("T (Rebaixamento)", f"{cj.transmissivity_m2_day:.2f} m²/dia",
                    f"R² = {cj.r_squared:.4f}", "alt"),
        unsafe_allow_html=True,
    )
    
    # Exibe T_rec se a fase de recuperação tiver sido carregada
    t_rec_str = f"{res_rec.transmissivity_m2_day:.2f} m²/dia" if res_rec else "Não carregado"
    sub_rec_str = f"R² = {res_rec.r_squared:.4f}" if res_rec else "recuperação Theis"
    c3.markdown(
        metric_card("T (Recuperação)", t_rec_str, sub_rec_str, "alt2"),
        unsafe_allow_html=True,
    )
    
    c4.markdown(
        metric_card("Armazenamento, S", f"{cj.storativity:.2e}", "adimensional"),
        unsafe_allow_html=True,
    )
    c5.markdown(
        metric_card("Rebaix. Previsto (12h)",
                    f"{s_12h:.2f} m" if s_12h is not None else "—",
                    "extrapolação Cooper-Jacob", "alt"),
        unsafe_allow_html=True,
    )

    with st.expander("📐 Ver Esquema do Poço", expanded=False):
        fig_esquema = sq.draw_well_schematic(
            type="single" if eh_poco_unico else "observation",
            NE=NE,
            ND=NE + rebaixamento_max_real,
            b=b if b is not None else 20.0,
            r=r,
            show_b=b is not None,
        )
        st.pyplot(fig_esquema, use_container_width=True)
        plt.close(fig_esquema)

    if eh_poco_unico:
        st.warning(
            "Aviso: Em testes de poço único, o Coeficiente de Armazenamento "
            "($S$) é altamente sensível a perdas de carga na parede do poço "
            "(skin effect) e deve ser interpretado apenas como ordem de "
            "grandeza. A Transmissividade ($T$) permanece válida."
        )

    if not cj.approximation_is_valid:
        st.warning(
            f"⚠️ A aproximação logarítmica de Cooper-Jacob pode não ser válida "
            f"neste trecho (u_max = {cj.u_max:.4f} ≥ 0,01). Considere descartar "
            "os tempos iniciais do ensaio no ajuste."
        )
    if show_projection and horizonte_ja_coberto:
        st.info(
            "ℹ️ O tempo limite da projeção já está coberto pelos dados medidos "
            "— nenhuma projeção adicional foi desenhada."
        )
    if pct_s_max_sobre_b is not None:
        if pct_s_max_sobre_b > 25.0:
            st.warning(
                f"⚠️ **Aviso de Validade Física (Jacob):** O rebaixamento máximo ({rebaixamento_max_real:.2f} m) "
                f"corresponde a **{pct_s_max_sobre_b:.1f}%** da espessura saturada ({b:.2f} m).\n\n"
                "Quando $s/b > 0,25$, a premissa de transmissividade constante perde a validade física "
                "e a correção de Jacob torna-se insuficiente. O modelo recomendado para este nível de "
                "dessaturação é Neuman ou Boulton (drenagem retardada)."
            )
        elif pct_s_max_sobre_b < 10.0:
            st.info(
                f"ℹ️ Aviso: O rebaixamento máximo representa apenas "
                f"{pct_s_max_sobre_b:.1f}% de b (< 10%). A correção de Jacob terá "
                "impacto negligenciável nos resultados de T e S — considere "
                "dispensá-la se a espessura saturada não for bem conhecida."
            )

    st.markdown("### Curva de Rebaixamento")
    fig = pl.plot_hydro_analysis(
        df_real=df_unificado,
        df_projection=df_projection,
        col_t="tempo_min", col_s="rebaixamento_m", col_origem="tipo_dado",
        col_t_proj="tempo_min", col_s_proj="rebaixamento_projetado_m",
        origens_sinteticas=("sintetico", "sintético"),
        show_projection=False,
        title="Análise do Teste de Bombeamento",
        Q=Q_m3_h, T=cj.transmissivity_m2_day, S=cj.storativity, r=r,
        s_12h=s_12h,
        save_path=None,
    )
    st.pyplot(fig)
    plt.close(fig)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    buf_png.seek(0)
    st.download_button(
        "⬇️ Baixar Gráfico em Alta Resolução (PNG, 300 DPI)",
        data=buf_png, file_name="analise_bombeamento.png", mime="image/png",
    )

# --- Gráfico e Análise de Recuperação de Theis ---
    if res_rec is not None and t_ratio_plot is not None:
        st.markdown("---")
        st.markdown("### Curva de Recuperação de Theis (Fase de Re-enchimento)")
        st.caption("Ajuste do rebaixamento residual $s''$ em função da razão de tempos $t/t'$.")

        fig_rec = pl.plot_recovery_theis(
            t_ratio=t_ratio_plot,
            residual_drawdown=s2_plot,
            fit_line=fit_line_rec,
            T_rec=res_rec.transmissivity_m2_day,
            title="Análise do Teste de Recuperação (Método de Theis)",
            save_path=None,
        )
        st.pyplot(fig_rec)
        plt.close(fig_rec)

        buf_png_rec = io.BytesIO()
        fig_rec.savefig(buf_png_rec, format="png", dpi=300, bbox_inches="tight")
        buf_png_rec.seek(0)
        st.download_button(
            "⬇️ Baixar Gráfico de Recuperação (PNG, 300 DPI)",
            data=buf_png_rec, file_name="recuperacao_theis.png", mime="image/png",
        )

with tab_tabela:
    st.markdown("### Dados Brutos e Calculados")
    st.caption("🔵 Medido em campo   ·   ⚪ Projeção/sintético (Cooper-Jacob)")

    tabela_exibicao = df_unificado.rename(columns={
        "tempo_min": "Tempo (min)",
        "nd_m": "Nível Dinâmico (m)",
        "rebaixamento_m": "Rebaixamento (m)",
        "vazao_m3_h": "Vazão (m³/h)",
        "tipo_dado": "Origem",
    })

    def _highlight_origem(row: pd.Series):
        if row["Origem"] == "Medido":
            return ["background-color: #e7f0fb"] * len(row)
        return ["background-color: #f2f2f2; color: #666666"] * len(row)

    styled = (
        tabela_exibicao.style
        .apply(_highlight_origem, axis=1)
        .format({
            "Tempo (min)": "{:.1f}",
            "Nível Dinâmico (m)": "{:.3f}",
            "Rebaixamento (m)": "{:.3f}",
            "Vazão (m³/h)": "{:.2f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=460)

    csv_bytes = tabela_exibicao.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar Tabela Completa (CSV)",
        data=csv_bytes, file_name="dados_ensaio_bombeamento.csv", mime="text/csv",
    )

    st.markdown("---")
    aviso_s_txt = (
        "\nAviso: Ensaio de poço único — S é sensível a perdas de carga na "
        "parede do poço (skin effect) e deve ser lido apenas como ordem de "
        "grandeza. T permanece válida.\n"
        if eh_poco_unico else ""
    )
    label_r = "Raio do Poço (r_w)" if eh_poco_unico else "Distância ao poço de observação (r)"
    relatorio = f"""RELATÓRIO — ANÁLISE DE ENSAIO DE BOMBEAMENTO
================================================================
Método: Cooper-Jacob (aproximação logarítmica de Theis)
Configuração do teste: {tipo_configuracao}

Resultados
----------
Vazão (Q):                     {Q_m3_h:.3f} m³/h  ({Q_l_s:.3f} L/s)
Transmissividade (T):          {cj.transmissivity_m2_day:.3f} m²/dia
Armazenamento (S):              {cj.storativity:.4e}
Condutividade (K):              {(cj.conductivity_m_day or float('nan')):.4f} m/dia
Coeficiente de determinação R²: {cj.r_squared:.5f}
u_max:                           {cj.u_max:.5f}  (aproximação válida: {cj.approximation_is_valid})
Pontos usados no ajuste:        {cj.n_points}
{aviso_s_txt}
Rebaixamento Máximo Real:       {rebaixamento_max_real:.3f} m  (em t = {ultimo_tempo_min:.1f} min)
Rebaixamento Previsto em 12h:   {(f"{s_12h:.3f} m" if s_12h is not None else "N/A")}

Parâmetros de entrada
----------------------
Nível Estático (NE):            {NE:.3f} m
Correção de Jacob aplicada:     {"Sim" if aplicar_correcao_jacob else "Não"}
Espessura Saturada (b):         {(f"{b:.3f} m" if b is not None else "N/A (correção não aplicada)")}
{label_r}:          {r:.3f} m
Volume do balde:                {volume_balde:.3f} L
Tempo de enchimento:            {tempo_enchimento:.3f} s
"""
    st.download_button(
        "⬇️ Baixar Relatório da Análise (TXT)",
        data=relatorio.encode("utf-8"), file_name="relatorio_analise.txt",
        mime="text/plain",
    )
