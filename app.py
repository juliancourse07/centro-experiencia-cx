import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.consultas import resumen, drivers, verbatims
from utils import nlp
from utils.copiloto import responder, ranking_negativos

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Centro de Experiencia CX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AZUL = "#072B7A"
AZUL_CLARO = "#0B4AE2"
NARANJA = "#F37021"
VERDE = "#00A651"
AMARILLO = "#F2B705"
ROJO = "#D64045"
GRIS = "#8A94A6"

BANDAS = {
    "NPS": {"rango": (-100, 100), "bajo": 50.0, "alto": 70.0, "invertido": False},
    "INS": {"rango": (0, 10), "bajo": 7.0, "alto": 9.0, "invertido": False},
    "CES": {"rango": (0, 5), "bajo": 2.5, "alto": 3.5, "invertido": True},
}

COLS_INTERMEDIARIO = ("intermediario", "nombre_intermediario", "cod_intermediario")
COLS_SUCURSAL = ("cod_suc", "sucursal", "codigo_sucursal")

# =====================================================
# CSS
# =====================================================

st.markdown(
    """
<style>
.main .block-container{max-width:1800px;padding-top:1rem;padding-bottom:3rem;}
.stApp{background:#F4F7FB;}
#MainMenu, footer {visibility:hidden;}

.hero{
    background:linear-gradient(135deg,#072B7A 0%,#0B4AE2 55%,#F37021 160%);
    border-radius:24px;padding:28px 34px;color:#fff;
    box-shadow:0 10px 30px rgba(7,43,122,.25);margin-bottom:22px;
}
.hero h1{margin:0;font-size:34px;font-weight:800;letter-spacing:-.5px;}
.hero .sub{opacity:.85;font-size:15px;margin-top:2px;}
.hero .big{font-size:22px;font-weight:600;margin-top:14px;}

.kpi{
    background:#fff;border-radius:18px;padding:18px 20px;height:100%;
    box-shadow:0 3px 16px rgba(7,43,122,.08);
    border-top:5px solid #F37021;transition:transform .15s ease;
}
.kpi:hover{transform:translateY(-3px);}
.kpi-title{color:#6B7280;font-size:12.5px;font-weight:600;
    text-transform:uppercase;letter-spacing:.4px;}
.kpi-value{font-size:38px;font-weight:800;color:#072B7A;line-height:1.15;margin:4px 0;}
.kpi-delta{font-size:13px;font-weight:600;}
.kpi-foot{font-size:11.5px;color:#9AA3B2;}

.badge{display:inline-block;padding:2px 10px;border-radius:999px;
    font-size:11px;font-weight:700;color:#fff;}

.insight{
    background:#fff;border-left:6px solid #072B7A;padding:14px 18px;
    border-radius:12px;margin-bottom:10px;
    box-shadow:0 2px 10px rgba(7,43,122,.06);font-size:14.5px;
}
.insight.warn{border-left-color:#F37021;}
.insight.bad{border-left-color:#D64045;}

/* contenedor del copiloto: el contenido se renderiza como Markdown nativo */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.cx-answer){
    background:#fff;border-left:6px solid #0B4AE2;border-radius:12px;
    box-shadow:0 2px 10px rgba(7,43,122,.06);
}
.cx-answer{font-size:13px;color:#0B4AE2;font-weight:700;
    text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px;}

.section{font-size:20px;font-weight:700;color:#072B7A;
    margin:26px 0 6px;padding-bottom:6px;border-bottom:2px solid #E3E9F5;}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# HELPERS
# =====================================================

NUM_COLS = [
    "encuestados",
    "avg_nps_score",
    "avg_ins",
    "avg_ces",
    "promotores",
    "pasivos",
    "detractores",
]


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte a numérico tolerando comas decimales, %, espacios y nulos textuales."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie
    limpio = (
        serie.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(r"\s", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": None, "nan": None, "None": None, "N/A": None, "-": None})
    )
    return pd.to_numeric(limpio, errors="coerce")


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = a_numero(df[c])
    cats = ("linea", "tipo_encuesta", "anio_mes") + COLS_SUCURSAL + COLS_INTERMEDIARIO
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


@st.cache_data(ttl=900, show_spinner="Cargando datos de experiencia…")
def cargar():
    return normalizar(resumen()), normalizar(drivers()), normalizar(verbatims())


def primera_col(df, candidatas):
    if df is None or getattr(df, "empty", True):
        return None
    return next((c for c in candidatas if c in df.columns), None)


def calcular_nps(d: pd.DataFrame):
    """NPS real = %Promotores - %Detractores. Fallback al promedio si no hay conteos."""
    if d is None or d.empty:
        return None
    if {"promotores", "detractores", "encuestados"}.issubset(d.columns):
        base = d["encuestados"].sum(skipna=True)
        if base and base > 0:
            promo = d["promotores"].sum(skipna=True)
            detr = d["detractores"].sum(skipna=True)
            return round((promo - detr) / base * 100, 1)
    if "avg_nps_score" in d.columns:
        val = d["avg_nps_score"].mean(skipna=True)
        return round(val, 1) if pd.notna(val) else None
    return None


def promedio_ponderado(d: pd.DataFrame, col: str):
    """Promedio ponderado por encuestados: el simple sobrevalora líneas pequeñas."""
    if d is None or d.empty or col not in d.columns:
        return None
    cols = [col] + (["encuestados"] if "encuestados" in d.columns else [])
    sub = d[cols].dropna()
    if sub.empty:
        return None
    if "encuestados" in sub.columns and sub["encuestados"].sum() > 0:
        return round((sub[col] * sub["encuestados"]).sum() / sub["encuestados"].sum(), 2)
    return round(sub[col].mean(), 2)


def semaforo(metrica: str, valor):
    """(etiqueta, color) según los rangos oficiales de la metodología."""
    if valor is None or pd.isna(valor):
        return "Sin dato", GRIS
    cfg = BANDAS[metrica]
    if cfg["invertido"]:
        if valor <= cfg["bajo"]:
            return "Bajo esfuerzo", VERDE
        if valor <= cfg["alto"]:
            return "Esfuerzo medio", AMARILLO
        return "Alto esfuerzo", ROJO
    if valor >= cfg["alto"]:
        return "Alto", VERDE
    if valor > cfg["bajo"]:
        return "Medio", AMARILLO
    return "Bajo", ROJO


def fmt(valor, dec=1):
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.{dec}f}"


def delta(actual, previo):
    if actual is None or previo is None or pd.isna(actual) or pd.isna(previo):
        return None
    return actual - previo


def card(titulo, valor, delta_val=None, pie="", color_borde=NARANJA,
         sufijo_delta="vs. período anterior"):
    if delta_val is None or pd.isna(delta_val):
        bloque = "<div class='kpi-delta' style='color:#9AA3B2'>— sin comparativo</div>"
    else:
        arriba = delta_val >= 0
        color = VERDE if arriba else ROJO
        flecha = "▲" if arriba else "▼"
        bloque = (
            f"<div class='kpi-delta' style='color:{color}'>{flecha} {abs(delta_val):,.1f} "
            f"<span style='color:#9AA3B2;font-weight:400'>{sufijo_delta}</span></div>"
        )
    st.markdown(
        f"<div class='kpi' style='border-top-color:{color_borde}'>"
        f"<div class='kpi-title'>{titulo}</div>"
        f"<div class='kpi-value'>{valor}</div>{bloque}"
        f"<div class='kpi-foot'>{pie}</div></div>",
        unsafe_allow_html=True,
    )


def gauge(nombre, valor, metrica):
    cfg = BANDAS[metrica]
    lo, hi = cfg["rango"]
    etiqueta, color = semaforo(metrica, valor)
    v = float(valor) if valor is not None and not pd.isna(valor) else lo

    if cfg["invertido"]:
        pasos = [
            {"range": [lo, cfg["bajo"]], "color": "#D8F5D8"},
            {"range": [cfg["bajo"], cfg["alto"]], "color": "#FFF4C2"},
            {"range": [cfg["alto"], hi], "color": "#FADADD"},
        ]
    else:
        pasos = [
            {"range": [lo, cfg["bajo"]], "color": "#FADADD"},
            {"range": [cfg["bajo"], cfg["alto"]], "color": "#FFF4C2"},
            {"range": [cfg["alto"], hi], "color": "#D8F5D8"},
        ]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"font": {"size": 46, "color": AZUL}},
            title={
                "text": f"<b>{nombre}</b><br>"
                        f"<span style='font-size:12px;color:{color}'>{etiqueta}</span>",
                "font": {"size": 16, "color": AZUL},
            },
            gauge={
                "axis": {"range": [lo, hi], "tickcolor": GRIS},
                "bar": {"color": AZUL, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": pasos,
                "threshold": {"line": {"color": NARANJA, "width": 4},
                              "thickness": 0.8, "value": v},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def sin_datos(msg="No hay datos para los filtros seleccionados."):
    st.markdown(f"<div class='insight warn'>⚠️ {msg}</div>", unsafe_allow_html=True)


# =====================================================
# DATA
# =====================================================

try:
    df, drv, verb = cargar()
except Exception as e:
    st.error("No fue posible cargar los datos desde el origen.")
    st.exception(e)
    st.stop()

if df is None or df.empty:
    st.error("La consulta de resumen no devolvió registros.")
    st.stop()

COL_SUC = primera_col(df, COLS_SUCURSAL)

# =====================================================
# FILTROS
# =====================================================

with st.sidebar:
    st.markdown(f"<h3 style='color:{AZUL};margin-bottom:0'>Filtros</h3>", unsafe_allow_html=True)
    st.caption("Aplican a todo el tablero")

    lineas = ["TODAS"] + sorted(df["linea"].dropna().unique().tolist())
    tipos = ["TODOS"] + sorted(df["tipo_encuesta"].dropna().unique().tolist())
    periodos = sorted(df["anio_mes"].dropna().unique().tolist(), reverse=True)

    linea = st.selectbox("Línea", lineas, key="f_linea")
    tipo = st.selectbox("Tipo de encuesta", tipos, key="f_tipo")
    periodo = st.selectbox("Período", periodos, key="f_periodo")

    if COL_SUC:
        base_suc = df if linea == "TODAS" else df[df["linea"] == linea]
        opciones_suc = sorted(base_suc[COL_SUC].dropna().astype(str).unique().tolist())
        sucursales = st.multiselect(
            f"Sucursal ({COL_SUC})", opciones_suc,
            help="Vacío = todas las sucursales", key="f_suc",
        )
    else:
        sucursales = []
        st.caption("ℹ️ El origen no expone `cod_suc`; filtro de sucursal deshabilitado.")

    st.divider()
    if st.button("↺ Limpiar filtros", use_container_width=True):
        st.session_state.update(f_linea="TODAS", f_tipo="TODOS",
                                f_periodo=periodos[0], f_suc=[])
        st.session_state.pop("verb_nlp", None)
        st.rerun()
    if st.button("🔄 Recargar datos", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("verb_nlp", None)
        st.rerun()

    st.divider()
    st.caption(
        "**Rangos oficiales**  \n"
        "NPS · Alto ≥70 · Medio 50,1–70 · Bajo <50  \n"
        "INS · Alto ≥9 · Medio 7,1–8,9 · Bajo <7  \n"
        "CES · Bajo esfuerzo <2,5 · Alto >3,5"
    )
    st.caption("🔑 LLM " + ("activo" if os.getenv("HF_TOKEN") else "en modo analítico (sin HF_TOKEN)"))


def aplicar_filtros(d, con_periodo=True, periodo_valor=None):
    if d is None or d.empty:
        return d
    out = d
    if linea != "TODAS" and "linea" in out.columns:
        out = out[out["linea"] == linea]
    if tipo != "TODOS" and "tipo_encuesta" in out.columns:
        out = out[out["tipo_encuesta"] == tipo]
    if con_periodo and "anio_mes" in out.columns:
        out = out[out["anio_mes"] == (periodo_valor or periodo)]
    if sucursales:
        col = primera_col(out, COLS_SUCURSAL)
        if col:
            out = out[out[col].astype(str).isin(sucursales)]
    return out


dfa = aplicar_filtros(df)
dfh = aplicar_filtros(df, con_periodo=False)

idx = periodos.index(periodo)
periodo_prev = periodos[idx + 1] if idx + 1 < len(periodos) else None
dfp = aplicar_filtros(df, periodo_valor=periodo_prev) if periodo_prev else pd.DataFrame()

verb_f = aplicar_filtros(verb) if verb is not None and not verb.empty else pd.DataFrame()
drv_f = aplicar_filtros(drv) if drv is not None and not drv.empty else pd.DataFrame()

# =====================================================
# KPIS
# =====================================================

total_resp = int(dfa["encuestados"].sum(skipna=True)) if not dfa.empty and "encuestados" in dfa.columns else 0
nps_real = calcular_nps(dfa)
ins = promedio_ponderado(dfa, "avg_ins")
ces = promedio_ponderado(dfa, "avg_ces")

total_prev = int(dfp["encuestados"].sum(skipna=True)) if not dfp.empty and "encuestados" in dfp.columns else None
nps_prev = calcular_nps(dfp) if not dfp.empty else None
ins_prev = promedio_ponderado(dfp, "avg_ins") if not dfp.empty else None
ces_prev = promedio_ponderado(dfp, "avg_ces") if not dfp.empty else None

# =====================================================
# HERO
# =====================================================

scope = []
if linea != "TODAS":
    scope.append(linea)
if tipo != "TODOS":
    scope.append(tipo)
if sucursales:
    scope.append(f"{len(sucursales)} sucursal(es)")
scope_txt = " · ".join(scope) if scope else "Todas las líneas y encuestas"

st.markdown(
    f"<div class='hero'>"
    f"<h1>📊 Centro de Experiencia CX</h1>"
    f"<div class='sub'>Seguros del Estado · {scope_txt} · Período {periodo}</div>"
    f"<div class='big'>{total_resp:,.0f} respuestas analizadas en el período</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if dfa.empty:
    sin_datos("No hay registros para esta combinación de filtros. Ajusta la selección en la barra lateral.")
    st.stop()

# =====================================================
# KPI CARDS
# =====================================================

et_nps, col_nps = semaforo("NPS", nps_real)
et_ins, col_ins = semaforo("INS", ins)
et_ces, col_ces = semaforo("CES", ces)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    card("🎯 NPS", fmt(nps_real), delta(nps_real, nps_prev),
         f"<span class='badge' style='background:{col_nps}'>{et_nps}</span>", col_nps)
with c2:
    card("😊 INS", fmt(ins, 2), delta(ins, ins_prev),
         f"<span class='badge' style='background:{col_ins}'>{et_ins}</span>", col_ins)
with c3:
    card("⚡ CES", fmt(ces, 2), delta(ces, ces_prev),
         f"<span class='badge' style='background:{col_ces}'>{et_ces}</span>", col_ces)
with c4:
    card("📨 Respuestas", f"{total_resp:,.0f}", delta(total_resp, total_prev),
         "Encuestas completadas", AZUL_CLARO)
with c5:
    card("💬 Verbatims", f"{len(verb_f):,.0f}", None, "Comentarios abiertos", NARANJA, "")

# =====================================================
# GAUGES
# =====================================================

g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(gauge("NPS", nps_real, "NPS"), use_container_width=True)
with g2:
    st.plotly_chart(gauge("INS", ins, "INS"), use_container_width=True)
with g3:
    st.plotly_chart(gauge("CES", ces, "CES"), use_container_width=True)

# =====================================================
# EVOLUCIÓN + COMPOSICIÓN NPS
# =====================================================

col_ev, col_dist = st.columns([1.35, 1])

with col_ev:
    st.markdown("<div class='section'>📈 Evolución</div>", unsafe_allow_html=True)
    if dfh.empty:
        sin_datos()
    else:
        agg = {c: "mean" for c in ("avg_ins", "avg_ces") if c in dfh.columns}
        trend = dfh.groupby("anio_mes", as_index=False).agg(agg) if agg else pd.DataFrame({"anio_mes": []})
        nps_mes = (
            dfh.groupby("anio_mes")[[c for c in NUM_COLS if c in dfh.columns]]
            .apply(calcular_nps)
            .reset_index(name="nps")
        )
        trend = nps_mes.merge(trend, on="anio_mes", how="left").sort_values("anio_mes")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=trend["anio_mes"], y=trend["nps"], name="NPS",
                       mode="lines+markers", line=dict(color=AZUL, width=3),
                       marker=dict(size=8)),
            secondary_y=False,
        )
        if "avg_ins" in trend.columns:
            fig.add_trace(
                go.Scatter(x=trend["anio_mes"], y=trend["avg_ins"], name="INS",
                           mode="lines+markers",
                           line=dict(color=NARANJA, width=3, dash="dot"),
                           marker=dict(size=8)),
                secondary_y=True,
            )
        fig.add_hline(y=70, line_dash="dash", line_color=VERDE, opacity=.5,
                      annotation_text="Meta NPS 70", annotation_position="top left")
        fig.update_yaxes(title_text="NPS", range=[-100, 100], secondary_y=False)
        fig.update_yaxes(title_text="INS", range=[0, 10], secondary_y=True, showgrid=False)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, use_container_width=True)

with col_dist:
    st.markdown("<div class='section'>🧩 Composición NPS</div>", unsafe_allow_html=True)
    cols_nps = [c for c in ("promotores", "pasivos", "detractores") if c in dfa.columns]
    if not cols_nps:
        sin_datos("El origen no expone promotores/pasivos/detractores.")
    else:
        dist = dfa.groupby("linea", as_index=False)[cols_nps].sum()
        tot = dist[cols_nps].sum(axis=1).replace(0, pd.NA)
        for c in cols_nps:
            dist[c] = dist[c] / tot * 100
        largo = dist.melt(id_vars="linea", value_vars=cols_nps,
                          var_name="Segmento", value_name="Porcentaje")
        fig = px.bar(
            largo, x="Porcentaje", y="linea", color="Segmento", orientation="h",
            color_discrete_map={"promotores": VERDE, "pasivos": AMARILLO, "detractores": ROJO},
            text=largo["Porcentaje"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else ""),
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          barmode="stack", plot_bgcolor="white",
                          paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="", yaxis_title="",
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# COMPARATIVOS
# =====================================================

col_r, col_h = st.columns(2)

with col_r:
    st.markdown("<div class='section'>📍 Comparativo por línea</div>", unsafe_allow_html=True)
    radar = dfa.groupby("linea", as_index=False)["avg_ins"].mean().dropna()
    if radar.empty:
        sin_datos()
    elif len(radar) < 3:
        st.bar_chart(radar.set_index("linea"), color=AZUL, height=340)
    else:
        fig = px.line_polar(radar, r="avg_ins", theta="linea", line_close=True, range_r=[0, 10])
        fig.update_traces(fill="toself", line_color=AZUL, fillcolor="rgba(11,74,226,.18)")
        fig.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with col_h:
    st.markdown("<div class='section'>🌡️ INS por línea y tipo</div>", unsafe_allow_html=True)
    if {"linea", "tipo_encuesta", "avg_ins"}.issubset(dfa.columns):
        piv = dfa.pivot_table(index="linea", columns="tipo_encuesta",
                              values="avg_ins", aggfunc="mean")
        if piv.empty:
            sin_datos()
        else:
            fig = px.imshow(piv, text_auto=".1f", aspect="auto", zmin=0, zmax=10,
                            color_continuous_scale=[[0, ROJO], [.7, AMARILLO], [1, VERDE]])
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="", yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        sin_datos()

# =====================================================
# DRIVERS
# =====================================================

st.markdown("<div class='section'>🔥 Drivers de experiencia</div>", unsafe_allow_html=True)

if drv_f.empty or "categoria" not in drv_f.columns:
    sin_datos("No hay drivers para los filtros seleccionados.")
else:
    col_impacto = primera_col(drv_f, ("avg_ins", "avg_nps_score", "impacto"))
    if col_impacto:
        rank = drv_f.groupby("categoria", as_index=False).agg(
            menciones=("categoria", "size"), impacto=(col_impacto, "mean"))
    else:
        rank = drv_f.groupby("categoria", as_index=False).agg(menciones=("categoria", "size"))
    rank["% del total"] = (rank["menciones"] / rank["menciones"].sum() * 100).round(1)

    t1, t2 = st.tabs(["🎯 Matriz de priorización", "📊 Pareto"])

    with t1:
        if not col_impacto:
            sin_datos("Falta una columna de impacto (avg_ins / avg_nps_score) en drivers().")
        else:
            mx, my = rank["menciones"].median(), rank["impacto"].median()
            fig = px.scatter(rank, x="menciones", y="impacto", size="menciones",
                             text="categoria", color="impacto",
                             color_continuous_scale=[[0, ROJO], [.5, AMARILLO], [1, VERDE]],
                             size_max=55)
            fig.update_traces(textposition="top center", textfont_size=11)
            fig.add_vline(x=mx, line_dash="dot", line_color=GRIS)
            fig.add_hline(y=my, line_dash="dot", line_color=GRIS)
            fig.add_annotation(x=rank["menciones"].max(), y=rank["impacto"].min(),
                               text="<b>ATACAR YA</b>", showarrow=False,
                               font=dict(color=ROJO, size=13), bgcolor="rgba(214,64,69,.12)")
            fig.add_annotation(x=rank["menciones"].max(), y=rank["impacto"].max(),
                               text="<b>MANTENER</b>", showarrow=False,
                               font=dict(color=VERDE, size=13), bgcolor="rgba(0,166,81,.12)")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                              plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="Volumen de menciones →",
                              yaxis_title="← Desempeño (peor abajo)",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Prioriza el cuadrante inferior derecho: alto volumen y bajo desempeño.")

    with t2:
        par = rank.sort_values("menciones", ascending=False).copy()
        par["acumulado"] = (par["menciones"].cumsum() / par["menciones"].sum() * 100).round(1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=par["categoria"], y=par["menciones"], name="Menciones",
                             marker_color=AZUL), secondary_y=False)
        fig.add_trace(go.Scatter(x=par["categoria"], y=par["acumulado"], name="% acumulado",
                                 mode="lines+markers", line=dict(color=NARANJA, width=3)),
                      secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color=ROJO, opacity=.6, secondary_y=True)
        fig.update_yaxes(title_text="Menciones", secondary_y=False)
        fig.update_yaxes(title_text="% acumulado", range=[0, 105], secondary_y=True, showgrid=False)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                          plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          hovermode="x unified",
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Regla 80/20: las categorías a la izquierda de la línea roja explican el 80% de las menciones.")

# =====================================================
# SENTIMIENTO (BERT)
# =====================================================

st.markdown("<div class='section'>🧠 Análisis de sentimiento (BERT)</div>", unsafe_allow_html=True)

if verb_f.empty:
    sin_datos("No hay comentarios para analizar.")
else:
    s1, s2, s3 = st.columns([1, 1, 1])
    muestra = s1.slider("Comentarios a analizar", 50, 1000, 300, step=50,
                        help="El modelo corre en CPU; más comentarios = más tiempo.")
    con_sub = s2.toggle("Incluir sub-sentimiento (zero-shot)", value=True)
    s3.write("")
    ejecutar = s3.button("▶️ Ejecutar análisis", type="primary", use_container_width=True)

    if ejecutar:
        try:
            st.session_state["verb_nlp"] = nlp.enriquecer(verb_f, muestra=muestra, con_sub=con_sub)
        except Exception as e:
            st.error("Falló el análisis de sentimiento.")
            st.exception(e)

    vnlp = st.session_state.get("verb_nlp")

    if vnlp is None:
        st.info("Pulsa **Ejecutar análisis** para clasificar los comentarios.")
    elif "polaridad" not in vnlp.columns:
        sin_datos("No se detectó una columna de texto en los verbatims.")
    else:
        dist = vnlp["polaridad"].value_counts()
        neg_pct = dist.get("Negativo", 0) / max(len(vnlp), 1) * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1: card("😊 Positivos", f"{dist.get('Positivo', 0):,.0f}", None, "Comentarios", VERDE, "")
        with k2: card("😐 Neutros", f"{dist.get('Neutro', 0):,.0f}", None, "Comentarios", AMARILLO, "")
        with k3: card("😠 Negativos", f"{dist.get('Negativo', 0):,.0f}", None, "Comentarios", ROJO, "")
        with k4: card("⚠️ Tasa negativa", f"{neg_pct:.1f}%", None, "Del total analizado", ROJO, "")

        n1, n2 = st.columns([1, 1.4])
        with n1:
            fig = px.pie(values=dist.values, names=dist.index, hole=.58, color=dist.index,
                         color_discrete_map={"Positivo": VERDE, "Neutro": AMARILLO, "Negativo": ROJO})
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with n2:
            if "subsentimiento" in vnlp.columns and vnlp["subsentimiento"].notna().any():
                sun = (vnlp.dropna(subset=["subsentimiento"])
                       .groupby(["polaridad", "subsentimiento"], as_index=False).size())
                fig = px.sunburst(sun, path=["polaridad", "subsentimiento"], values="size",
                                  color="polaridad",
                                  color_discrete_map={"Positivo": VERDE, "Neutro": AMARILLO,
                                                      "Negativo": ROJO, "(?)": GRIS})
                fig.update_traces(insidetextorientation="radial")
                fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Activa el sub-sentimiento para ver el desglose.")

        opciones_dim = [c for c in (COLS_INTERMEDIARIO + COLS_SUCURSAL + ("linea", "tipo_encuesta"))
                        if c in vnlp.columns]
        if opciones_dim:
            dim_col = st.selectbox("Agrupar comentarios negativos por", opciones_dim, key="f_dim_neg")
            rk = ranking_negativos(vnlp, dim_col, top=15)
            if rk.empty:
                sin_datos("Volumen insuficiente para el ranking (mínimo 5 comentarios por grupo).")
            else:
                fig = px.bar(rk.sort_values("negativos"), x="negativos", y=dim_col,
                             orientation="h", text="% negativos", color="% negativos",
                             color_continuous_scale=[[0, AMARILLO], [1, ROJO]])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(height=460, margin=dict(l=10, r=40, t=20, b=10),
                                  plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis_title="Comentarios negativos", yaxis_title="",
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rk, use_container_width=True, hide_index=True)
        else:
            st.caption("ℹ️ Los verbatims no traen columnas de intermediario ni sucursal.")

# =====================================================
# INSIGHTS
# =====================================================

st.markdown("<div class='section'>🧠 Insights automáticos</div>", unsafe_allow_html=True)

insights = []
serie_ins = dfa.groupby("linea")["avg_ins"].mean().dropna()
if not serie_ins.empty:
    mejor, peor = serie_ins.idxmax(), serie_ins.idxmin()
    insights.append(("", f"La línea con mejor experiencia es <b>{mejor}</b> (INS {serie_ins.max():.2f})."))
    if mejor != peor:
        insights.append(("warn", f"La línea con mayor oportunidad es <b>{peor}</b> (INS {serie_ins.min():.2f})."))

d_nps = delta(nps_real, nps_prev)
if d_nps is not None:
    if d_nps >= 0:
        insights.append(("", f"El NPS subió <b>{d_nps:+.1f} puntos</b> frente a {periodo_prev}."))
    else:
        insights.append(("bad", f"El NPS cayó <b>{d_nps:.1f} puntos</b> frente a {periodo_prev}. Revisar drivers."))

if ces is not None and ces > BANDAS["CES"]["alto"]:
    insights.append(("bad", f"El CES ({ces:.2f}) está en zona de <b>alto esfuerzo</b>. "
                            "Priorizar simplificación de procesos."))

insights.append(("", f"Se analizaron <b>{total_resp:,.0f}</b> respuestas y <b>{len(verb_f):,.0f}</b> verbatims."))

for clase, txt in insights:
    icono = {"bad": "🚨", "warn": "⚠️", "": "✅"}[clase]
    st.markdown(f"<div class='insight {clase}'>{icono} {txt}</div>", unsafe_allow_html=True)

# =====================================================
# VERBATIMS
# =====================================================

st.markdown("<div class='section'>💬 Verbatims</div>", unsafe_allow_html=True)

if verb_f.empty:
    sin_datos("No hay comentarios para los filtros seleccionados.")
else:
    vnlp = st.session_state.get("verb_nlp")
    fuente_tabla = vnlp if vnlp is not None and "polaridad" in getattr(vnlp, "columns", []) else verb_f

    v1, v2, v3 = st.columns([2.5, 1, 1])
    busqueda = v1.text_input("Buscar en los comentarios", placeholder="Ej.: demora, atención, portal…")
    if "polaridad" in fuente_tabla.columns:
        pol = v2.multiselect("Polaridad", ["Positivo", "Neutro", "Negativo"], key="f_pol")
    else:
        pol = []
        v2.caption("")
    limite = v3.number_input("Filas", 50, 5000, 500, step=50)

    vista = fuente_tabla
    if pol:
        vista = vista[vista["polaridad"].isin(pol)]
    if busqueda:
        cols_txt = vista.select_dtypes(include="object").columns
        mask = pd.Series(False, index=vista.index)
        for c in cols_txt:
            mask |= vista[c].astype(str).str.contains(busqueda, case=False, na=False)
        vista = vista[mask]

    st.caption(f"{len(vista):,.0f} comentarios coinciden con los criterios.")
    st.dataframe(vista.head(int(limite)), use_container_width=True, height=420)

    buff = io.StringIO()
    vista.to_csv(buff, index=False)
    st.download_button("⬇️ Descargar CSV", buff.getvalue(),
                       file_name=f"verbatims_{periodo}_{linea}_{tipo}.csv", mime="text/csv")

# =====================================================
# COPILOTO
# =====================================================

st.markdown("<div class='section'>🤖 Copiloto CX</div>", unsafe_allow_html=True)

cp1, cp2 = st.columns([2, 1])

with cp1:
    pregunta = st.text_area(
        "Pregúntale al copiloto sobre la experiencia de tus clientes",
        height=130,
        placeholder="Ej.: ¿Qué intermediarios tienen mayor número de comentarios malos?",
    )
    if st.button("Analizar", type="primary"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta para analizar.")
        else:
            fuente = st.session_state.get("verb_nlp")
            if fuente is None or "polaridad" not in getattr(fuente, "columns", []):
                st.warning(
                    "Primero ejecuta el **Análisis de sentimiento** para que el copiloto "
                    "pueda distinguir comentarios positivos de negativos."
                )
            else:
                kpis = {
                    "nps": nps_real, "ins": ins, "ces": ces,
                    "respuestas": total_resp, "verbatims": len(verb_f),
                    "filtros": f"{linea} · {tipo} · {periodo}"
                    + (f" · sucursales: {', '.join(sucursales)}" if sucursales else ""),
                }
                with st.spinner("Analizando…"):
                    texto, tabla = responder(pregunta, fuente, kpis)

                # Markdown nativo: el LLM devuelve viñetas y negritas
                with st.container(border=True):
                    st.markdown("<div class='cx-answer'>🤖 Respuesta del copiloto</div>",
                                unsafe_allow_html=True)
                    st.markdown(texto)
                    if tabla is not None and not tabla.empty:
                        st.dataframe(tabla, use_container_width=True, hide_index=True)
                        b = io.StringIO()
                        tabla.to_csv(b, index=False)
                        st.download_button("⬇️ Descargar este ranking", b.getvalue(),
                                           file_name="ranking_negativos.csv", mime="text/csv")

with cp2:
    st.markdown(
        "<div class='insight'>"
        "<b>Ejemplos</b><br>"
        "• ¿Qué intermediarios tienen más comentarios malos?<br>"
        "• ¿Qué sucursales concentran las quejas?<br>"
        "• Resume los comentarios de Vida.<br>"
        "• Principales dolores de Clientes.<br>"
        "• Riesgos por línea."
        "</div>",
        unsafe_allow_html=True,
    )