import logging

import pandas as pd
import streamlit as st

from utils import nlp
from utils.consultas import drivers, resumen, resumen_por_sucursal, tipos_encuesta_disponibles, verbatims
from utils.copiloto import responder
from utils.estilos import PALETA, aplicar_estilos
from utils.graficos import (
    figura_composicion_nps,
    figura_donut_polaridad,
    figura_drivers_matriz,
    figura_drivers_pareto,
    figura_evolucion,
    figura_gauge,
    figura_heatmap_ins,
    figura_radar_linea,
    figura_sunburst_sentimiento,
    figura_temas_negativos,
)
from utils.iconos import icono, titulo_seccion
from utils.informe import (
    dataframe_a_csv_bytes,
    dataframes_a_xlsx_bytes,
    generar_informe_excel,
    generar_informe_html,
    nombre_archivo_base,
)
from utils.tablero import (
    aplicar_filtros,
    calcular_nps,
    card,
    construir_kpis,
    construir_opciones_tipo,
    filtros_activos_texto,
    fmt,
    fuente_datos,
    insights_tablero,
    normalizar,
    sin_datos,
    sucursales_disponibles,
    valores_union,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Centro de Experiencia CX",
    layout="wide",
    initial_sidebar_state="expanded",
)
aplicar_estilos()


@st.cache_data(ttl=900, show_spinner="Cargando datos de experiencia…")
def cargar_resumen() -> pd.DataFrame:
    return normalizar(resumen())


@st.cache_data(ttl=900, show_spinner="Cargando resumen por sucursal…")
def cargar_resumen_sucursal() -> pd.DataFrame:
    return normalizar(resumen_por_sucursal())


@st.cache_data(ttl=900, show_spinner=False)
def cargar_drivers() -> pd.DataFrame:
    return normalizar(drivers())


@st.cache_data(ttl=900, show_spinner=False)
def cargar_verbatims() -> pd.DataFrame:
    return normalizar(verbatims())


try:
    df_base = cargar_resumen()
    df_sucursal = cargar_resumen_sucursal()
    drv = cargar_drivers()
    verb = cargar_verbatims()
except Exception as error:
    LOGGER.exception("Fallo cargando datasets base")
    st.error("No fue posible cargar los datos desde Databricks SQL. Revisa la conexión y vuelve a intentar.")
    st.exception(error)
    st.stop()

if df_base is None or df_base.empty:
    st.error("La consulta principal no devolvió registros.")
    st.stop()

tipos_maestros = tipos_encuesta_disponibles()
lineas = ["TODAS", *valores_union(df_base, df_sucursal, drv, verb, columna="linea")]
periodos = sorted(valores_union(df_base, df_sucursal, drv, verb, columna="anio_mes"), reverse=True)
if not periodos:
    st.error("No hay períodos disponibles en las fuentes consultadas.")
    st.stop()

with st.sidebar:
    st.markdown(f"<h3 style='color:{PALETA['azul']};margin-bottom:0'>Filtros</h3>", unsafe_allow_html=True)
    tipo = st.selectbox(
        "Tipo de encuesta",
        construir_opciones_tipo(df_base, df_sucursal, verb, tipos_maestros),
        key="f_tipo",
    )
    periodo = st.selectbox("Período", periodos, key="f_periodo")
    linea = st.selectbox("Línea", lineas, key="f_linea")
    if tipo == "INTERMEDIARIO":
        opciones_suc = sucursales_disponibles(df_sucursal, linea=linea, periodo=periodo)
        actual_suc = st.session_state.get("f_suc", "TODAS")
        if actual_suc not in opciones_suc:
            st.session_state["f_suc"] = "TODAS"
        sucursal = st.selectbox("Sucursal", opciones_suc, key="f_suc")
    else:
        st.session_state.pop("f_suc", None)
        sucursal = "TODAS"
    if st.button("Limpiar filtros", use_container_width=True):
        st.session_state.update(f_tipo="TODOS", f_periodo=periodos[0], f_linea="TODAS")
        st.session_state.pop("f_suc", None)
        st.session_state.pop("verb_nlp", None)
        st.session_state.pop("verb_nlp_firma", None)
        st.rerun()
    if st.button("Recargar datos", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("verb_nlp", None)
        st.session_state.pop("verb_nlp_firma", None)
        st.rerun()
    st.caption(
        "**Rangos oficiales**  \n"
        "NPS · Alto ≥70 · Medio 50,1–70 · Bajo <50  \n"
        "INS · Alto ≥9 · Medio 7,1–8,9 · Bajo <7  \n"
        "CES · Bajo esfuerzo <2,5 · Alto >3,5"
    )


df_fuente = fuente_datos(tipo, sucursal, df_base, df_sucursal)
dfa = aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
dfh = aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal, con_periodo=False)
idx = periodos.index(periodo)
periodo_prev = periodos[idx + 1] if idx + 1 < len(periodos) else None
dfp = (
    aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal, periodo_valor=periodo_prev)
    if periodo_prev
    else pd.DataFrame()
)
verb_f = aplicar_filtros(verb, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
drv_f = aplicar_filtros(drv, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
filtros_texto = filtros_activos_texto(linea, tipo, periodo, sucursal)
kpis = construir_kpis(dfa, dfp, verb_f)
alcance = [
    valor
    for valor in [
        linea if linea != "TODAS" else None,
        tipo if tipo != "TODOS" else None,
        f"Sucursal {sucursal}" if sucursal != "TODAS" else None,
    ]
    if valor
]

st.markdown(
    f"<div class='hero'><h1>Centro de Experiencia CX</h1>"
    f"<div class='sub'>Seguros del Estado · {' · '.join(alcance) if alcance else 'Cobertura general'} · Período {periodo}</div>"
    f"<div class='big'>{kpis['respuestas']:,.0f} respuestas analizadas</div></div>",
    unsafe_allow_html=True,
)

if dfa.empty:
    sin_datos(
        f"No hay datos para los filtros activos ({filtros_texto}). Si elegiste INTERMEDIARIO, la app ya intentó el respaldo con KPIs por sucursal."
    )
    st.stop()

vnlp_base = st.session_state.get("verb_nlp") if st.session_state.get("verb_nlp_firma", ())[:4] == (linea, tipo, periodo, sucursal) else None

tab_tablero, tab_drivers, tab_voz, tab_base, tab_informe, tab_copiloto = st.tabs(
    ["Tablero general", "Drivers", "Voz del cliente", "Base general", "Informe", "Copiloto CX"]
)

with tab_tablero:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("NPS", fmt(kpis["nps"]), kpis["delta_nps"], f"<span class='badge' style='background:{kpis['color_nps']}'>{kpis['estado_nps']}</span>", kpis["color_nps"], icon_name="nps")
    with c2:
        card("INS", fmt(kpis["ins"], 2), kpis["delta_ins"], f"<span class='badge' style='background:{kpis['color_ins']}'>{kpis['estado_ins']}</span>", kpis["color_ins"], icon_name="ins")
    with c3:
        card("CES", fmt(kpis["ces"], 2), kpis["delta_ces"], f"<span class='badge' style='background:{kpis['color_ces']}'>{kpis['estado_ces']}</span>", kpis["color_ces"], icon_name="termometro")
    with c4:
        card("Respuestas", f"{kpis['respuestas']:,.0f}", kpis["delta_respuestas"], "Encuestas completadas", PALETA["azul_claro"], icon_name="respuestas")
    with c5:
        card("Verbatims", f"{kpis['verbatims']:,.0f}", None, "Comentarios abiertos", PALETA["naranja"], icon_name="verbatims")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(figura_gauge("NPS", kpis["nps"], "NPS"), use_container_width=True)
    with g2:
        st.plotly_chart(figura_gauge("INS", kpis["ins"], "INS"), use_container_width=True)
    with g3:
        st.plotly_chart(figura_gauge("CES", kpis["ces"], "CES"), use_container_width=True)
    col_ev, col_dist = st.columns([1.35, 1])
    with col_ev:
        titulo_seccion("evolucion", "Evolución")
        if not dfh.empty:
            st.plotly_chart(figura_evolucion(dfh), use_container_width=True)
        else:
            sin_datos("Sin histórico disponible para los filtros activos.")
    with col_dist:
        titulo_seccion("dashboard", "Composición NPS")
        if {"promotores", "neutros", "detractores"}.issubset(dfa.columns):
            st.plotly_chart(figura_composicion_nps(dfa), use_container_width=True)
        else:
            sin_datos("El origen no expone la composición NPS requerida.")
    col_r, col_h = st.columns(2)
    with col_r:
        titulo_seccion("radar", "Comparativo por línea")
        radar = dfa.groupby("linea", as_index=False)["avg_ins"].mean().dropna()
        if len(radar) >= 3:
            st.plotly_chart(figura_radar_linea(dfa), use_container_width=True)
        elif not radar.empty:
            st.bar_chart(radar.set_index("linea"), color=PALETA["azul"], height=340)
        else:
            sin_datos("No hay datos comparables por línea.")
    with col_h:
        titulo_seccion("termometro", "INS por línea y tipo")
        if {"linea", "tipo_encuesta", "avg_ins"}.issubset(dfa.columns):
            st.plotly_chart(figura_heatmap_ins(dfa), use_container_width=True)
        else:
            sin_datos("Sin datos suficientes para el mapa de calor.")
    titulo_seccion("cerebro", "Insights automáticos")
    for _, fila in insights_tablero(kpis, dfa, periodo_prev).iterrows():
        clase = {"positivo": "", "alerta": "warn", "riesgo": "bad"}.get(fila["tipo"], "")
        nombre_icono = "check" if clase == "" else "alerta"
        st.markdown(
            f"<div class='insight {clase}'>{icono(nombre_icono, size=16)} {fila['detalle']}</div>",
            unsafe_allow_html=True,
        )

with tab_drivers:
    titulo_seccion("drivers", "Drivers de experiencia")
    if drv_f.empty or "categoria" not in drv_f.columns:
        sin_datos("No hay drivers disponibles con los filtros seleccionados.")
    else:
        col_impacto = next((col for col in ("avg_ins", "avg_nps_score", "impacto") if col in drv_f.columns), None)
        if col_impacto:
            rank = drv_f.groupby("categoria", as_index=False).agg(menciones=("categoria", "size"), impacto=(col_impacto, "mean"))
            td1, td2 = st.tabs(["Matriz de priorización", "Pareto"])
            with td1:
                st.plotly_chart(figura_drivers_matriz(rank), use_container_width=True)
            with td2:
                st.plotly_chart(figura_drivers_pareto(rank[["categoria", "menciones"]]), use_container_width=True)
        else:
            rank = drv_f.groupby("categoria", as_index=False).agg(menciones=("categoria", "size"))
            st.plotly_chart(figura_drivers_pareto(rank[["categoria", "menciones"]]), use_container_width=True)
        st.dataframe(rank.sort_values("menciones", ascending=False), use_container_width=True, hide_index=True)

with tab_voz:
    titulo_seccion("cerebro", "Voz del cliente · Análisis de sentimiento")
    st.caption(
        "Clasificación jerárquica de polaridad, emoción y tema. El modo avanzado con zero-shot es opcional porque consume más CPU."
    )
    if verb_f.empty:
        sin_datos("No hay verbatims disponibles para analizar con los filtros activos.")
    else:
        s1, s2, s3 = st.columns([1, 1, 1])
        max_muestra = min(2000, max(50, len(verb_f)))
        muestra = s1.slider("Comentarios a analizar", 50, max_muestra, min(300, max_muestra), step=25)
        batch = s2.selectbox("Tamaño de bloque", [16, 32, 64], index=1)
        avanzado = s3.toggle("Usar zero-shot avanzado (más lento)", value=False)
        firma = (linea, tipo, periodo, sucursal, muestra, batch, avanzado)
        if st.button("Ejecutar análisis de sentimiento", type="primary"):
            try:
                st.session_state["verb_nlp"] = nlp.enriquecer(verb_f, muestra=muestra, con_zero_shot=avanzado, batch_size=batch)
                st.session_state["verb_nlp_firma"] = firma
            except Exception as error:
                LOGGER.exception("Fallo el análisis de sentimiento")
                st.error("Falló el análisis de sentimiento. Reduce la muestra o revisa el modelo configurado.")
                st.exception(error)
        vnlp = st.session_state.get("verb_nlp") if st.session_state.get("verb_nlp_firma") == firma else None
        if vnlp is None:
            st.info("Ejecuta el análisis para ver polaridad, emoción y tema con la muestra actual.")
        else:
            dist = vnlp["polaridad"].fillna("Sin clasificar").value_counts()
            neg_pct = dist.get("Negativo", 0) / max(len(vnlp), 1) * 100
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                card("Positivos", f"{dist.get('Positivo', 0):,.0f}", None, "Comentarios", PALETA["verde"], icon_name="check")
            with k2:
                card("Neutros", f"{dist.get('Neutro', 0):,.0f}", None, "Comentarios", PALETA["amarillo"], icon_name="comentarios")
            with k3:
                card("Negativos", f"{dist.get('Negativo', 0):,.0f}", None, "Comentarios", PALETA["rojo"], icon_name="alerta")
            with k4:
                card("Tasa negativa", f"{neg_pct:.1f}%", None, "Del total analizado", PALETA["rojo"], icon_name="termometro")
            n1, n2 = st.columns([1, 1.4])
            with n1:
                st.plotly_chart(figura_donut_polaridad(vnlp), use_container_width=True)
            with n2:
                st.plotly_chart(figura_sunburst_sentimiento(vnlp), use_container_width=True)
            titulo_seccion("verbatims", "Temas más mencionados dentro de negativos")
            st.plotly_chart(figura_temas_negativos(vnlp), use_container_width=True)
            columnas_vista = [col for col in ["_texto", "polaridad", "emocion", "tema", "confianza"] if col in vnlp.columns]
            st.dataframe(vnlp[columnas_vista].head(200), use_container_width=True, hide_index=True)

vnlp_base = st.session_state.get("verb_nlp") if st.session_state.get("verb_nlp_firma", ())[:4] == (linea, tipo, periodo, sucursal) else None
base_datasets = {
    "KPIs por sucursal": aplicar_filtros(df_sucursal, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal),
    "Resumen": aplicar_filtros(df_base, linea=linea, tipo=tipo, periodo=periodo, sucursal="TODAS"),
    "Drivers": drv_f,
    "Verbatims": vnlp_base if vnlp_base is not None else verb_f,
}

with tab_base:
    titulo_seccion("dashboard", "Base general")
    dataset_nombre = st.selectbox("Dataset", list(base_datasets.keys()))
    dataset = base_datasets[dataset_nombre].copy()
    columnas = st.multiselect("Columnas", dataset.columns.tolist(), default=dataset.columns.tolist())
    busqueda = st.text_input("Buscar texto libre", placeholder="Escribe una palabra o frase")
    limite_default = min(500, max(50, len(dataset)))
    limite = st.number_input("Número de filas", min_value=50, max_value=5000, value=limite_default, step=50)
    vista = dataset[columnas] if columnas else dataset
    if busqueda:
        mask = vista.astype(str).apply(lambda serie: serie.str.contains(busqueda, case=False, na=False))
        vista = vista[mask.any(axis=1)]
    st.caption(f"Registros filtrados: {len(vista):,.0f}")
    st.dataframe(vista.head(int(limite)), use_container_width=True, hide_index=True)
    nombre_csv = nombre_archivo_base(dataset_nombre, periodo, linea, tipo, sucursal, "csv")
    nombre_xlsx = nombre_archivo_base(dataset_nombre, periodo, linea, tipo, sucursal, "xlsx")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Descargar CSV", data=dataframe_a_csv_bytes(vista), file_name=nombre_csv, mime="text/csv")
    with b2:
        st.download_button(
            "Descargar XLSX",
            data=dataframes_a_xlsx_bytes({dataset_nombre: vista}),
            file_name=nombre_xlsx,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with tab_informe:
    titulo_seccion("dashboard", "Informe automatizado")
    filtros = {
        "tipo": tipo,
        "período": periodo,
        "línea": linea,
        "sucursal": sucursal,
        "fuente_kpis": "resumen_por_sucursal" if df_fuente is df_sucursal else "resumen",
    }
    insights = insights_tablero(kpis, dfa, periodo_prev)
    tablas = {
        "comparativo_lineas": dfa,
        "drivers": drv_f,
        "sentimiento": vnlp_base if vnlp_base is not None else pd.DataFrame(),
        "insights": insights,
        "verbatims": vnlp_base if vnlp_base is not None else verb_f,
    }
    figuras = {
        "Evolución": figura_evolucion(dfh),
        "Composición NPS": figura_composicion_nps(dfa) if {"promotores", "neutros", "detractores"}.issubset(dfa.columns) else None,
        "Comparativo por línea": figura_radar_linea(dfa) if len(dfa.groupby("linea")["avg_ins"].mean().dropna()) >= 3 else None,
        "Drivers": figura_drivers_pareto(drv_f.groupby("categoria", as_index=False).size().rename(columns={"size": "menciones"})[["categoria", "menciones"]]) if not drv_f.empty and "categoria" in drv_f.columns else None,
        "Sentimiento": figura_sunburst_sentimiento(vnlp_base) if vnlp_base is not None and not vnlp_base.empty else None,
        "Negativos por tema": figura_temas_negativos(vnlp_base) if vnlp_base is not None and not vnlp_base.empty else None,
    }
    st.write("Portada: **Seguros del Estado · Centro de Experiencia CX**")
    st.write(f"Alcance actual: {filtros_texto}")
    st.dataframe(insights, use_container_width=True, hide_index=True)
    i1, i2 = st.columns(2)
    with i1:
        st.download_button(
            "Descargar informe XLSX",
            data=generar_informe_excel(filtros, kpis, tablas),
            file_name=nombre_archivo_base("informe", periodo, linea, tipo, sucursal, "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with i2:
        st.download_button(
            "Descargar informe HTML",
            data=generar_informe_html(filtros, kpis, tablas, figuras),
            file_name=nombre_archivo_base("informe", periodo, linea, tipo, sucursal, "html"),
            mime="text/html",
        )

with tab_copiloto:
    titulo_seccion("bot", "Copiloto CX")
    pregunta = st.text_area(
        "Pregúntale al copiloto sobre la experiencia actual",
        height=120,
        placeholder="Ej.: ¿Qué sucursales concentran más comentarios negativos?",
    )
    if st.button("Analizar con copiloto", type="primary"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta para analizar.")
        elif vnlp_base is None or "polaridad" not in vnlp_base.columns:
            st.warning("Ejecuta antes el análisis de sentimiento en la pestaña Voz del cliente.")
        else:
            with st.spinner("Analizando contexto…"):
                texto, tabla = responder(pregunta, vnlp_base, {**kpis, "filtros": filtros_texto})
            with st.container(border=True):
                st.markdown("<div class='cx-answer'>Respuesta del copiloto</div>", unsafe_allow_html=True)
                st.markdown(texto)
                if tabla is not None and not tabla.empty:
                    st.dataframe(tabla, use_container_width=True, hide_index=True)
