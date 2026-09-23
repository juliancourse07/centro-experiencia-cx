import logging

import pandas as pd
import streamlit as st

from utils import nlp
from utils.consultas import drivers, resumen, resumen_por_sucursal, tipos_encuesta_disponibles, verbatims
from utils.copiloto import responder
from utils.estilos import PALETA, aplicar_estilos
from utils.graficos import (
    figura_composicion_nps,
    figura_delta_lineas,
    figura_donut_polaridad,
    figura_dispersion_lineas,
    figura_drivers_matriz,
    figura_drivers_pareto,
    figura_evolucion,
    figura_gauge,
    figura_heatmap_ins,
    figura_radar_linea,
    figura_ranking_lineas,
    figura_riesgo_sucursal,
    figura_sunburst_sentimiento,
    figura_temas_negativos,
    figura_vacia,
)
from utils.iconos import icono, titulo_seccion
from utils.informe import (
    construir_figuras_informe,
    dataframe_a_csv_bytes,
    dataframes_a_xlsx_bytes,
    generar_informe_excel,
    generar_informe_html,
    nombre_archivo_base,
)
from utils.tablero import (
    aplicar_filtros,
    card,
    comparativo_lineas,
    construir_kpis,
    construir_opciones_tipo,
    deduplicar_columnas,
    diagnostico_cod_suc,
    diagnostico_tipo_periodo,
    diagnostico_tipos,
    es_intermediario,
    filtros_activos_texto,
    fmt,
    fuente_datos,
    insights_tablero,
    normalizar,
    normalizar_tipo,
    opciones_linea,
    opciones_periodo,
    riesgo_sucursal_por_indicador,
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


def obtener_nlp_vigente(
    linea: str,
    tipo: str,
    periodo: str,
    sucursal: str,
    muestra: int | None = None,
    batch: int | None = None,
    avanzado: bool | None = None,
) -> pd.DataFrame | None:
    """Devuelve el NLP en caché solo si coincide con los filtros vigentes."""
    firma = st.session_state.get("verb_nlp_firma", ())
    if tuple(firma[:4]) != (linea, tipo, periodo, sucursal):
        return None
    if None not in (muestra, batch, avanzado) and tuple(firma) != (linea, tipo, periodo, sucursal, muestra, batch, avanzado):
        return None
    return st.session_state.get("verb_nlp")


def mostrar_error_tab(nombre: str, error: Exception) -> None:
    """Muestra un error accionable sin exponer el traceback en abierto."""
    LOGGER.exception("Fallo renderizando %s", nombre)
    st.error(f"No fue posible renderizar {nombre.lower()} con los filtros seleccionados. Revisa el diagnóstico del sidebar o cambia el período.")
    with st.expander("Detalle técnico"):
        st.exception(error)


def descripcion_dataset(nombre: str) -> str:
    descripciones = {
        "KPIs por sucursal": "Una fila por mes × línea × tipo × sucursal × canal. Reagregada desde gold_cx_kpis; es la única base con apertura por sucursal.",
        "Resumen": "Agregado mensual oficial por línea y tipo desde gold_cx_resumen. Úsalo para seguimiento ejecutivo sin detalle de sucursal.",
        "Drivers": "Una fila por driver mencionado. Sirve para priorizar dolores y puntos de contacto más repetidos.",
        "Verbatims": "Una fila por comentario abierto. Incluye polaridad, emoción y tema cuando ya ejecutaste el análisis de sentimiento.",
    }
    return descripciones.get(nombre, "")


def lectura_comparativo(df: pd.DataFrame) -> str:
    """Construye una lectura ejecutiva automática del comparativo por línea."""
    if df is None or df.empty or "linea" not in df.columns:
        return "No hay suficientes datos para comparar líneas en el período seleccionado."
    detalle = df[df["linea"] != "TOTAL / Promedio compañía"].copy()
    total = df[df["linea"] == "TOTAL / Promedio compañía"]
    if detalle.empty or total.empty or "nps" not in detalle.columns:
        return "No hay suficientes datos para comparar líneas en el período seleccionado."
    lider = detalle.sort_values("nps", ascending=False, na_position="last").iloc[0]
    rezago = detalle.sort_values("nps", ascending=True, na_position="last").iloc[0]
    promedio = total["nps"].iloc[0]
    mejoro = detalle.dropna(subset=["delta_nps"]).sort_values("delta_nps", ascending=False).head(1)
    empeoro = detalle.dropna(subset=["delta_nps"]).sort_values("delta_nps", ascending=True).head(1)
    frases = [
        f"{lider['linea']} lidera el período con NPS {lider['nps']:.1f}, mientras {rezago['linea']} registra el menor desempeño con {rezago['nps']:.1f}.",
        f"La brecha entre ambas líneas es de {lider['nps'] - rezago['nps']:.1f} puntos y el promedio de la compañía se ubica en {promedio:.1f}.",
    ]
    if not mejoro.empty:
        fila = mejoro.iloc[0]
        frases.append(f"La mayor mejora frente al período anterior la presenta {fila['linea']} ({fila['delta_nps']:+.1f} pts).")
    if not empeoro.empty and empeoro.iloc[0]["delta_nps"] < 0:
        fila = empeoro.iloc[0]
        frases.append(f"La caída más fuerte la muestra {fila['linea']} ({fila['delta_nps']:+.1f} pts).")
    return " ".join(frases)


def diagnostico_vacio(tipo: str, fuente_label: str, df_base: pd.DataFrame, df_sucursal: pd.DataFrame, drv: pd.DataFrame, verb: pd.DataFrame) -> str:
    """Genera un mensaje accionable cuando un filtro queda sin datos."""
    conteos = {
        "gold_cx_resumen": 0 if df_base is None else len(df_base),
        "gold_cx_kpis": 0 if df_sucursal is None else len(df_sucursal),
        "gold_cx_drivers": 0 if drv is None else len(drv),
        "gold_cx_verbatims": 0 if verb is None else len(verb),
    }
    tipos_kpis = ", ".join(sorted(df_sucursal["tipo_encuesta"].dropna().astype(str).unique())) if df_sucursal is not None and not df_sucursal.empty and "tipo_encuesta" in df_sucursal.columns else "SIN DATOS"
    return (
        f"No hay datos para los filtros activos. Fuente usada: {fuente_label}. "
        f"Filas cargadas sin filtrar → resumen={conteos['gold_cx_resumen']:,}, kpis={conteos['gold_cx_kpis']:,}, "
        f"drivers={conteos['gold_cx_drivers']:,}, verbatims={conteos['gold_cx_verbatims']:,}. "
        f"Valores detectados de tipo_encuesta en gold_cx_kpis: {tipos_kpis}. "
        f"Si elegiste {tipo}, revisa el expander Diagnóstico del sidebar para validar períodos y literales disponibles."
    )


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
tipos_opciones = construir_opciones_tipo(df_base, df_sucursal, verb, tipos_maestros)
diagnostico_tablas = pd.concat(
    [
        diagnostico_tipos("gold_cx_resumen", df_base),
        diagnostico_tipos("gold_cx_kpis", df_sucursal),
        diagnostico_tipos("gold_cx_drivers", drv),
        diagnostico_tipos("gold_cx_verbatims", verb),
    ],
    ignore_index=True,
)
diagnostico_conteos = pd.concat(
    [
        diagnostico_tipo_periodo("gold_cx_resumen", df_base),
        diagnostico_tipo_periodo("gold_cx_kpis", df_sucursal),
        diagnostico_tipo_periodo("gold_cx_drivers", drv),
        diagnostico_tipo_periodo("gold_cx_verbatims", verb),
    ],
    ignore_index=True,
)

with st.sidebar:
    st.markdown(f"<h3 style='color:{PALETA['azul']};margin-bottom:0'>Filtros</h3>", unsafe_allow_html=True)
    tipo = st.selectbox(
        "Tipo de encuesta",
        tipos_opciones,
        key="f_tipo",
        help="Define el universo analizado. INTERMEDIARIO reconoce variantes históricas como asesor, corredor o agente.",
    )
    periodos = opciones_periodo(tipo, df_base, df_sucursal)
    if not periodos:
        periodos = sorted(valores_union(df_base, df_sucursal, drv, verb, columna="anio_mes"), reverse=True)
    if not periodos:
        st.error("No hay períodos disponibles en las fuentes consultadas.")
        st.stop()
    periodo_guardado = st.session_state.get("f_periodo")
    if periodo_guardado not in periodos:
        if periodo_guardado:
            st.info(f"El período {periodo_guardado} no tiene datos para {normalizar_tipo(tipo)}; se seleccionó {periodos[0]}.")
        st.session_state["f_periodo"] = periodos[0]
    periodo = st.selectbox("Período", periodos, key="f_periodo", help="Muestra solo períodos con datos para el tipo seleccionado.")

    opciones_comp = ["Sin comparación", *periodos]
    periodo_comp_guardado = st.session_state.get("f_periodo_comp", "Sin comparación")
    if periodo_comp_guardado not in opciones_comp:
        st.session_state["f_periodo_comp"] = "Sin comparación"
    periodo_comparativo = st.selectbox(
        "Período comparativo",
        opciones_comp,
        key="f_periodo_comp",
        help="Segundo período para comparar con el general. Si se deja en 'Sin comparación', la aplicación conserva el comportamiento actual.",
    )
    if periodo_comparativo == "Sin comparación":
        periodo_comparativo = None

    lineas = opciones_linea(tipo, periodo, df_base, df_sucursal)
    linea_guardada = st.session_state.get("f_linea", "TODAS")
    if linea_guardada not in lineas:
        st.session_state["f_linea"] = "TODAS"
    linea = st.selectbox("Línea", lineas, key="f_linea", help="Permite comparar una línea específica o todo el portafolio disponible para el tipo y período.")
    if es_intermediario(tipo):
        opciones_suc, conteos_suc, suc_historicas = sucursales_disponibles(df_sucursal, linea=linea, periodo=periodo)
        actual_suc = st.session_state.get("f_suc", "TODAS")
        if actual_suc not in opciones_suc:
            st.session_state["f_suc"] = "TODAS"
        sucursal = st.selectbox(
            "Sucursal",
            opciones_suc,
            key="f_suc",
            help="Sucursal disponible solo para intermediarios. Si el período no trae coincidencias, se muestran sucursales históricas del tipo.",
            format_func=lambda valor: (
                valor
                if valor == "TODAS"
                else f"{valor} — {conteos_suc.get(valor, 0):,} respuestas"
            ),
        )
        if suc_historicas:
            st.caption("Se muestran todas las sucursales históricas del tipo porque el período seleccionado no trae coincidencias.")
    else:
        st.session_state.pop("f_suc", None)
        sucursal = "TODAS"
    if st.button("Limpiar filtros", use_container_width=True):
        st.session_state.update(f_tipo="TODOS", f_periodo=periodos[0], f_linea="TODAS", f_periodo_comp="Sin comparación")
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
    with st.expander("Diagnóstico", expanded=False):
        st.caption("Solo lectura para depurar disponibilidad de datos sin acceso directo a Databricks.")
        st.dataframe(deduplicar_columnas(diagnostico_tablas), use_container_width=True, hide_index=True)
        st.dataframe(deduplicar_columnas(diagnostico_conteos.head(200)), use_container_width=True, hide_index=True)
        st.dataframe(deduplicar_columnas(diagnostico_cod_suc(df_sucursal)), use_container_width=True, hide_index=True)


df_fuente = fuente_datos(tipo, sucursal, df_base, df_sucursal)
dfa = aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
dfh = aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal, con_periodo=False)
idx = periodos.index(periodo)
periodo_prev = periodos[idx + 1] if idx + 1 < len(periodos) else None
periodo_comp_efectivo = periodo_comparativo if periodo_comparativo and periodo_comparativo != periodo else None
if not periodo_comp_efectivo:
    periodo_comp_efectivo = periodo_prev

dfp = (
    aplicar_filtros(df_fuente, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal, periodo_valor=periodo_comp_efectivo)
    if periodo_comp_efectivo
    else pd.DataFrame()
)
verb_f = aplicar_filtros(verb, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
drv_f = aplicar_filtros(drv, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal)
filtros_texto = filtros_activos_texto(linea, tipo, periodo, sucursal, periodo_comparativo)
kpis = construir_kpis(dfa, dfp, verb_f)
comparativo = comparativo_lineas(dfa, dfp)
if not comparativo.empty and not verb_f.empty and "linea" in verb_f.columns:
    verb_linea = verb_f.groupby("linea").size().reset_index(name="verbatims")
    comparativo = deduplicar_columnas(comparativo.merge(verb_linea, on="linea", how="left"))
    comparativo["verbatims"] = comparativo["verbatims"].fillna(comparativo["respuestas"] if "respuestas" in comparativo.columns else 0)
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
    st.info(
        diagnostico_vacio(
            tipo=tipo,
            fuente_label="gold_cx_kpis" if df_fuente is df_sucursal else "gold_cx_resumen",
            df_base=df_base,
            df_sucursal=df_sucursal,
            drv=drv,
            verb=verb,
        )
    )

vnlp_base = obtener_nlp_vigente(linea, tipo, periodo, sucursal)

tab_tablero, tab_drivers, tab_voz, tab_base, tab_informe, tab_copiloto = st.tabs(
    ["Tablero general", "Drivers", "Voz del cliente", "Base general", "Informe", "Copiloto CX"]
)

with tab_tablero:
    try:
        with st.expander("Glosario ejecutivo", expanded=False):
            st.markdown(
                "- **NPS**: saldo neto entre promotores y detractores. Alto ≥ 70, medio 50,1–70, bajo < 50.\n"
                "- **INS**: índice integral de satisfacción en escala 0–10. Alto ≥ 9, medio 7,1–8,9, bajo < 7.\n"
                "- **CES**: esfuerzo percibido; entre más bajo, mejor. Bajo esfuerzo < 2,5 y alto esfuerzo > 3,5."
            )
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
            st.plotly_chart(figura_evolucion(dfh), use_container_width=True)
        with col_dist:
            titulo_seccion("dashboard", "Composición NPS")
            st.plotly_chart(figura_composicion_nps(dfa), use_container_width=True)
        col_r, col_h = st.columns(2)
        with col_r:
            titulo_seccion("radar", "Comparativo por línea")
            radar = dfa.groupby("linea", as_index=False)["avg_ins"].mean().dropna() if {"linea", "avg_ins"}.issubset(dfa.columns) else pd.DataFrame()
            if len(radar) >= 3:
                st.plotly_chart(figura_radar_linea(dfa), use_container_width=True)
            elif not radar.empty:
                st.bar_chart(radar.set_index("linea"), color=PALETA["azul"], height=340)
            else:
                st.plotly_chart(figura_vacia(), use_container_width=True)
        with col_h:
            titulo_seccion("termometro", "INS por línea y tipo")
            st.plotly_chart(figura_heatmap_ins(dfa), use_container_width=True)

        if es_intermediario(tipo):
            riesgo_df = riesgo_sucursal_por_indicador(
                dfa,
                dfp if not dfp.empty else None,
                linea=linea,
                tipo=tipo,
                periodo_actual=periodo,
                periodo_previo=periodo_comparativo or periodo_prev,
            )
            if not riesgo_df.empty:
                titulo_seccion("alerta", "Riesgo por indicador por sucursal")
                st.caption(f"Comparación activa: {periodo} vs. {periodo_comparativo or (periodo_prev or 'sin comparación')}")
                r1, r2 = st.columns([1.2, 1.8])
                with r1:
                    st.plotly_chart(figura_riesgo_sucursal(riesgo_df), use_container_width=True)
                with r2:
                    st.dataframe(
                        riesgo_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "valor_actual": st.column_config.NumberColumn("Valor actual", format="%.2f"),
                            "meta": st.column_config.NumberColumn("Meta", format="%.2f"),
                            "variacion": st.column_config.NumberColumn("Variación", format="%+,.2f"),
                        },
                    )

        titulo_seccion("comparativo", "Comparativo entre líneas")
        st.markdown(lectura_comparativo(comparativo))
        st.dataframe(
            deduplicar_columnas(comparativo),
            use_container_width=True,
            hide_index=True,
            column_config={
                "nps": st.column_config.NumberColumn("NPS", format="%.1f"),
                "ins": st.column_config.NumberColumn("INS", format="%.2f"),
                "ces": st.column_config.NumberColumn("CES", format="%.2f"),
                "respuestas": st.column_config.NumberColumn("Respuestas", format="%d"),
                "pct_promotores": st.column_config.ProgressColumn("% Promotores", min_value=0, max_value=100, format="%.1f%%"),
                "pct_neutros": st.column_config.ProgressColumn("% Neutros", min_value=0, max_value=100, format="%.1f%%"),
                "pct_detractores": st.column_config.ProgressColumn("% Detractores", min_value=0, max_value=100, format="%.1f%%"),
                "delta_nps": st.column_config.NumberColumn("Δ NPS", format="%+.1f"),
                "delta_ins": st.column_config.NumberColumn("Δ INS", format="%+.2f"),
                "delta_ces": st.column_config.NumberColumn("Δ CES", format="%+.2f"),
            },
        )
        c_rank, c_disp = st.columns(2)
        with c_rank:
            st.plotly_chart(figura_ranking_lineas(comparativo), use_container_width=True)
        with c_disp:
            st.plotly_chart(figura_dispersion_lineas(comparativo), use_container_width=True)
        st.plotly_chart(figura_delta_lineas(comparativo), use_container_width=True)
        titulo_seccion("cerebro", "Insights automáticos")
        for _, fila in insights_tablero(kpis, dfa, periodo_prev).iterrows():
            clase = {"positivo": "", "alerta": "warn", "riesgo": "bad"}.get(fila["tipo"], "")
            nombre_icono = "check" if clase == "" else "alerta"
            st.markdown(
                f"<div class='insight {clase}'>{icono(nombre_icono, size=16)} {fila['detalle']}</div>",
                unsafe_allow_html=True,
            )
    except Exception as error:
        mostrar_error_tab("Tablero general", error)

with tab_drivers:
    try:
        titulo_seccion("drivers", "Drivers de experiencia")
        with st.expander("¿Cómo leer esta vista?", expanded=False):
            st.markdown(
                "- **Matriz de priorización**: arriba a la derecha están los temas con mayor impacto y más menciones; son la primera prioridad.\n"
                "- **Pareto 80/20**: identifica el pequeño grupo de drivers que concentra la mayoría de menciones para enfocar acciones."
            )
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
            st.dataframe(deduplicar_columnas(rank.sort_values("menciones", ascending=False)), use_container_width=True, hide_index=True)
    except Exception as error:
        mostrar_error_tab("Drivers", error)

with tab_voz:
    try:
        titulo_seccion("cerebro", "Voz del cliente · Análisis de sentimiento")
        with st.expander("¿Cómo interpretar polaridad, emoción y tema?", expanded=False):
            st.markdown(
                "- **Polaridad** resume si el comentario es positivo, neutro o negativo.\n"
                "- **Emoción** profundiza la intención dominante dentro de cada polaridad.\n"
                "- **Tema** identifica el asunto principal del comentario.\n"
                "- En el sunburst, cada anillo agrega detalle. **Score/confianza** indica qué tan seguro estuvo el modelo."
            )
        if verb_f.empty:
            sin_datos("No hay verbatims disponibles para analizar con los filtros activos.")
        else:
            s1, s2, s3 = st.columns([1, 1, 1])
            max_muestra = min(2000, max(50, len(verb_f)))
            muestra = s1.slider("Comentarios a analizar", 50, max_muestra, min(300, max_muestra), step=25, help="Controla cuántos comentarios abiertos se procesan en esta ejecución.")
            batch = s2.selectbox("Tamaño de bloque", [16, 32, 64], index=1, help="Entre mayor sea el bloque, más rápido procesa, pero usa más memoria.")
            avanzado = s3.toggle("Usar zero-shot avanzado (más lento)", value=False, help="Refina el tema con un clasificador adicional. Úsalo solo cuando necesites más detalle.")
            firma = (linea, tipo, periodo, sucursal, muestra, batch, avanzado)
            if st.button("Ejecutar análisis de sentimiento", type="primary"):
                try:
                    st.session_state["verb_nlp"] = nlp.enriquecer(verb_f, muestra=muestra, con_zero_shot=avanzado, batch_size=batch)
                    st.session_state["verb_nlp_firma"] = firma
                except Exception as error:
                    LOGGER.exception("Fallo el análisis de sentimiento")
                    st.error("Falló el análisis de sentimiento. Reduce la muestra o revisa el modelo configurado.")
                    with st.expander("Detalle técnico"):
                        st.exception(error)
            vnlp = obtener_nlp_vigente(linea, tipo, periodo, sucursal, muestra, batch, avanzado)
            if vnlp is None:
                st.info("Ejecuta el análisis para ver polaridad, emoción y tema con la muestra actual.")
            else:
                dist = vnlp["polaridad"].fillna("Sin clasificar").value_counts() if "polaridad" in vnlp.columns else pd.Series(dtype=int)
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
                st.dataframe(deduplicar_columnas(vnlp[columnas_vista].head(200)), use_container_width=True, hide_index=True)
    except Exception as error:
        mostrar_error_tab("Voz del cliente", error)

vnlp_base = obtener_nlp_vigente(linea, tipo, periodo, sucursal)
base_datasets = {
    "KPIs por sucursal": aplicar_filtros(df_sucursal, linea=linea, tipo=tipo, periodo=periodo, sucursal=sucursal),
    "Resumen": aplicar_filtros(df_base, linea=linea, tipo=tipo, periodo=periodo, sucursal="TODAS"),
    "Drivers": drv_f,
    "Verbatims": vnlp_base if vnlp_base is not None else verb_f,
}

with tab_base:
    try:
        titulo_seccion("dashboard", "Base general")
        with st.expander("¿Qué contiene cada base?", expanded=False):
            referencia = pd.DataFrame(
                [
                    {
                        "dataset": "KPIs por sucursal",
                        "grano": "Mes × línea × tipo × sucursal × canal",
                        "origen": "gold_cx_kpis",
                        "columnas clave": "anio_mes, linea, tipo_encuesta, cod_suc, avg_nps_score, avg_ins, avg_ces",
                        "cuándo usarla": descripcion_dataset("KPIs por sucursal"),
                    },
                    {
                        "dataset": "Resumen",
                        "grano": "Mes × línea × tipo",
                        "origen": "gold_cx_resumen",
                        "columnas clave": "anio_mes, linea, tipo_encuesta, encuestados, avg_nps_score, avg_ins, avg_ces",
                        "cuándo usarla": descripcion_dataset("Resumen"),
                    },
                    {
                        "dataset": "Drivers",
                        "grano": "Una fila por driver mencionado",
                        "origen": "gold_cx_drivers",
                        "columnas clave": "anio_mes, linea, tipo_encuesta, categoria, tipo_driver",
                        "cuándo usarla": descripcion_dataset("Drivers"),
                    },
                    {
                        "dataset": "Verbatims",
                        "grano": "Una fila por comentario abierto",
                        "origen": "gold_cx_verbatims",
                        "columnas clave": "anio_mes, linea, tipo_encuesta, cod_suc, texto, polaridad, emocion, tema",
                        "cuándo usarla": descripcion_dataset("Verbatims"),
                    },
                ]
            )
            st.dataframe(deduplicar_columnas(referencia), use_container_width=True, hide_index=True)
        dataset_nombre = st.selectbox("Dataset", list(base_datasets.keys()), help="Selecciona la base que quieres revisar o exportar.")
        st.info(descripcion_dataset(dataset_nombre))
        dataset = deduplicar_columnas(base_datasets[dataset_nombre].copy())
        columnas = st.multiselect("Columnas", dataset.columns.tolist(), default=dataset.columns.tolist(), help="Elige solo las columnas relevantes para la extracción.")
        busqueda = st.text_input("Buscar texto libre", placeholder="Escribe una palabra o frase")
        limite_default = min(500, max(50, len(dataset))) if len(dataset) else 50
        limite = st.number_input("Número de filas", min_value=50, max_value=5000, value=limite_default, step=50, help="Controla cuántas filas mostrar en pantalla.")
        vista = dataset[columnas] if columnas and not dataset.empty else dataset
        if busqueda and not vista.empty:
            mask = vista.astype(str).apply(lambda serie: serie.str.contains(busqueda, case=False, na=False))
            vista = vista[mask.any(axis=1)]
        vista = deduplicar_columnas(vista)
        st.caption(f"Registros filtrados: {len(vista):,.0f}")
        if vista.empty:
            sin_datos("El dataset seleccionado no tiene filas para los filtros actuales.")
        else:
            st.dataframe(deduplicar_columnas(vista.head(int(limite))), use_container_width=True, hide_index=True)
        nombre_csv = nombre_archivo_base(dataset_nombre, periodo, linea, tipo, sucursal, "csv")
        nombre_xlsx = nombre_archivo_base(dataset_nombre, periodo, linea, tipo, sucursal, "xlsx")
        b1, b2 = st.columns(2)
        with b1:
            st.download_button("Descargar CSV", data=dataframe_a_csv_bytes(vista), file_name=nombre_csv, mime="text/csv", disabled=vista.empty)
        with b2:
            st.download_button(
                "Descargar XLSX",
                data=dataframes_a_xlsx_bytes({dataset_nombre: vista}),
                file_name=nombre_xlsx,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=vista.empty,
            )
    except Exception as error:
        mostrar_error_tab("Base general", error)

with tab_informe:
    try:
        titulo_seccion("dashboard", "Informe automatizado")
        with st.expander("¿Qué incluye cada exportable?", expanded=False):
            st.markdown(
                "- **XLSX**: metadatos, KPIs, comparativo entre líneas, drivers, insights, sentimiento y verbatims filtrados.\n"
                "- **HTML**: portada ejecutiva, KPIs, gráficos visibles y tablas resumidas listas para imprimir o convertir a PDF."
            )
        filtros = {
            "tipo": tipo,
            "período": periodo,
            "línea": linea,
            "sucursal": sucursal,
            "fuente_kpis": "resumen_por_sucursal" if df_fuente is df_sucursal else "resumen",
            "comparativo": periodo_comparativo or (periodo_prev or "sin comparación"),
        }
        insights = deduplicar_columnas(insights_tablero(kpis, dfa, periodo_prev))
        tablas = {
            "comparativo_lineas": deduplicar_columnas(comparativo),
            "drivers": deduplicar_columnas(drv_f),
            "sentimiento": deduplicar_columnas(vnlp_base if vnlp_base is not None else pd.DataFrame()),
            "insights": insights,
            "verbatims": deduplicar_columnas(vnlp_base if vnlp_base is not None else verb_f),
        }
        figuras = construir_figuras_informe(dfh, comparativo, dfa, drv_f, vnlp_base)
        st.write("Portada: **Seguros del Estado · Centro de Experiencia CX**")
        st.write(f"Alcance actual: {filtros_texto}")
        st.dataframe(deduplicar_columnas(insights), use_container_width=True, hide_index=True)
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
    except Exception as error:
        mostrar_error_tab("Informe", error)

with tab_copiloto:
    try:
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
                        st.dataframe(deduplicar_columnas(tabla), use_container_width=True, hide_index=True)
    except Exception as error:
        mostrar_error_tab("Copiloto CX", error)
