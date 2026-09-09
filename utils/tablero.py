"""Helpers puros para filtros, métricas y selección de fuentes del tablero."""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

from utils.estilos import BANDAS, PALETA
from utils.iconos import texto_icono

NUM_COLS = [
    "encuestados",
    "avg_nps_score",
    "avg_ins",
    "avg_ces",
    "promotores",
    "neutros",
    "detractores",
]
COLS_INTERMEDIARIO = ("intermediario", "nombre_intermediario", "cod_intermediario")
COLS_SUCURSAL = ("cod_suc", "sucursal", "codigo_sucursal")


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte una serie a numérico tolerando comas, porcentajes y vacíos."""
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
    """Normaliza tipos de datos y limpia categorías clave."""
    if df is None or df.empty:
        return df
    salida = df.copy()
    for columna in NUM_COLS:
        if columna in salida.columns:
            salida[columna] = a_numero(salida[columna])
    for columna in ("linea", "tipo_encuesta", "anio_mes", *COLS_SUCURSAL, *COLS_INTERMEDIARIO):
        if columna in salida.columns:
            salida[columna] = salida[columna].astype(str).str.strip()
    return salida


def primera_col(df: pd.DataFrame, candidatas: Sequence[str]) -> str | None:
    """Devuelve la primera columna disponible dentro de una lista de candidatas."""
    if df is None or getattr(df, "empty", True):
        return None
    return next((columna for columna in candidatas if columna in df.columns), None)


def _normalizar_valores(df: pd.DataFrame, columna: str) -> set[str]:
    if df is None or df.empty or columna not in df.columns:
        return set()
    serie = df[columna].dropna().astype(str).str.strip().str.upper()
    return {valor for valor in serie if valor and valor not in {"NAN", "NONE"}}


def construir_opciones_tipo(
    df_resumen: pd.DataFrame,
    df_sucursal: pd.DataFrame,
    df_verbatims: pd.DataFrame,
    tipos_base: Iterable[str] | None = None,
) -> list[str]:
    """Construye el selector de tipo de encuesta a partir de todas las fuentes."""
    valores = set(valor.strip().upper() for valor in (tipos_base or []) if str(valor).strip())
    valores |= _normalizar_valores(df_resumen, "tipo_encuesta")
    valores |= _normalizar_valores(df_sucursal, "tipo_encuesta")
    valores |= _normalizar_valores(df_verbatims, "tipo_encuesta")
    preferidos = [valor for valor in ("CLIENTE", "INTERMEDIARIO") if valor in valores]
    restantes = sorted(valor for valor in valores if valor not in set(preferidos))
    return ["TODOS", *preferidos, *restantes]


def ordenar_sucursales(valores: Iterable[str]) -> list[str]:
    """Ordena códigos de sucursal priorizando orden numérico cuando aplica."""
    unicos = {str(valor).strip() for valor in valores if str(valor).strip()}

    def clave(valor: str) -> tuple[int, int | float, str]:
        return (0, int(valor), valor) if valor.isdigit() else (1, float("inf"), valor)

    return sorted(unicos, key=clave)


def sucursales_disponibles(
    df_sucursal: pd.DataFrame,
    linea: str,
    periodo: str,
) -> list[str]:
    """Devuelve sucursales válidas para intermediarios con filtros activos."""
    if df_sucursal is None or df_sucursal.empty:
        return ["TODAS"]
    columna = primera_col(df_sucursal, COLS_SUCURSAL)
    if not columna:
        return ["TODAS"]
    datos = df_sucursal.copy()
    if "tipo_encuesta" in datos.columns:
        datos = datos[datos["tipo_encuesta"] == "INTERMEDIARIO"]
    if linea != "TODAS" and "linea" in datos.columns:
        datos = datos[datos["linea"] == linea]
    if periodo and "anio_mes" in datos.columns:
        datos = datos[datos["anio_mes"] == periodo]
    return ["TODAS", *ordenar_sucursales(datos[columna].dropna().astype(str).tolist())]


def fuente_datos(
    tipo: str,
    sucursal: str,
    df_resumen: pd.DataFrame,
    df_resumen_sucursal: pd.DataFrame,
) -> pd.DataFrame:
    """Selecciona una única fuente base para KPIs y comparativos.

    Usa resumen por sucursal cuando se filtra una sucursal concreta o cuando el
    tipo INTERMEDIARIO no está presente en el resumen agregado mensual.
    """
    if sucursal != "TODAS" and df_resumen_sucursal is not None and not df_resumen_sucursal.empty:
        return df_resumen_sucursal
    if tipo == "INTERMEDIARIO" and df_resumen_sucursal is not None and not df_resumen_sucursal.empty:
        disponibles = _normalizar_valores(df_resumen, "tipo_encuesta")
        if "INTERMEDIARIO" not in disponibles:
            return df_resumen_sucursal
    return df_resumen


def aplicar_filtros(
    df: pd.DataFrame,
    linea: str = "TODAS",
    tipo: str = "TODOS",
    periodo: str | None = None,
    sucursal: str = "TODAS",
    con_periodo: bool = True,
    periodo_valor: str | None = None,
) -> pd.DataFrame:
    """Aplica los filtros de tablero a un dataframe homogéneo."""
    if df is None or df.empty:
        return df
    salida = df.copy()
    if linea != "TODAS" and "linea" in salida.columns:
        salida = salida[salida["linea"] == linea]
    if tipo != "TODOS" and "tipo_encuesta" in salida.columns:
        salida = salida[salida["tipo_encuesta"] == tipo]
    if con_periodo and periodo and "anio_mes" in salida.columns:
        salida = salida[salida["anio_mes"] == (periodo_valor or periodo)]
    if sucursal != "TODAS":
        columna = primera_col(salida, COLS_SUCURSAL)
        if columna:
            salida = salida[salida[columna].astype(str) == sucursal]
    return salida


def calcular_nps(df: pd.DataFrame) -> float | None:
    """Calcula NPS real con conteos o usa el promedio como respaldo."""
    if df is None or df.empty:
        return None
    if {"promotores", "detractores", "encuestados"}.issubset(df.columns):
        base = df["encuestados"].sum(skipna=True)
        if base and base > 0:
            return round((df["promotores"].sum(skipna=True) - df["detractores"].sum(skipna=True)) / base * 100, 1)
    if "avg_nps_score" in df.columns:
        valor = df["avg_nps_score"].mean(skipna=True)
        return round(valor, 1) if pd.notna(valor) else None
    return None


def promedio_ponderado(df: pd.DataFrame, columna: str) -> float | None:
    """Calcula un promedio ponderado por encuestados cuando es posible."""
    if df is None or df.empty or columna not in df.columns:
        return None
    cols = [columna] + (["encuestados"] if "encuestados" in df.columns else [])
    sub = df[cols].dropna()
    if sub.empty:
        return None
    if "encuestados" in sub.columns and sub["encuestados"].sum() > 0:
        return round((sub[columna] * sub["encuestados"]).sum() / sub["encuestados"].sum(), 2)
    return round(sub[columna].mean(), 2)


def semaforo(metrica: str, valor: float | None) -> tuple[str, str]:
    """Clasifica una métrica según las bandas oficiales."""
    if valor is None or pd.isna(valor):
        return "Sin dato", PALETA["gris"]
    config = BANDAS[metrica]
    if config["invertido"]:
        if valor <= config["bajo"]:
            return "Bajo esfuerzo", PALETA["verde"]
        if valor <= config["alto"]:
            return "Esfuerzo medio", PALETA["amarillo"]
        return "Alto esfuerzo", PALETA["rojo"]
    if valor >= config["alto"]:
        return "Alto", PALETA["verde"]
    if valor > config["bajo"]:
        return "Medio", PALETA["amarillo"]
    return "Bajo", PALETA["rojo"]


def fmt(valor: float | int | None, decimales: int = 1) -> str:
    """Formatea números para visualización."""
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.{decimales}f}"


def delta(actual: float | int | None, previo: float | int | None) -> float | None:
    """Calcula la variación absoluta entre dos métricas."""
    if actual is None or previo is None or pd.isna(actual) or pd.isna(previo):
        return None
    return actual - previo


def filtros_activos_texto(linea: str, tipo: str, periodo: str, sucursal: str) -> str:
    """Convierte los filtros activos en una cadena legible."""
    filtros = [f"línea={linea}", f"tipo={tipo}", f"período={periodo}"]
    if sucursal != "TODAS":
        filtros.append(f"sucursal={sucursal}")
    return ", ".join(filtros)


def card(
    titulo: str,
    valor: str,
    delta_val: float | None = None,
    pie: str = "",
    color_borde: str | None = None,
    sufijo_delta: str = "vs. período anterior",
    icon_name: str | None = None,
) -> None:
    """Renderiza una tarjeta KPI consistente con la identidad visual."""
    titulo_html = texto_icono(icon_name, titulo, size=16) if icon_name else escape(titulo)
    if delta_val is None or pd.isna(delta_val):
        bloque = "<div class='kpi-delta' style='color:#9AA3B2'>— sin comparativo</div>"
    else:
        arriba = delta_val >= 0
        color = PALETA["verde"] if arriba else PALETA["rojo"]
        flecha = "▲" if arriba else "▼"
        bloque = (
            f"<div class='kpi-delta' style='color:{color}'>{flecha} {abs(delta_val):,.1f} "
            f"<span style='color:#9AA3B2;font-weight:400'>{sufijo_delta}</span></div>"
        )
    st.markdown(
        f"<div class='kpi' style='border-top-color:{color_borde or PALETA['naranja']}'>"
        f"<div class='kpi-title'>{titulo_html}</div>"
        f"<div class='kpi-value'>{valor}</div>{bloque}"
        f"<div class='kpi-foot'>{pie}</div></div>",
        unsafe_allow_html=True,
    )



def sin_datos(mensaje: str) -> None:
    """Muestra un mensaje consistente cuando una sección no tiene datos."""
    from utils.iconos import icono

    st.markdown(
        f"<div class='insight warn'>{icono('alerta', size=16)} {mensaje}</div>",
        unsafe_allow_html=True,
    )



def valores_union(*dataframes: pd.DataFrame, columna: str) -> list[str]:
    """Une valores no nulos de una columna presentes en varios dataframes."""
    valores: set[str] = set()
    for df in dataframes:
        if df is not None and not df.empty and columna in df.columns:
            serie = df[columna].dropna().astype(str).str.strip()
            valores |= {valor for valor in serie if valor and valor not in {"nan", "None"}}
    return sorted(valores)



def construir_kpis(actual: pd.DataFrame, previo: pd.DataFrame, verb_filtrado: pd.DataFrame) -> dict[str, object]:
    """Resume KPIs y variaciones para el período activo."""
    nps_actual = calcular_nps(actual)
    ins_actual = promedio_ponderado(actual, "avg_ins")
    ces_actual = promedio_ponderado(actual, "avg_ces")
    nps_prev = calcular_nps(previo)
    ins_prev = promedio_ponderado(previo, "avg_ins")
    ces_prev = promedio_ponderado(previo, "avg_ces")
    respuestas = int(actual["encuestados"].sum()) if not actual.empty and "encuestados" in actual.columns else 0
    respuestas_prev = int(previo["encuestados"].sum()) if not previo.empty and "encuestados" in previo.columns else None
    estado_nps, color_nps = semaforo("NPS", nps_actual)
    estado_ins, color_ins = semaforo("INS", ins_actual)
    estado_ces, color_ces = semaforo("CES", ces_actual)
    return {
        "nps": nps_actual,
        "ins": ins_actual,
        "ces": ces_actual,
        "respuestas": respuestas,
        "verbatims": len(verb_filtrado),
        "delta_nps": delta(nps_actual, nps_prev),
        "delta_ins": delta(ins_actual, ins_prev),
        "delta_ces": delta(ces_actual, ces_prev),
        "delta_respuestas": delta(respuestas, respuestas_prev),
        "estado_nps": estado_nps,
        "estado_ins": estado_ins,
        "estado_ces": estado_ces,
        "color_nps": color_nps,
        "color_ins": color_ins,
        "color_ces": color_ces,
    }



def insights_tablero(kpis: dict[str, object], actual: pd.DataFrame, periodo_prev: str | None) -> pd.DataFrame:
    """Genera insights automáticos exportables del tablero."""
    hallazgos: list[dict[str, str]] = []
    serie_ins = (
        actual.groupby("linea")["avg_ins"].mean().dropna()
        if not actual.empty and "avg_ins" in actual.columns
        else pd.Series(dtype=float)
    )
    if not serie_ins.empty:
        hallazgos.append({"tipo": "positivo", "detalle": f"La línea con mejor experiencia es {serie_ins.idxmax()} (INS {serie_ins.max():.2f})."})
        if len(serie_ins) > 1:
            hallazgos.append({"tipo": "alerta", "detalle": f"La mayor oportunidad está en {serie_ins.idxmin()} (INS {serie_ins.min():.2f})."})
    if kpis["delta_nps"] is not None and periodo_prev:
        verbo = "subió" if kpis["delta_nps"] >= 0 else "cayó"
        hallazgos.append({
            "tipo": "alerta" if kpis["delta_nps"] < 0 else "positivo",
            "detalle": f"El NPS {verbo} {kpis['delta_nps']:+.1f} puntos frente a {periodo_prev}.",
        })
    if kpis["ces"] is not None and kpis["ces"] > 3.5:
        hallazgos.append({"tipo": "riesgo", "detalle": f"El CES ({kpis['ces']:.2f}) está en zona de alto esfuerzo."})
    hallazgos.append({"tipo": "positivo", "detalle": f"Se analizaron {kpis['respuestas']:,.0f} respuestas y {kpis['verbatims']:,.0f} verbatims."})
    return pd.DataFrame(hallazgos)
