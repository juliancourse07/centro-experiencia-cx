"""
Consultas a la capa Gold del Centro de Experiencia CX.

Tablas Gold:
  gold_cx_kpis      -> 1 fila por encuesta (base maestra)
  gold_cx_resumen   -> agregado anio_mes x tipo_encuesta x linea
  gold_cx_drivers   -> 1 fila por driver mencionado
  gold_cx_verbatims -> 1 fila por comentario

Notas de diseño:
  - gold_cx_resumen NO expone cod_suc. Para filtrar por sucursal se usa
    resumen_por_sucursal(), que reagrega gold_cx_kpis en vivo.
  - drivers y verbatims traen `fecha`, no `anio_mes`: se deriva en SQL
    para que el filtro de período del tablero funcione igual en todas las tablas.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from utils.warehouse import query

CATALOGO = os.getenv("CX_CATALOGO", "hive_metastore")
ESQUEMA = os.getenv("CX_ESQUEMA", "desarrollo_dmvicecomhechos")

T_KPIS = f"{CATALOGO}.{ESQUEMA}.gold_cx_kpis"
T_RESUMEN = f"{CATALOGO}.{ESQUEMA}.gold_cx_resumen"
T_DRIVERS = f"{CATALOGO}.{ESQUEMA}.gold_cx_drivers"
T_VERBATIMS = f"{CATALOGO}.{ESQUEMA}.gold_cx_verbatims"

TTL = 900
EXPR_FECHA = """
COALESCE(
    TRY_TO_DATE(TRIM({col}), 'M/d/yy'),
    TRY_TO_DATE(TRIM({col}), 'yyyy-MM-dd'),
    TRY_CAST({col} AS DATE)
)
""".strip()


def expr_fecha(columna: str = "fecha") -> str:
    """Devuelve la expresión SQL robusta para parsear fechas."""
    return EXPR_FECHA.format(col=columna)


@st.cache_data(ttl=TTL, show_spinner=False)
def tipos_encuesta_disponibles() -> list[str]:
    """Consulta los tipos de encuesta disponibles desde la fuente maestra."""
    df = query(f"""
        SELECT DISTINCT UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta
        FROM {T_KPIS}
        WHERE tipo_encuesta IS NOT NULL
          AND TRIM(tipo_encuesta) <> ''
        ORDER BY 1
    """)
    if df is None or df.empty or "tipo_encuesta" not in df.columns:
        return []
    return df["tipo_encuesta"].dropna().astype(str).str.strip().tolist()


@st.cache_data(ttl=TTL, show_spinner=False)
def resumen() -> pd.DataFrame:
    """Agregado mensual por línea y tipo de encuesta."""
    return query(f"""
        SELECT
            CAST(anio  AS INT)        AS anio,
            CAST(mes   AS INT)        AS mes,
            CAST(anio_mes AS STRING)  AS anio_mes,
            UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta,
            UPPER(TRIM(linea))         AS linea,
            CAST(encuestados   AS BIGINT) AS encuestados,
            CAST(avg_nps_score AS DOUBLE) AS avg_nps_score,
            CAST(avg_ins       AS DOUBLE) AS avg_ins,
            CAST(avg_ces       AS DOUBLE) AS avg_ces,
            CAST(promotores    AS BIGINT) AS promotores,
            CAST(neutros       AS BIGINT) AS neutros,
            CAST(detractores   AS BIGINT) AS detractores
        FROM {T_RESUMEN}
        WHERE anio_mes IS NOT NULL
        ORDER BY anio_mes DESC, tipo_encuesta, linea
    """)


@st.cache_data(ttl=TTL, show_spinner="Agregando por sucursal…")
def resumen_por_sucursal() -> pd.DataFrame:
    """Reagrega KPIs abriendo sucursal para soportar el filtro de intermediarios."""
    return query(f"""
        SELECT
            CAST(anio AS INT)         AS anio,
            CAST(mes  AS INT)         AS mes,
            CAST(anio_mes AS STRING)  AS anio_mes,
            UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta,
            UPPER(TRIM(linea))         AS linea,
            COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)), ''), 'SIN SUCURSAL') AS cod_suc,
            COALESCE(NULLIF(TRIM(canal), ''), 'SIN CANAL') AS canal,
            COUNT(*) AS encuestados,
            AVG(CAST(nps_score AS DOUBLE)) AS avg_nps_score,
            AVG(CAST(ins_score AS DOUBLE)) AS avg_ins,
            AVG(CAST(ces_promedio AS DOUBLE)) AS avg_ces,
            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'PROMOTOR'  THEN 1 ELSE 0 END) AS promotores,
            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'NEUTRO'    THEN 1 ELSE 0 END) AS neutros,
            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'DETRACTOR' THEN 1 ELSE 0 END) AS detractores
        FROM {T_KPIS}
        WHERE anio_mes IS NOT NULL
        GROUP BY anio, mes, anio_mes, tipo_encuesta, linea, cod_suc, canal
        ORDER BY anio_mes DESC, linea, cod_suc
    """)


@st.cache_data(ttl=TTL, show_spinner=False)
def drivers() -> pd.DataFrame:
    """Consulta drivers con fecha parseada y anio_mes derivado."""
    fecha_expr = expr_fecha()
    return query(f"""
        WITH base AS (
            SELECT *, {fecha_expr} AS fecha_parsed
            FROM {T_DRIVERS}
        )
        SELECT
            fecha_parsed AS fecha,
            DATE_FORMAT(fecha_parsed, 'yyyy-MM') AS anio_mes,
            UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta,
            UPPER(TRIM(linea)) AS linea,
            UPPER(TRIM(tipo_driver)) AS tipo_driver,
            INITCAP(TRIM(categoria)) AS categoria,
            CASE WHEN UPPER(tipo_driver) LIKE 'DOLOR%'
                 THEN 'DOLOR' ELSE 'PUNTO_CONTACTO' END AS familia,
            CASE WHEN UPPER(tipo_driver) LIKE '%CES'
                 THEN 'CES' ELSE 'NPS' END AS metrica
        FROM base
        WHERE categoria IS NOT NULL
          AND TRIM(categoria) <> ''
          AND UPPER(TRIM(categoria)) NOT IN ('N/A', 'NA', 'NULL', 'NINGUNO', 'SIN DATO')
          AND fecha_parsed IS NOT NULL
    """)


@st.cache_data(ttl=TTL, show_spinner=False)
def verbatims(limite: int = 20000) -> pd.DataFrame:
    """Consulta comentarios abiertos con filtros homogéneos para el tablero."""
    fecha_expr = expr_fecha()
    return query(f"""
        WITH base AS (
            SELECT *, {fecha_expr} AS fecha_parsed
            FROM {T_VERBATIMS}
        )
        SELECT
            fecha_parsed AS fecha,
            DATE_FORMAT(fecha_parsed, 'yyyy-MM') AS anio_mes,
            UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta,
            UPPER(TRIM(linea)) AS linea,
            documento,
            TRIM(nombre_completo) AS nombre_completo,
            COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)), ''), 'SIN SUCURSAL') AS cod_suc,
            COALESCE(NULLIF(TRIM(canal), ''), 'SIN CANAL') AS canal,
            INITCAP(TRIM(nps_categoria)) AS nps_categoria,
            UPPER(TRIM(tipo_comentario)) AS tipo_comentario,
            TRIM(texto) AS texto,
            CASE WHEN UPPER(TRIM(tipo_encuesta)) = 'INTERMEDIARIO'
                 THEN COALESCE(NULLIF(TRIM(nombre_completo), ''), CAST(documento AS STRING), 'SIN NOMBRE')
                 ELSE NULL END AS intermediario,
            CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR' THEN 1 ELSE 0 END AS es_negativo
        FROM base
        WHERE texto IS NOT NULL
          AND LENGTH(TRIM(texto)) > 3
          AND UPPER(TRIM(texto)) NOT IN ('N/A','NA','NINGUNO','NINGUNA','NO','SIN COMENTARIO','.','-')
          AND fecha_parsed IS NOT NULL
        ORDER BY fecha_parsed DESC
        LIMIT {int(limite)}
    """)


@st.cache_data(ttl=TTL, show_spinner=False)
def ranking_negativos_sql(
    dimension: str = "intermediario",
    anio_mes: str | None = None,
    linea: str | None = None,
    minimo: int = 5,
    top: int = 20,
) -> pd.DataFrame:
    """Calcula el ranking de negativos directamente en el warehouse."""
    permitidas = {
        "intermediario": "COALESCE(NULLIF(TRIM(nombre_completo),''), CAST(documento AS STRING))",
        "cod_suc": "COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)),''), 'SIN SUCURSAL')",
        "canal": "COALESCE(NULLIF(TRIM(canal),''), 'SIN CANAL')",
        "linea": "UPPER(TRIM(linea))",
    }
    if dimension not in permitidas:
        raise ValueError(f"Dimensión no permitida: {dimension}")
    expr = permitidas[dimension]
    filtros = ["texto IS NOT NULL", "LENGTH(TRIM(texto)) > 3"]
    if dimension == "intermediario":
        filtros.append("UPPER(TRIM(tipo_encuesta)) = 'INTERMEDIARIO'")
    fecha_expr = expr_fecha()
    filtros.append("fecha_parsed IS NOT NULL")
    if anio_mes:
        filtros.append(f"DATE_FORMAT(fecha_parsed, 'yyyy-MM') = '{anio_mes}'")
    if linea and linea.upper() != "TODAS":
        filtros.append(f"UPPER(TRIM(linea)) = '{linea.upper()}'")
    where = " AND ".join(filtros)
    return query(f"""
        WITH base AS (
            SELECT *, {fecha_expr} AS fecha_parsed
            FROM {T_VERBATIMS}
        )
        SELECT
            {expr} AS {dimension},
            COUNT(*) AS total,
            SUM(CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR' THEN 1 ELSE 0 END) AS negativos,
            ROUND(SUM(CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS porcentaje_negativos
        FROM base
        WHERE {where}
        GROUP BY {expr}
        HAVING COUNT(*) >= {int(minimo)}
        ORDER BY negativos DESC, porcentaje_negativos DESC, total DESC
        LIMIT {int(top)}
    """)


@st.cache_data(ttl=TTL, show_spinner=False)
def diagnostico() -> pd.DataFrame:
    """Resume parseabilidad de fechas y conteos de tablas Gold."""
    consultas = {
        "gold_cx_resumen": "anio_mes IS NOT NULL",
        "gold_cx_kpis": "anio_mes IS NOT NULL",
        "gold_cx_drivers": expr_fecha(),
        "gold_cx_verbatims": expr_fecha(),
    }
    filas: list[dict[str, object]] = []
    for tabla, condicion in consultas.items():
        if "(" in condicion:
            sql = f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN {condicion} IS NOT NULL THEN 1 ELSE 0 END) AS fecha_parseable,
                    SUM(CASE WHEN {condicion} IS NULL THEN 1 ELSE 0 END) AS fecha_no_parseable
                FROM {CATALOGO}.{ESQUEMA}.{tabla}
            """
        else:
            sql = f"""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) AS fecha_parseable,
                    0 AS fecha_no_parseable
                FROM {CATALOGO}.{ESQUEMA}.{tabla}
                WHERE {condicion}
            """
        resultado = query(sql)
        registro = resultado.iloc[0].to_dict() if resultado is not None and not resultado.empty else {}
        filas.append({"tabla": tabla, **registro})
    return pd.DataFrame(filas)
