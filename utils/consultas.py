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

import os

import pandas as pd
import streamlit as st

from utils.warehouse import query  # conector OAuth existente

# =====================================================
# CONFIGURACIÓN DE TABLAS
# =====================================================

CATALOGO = os.getenv("CX_CATALOGO", "hive_metastore")
ESQUEMA = os.getenv("CX_ESQUEMA", "desarrollo_dmvicecomhechos")

T_KPIS = f"{CATALOGO}.{ESQUEMA}.gold_cx_kpis"
T_RESUMEN = f"{CATALOGO}.{ESQUEMA}.gold_cx_resumen"
T_DRIVERS = f"{CATALOGO}.{ESQUEMA}.gold_cx_drivers"
T_VERBATIMS = f"{CATALOGO}.{ESQUEMA}.gold_cx_verbatims"

TTL = 900  # 15 min


# =====================================================
# 1. RESUMEN (agregado mensual)
# =====================================================

@st.cache_data(ttl=TTL, show_spinner=False)
def resumen() -> pd.DataFrame:
    """
    Agregado mensual por línea y tipo de encuesta.
    Fuente del 80% del tablero: KPIs, gauges, evolución, radar, heatmap.
    NO contiene cod_suc.
    """
    return query(f"""
        SELECT
            CAST(anio  AS INT)          AS anio,
            CAST(mes   AS INT)          AS mes,
            CAST(anio_mes AS STRING)    AS anio_mes,
            UPPER(TRIM(tipo_encuesta))  AS tipo_encuesta,
            UPPER(TRIM(linea))          AS linea,
            CAST(encuestados   AS BIGINT)  AS encuestados,
            CAST(avg_nps_score AS DOUBLE)  AS avg_nps_score,
            CAST(avg_ins       AS DOUBLE)  AS avg_ins,
            CAST(avg_ces       AS DOUBLE)  AS avg_ces,
            CAST(promotores    AS BIGINT)  AS promotores,
            CAST(neutros       AS BIGINT)  AS neutros,
            CAST(detractores   AS BIGINT)  AS detractores
        FROM {T_RESUMEN}
        WHERE anio_mes IS NOT NULL
        ORDER BY anio_mes DESC, tipo_encuesta, linea
    """)


# =====================================================
# 2. RESUMEN POR SUCURSAL
# =====================================================

@st.cache_data(ttl=TTL, show_spinner="Agregando por sucursal…")
def resumen_por_sucursal() -> pd.DataFrame:
    """
    Mismo grano que resumen() pero abriendo cod_suc y canal.
    Reagrega gold_cx_kpis en vivo porque gold_cx_resumen no tiene sucursal.

    RECOMENDACIÓN: materializa esto como gold_cx_resumen_suc en tu notebook
    de Gold para evitar el costo de reagregar en cada consulta.
    """
    return query(f"""
        SELECT
            CAST(anio AS INT)            AS anio,
            CAST(mes  AS INT)            AS mes,
            CAST(anio_mes AS STRING)     AS anio_mes,
            UPPER(TRIM(tipo_encuesta))   AS tipo_encuesta,
            UPPER(TRIM(linea))           AS linea,
            COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)), ''), 'SIN SUCURSAL') AS cod_suc,
            COALESCE(NULLIF(TRIM(canal), ''), 'SIN CANAL')                      AS canal,

            COUNT(*)                     AS encuestados,
            AVG(CAST(nps_score    AS DOUBLE)) AS avg_nps_score,
            AVG(CAST(ins_score    AS DOUBLE)) AS avg_ins,
            AVG(CAST(ces_promedio AS DOUBLE)) AS avg_ces,

            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'PROMOTOR'  THEN 1 ELSE 0 END) AS promotores,
            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'NEUTRO'    THEN 1 ELSE 0 END) AS neutros,
            SUM(CASE WHEN UPPER(TRIM(nps_categoria)) = 'DETRACTOR' THEN 1 ELSE 0 END) AS detractores
        FROM {T_KPIS}
        WHERE anio_mes IS NOT NULL
        GROUP BY anio, mes, anio_mes, tipo_encuesta, linea, cod_suc, canal
        ORDER BY anio_mes DESC, linea, cod_suc
    """)


# =====================================================
# 3. DRIVERS
# =====================================================

@st.cache_data(ttl=TTL, show_spinner=False)
def drivers() -> pd.DataFrame:
    """
    Dolores y puntos de contacto.
    Deriva anio_mes desde fecha y descompone tipo_driver en:
      familia -> DOLOR | PUNTO_CONTACTO
      metrica -> NPS | CES
    Eso habilita la matriz Dolor vs. Punto de contacto y el Pareto.
    """
    return query(f"""
        SELECT
            CAST(fecha AS DATE)                                AS fecha,
            DATE_FORMAT(CAST(fecha AS DATE), 'yyyy-MM')        AS anio_mes,
            UPPER(TRIM(tipo_encuesta))                         AS tipo_encuesta,
            UPPER(TRIM(linea))                                 AS linea,
            UPPER(TRIM(tipo_driver))                           AS tipo_driver,
            INITCAP(TRIM(categoria))                           AS categoria,

            CASE WHEN UPPER(tipo_driver) LIKE 'DOLOR%'
                 THEN 'DOLOR' ELSE 'PUNTO_CONTACTO' END        AS familia,
            CASE WHEN UPPER(tipo_driver) LIKE '%CES'
                 THEN 'CES' ELSE 'NPS' END                     AS metrica
        FROM {T_DRIVERS}
        WHERE categoria IS NOT NULL
          AND TRIM(categoria) <> ''
          AND UPPER(TRIM(categoria)) NOT IN ('N/A', 'NA', 'NULL', 'NINGUNO', 'SIN DATO')
          AND fecha IS NOT NULL
    """)


# =====================================================
# 4. VERBATIMS
# =====================================================

@st.cache_data(ttl=TTL, show_spinner=False)
def verbatims(limite: int = 20000) -> pd.DataFrame:
    """
    Comentarios abiertos. Alimenta tabla, sentimiento BERT y copiloto.

    Columna derivada `intermediario`: para tipo_encuesta = INTERMEDIARIO
    toma nombre_completo (o documento como respaldo). Es lo que habilita
    el ranking de "intermediarios con más comentarios malos".

    `es_negativo` viene de tipo_comentario, ya etiquetado por la encuesta:
    el ranking funciona SIN ejecutar ningún modelo de IA.
    """
    return query(f"""
        SELECT
            CAST(fecha AS DATE)                          AS fecha,
            DATE_FORMAT(CAST(fecha AS DATE), 'yyyy-MM')  AS anio_mes,
            UPPER(TRIM(tipo_encuesta))                   AS tipo_encuesta,
            UPPER(TRIM(linea))                           AS linea,

            documento,
            TRIM(nombre_completo)                        AS nombre_completo,
            COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)), ''), 'SIN SUCURSAL') AS cod_suc,
            COALESCE(NULLIF(TRIM(canal), ''), 'SIN CANAL')                      AS canal,

            INITCAP(TRIM(nps_categoria))                 AS nps_categoria,
            UPPER(TRIM(tipo_comentario))                 AS tipo_comentario,
            TRIM(texto)                                  AS texto,

            CASE WHEN UPPER(TRIM(tipo_encuesta)) = 'INTERMEDIARIO'
                 THEN COALESCE(NULLIF(TRIM(nombre_completo), ''),
                               CAST(documento AS STRING),
                               'SIN NOMBRE')
                 ELSE NULL END                           AS intermediario,

            CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR'
                 THEN 1 ELSE 0 END                       AS es_negativo
        FROM {T_VERBATIMS}
        WHERE texto IS NOT NULL
          AND LENGTH(TRIM(texto)) > 3
          AND UPPER(TRIM(texto)) NOT IN ('N/A','NA','NINGUNO','NINGUNA','NO','SIN COMENTARIO','.','-')
          AND fecha IS NOT NULL
        ORDER BY fecha DESC
        LIMIT {int(limite)}
    """)


# =====================================================
# 5. RANKING DE NEGATIVOS (agregado en el warehouse)
# =====================================================

@st.cache_data(ttl=TTL, show_spinner=False)
def ranking_negativos_sql(dimension: str = "intermediario",
                          anio_mes: str | None = None,
                          linea: str | None = None,
                          minimo: int = 5,
                          top: int = 20) -> pd.DataFrame:
    """
    Ranking de comentarios negativos calculado en el warehouse.
    Úsalo cuando el volumen de verbatims sea grande: evita traer
    todas las filas a la app solo para agrupar.

    dimension admitida: intermediario | cod_suc | canal | linea
    """
    permitidas = {
        "intermediario": "COALESCE(NULLIF(TRIM(nombre_completo),''), CAST(documento AS STRING))",
        "cod_suc": "COALESCE(NULLIF(TRIM(CAST(cod_suc AS STRING)),''), 'SIN SUCURSAL')",
        "canal": "COALESCE(NULLIF(TRIM(canal),''), 'SIN CANAL')",
        "linea": "UPPER(TRIM(linea))",
    }
    if dimension not in permitidas:
        raise ValueError(f"Dimensión no permitida: {dimension}")  # evita inyección SQL

    expr = permitidas[dimension]
    filtros = ["texto IS NOT NULL", "LENGTH(TRIM(texto)) > 3"]
    if dimension == "intermediario":
        filtros.append("UPPER(TRIM(tipo_encuesta)) = 'INTERMEDIARIO'")
    if anio_mes:
        filtros.append(f"DATE_FORMAT(CAST(fecha AS DATE), 'yyyy-MM') = '{anio_mes}'")
    if linea and linea.upper() != "TODAS":
        filtros.append(f"UPPER(TRIM(linea)) = '{linea.upper()}'")

    where = " AND ".join(filtros)

    return query(f"""
        SELECT
            {expr} AS {dimension},
            COUNT(*) AS total,
            SUM(CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR' THEN 1 ELSE 0 END) AS negativos,
            ROUND(
                100.0 * SUM(CASE WHEN UPPER(TRIM(tipo_comentario)) = 'DETRACTOR' THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 1
            ) AS pct_negativos
        FROM {T_VERBATIMS}
        WHERE {where}
        GROUP BY {expr}
        HAVING COUNT(*) >= {int(minimo)}
        ORDER BY negativos DESC, pct_negativos DESC
        LIMIT {int(top)}
    """)


# =====================================================
# 6. DIAGNÓSTICO
# =====================================================

@st.cache_data(ttl=TTL, show_spinner=False)
def diagnostico() -> pd.DataFrame:
    """Verifica que las 4 tablas Gold existan y tengan datos. Útil al desplegar."""
    filas = []
    for nombre, tabla in [("gold_cx_kpis", T_KPIS), ("gold_cx_resumen", T_RESUMEN),
                          ("gold_cx_drivers", T_DRIVERS), ("gold_cx_verbatims", T_VERBATIMS)]:
        try:
            r = query(f"SELECT COUNT(*) AS filas FROM {tabla}")
            filas.append({"tabla": nombre, "estado": "✅ OK", "filas": int(r.iloc[0]["filas"])})
        except Exception as e:
            filas.append({"tabla": nombre, "estado": f"❌ {type(e).__name__}", "filas": 0,
                          "detalle": str(e)[:200]})
    return pd.DataFrame(filas)