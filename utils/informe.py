"""Funciones puras para exportar bases e informes del tablero CX."""

from __future__ import annotations

import io
from datetime import datetime
from html import escape

import pandas as pd

from utils.estilos import PALETA
from utils.tablero import deduplicar_columnas


def slug(valor: str) -> str:
    """Normaliza una cadena para usarla en nombres de archivo."""
    limpio = "_".join(str(valor).strip().lower().split())
    return "".join(caracter for caracter in limpio if caracter.isalnum() or caracter in {"_", "-"}) or "todos"


def nombre_archivo_base(dataset: str, periodo: str, linea: str, tipo: str, sucursal: str, extension: str) -> str:
    """Construye el nombre autodescriptivo de una exportación."""
    partes = ["base_cx", slug(dataset), slug(periodo), slug(linea), slug(tipo)]
    if sucursal != "TODAS":
        partes.append(f"suc{slug(sucursal)}")
    return f"{'_'.join(partes)}.{extension}"


def dataframe_a_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa un dataframe a CSV compatible con Excel en español."""
    df = deduplicar_columnas(df) if df is not None else pd.DataFrame()
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def dataframes_a_xlsx_bytes(hojas: dict[str, pd.DataFrame]) -> bytes:
    """Serializa varias hojas en un único archivo XLSX."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for nombre, dataframe in hojas.items():
            deduplicar_columnas(dataframe if dataframe is not None else pd.DataFrame()).to_excel(writer, sheet_name=nombre[:31], index=False)
    return buffer.getvalue()


def dataframe_kpis(kpis: dict[str, object]) -> pd.DataFrame:
    """Convierte el resumen de KPIs en una tabla exportable."""
    return pd.DataFrame(
        [
            {"indicador": "NPS", "valor": kpis.get("nps"), "variacion": kpis.get("delta_nps"), "estado": kpis.get("estado_nps")},
            {"indicador": "INS", "valor": kpis.get("ins"), "variacion": kpis.get("delta_ins"), "estado": kpis.get("estado_ins")},
            {"indicador": "CES", "valor": kpis.get("ces"), "variacion": kpis.get("delta_ces"), "estado": kpis.get("estado_ces")},
            {"indicador": "Respuestas", "valor": kpis.get("respuestas"), "variacion": kpis.get("delta_respuestas"), "estado": ""},
            {"indicador": "Verbatims", "valor": kpis.get("verbatims"), "variacion": None, "estado": ""},
        ]
    )


def dataframe_metadatos(filtros: dict[str, str]) -> pd.DataFrame:
    """Convierte filtros y marcas de tiempo en hoja de metadatos."""
    return pd.DataFrame(
        {
            "campo": [*filtros.keys(), "generado_en"],
            "valor": [*filtros.values(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }
    )


def generar_informe_excel(
    filtros: dict[str, str],
    kpis: dict[str, object],
    secciones: dict[str, pd.DataFrame],
) -> bytes:
    """Genera un XLSX multihoja del informe actual."""
    hojas = {
        "metadatos": dataframe_metadatos(filtros),
        "kpis": dataframe_kpis(kpis),
    }
    hojas.update({nombre: deduplicar_columnas(df) for nombre, df in secciones.items() if df is not None})
    return dataframes_a_xlsx_bytes(hojas)


def _tabla_html(df: pd.DataFrame, limite: int = 50) -> str:
    if df is None or df.empty:
        return "<p>Sin datos para esta sección.</p>"
    return deduplicar_columnas(df).head(limite).to_html(index=False, border=0, classes="tabla")


def construir_figuras_informe(
    historico: pd.DataFrame,
    comparativo: pd.DataFrame,
    detalle: pd.DataFrame,
    drivers_df: pd.DataFrame,
    sentimiento_df: pd.DataFrame | None,
) -> dict[str, object]:
    """Arma las figuras del informe de forma perezosa y tolerante."""
    from utils.graficos import (
        figura_composicion_nps,
        figura_delta_lineas,
        figura_dispersion_lineas,
        figura_drivers_pareto,
        figura_evolucion,
        figura_ranking_lineas,
        figura_sunburst_sentimiento,
        figura_temas_negativos,
    )

    figuras: dict[str, object] = {}
    figuras["Evolución"] = figura_evolucion(historico)
    figuras["Composición NPS"] = figura_composicion_nps(detalle)
    figuras["Ranking entre líneas"] = figura_ranking_lineas(comparativo)
    figuras["Volumen vs. desempeño"] = figura_dispersion_lineas(comparativo)
    figuras["Brecha vs. compañía"] = figura_delta_lineas(comparativo)
    if drivers_df is not None and not drivers_df.empty and "categoria" in drivers_df.columns:
        pareto = drivers_df.groupby("categoria", as_index=False).size().rename(columns={"size": "menciones"})
        figuras["Drivers"] = figura_drivers_pareto(pareto[["categoria", "menciones"]])
    else:
        figuras["Drivers"] = None
    if sentimiento_df is not None and not sentimiento_df.empty:
        figuras["Sentimiento"] = figura_sunburst_sentimiento(sentimiento_df)
        figuras["Negativos por tema"] = figura_temas_negativos(sentimiento_df)
    else:
        figuras["Sentimiento"] = None
        figuras["Negativos por tema"] = None
    return figuras


def generar_informe_html(
    filtros: dict[str, str],
    kpis: dict[str, object],
    tablas: dict[str, pd.DataFrame],
    figuras: dict[str, object],
) -> str:
    """Construye un informe HTML autocontenido imprimible a PDF."""
    kpis_html = deduplicar_columnas(dataframe_kpis(kpis)).to_html(index=False, border=0, classes="tabla")
    figuras_html = []
    include_js = True
    for nombre, figura in figuras.items():
        if figura is None:
            continue
        figuras_html.append(f"<h2>{escape(nombre)}</h2>" + figura.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False
    tablas_html = "".join(f"<h2>{escape(nombre)}</h2>{_tabla_html(df)}" for nombre, df in tablas.items())
    alcance = " · ".join(f"{escape(clave)}: {escape(str(valor))}" for clave, valor in filtros.items())
    generado = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8" />
      <title>Informe Centro de Experiencia CX</title>
      <style>
        body{{font-family:Arial,sans-serif;background:{PALETA['fondo']};color:{PALETA['azul']};margin:0;padding:0 24px 32px;}}
        .portada{{background:linear-gradient(135deg,{PALETA['azul']} 0%,{PALETA['azul_claro']} 55%,{PALETA['naranja']} 160%);color:#fff;padding:28px;border-radius:20px;margin:24px 0;}}
        .tabla{{width:100%;border-collapse:collapse;background:#fff;margin-bottom:24px;}}
        .tabla th,.tabla td{{padding:8px 10px;border:1px solid {PALETA['borde']};font-size:13px;}}
        h2{{border-bottom:2px solid {PALETA['borde']};padding-bottom:6px;}}
      </style>
    </head>
    <body>
      <div class="portada">
        <h1>Seguros del Estado · Centro de Experiencia CX</h1>
        <p>Alcance: {alcance}</p>
        <p>Generado: {generado}</p>
      </div>
      <h2>KPIs</h2>
      {kpis_html}
      {''.join(figuras_html)}
      {tablas_html}
    </body>
    </html>
    """
