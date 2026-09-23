"""Helpers puros para filtros, métricas y selección de fuentes del tablero."""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence
import unicodedata

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
COLS_NLP = ("polaridad", "emocion", "tema", "score", "confianza")
TIPOS_INTERMEDIARIO = {"ASESOR", "CORREDOR", "AGENTE"}


def deduplicar_columnas(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Devuelve una copia sin columnas duplicadas preservando la primera ocurrencia."""
    if df is None or not hasattr(df, "columns"):
        return df
    if getattr(df.columns, "is_unique", True):
        return df.copy()
    return df.loc[:, ~df.columns.duplicated()].copy()


def quitar_tildes(valor: object) -> str:
    """Elimina tildes y normaliza espacios para comparaciones robustas."""
    texto = unicodedata.normalize("NFKD", str(valor or ""))
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def normalizar_tipo(valor: object) -> str:
    """Convierte variantes del tipo de encuesta a una forma canónica."""
    texto = quitar_tildes(valor).strip().upper()
    if not texto or texto in {"NAN", "NONE", "NULL"}:
        return ""
    if texto.startswith("CLIENT"):
        return "CLIENTE"
    if texto.startswith("INTERMEDIARI") or texto in TIPOS_INTERMEDIARIO:
        return "INTERMEDIARIO"
    return texto


def es_intermediario(valor: object) -> bool:
    """Reconoce variantes históricas del literal de intermediarios."""
    texto = quitar_tildes(valor).strip().upper()
    return bool(texto) and (texto.startswith("INTERMEDIARI") or texto in TIPOS_INTERMEDIARIO)


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
    salida = deduplicar_columnas(df)
    for columna in NUM_COLS:
        if columna in salida.columns:
            salida[columna] = a_numero(salida[columna])
    for columna in ("linea", "anio_mes", *COLS_SUCURSAL, *COLS_INTERMEDIARIO):
        if columna in salida.columns:
            salida[columna] = salida[columna].astype(str).str.strip()
    if "tipo_encuesta" in salida.columns:
        salida["tipo_encuesta"] = salida["tipo_encuesta"].map(normalizar_tipo)
    if "linea" in salida.columns:
        salida["linea"] = salida["linea"].astype(str).str.strip().str.upper()
    return salida


def primera_col(df: pd.DataFrame, candidatas: Sequence[str]) -> str | None:
    """Devuelve la primera columna disponible dentro de una lista de candidatas."""
    if df is None or getattr(df, "empty", True):
        return None
    return next((columna for columna in candidatas if columna in df.columns), None)


def _normalizar_valores(df: pd.DataFrame, columna: str) -> set[str]:
    if df is None or df.empty or columna not in df.columns:
        return set()
    serie = df[columna].dropna().map(normalizar_tipo if columna == "tipo_encuesta" else lambda valor: str(valor).strip().upper())
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
) -> tuple[list[str], dict[str, int], bool]:
    """Devuelve sucursales válidas para intermediarios con filtros activos."""
    if df_sucursal is None or df_sucursal.empty:
        return ["TODAS"], {}, False
    columna = primera_col(df_sucursal, COLS_SUCURSAL)
    if not columna:
        return ["TODAS"], {}, False
    datos = deduplicar_columnas(df_sucursal)
    if "tipo_encuesta" in datos.columns:
        datos = datos[datos["tipo_encuesta"].map(es_intermediario)]
    if linea != "TODAS" and "linea" in datos.columns:
        datos = datos[datos["linea"] == linea]
    historico = False
    datos_periodo = datos
    if periodo and "anio_mes" in datos.columns:
        datos_periodo = datos[datos["anio_mes"] == periodo]
        if not datos_periodo.empty:
            datos = datos_periodo
        else:
            historico = True
    conteos = (
        datos.groupby(columna)["encuestados"].sum().fillna(0).astype(int).to_dict()
        if "encuestados" in datos.columns
        else datos.groupby(columna).size().astype(int).to_dict()
    )
    valores = [str(valor).strip() for valor in conteos if str(valor).strip()]
    if len(valores) > 1:
        valores = [valor for valor in valores if valor != "SIN SUCURSAL"]
    return ["TODAS", *ordenar_sucursales(valores)], conteos, historico


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
    if es_intermediario(tipo) and df_resumen_sucursal is not None and not df_resumen_sucursal.empty:
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
        tipo_objetivo = normalizar_tipo(tipo)
        if es_intermediario(tipo_objetivo):
            salida = salida[salida["tipo_encuesta"].map(es_intermediario)]
        else:
            salida = salida[salida["tipo_encuesta"].map(normalizar_tipo) == tipo_objetivo]
    if con_periodo and periodo and "anio_mes" in salida.columns:
        salida = salida[salida["anio_mes"] == (periodo_valor or periodo)]
    if sucursal != "TODAS":
        columna = primera_col(salida, COLS_SUCURSAL)
        if columna:
            salida = salida[salida[columna].astype(str) == sucursal]
    return deduplicar_columnas(salida)


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


def filtros_activos_texto(
    linea: str,
    tipo: str,
    periodo: str,
    sucursal: str,
    periodo_comparativo: str | None = None,
) -> str:
    """Convierte los filtros activos en una cadena legible."""
    filtros = [f"línea={linea}", f"tipo={tipo}", f"período={periodo}"]
    if periodo_comparativo and periodo_comparativo != "SIN COMPARACIÓN":
        filtros.append(f"comparativo={periodo_comparativo}")
    if sucursal != "TODAS":
        filtros.append(f"sucursal={sucursal}")
    return ", ".join(filtros)


def periodo_comparativo_resuelto(
    periodo_actual: str,
    periodos: Sequence[str],
    periodo_comparativo: str | None,
    modo_comparacion: str = "Período anterior",
) -> tuple[str | None, str]:
    """Resuelve el período de comparación respetando el comportamiento actual por defecto."""
    if not periodos:
        return None, "Sin comparación"
    if modo_comparacion == "Sin comparación":
        return None, "Sin comparación"
    if modo_comparacion == "Periodo específico":
        if not periodo_comparativo or periodo_comparativo == "SIN COMPARACIÓN":
            return None, "Sin comparación"
        if periodo_comparativo == periodo_actual:
            return None, "Sin comparación"
        return str(periodo_comparativo), str(periodo_comparativo)
    idx = periodos.index(periodo_actual) if periodo_actual in periodos else 0
    if idx + 1 < len(periodos):
        return periodos[idx + 1], periodos[idx + 1]
    return None, "Sin comparación"


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


def opciones_periodo(tipo: str, df_resumen: pd.DataFrame, df_resumen_sucursal: pd.DataFrame) -> list[str]:
    """Lista períodos válidos para el tipo seleccionado usando la fuente efectiva."""
    fuente = fuente_datos(tipo, "TODAS", df_resumen, df_resumen_sucursal)
    base = aplicar_filtros(fuente, tipo=tipo, con_periodo=False)
    return sorted(valores_union(base, columna="anio_mes"), reverse=True)


def opciones_linea(tipo: str, periodo: str, df_resumen: pd.DataFrame, df_resumen_sucursal: pd.DataFrame) -> list[str]:
    """Lista líneas válidas para el tipo y período seleccionados."""
    fuente = fuente_datos(tipo, "TODAS", df_resumen, df_resumen_sucursal)
    base = aplicar_filtros(fuente, tipo=tipo, periodo=periodo, con_periodo=bool(periodo))
    if base is None or base.empty:
        base = aplicar_filtros(fuente, tipo=tipo, con_periodo=False)
    return ["TODAS", *valores_union(base, columna="linea")]


def diagnostico_tipos(nombre: str, df: pd.DataFrame) -> pd.DataFrame:
    """Resume tipos de encuesta distintos presentes en una tabla cargada."""
    if df is None or df.empty or "tipo_encuesta" not in df.columns:
        return pd.DataFrame([{"tabla": nombre, "tipo_encuesta": "SIN DATOS"}])
    valores = sorted({normalizar_tipo(valor) for valor in df["tipo_encuesta"].dropna() if normalizar_tipo(valor)})
    return pd.DataFrame({"tabla": [nombre] * len(valores), "tipo_encuesta": valores})


def diagnostico_tipo_periodo(nombre: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta filas por tipo y período para diagnóstico del tablero."""
    columnas = {"tipo_encuesta", "anio_mes"}
    if df is None or df.empty or not columnas.issubset(df.columns):
        return pd.DataFrame(columns=["tabla", "tipo_encuesta", "anio_mes", "filas"])
    salida = (
        df.groupby(["tipo_encuesta", "anio_mes"], dropna=False)
        .size()
        .reset_index(name="filas")
        .sort_values(["anio_mes", "tipo_encuesta"], ascending=[False, True])
    )
    salida.insert(0, "tabla", nombre)
    return salida


def diagnostico_cod_suc(df: pd.DataFrame) -> pd.DataFrame:
    """Indica si los registros intermediarios tienen códigos de sucursal."""
    columnas = {"tipo_encuesta", "cod_suc"}
    if df is None or df.empty or not columnas.issubset(df.columns):
        return pd.DataFrame([{"escenario": "INTERMEDIARIO", "registros": 0, "cod_suc_no_nulo": 0}])
    inter = df[df["tipo_encuesta"].map(es_intermediario)]
    return pd.DataFrame(
        [
            {
                "escenario": "INTERMEDIARIO",
                "registros": int(len(inter)),
                "cod_suc_no_nulo": int(inter["cod_suc"].fillna("").astype(str).str.strip().ne("").sum()),
            }
        ]
    )


def comparativo_lineas(df_actual: pd.DataFrame, df_previo: pd.DataFrame) -> pd.DataFrame:
    """Calcula un comparativo entre líneas con referencia total y variación previa."""

    def _resumir(df: pd.DataFrame | None) -> pd.DataFrame:
        columnas = [
            "linea",
            "nps",
            "ins",
            "ces",
            "respuestas",
            "promotores",
            "neutros",
            "detractores",
            "pct_promotores",
            "pct_neutros",
            "pct_detractores",
        ]
        if df is None or df.empty or "linea" not in df.columns:
            return pd.DataFrame(columns=columnas)
        filas: list[dict[str, float | int | str | None]] = []
        for linea_nombre, grupo in df.groupby("linea", dropna=False):
            respuestas = int(grupo["encuestados"].sum()) if "encuestados" in grupo.columns else int(len(grupo))
            promotores = float(grupo["promotores"].sum()) if "promotores" in grupo.columns else 0.0
            neutros = float(grupo["neutros"].sum()) if "neutros" in grupo.columns else 0.0
            detractores = float(grupo["detractores"].sum()) if "detractores" in grupo.columns else 0.0
            filas.append(
                {
                    "linea": linea_nombre,
                    "nps": calcular_nps(grupo),
                    "ins": promedio_ponderado(grupo, "avg_ins"),
                    "ces": promedio_ponderado(grupo, "avg_ces"),
                    "respuestas": respuestas,
                    "promotores": promotores,
                    "neutros": neutros,
                    "detractores": detractores,
                    "pct_promotores": round(promotores / respuestas * 100, 1) if respuestas else None,
                    "pct_neutros": round(neutros / respuestas * 100, 1) if respuestas else None,
                    "pct_detractores": round(detractores / respuestas * 100, 1) if respuestas else None,
                }
            )
        salida = pd.DataFrame(filas)
        if salida.empty:
            return salida.reindex(columns=columnas)
        total = _resumir_total(df)
        return pd.concat([salida, pd.DataFrame([total])], ignore_index=True).reindex(columns=columnas)

    def _resumir_total(df: pd.DataFrame) -> dict[str, float | int | str | None]:
        respuestas = int(df["encuestados"].sum()) if "encuestados" in df.columns else int(len(df))
        promotores = float(df["promotores"].sum()) if "promotores" in df.columns else 0.0
        neutros = float(df["neutros"].sum()) if "neutros" in df.columns else 0.0
        detractores = float(df["detractores"].sum()) if "detractores" in df.columns else 0.0
        return {
            "linea": "TOTAL / Promedio compañía",
            "nps": calcular_nps(df),
            "ins": promedio_ponderado(df, "avg_ins"),
            "ces": promedio_ponderado(df, "avg_ces"),
            "respuestas": respuestas,
            "promotores": promotores,
            "neutros": neutros,
            "detractores": detractores,
            "pct_promotores": round(promotores / respuestas * 100, 1) if respuestas else None,
            "pct_neutros": round(neutros / respuestas * 100, 1) if respuestas else None,
            "pct_detractores": round(detractores / respuestas * 100, 1) if respuestas else None,
        }

    actual = _resumir(df_actual)
    previo = _resumir(df_previo)[["linea", "nps", "ins", "ces"]].rename(
        columns={"nps": "nps_previo", "ins": "ins_previo", "ces": "ces_previo"}
    )
    if actual.empty:
        return pd.DataFrame(
            columns=[
                "linea",
                "nps",
                "ins",
                "ces",
                "respuestas",
                "pct_promotores",
                "pct_neutros",
                "pct_detractores",
                "delta_nps",
                "delta_ins",
                "delta_ces",
            ]
        )
    salida = actual.merge(previo, on="linea", how="left")
    salida["delta_nps"] = salida.apply(lambda fila: delta(fila["nps"], fila.get("nps_previo")), axis=1)
    salida["delta_ins"] = salida.apply(lambda fila: delta(fila["ins"], fila.get("ins_previo")), axis=1)
    salida["delta_ces"] = salida.apply(lambda fila: delta(fila["ces"], fila.get("ces_previo")), axis=1)
    salida = salida.sort_values(["linea"], kind="stable").reset_index(drop=True)
    total = salida[salida["linea"] == "TOTAL / Promedio compañía"]
    detalle = salida[salida["linea"] != "TOTAL / Promedio compañía"].sort_values("nps", ascending=False, na_position="last")
    return deduplicar_columnas(pd.concat([detalle, total], ignore_index=True))


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


def riesgo_sucursal_por_indicador(
    df_actual: pd.DataFrame,
    df_previo: pd.DataFrame | None = None,
    linea: str = "TODAS",
    tipo: str = "TODOS",
    periodo_actual: str | None = None,
    periodo_previo: str | None = None,
) -> pd.DataFrame:
    """Construye una vista ejecutiva de riesgo por sucursal e indicador usando las mismas reglas del tablero."""
    if df_actual is None or df_actual.empty:
        return pd.DataFrame(columns=["sucursal", "indicador", "valor_actual", "meta", "estado", "variacion", "periodo_comparativo"])
    datos = df_actual.copy()
    if linea != "TODAS" and "linea" in datos.columns:
        datos = datos[datos["linea"] == linea]
    if tipo != "TODOS" and "tipo_encuesta" in datos.columns:
        tipo_objetivo = normalizar_tipo(tipo)
        if es_intermediario(tipo_objetivo):
            datos = datos[datos["tipo_encuesta"].map(es_intermediario)]
        else:
            datos = datos[datos["tipo_encuesta"].map(normalizar_tipo) == tipo_objetivo]
    sucursal_col = primera_col(datos, COLS_SUCURSAL)
    if not sucursal_col:
        return pd.DataFrame(columns=["sucursal", "indicador", "valor_actual", "meta", "estado", "variacion", "periodo_comparativo"])
    base = datos.groupby(sucursal_col, dropna=False).agg(
        encuestados=("encuestados", "sum"),
        promotores=("promotores", "sum"),
        neutros=("neutros", "sum"),
        detractores=("detractores", "sum"),
        avg_ins=("avg_ins", lambda s: (s * datos.loc[s.index, "encuestados"]).sum() / datos.loc[s.index, "encuestados"].sum() if "encuestados" in datos.columns and datos.loc[s.index, "encuestados"].sum() else s.mean()),
        avg_ces=("avg_ces", lambda s: (s * datos.loc[s.index, "encuestados"]).sum() / datos.loc[s.index, "encuestados"].sum() if "encuestados" in datos.columns and datos.loc[s.index, "encuestados"].sum() else s.mean()),
    ) if "encuestados" in datos.columns else datos.groupby(sucursal_col).size().reset_index(name="encuestados")
    if isinstance(base, pd.DataFrame) and base.empty:
        return pd.DataFrame(columns=["sucursal", "indicador", "valor_actual", "meta", "estado", "variacion", "periodo_comparativo"])
    if not isinstance(base, pd.DataFrame):
        return pd.DataFrame(columns=["sucursal", "indicador", "valor_actual", "meta", "estado", "variacion", "periodo_comparativo"])
    filas: list[dict[str, object]] = []
    for sucursal, fila in base.iterrows():
        valores = {
            "NPS": calcular_nps(pd.DataFrame({
                "encuestados": [fila.get("encuestados", 0)],
                "promotores": [fila.get("promotores", 0)],
                "detractores": [fila.get("detractores", 0)],
            })),
            "INS": promedio_ponderado(pd.DataFrame({
                "encuestados": [fila.get("encuestados", 0)],
                "avg_ins": [fila.get("avg_ins")],
            }), "avg_ins"),
            "CES": promedio_ponderado(pd.DataFrame({
                "encuestados": [fila.get("encuestados", 0)],
                "avg_ces": [fila.get("avg_ces")],
            }), "avg_ces"),
        }
        metas = {"NPS": 70.0, "INS": 9.0, "CES": 2.5}
        previa = None
        if df_previo is not None and not df_previo.empty:
            previo_df = df_previo.copy()
            if linea != "TODAS" and "linea" in previo_df.columns:
                previo_df = previo_df[previo_df["linea"] == linea]
            if tipo != "TODOS" and "tipo_encuesta" in previo_df.columns:
                tipo_objetivo = normalizar_tipo(tipo)
                if es_intermediario(tipo_objetivo):
                    previo_df = previo_df[previo_df["tipo_encuesta"].map(es_intermediario)]
                else:
                    previo_df = previo_df[previo_df["tipo_encuesta"].map(normalizar_tipo) == tipo_objetivo]
            prev_sucursal_col = primera_col(previo_df, COLS_SUCURSAL)
            if prev_sucursal_col:
                prev_group = previo_df[previo_df[prev_sucursal_col].astype(str) == str(sucursal)]
                if not prev_group.empty:
                    previa = {
                        "NPS": calcular_nps(prev_group),
                        "INS": promedio_ponderado(prev_group, "avg_ins"),
                        "CES": promedio_ponderado(prev_group, "avg_ces"),
                    }
        for indicador, valor in valores.items():
            if valor is None or pd.isna(valor):
                continue
            estado, _ = semaforo(indicador, valor)
            variacion = None if previa is None or previa.get(indicador) is None else delta(valor, previa.get(indicador))
            filas.append(
                {
                    "sucursal": str(sucursal),
                    "indicador": indicador,
                    "valor_actual": float(valor),
                    "meta": float(metas[indicador]),
                    "estado": estado,
                    "variacion": None if variacion is None else float(variacion),
                    "periodo_comparativo": periodo_previo or "sin comparación",
                }
            )
    return pd.DataFrame(filas).sort_values(["sucursal", "indicador"]).reset_index(drop=True)


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
