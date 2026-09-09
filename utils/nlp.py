"""Análisis de sentimiento jerárquico para verbatims CX."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterable

import pandas as pd
import streamlit as st

from utils.tablero import COLS_NLP, deduplicar_columnas

MODELO_SENT = os.getenv("CX_MODELO_SENT", "pysentimiento/robertuito-sentiment-analysis")
MODELO_ZS = os.getenv("CX_MODELO_ZS", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

TAXONOMIA = {
    "Positivo": {
        "emociones": ["Satisfacción", "Confianza", "Gratitud", "Entusiasmo"],
        "default": "Satisfacción",
    },
    "Neutro": {
        "emociones": ["Indiferencia", "Consulta/Solicitud", "Sugerencia"],
        "default": "Consulta/Solicitud",
    },
    "Negativo": {
        "emociones": ["Frustración", "Decepción", "Enojo", "Desconfianza", "Preocupación"],
        "default": "Frustración",
    },
}

TEMAS = {
    "demoras": ["demora", "demorado", "tard", "espera", "tiempo", "nunca respon"],
    "trato del asesor": ["asesor", "atención", "trato", "amable", "servicio", "ejecutivo"],
    "indemnización de siniestro": ["siniestro", "indemnización", "reembolso", "reclam", "pago del siniestro"],
    "precio/prima": ["precio", "prima", "costo", "caro", "tarifa", "valor"],
    "claridad de la información": ["información", "explic", "claridad", "entend", "confuso", "detalle"],
    "portal web o app": ["portal", "web", "app", "aplicación", "página", "plataforma"],
    "cobertura del producto": ["cobertura", "producto", "póliza", "beneficio", "amparo"],
    "gestión del intermediario": ["intermediario", "corredor", "agente", "gestión", "acompañamiento"],
    "trámites y documentación": ["trámite", "document", "papel", "radicar", "formulario", "proceso"],
}

PATRONES_EMOCION = {
    "Satisfacción": ["satisfe", "content", "excelente", "muy bien", "feliz"],
    "Confianza": ["conf", "seguro", "tranquil", "respaldo"],
    "Gratitud": ["gracias", "agrade", "reconozco"],
    "Entusiasmo": ["encant", "maravill", "emocion", "excelentísimo"],
    "Indiferencia": ["normal", "igual", "da lo mismo", "me es indiferente"],
    "Consulta/Solicitud": ["quisiera", "necesito", "solicito", "consulta", "pregunta", "informar"],
    "Sugerencia": ["deberían", "sugiero", "podrían", "recomiendo", "mejorar"],
    "Frustración": ["frustr", "molest", "incómodo", "no resuel", "no sirve"],
    "Decepción": ["decepcion", "esperaba", "no cumpl", "quedé mal"],
    "Enojo": ["enoj", "indign", "terrible", "pésimo", "fatal"],
    "Desconfianza": ["desconf", "duda", "no creo", "incertidumbre"],
    "Preocupación": ["preocup", "riesgo", "miedo", "urgente", "grave"],
}

MAPA_ETIQUETAS = {
    "POS": ("Positivo", 1.0),
    "NEU": ("Neutro", 0.0),
    "NEG": ("Negativo", -1.0),
    "1 star": ("Negativo", -1.0),
    "2 stars": ("Negativo", -0.5),
    "3 stars": ("Neutro", 0.0),
    "4 stars": ("Positivo", 0.5),
    "5 stars": ("Positivo", 1.0),
}


def limpiar(texto: str) -> str:
    """Limpia y recorta un comentario para inferencia."""
    if not isinstance(texto, str):
        return ""
    return re.sub(r"\s+", " ", texto).strip()[:2048]


def col_texto(df: pd.DataFrame) -> str | None:
    """Detecta la columna que contiene el comentario."""
    candidatas = ["verbatim", "comentario", "texto", "respuesta_abierta", "observacion", "comment"]
    for columna in candidatas:
        if columna in df.columns:
            return columna
    objetos = df.select_dtypes(include="object")
    if objetos.empty:
        return None
    return objetos.apply(lambda serie: serie.astype(str).str.len().mean()).idxmax()


@st.cache_resource(show_spinner="Cargando modelo de sentimiento…")
def pipe_sentimiento():
    """Carga el pipeline de sentimiento liviano en español."""
    import torch
    from transformers import pipeline

    torch.set_num_threads(os.cpu_count() or 1)
    return pipeline("sentiment-analysis", model=MODELO_SENT, truncation=True, max_length=256)


@st.cache_resource(show_spinner="Cargando clasificador avanzado…")
def pipe_zeroshot():
    """Carga el zero-shot para uso explícito y opcional."""
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=MODELO_ZS)


@st.cache_resource
def cache_resultados() -> dict[str, dict[str, object]]:
    """Cache de resultados por hash de texto individual."""
    return {}


def hash_texto(texto: str) -> str:
    """Genera un hash estable de un comentario limpio."""
    return hashlib.md5(limpiar(texto).encode("utf-8")).hexdigest()


def mapear_modelo(label: str, score: float) -> tuple[str, float, float]:
    """Normaliza etiquetas de modelos POS/NEU/NEG o estrellas."""
    polaridad, score_base = MAPA_ETIQUETAS.get(label, MAPA_ETIQUETAS.get(label.lower(), (label, 0.0)))
    confianza = round(float(score), 3)
    return polaridad, round(score_base * confianza, 3), confianza


def clasificar_tema(texto: str) -> str:
    """Asigna el tema predominante mediante reglas ponderadas."""
    texto_n = limpiar(texto).lower()
    puntajes = {}
    for tema, claves in TEMAS.items():
        puntajes[tema] = sum(2 if clave in texto_n else 0 for clave in claves)
    mejor, puntaje = max(puntajes.items(), key=lambda item: item[1])
    return mejor if puntaje > 0 else "claridad de la información"


def clasificar_emocion(texto: str, polaridad: str) -> str:
    """Determina la emoción más compatible con la polaridad detectada."""
    texto_n = limpiar(texto).lower()
    opciones = TAXONOMIA.get(polaridad, TAXONOMIA["Neutro"])
    mejor = opciones["default"]
    puntaje_mejor = 0
    for emocion in opciones["emociones"]:
        puntaje = sum(2 if clave in texto_n else 0 for clave in PATRONES_EMOCION.get(emocion, []))
        if puntaje > puntaje_mejor:
            mejor = emocion
            puntaje_mejor = puntaje
    return validar_emocion(polaridad, mejor)


def validar_emocion(polaridad: str, emocion: str) -> str:
    """Garantiza que la emoción pertenezca a la polaridad indicada."""
    opciones = TAXONOMIA.get(polaridad, TAXONOMIA["Neutro"])
    return emocion if emocion in opciones["emociones"] else opciones["default"]


def _clasificacion_base(texto: str, resultado: dict[str, object]) -> dict[str, object]:
    polaridad, score, confianza = mapear_modelo(str(resultado.get("label")), float(resultado.get("score", 0.0)))
    emocion = clasificar_emocion(texto, polaridad)
    tema = clasificar_tema(texto)
    return {
        "polaridad": polaridad,
        "emocion": validar_emocion(polaridad, emocion),
        "tema": tema,
        "score": score,
        "confianza": confianza,
    }


def _aplicar_zero_shot(textos: list[str], resultados: list[dict[str, object]], batch_size: int, progress, inicio: float, fin: float) -> list[dict[str, object]]:
    """Refina tema y emoción con zero-shot opcional."""
    pipe = pipe_zeroshot()
    temas = list(TEMAS.keys())
    salida = []
    total = max(1, math.ceil(len(textos) / batch_size))
    for indice in range(total):
        lote = textos[indice * batch_size:(indice + 1) * batch_size]
        if not lote:
            continue
        respuesta = pipe(lote, candidate_labels=temas, hypothesis_template="Este comentario trata principalmente sobre {}.", multi_label=False)
        if isinstance(respuesta, dict):
            respuesta = [respuesta]
        for base, clasificacion in zip(resultados[indice * batch_size:(indice + 1) * batch_size], respuesta):
            base = dict(base)
            base["tema"] = clasificacion["labels"][0]
            salida.append(base)
        progress.progress(inicio + ((indice + 1) / total) * (fin - inicio), text=f"Clasificando tema avanzado · bloque {indice + 1}/{total}")
    return salida


def enriquecer(
    df: pd.DataFrame,
    muestra: int = 300,
    con_zero_shot: bool = False,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Añade polaridad, emoción y tema con progreso visible por bloques."""
    if df is None or df.empty:
        return df
    df = deduplicar_columnas(df)
    columna = col_texto(df)
    if columna is None:
        return df
    base = df.head(muestra).copy().reset_index(drop=True)
    base["_texto"] = base[columna].astype(str)
    hashes = [hash_texto(texto) for texto in base["_texto"]]
    cache = cache_resultados()
    faltantes = [(indice, texto, hashes[indice]) for indice, texto in enumerate(base["_texto"].tolist()) if hashes[indice] not in cache and len(limpiar(texto)) > 3]
    faltantes.sort(key=lambda item: len(item[1]))
    progress = st.progress(0.0, text="Preparando comentarios…")
    with st.status("Analizando comentarios", expanded=True) as estado:
        estado.write(f"Se revisarán {len(base):,.0f} comentarios. Nuevos por procesar: {len(faltantes):,.0f}.")
        if faltantes:
            import torch

            pipe = pipe_sentimiento()
            textos = [texto for _, texto, _ in faltantes]
            total = max(1, math.ceil(len(textos) / batch_size))
            procesados = []
            with torch.inference_mode():
                for indice in range(total):
                    lote = textos[indice * batch_size:(indice + 1) * batch_size]
                    if not lote:
                        continue
                    resultados = pipe(lote, batch_size=batch_size, truncation=True, padding=True, max_length=256)
                    if isinstance(resultados, dict):
                        resultados = [resultados]
                    procesados.extend(resultados)
                    progress.progress((indice + 1) / total * 0.8, text=f"Analizando sentimiento · bloque {indice + 1}/{total}")
            clasificados = [_clasificacion_base(texto, resultado) for texto, resultado in zip(textos, procesados)]
            if con_zero_shot:
                estado.write("Refinando temas con clasificador avanzado (más lento).")
                clasificados = _aplicar_zero_shot(textos, clasificados, batch_size, progress, 0.8, 1.0)
            for (_, texto, hash_id), resultado in zip(faltantes, clasificados):
                cache[hash_id] = dict(resultado)
        progress.progress(1.0, text="Análisis completado")
        estado.update(label="Análisis de sentimiento completado", state="complete")
    base = base.drop(columns=[columna for columna in COLS_NLP if columna in base.columns], errors="ignore")
    filas = []
    for hash_id, texto in zip(hashes, base["_texto"]):
        if hash_id in cache:
            filas.append(cache[hash_id])
        else:
            filas.append({
                "polaridad": None,
                "emocion": None,
                "tema": None,
                "score": None,
                "confianza": None,
            })
    resultado = pd.concat([base, pd.DataFrame(filas)], axis=1)
    return deduplicar_columnas(resultado)
