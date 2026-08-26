"""Análisis de sentimiento y sub-sentimiento sobre verbatims CX."""

import hashlib
import os
import re

import pandas as pd
import streamlit as st

MODELO_SENT = os.getenv("CX_MODELO_SENT", "nlptown/bert-base-multilingual-uncased-sentiment")
MODELO_ZS = os.getenv("CX_MODELO_ZS", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# Sub-sentimientos: ajusta esta lista a tus categorías de negocio
SUBCATEGORIAS = [
    "demoras en la respuesta",
    "trato del asesor",
    "proceso de indemnización de siniestro",
    "precio o valor de la prima",
    "claridad de la información",
    "portal web o app",
    "cobertura del producto",
    "gestión del intermediario",
    "trámites y documentación",
]

EMOCIONES = ["frustración", "enojo", "satisfacción", "confianza", "decepción", "sorpresa"]

MAP_ESTRELLAS = {1: "Muy negativo", 2: "Negativo", 3: "Neutro", 4: "Positivo", 5: "Muy positivo"}
MAP_POLARIDAD = {
    "Muy negativo": "Negativo", "Negativo": "Negativo",
    "Neutro": "Neutro",
    "Positivo": "Positivo", "Muy positivo": "Positivo",
}


def limpiar(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    t = re.sub(r"\s+", " ", texto).strip()
    return t[:512]  # límite de tokens de BERT


def col_texto(df: pd.DataFrame) -> str | None:
    """Detecta la columna que contiene el comentario."""
    candidatas = ["verbatim", "comentario", "texto", "respuesta_abierta", "observacion", "comment"]
    for c in candidatas:
        if c in df.columns:
            return c
    obj = df.select_dtypes(include="object")
    if obj.empty:
        return None
    return obj.apply(lambda s: s.astype(str).str.len().mean()).idxmax()


@st.cache_resource(show_spinner="Cargando modelo de sentimiento…")
def pipe_sentimiento():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODELO_SENT, truncation=True, max_length=512)


@st.cache_resource(show_spinner="Cargando modelo de sub-sentimiento…")
def pipe_zeroshot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=MODELO_ZS)


def _hash(textos) -> str:
    return hashlib.md5("|".join(map(str, textos)).encode()).hexdigest()


@st.cache_data(ttl=86400, show_spinner="Analizando sentimiento…")
def analizar_sentimiento(textos: list[str], _clave: str) -> pd.DataFrame:
    """Devuelve sentimiento por texto. `_clave` fuerza el cacheo por lote."""
    limpios = [limpiar(t) for t in textos]
    validos = [i for i, t in enumerate(limpios) if len(t) > 3]

    filas = [{"sentimiento": None, "polaridad": None, "score": None, "estrellas": None}] * len(limpios)
    filas = [dict(f) for f in filas]

    if validos:
        pipe = pipe_sentimiento()
        res = pipe([limpios[i] for i in validos], batch_size=16)
        for i, r in zip(validos, res):
            label = r["label"]
            estrellas = int(re.search(r"\d", label).group()) if re.search(r"\d", label) else None
            if estrellas:
                sent = MAP_ESTRELLAS[estrellas]
            else:  # robertuito: POS/NEU/NEG
                sent = {"POS": "Positivo", "NEU": "Neutro", "NEG": "Negativo"}.get(label, label)
            filas[i] = {
                "sentimiento": sent,
                "polaridad": MAP_POLARIDAD.get(sent, sent),
                "score": round(float(r["score"]), 3),
                "estrellas": estrellas,
            }
    return pd.DataFrame(filas)


@st.cache_data(ttl=86400, show_spinner="Clasificando sub-sentimientos…")
def analizar_subsentimiento(
    textos: list[str], _clave: str, etiquetas: tuple[str, ...], multi: bool = False
) -> pd.DataFrame:
    limpios = [limpiar(t) for t in textos]
    validos = [i for i, t in enumerate(limpios) if len(t) > 3]

    filas = [{"subsentimiento": None, "sub_score": None} for _ in limpios]
    if validos:
        pipe = pipe_zeroshot()
        res = pipe(
            [limpios[i] for i in validos],
            candidate_labels=list(etiquetas),
            hypothesis_template="Este comentario trata sobre {}.",
            multi_label=multi,
        )
        if isinstance(res, dict):
            res = [res]
        for i, r in zip(validos, res):
            filas[i] = {"subsentimiento": r["labels"][0], "sub_score": round(float(r["scores"][0]), 3)}
    return pd.DataFrame(filas)


def enriquecer(df: pd.DataFrame, muestra: int = 300, con_sub: bool = True) -> pd.DataFrame:
    """Añade columnas de sentimiento y sub-sentimiento a los verbatims."""
    if df is None or df.empty:
        return df

    col = col_texto(df)
    if col is None:
        return df

    base = df.head(muestra).copy().reset_index(drop=True)
    textos = base[col].astype(str).tolist()
    clave = _hash(textos)

    sent = analizar_sentimiento(textos, clave)
    base = pd.concat([base, sent], axis=1)

    if con_sub:
        sub = analizar_subsentimiento(textos, clave, tuple(SUBCATEGORIAS))
        base = pd.concat([base, sub], axis=1)

    base["_texto"] = base[col]
    return base