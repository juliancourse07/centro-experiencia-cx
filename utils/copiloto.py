"""Copiloto CX: responde con analítica real y redacta con un LLM opcional."""

from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

MODELO_LLM = os.getenv("CX_MODELO_LLM", "meta-llama/Llama-3.3-70B-Instruct")
_ESPACIOS = re.compile(r"\s+")
SYSTEM = """Eres un analista de experiencia de cliente (CX) de una aseguradora colombiana.
Respondes en español, en máximo 6 viñetas, con foco en acción de negocio.
Usa ÚNICAMENTE los datos del contexto. Si el dato no está, dilo explícitamente.
No inventes cifras, nombres de intermediarios ni porcentajes."""


def _token() -> str | None:
    return (st.secrets.get("HF_TOKEN", None) if hasattr(st, "secrets") else None) or os.getenv("HF_TOKEN")


@st.cache_resource
def cliente_llm():
    """Devuelve un cliente de inferencia de Hugging Face cuando hay token."""
    token = _token()
    if not token:
        return None
    from huggingface_hub import InferenceClient

    return InferenceClient(model=MODELO_LLM, token=token)


def _limpiar(texto: object, limite: int = 220) -> str:
    return _ESPACIOS.sub(" ", str(texto)).strip()[:limite]


def ranking_negativos(verb: pd.DataFrame, dimension: str, top: int = 10) -> pd.DataFrame:
    """Ranking por volumen y tasa de comentarios negativos."""
    if verb is None or verb.empty or dimension not in verb.columns or "polaridad" not in verb.columns:
        return pd.DataFrame()
    grupos = verb.groupby(dimension)
    salida = pd.DataFrame(
        {
            "total": grupos.size(),
            "negativos": grupos["polaridad"].apply(lambda serie: (serie == "Negativo").sum()),
        }
    ).reset_index()
    salida["% negativos"] = (salida["negativos"] / salida["total"] * 100).round(1)
    return salida[salida["total"] >= 5].sort_values(["negativos", "% negativos"], ascending=False).head(top)


def detectar_intencion(pregunta: str) -> str | None:
    """Infiera el tipo de análisis pedido en lenguaje natural."""
    pregunta_n = pregunta.lower()
    negativa = any(palabra in pregunta_n for palabra in ["malo", "negativ", "queja", "detractor", "dolor", "peor", "insatisf"])
    if negativa and any(palabra in pregunta_n for palabra in ["intermediar", "asesor", "agente", "corredor"]):
        return "neg_intermediario"
    if negativa and any(palabra in pregunta_n for palabra in ["sucursal", "cod_suc", "oficina"]):
        return "neg_sucursal"
    if negativa and any(palabra in pregunta_n for palabra in ["linea", "línea", "producto", "ramo"]):
        return "neg_linea"
    return "neg_general" if negativa else None


def construir_contexto(verb: pd.DataFrame, kpis: dict[str, object], pregunta: str) -> tuple[str, pd.DataFrame]:
    """Arma el contexto estructurado para responder preguntas del copiloto."""
    partes = [
        "KPIs del período filtrado:",
        f"- NPS: {kpis.get('nps')}, INS: {kpis.get('ins')}, CES: {kpis.get('ces')}",
        f"- Respuestas: {kpis.get('respuestas')}, Verbatims: {kpis.get('verbatims')}",
        f"- Filtros activos: {kpis.get('filtros')}",
    ]
    tabla = pd.DataFrame()
    if verb is None or verb.empty:
        partes.append("\nNo hay verbatims disponibles para el filtro actual.")
        return "\n".join(partes), tabla
    intencion = detectar_intencion(pregunta)
    dimension = {
        "neg_intermediario": next((col for col in ("intermediario", "nombre_intermediario", "cod_intermediario") if col in verb.columns), None),
        "neg_sucursal": next((col for col in ("cod_suc", "sucursal") if col in verb.columns), None),
        "neg_linea": "linea" if "linea" in verb.columns else None,
    }.get(intencion)
    if dimension:
        tabla = ranking_negativos(verb, dimension)
        if not tabla.empty:
            partes.append(f"\nRanking de comentarios negativos por {dimension} (mín. 5 comentarios):")
            partes.append(tabla.to_string(index=False))
    if "polaridad" in verb.columns:
        partes.append("\nDistribución de sentimiento:")
        partes.append(verb["polaridad"].value_counts().to_string())
    if "tema" in verb.columns:
        partes.append("\nTemas más mencionados:")
        partes.append(verb["tema"].value_counts().head(8).to_string())
    if "emocion" in verb.columns:
        partes.append("\nEmociones detectadas:")
        partes.append(verb["emocion"].value_counts().head(8).to_string())
    if "_texto" in verb.columns and "polaridad" in verb.columns:
        muestras = verb[verb["polaridad"] == "Negativo"]["_texto"].head(15).tolist()
        if muestras:
            partes.append("\nComentarios negativos de ejemplo:")
            partes.extend(f"- {_limpiar(texto)}" for texto in muestras)
    return "\n".join(partes), tabla


def responder(pregunta: str, verb: pd.DataFrame, kpis: dict[str, object]) -> tuple[str, pd.DataFrame]:
    """Responde usando analítica determinística y, si existe, un LLM."""
    contexto, tabla = construir_contexto(verb, kpis, pregunta)
    cliente = cliente_llm()
    if cliente is None:
        base = "**Modo analítico** (sin `HF_TOKEN` configurado).\n\n"
        if not tabla.empty:
            fila = tabla.iloc[0]
            base += (
                f"El mayor volumen de comentarios negativos lo concentra **{fila.iloc[0]}** con **{int(fila['negativos'])}** "
                f"de {int(fila['total'])} comentarios ({fila['% negativos']}% negativos)."
            )
        else:
            base += "No pude derivar un ranking con la dimensión solicitada."
        return base, tabla
    try:
        respuesta = cliente.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"},
            ],
            max_tokens=700,
            temperature=0.2,
        )
        return respuesta.choices[0].message.content, tabla
    except Exception as error:
        return f"No fue posible consultar el modelo `{MODELO_LLM}`: {error}", tabla
