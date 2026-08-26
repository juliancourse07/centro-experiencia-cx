"""Copiloto CX: responde con analítica real y redacta con un LLM gratuito."""

import os
import re

import pandas as pd
import streamlit as st

MODELO_LLM = os.getenv("CX_MODELO_LLM", "meta-llama/Llama-3.3-70B-Instruct")

# Compilado a nivel de módulo: NO puede ir dentro de un f-string (SyntaxError en Python < 3.12)
_ESPACIOS = re.compile(r"\s+")

SYSTEM = """Eres un analista de experiencia de cliente (CX) de una aseguradora colombiana.
Respondes en español, en máximo 6 viñetas, con foco en acción de negocio.
Usa ÚNICAMENTE los datos del contexto. Si el dato no está, dilo explícitamente.
No inventes cifras, nombres de intermediarios ni porcentajes."""


def _token():
    return (
        st.secrets.get("HF_TOKEN", None)
        if hasattr(st, "secrets") else None
    ) or os.getenv("HF_TOKEN")


@st.cache_resource
def cliente_llm():
    tok = _token()
    if not tok:
        return None
    from huggingface_hub import InferenceClient
    return InferenceClient(model=MODELO_LLM, token=tok)


def _limpiar(texto, limite: int = 220) -> str:
    """Colapsa espacios y recorta. Se usa fuera de f-strings a propósito."""
    return _ESPACIOS.sub(" ", str(texto)).strip()[:limite]


# ---------------------------------------------------------------
# Analítica determinística: esto NO lo hace el LLM
# ---------------------------------------------------------------

def ranking_negativos(verb: pd.DataFrame, dimension: str, top: int = 10) -> pd.DataFrame:
    """Ranking por volumen y tasa de comentarios negativos."""
    if verb is None or verb.empty or dimension not in verb.columns:
        return pd.DataFrame()
    if "polaridad" not in verb.columns:
        return pd.DataFrame()

    g = verb.groupby(dimension)
    out = pd.DataFrame({
        "total": g.size(),
        "negativos": g["polaridad"].apply(lambda s: (s == "Negativo").sum()),
    }).reset_index()
    out["% negativos"] = (out["negativos"] / out["total"] * 100).round(1)
    # Solo dimensiones con volumen mínimo, para no premiar ruido de 2 comentarios
    out = out[out["total"] >= 5]
    return out.sort_values(["negativos", "% negativos"], ascending=False).head(top)


def detectar_intencion(pregunta: str):
    p = pregunta.lower()
    neg = any(w in p for w in ["malo", "negativ", "queja", "detractor", "dolor", "peor", "insatisf"])
    if neg and any(w in p for w in ["intermediar", "asesor", "agente", "corredor"]):
        return "neg_intermediario"
    if neg and any(w in p for w in ["sucursal", "cod_suc", "oficina"]):
        return "neg_sucursal"
    if neg and any(w in p for w in ["linea", "línea", "producto", "ramo"]):
        return "neg_linea"
    if neg:
        return "neg_general"
    return None


def construir_contexto(verb: pd.DataFrame, kpis: dict, pregunta: str):
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

    dim = {
        "neg_intermediario": next(
            (c for c in ("intermediario", "nombre_intermediario", "cod_intermediario") if c in verb.columns), None
        ),
        "neg_sucursal": next((c for c in ("cod_suc", "sucursal") if c in verb.columns), None),
        "neg_linea": "linea" if "linea" in verb.columns else None,
    }.get(intencion)

    if dim:
        tabla = ranking_negativos(verb, dim)
        if not tabla.empty:
            partes.append(f"\nRanking de comentarios negativos por {dim} (mín. 5 comentarios):")
            partes.append(tabla.to_string(index=False))

    if "polaridad" in verb.columns:
        partes.append("\nDistribución de sentimiento:")
        partes.append(verb["polaridad"].value_counts().to_string())
    if "subsentimiento" in verb.columns:
        partes.append("\nTop sub-sentimientos:")
        partes.append(verb["subsentimiento"].value_counts().head(8).to_string())

    if "_texto" in verb.columns and "polaridad" in verb.columns:
        muestras = verb[verb["polaridad"] == "Negativo"]["_texto"].head(15).tolist()
        if muestras:
            partes.append("\nComentarios negativos de ejemplo:")
            for m in muestras:
                partes.append("- " + _limpiar(m))

    return "\n".join(partes), tabla


def responder(pregunta: str, verb: pd.DataFrame, kpis: dict):
    contexto, tabla = construir_contexto(verb, kpis, pregunta)
    cli = cliente_llm()

    if cli is None:
        base = "**Modo analítico** (sin `HF_TOKEN` configurado, no se generó redacción con LLM).\n\n"
        if not tabla.empty:
            fila = tabla.iloc[0]
            base += (
                f"El mayor volumen de comentarios negativos lo concentra "
                f"**{fila.iloc[0]}** con **{int(fila['negativos'])}** de {int(fila['total'])} "
                f"comentarios ({fila['% negativos']}% negativos).\n\n"
                "Ver tabla completa abajo."
            )
        else:
            base += "No pude derivar un ranking. Verifica que los verbatims tengan la dimensión consultada."
        return base, tabla

    try:
        resp = cli.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"},
            ],
            max_tokens=700,
            temperature=0.2,
        )
        return resp.choices[0].message.content, tabla
    except Exception as e:
        return f"No fue posible consultar el modelo (`{MODELO_LLM}`): {e}", tabla
