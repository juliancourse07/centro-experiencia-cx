"""Constructores de gráficos Plotly reutilizables para el tablero CX."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.estilos import BANDAS, PALETA

COLORES_POLARIDAD = {
    "Positivo": PALETA["verde"],
    "Neutro": PALETA["amarillo"],
    "Negativo": PALETA["rojo"],
}

COLORES_EMOCION = {
    "Satisfacción": "#00A651",
    "Confianza": "#1FB36D",
    "Gratitud": "#54C78D",
    "Entusiasmo": "#89DBAE",
    "Indiferencia": "#F2B705",
    "Consulta/Solicitud": "#F5C94A",
    "Sugerencia": "#F8DB81",
    "Frustración": "#D64045",
    "Decepción": "#DE6165",
    "Enojo": "#E57F82",
    "Desconfianza": "#EC9FA1",
    "Preocupación": "#F2BEC0",
}


def figura_gauge(nombre: str, valor: float | None, metrica: str) -> go.Figure:
    """Construye un velocímetro corporativo para una métrica CX."""
    config = BANDAS[metrica]
    minimo, maximo = config["rango"]
    etiqueta, color = ("Sin dato", PALETA["gris"]) if valor is None else (None, None)
    if valor is not None:
        from utils.tablero import semaforo

        etiqueta, color = semaforo(metrica, valor)
    v = float(valor) if valor is not None and pd.notna(valor) else minimo
    pasos = (
        [
            {"range": [minimo, config["bajo"]], "color": "#D8F5D8"},
            {"range": [config["bajo"], config["alto"]], "color": "#FFF4C2"},
            {"range": [config["alto"], maximo], "color": "#FADADD"},
        ]
        if config["invertido"]
        else [
            {"range": [minimo, config["bajo"]], "color": "#FADADD"},
            {"range": [config["bajo"], config["alto"]], "color": "#FFF4C2"},
            {"range": [config["alto"], maximo], "color": "#D8F5D8"},
        ]
    )
    figura = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"font": {"size": 46, "color": PALETA["azul"]}},
            title={
                "text": f"<b>{nombre}</b><br><span style='font-size:12px;color:{color}'>{etiqueta}</span>",
                "font": {"size": 16, "color": PALETA["azul"]},
            },
            gauge={
                "axis": {"range": [minimo, maximo], "tickcolor": PALETA["gris"]},
                "bar": {"color": PALETA["azul"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": pasos,
                "threshold": {"line": {"color": PALETA["naranja"], "width": 4}, "thickness": 0.8, "value": v},
            },
        )
    )
    figura.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return figura


def figura_evolucion(df: pd.DataFrame) -> go.Figure:
    """Grafica la evolución mensual de NPS e INS."""
    from utils.tablero import calcular_nps

    agregados = {col: "mean" for col in ("avg_ins", "avg_ces") if col in df.columns}
    tendencia = df.groupby("anio_mes", as_index=False).agg(agregados) if agregados else pd.DataFrame({"anio_mes": []})
    nps_mes = df.groupby("anio_mes").apply(calcular_nps).reset_index(name="nps")
    tendencia = nps_mes.merge(tendencia, on="anio_mes", how="left").sort_values("anio_mes")
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(
        go.Scatter(
            x=tendencia["anio_mes"], y=tendencia["nps"], name="NPS", mode="lines+markers",
            line=dict(color=PALETA["azul"], width=3), marker=dict(size=8)
        ),
        secondary_y=False,
    )
    if "avg_ins" in tendencia.columns:
        figura.add_trace(
            go.Scatter(
                x=tendencia["anio_mes"], y=tendencia["avg_ins"], name="INS", mode="lines+markers",
                line=dict(color=PALETA["naranja"], width=3, dash="dot"), marker=dict(size=8)
            ),
            secondary_y=True,
        )
    figura.add_hline(y=70, line_dash="dash", line_color=PALETA["verde"], opacity=0.5, annotation_text="Meta NPS 70", annotation_position="top left")
    figura.update_yaxes(title_text="NPS", range=[-100, 100], secondary_y=False)
    figura.update_yaxes(title_text="INS", range=[0, 10], secondary_y=True, showgrid=False)
    figura.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0))
    return figura


def figura_composicion_nps(df: pd.DataFrame) -> go.Figure:
    """Grafica la composición porcentual de promotores, neutros y detractores."""
    columnas = [col for col in ("promotores", "neutros", "detractores") if col in df.columns]
    dist = df.groupby("linea", as_index=False)[columnas].sum()
    total = dist[columnas].sum(axis=1).replace(0, pd.NA)
    for columna in columnas:
        dist[columna] = dist[columna] / total * 100
    largo = dist.melt(id_vars="linea", value_vars=columnas, var_name="Segmento", value_name="Porcentaje")
    figura = px.bar(
        largo,
        x="Porcentaje",
        y="linea",
        color="Segmento",
        orientation="h",
        color_discrete_map={"promotores": PALETA["verde"], "neutros": PALETA["amarillo"], "detractores": PALETA["rojo"]},
        text=largo["Porcentaje"].map(lambda valor: f"{valor:.0f}%" if pd.notna(valor) else ""),
    )
    figura.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), barmode="stack", plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="", legend=dict(orientation="h", y=1.12, x=0))
    return figura


def figura_radar_linea(df: pd.DataFrame) -> go.Figure:
    """Compara el INS promedio por línea."""
    radar = df.groupby("linea", as_index=False)["avg_ins"].mean().dropna()
    figura = px.line_polar(radar, r="avg_ins", theta="linea", line_close=True, range_r=[0, 10])
    figura.update_traces(fill="toself", line_color=PALETA["azul"], fillcolor="rgba(11,74,226,.18)")
    figura.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
    return figura


def figura_heatmap_ins(df: pd.DataFrame) -> go.Figure:
    """Crea el mapa de calor INS por línea y tipo de encuesta."""
    pivote = df.pivot_table(index="linea", columns="tipo_encuesta", values="avg_ins", aggfunc="mean")
    figura = px.imshow(pivote, text_auto=".1f", aspect="auto", zmin=0, zmax=10, color_continuous_scale=[[0, PALETA["rojo"]], [0.7, PALETA["amarillo"]], [1, PALETA["verde"]]])
    figura.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="", coloraxis_showscale=False)
    return figura


def figura_drivers_matriz(rank: pd.DataFrame) -> go.Figure:
    """Crea la matriz de priorización de drivers."""
    mediana_x, mediana_y = rank["menciones"].median(), rank["impacto"].median()
    figura = px.scatter(rank, x="menciones", y="impacto", size="menciones", text="categoria", color="impacto", color_continuous_scale=[[0, PALETA["rojo"]], [0.5, PALETA["amarillo"]], [1, PALETA["verde"]]], size_max=55)
    figura.update_traces(textposition="top center", textfont_size=11)
    figura.add_vline(x=mediana_x, line_dash="dot", line_color=PALETA["gris"])
    figura.add_hline(y=mediana_y, line_dash="dot", line_color=PALETA["gris"])
    figura.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Volumen de menciones →", yaxis_title="← Desempeño (peor abajo)", coloraxis_showscale=False)
    return figura


def figura_drivers_pareto(rank: pd.DataFrame) -> go.Figure:
    """Crea un Pareto de menciones por driver."""
    pareto = rank.sort_values("menciones", ascending=False).copy()
    pareto["acumulado"] = (pareto["menciones"].cumsum() / pareto["menciones"].sum() * 100).round(1)
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(go.Bar(x=pareto["categoria"], y=pareto["menciones"], name="Menciones", marker_color=PALETA["azul"]), secondary_y=False)
    figura.add_trace(go.Scatter(x=pareto["categoria"], y=pareto["acumulado"], name="% acumulado", mode="lines+markers", line=dict(color=PALETA["naranja"], width=3)), secondary_y=True)
    figura.add_hline(y=80, line_dash="dash", line_color=PALETA["rojo"], opacity=0.6, secondary_y=True)
    figura.update_yaxes(title_text="Menciones", secondary_y=False)
    figura.update_yaxes(title_text="% acumulado", range=[0, 105], secondary_y=True, showgrid=False)
    figura.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0))
    return figura


def figura_donut_polaridad(df: pd.DataFrame) -> go.Figure:
    """Crea el donut de polaridad del sentimiento."""
    dist = df["polaridad"].fillna("Sin clasificar").value_counts().rename_axis("polaridad").reset_index(name="total")
    mapa = {**COLORES_POLARIDAD, "Sin clasificar": PALETA["gris"]}
    figura = px.pie(dist, values="total", names="polaridad", hole=0.58, color="polaridad", color_discrete_map=mapa)
    figura.update_traces(textinfo="percent+label")
    figura.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return figura


def figura_sunburst_sentimiento(df: pd.DataFrame) -> go.Figure:
    """Crea la jerarquía polaridad → emoción → tema."""
    base = df.fillna({"polaridad": "Sin clasificar", "emocion": "Sin clasificar", "tema": "Sin clasificar"})
    resumen = base.groupby(["polaridad", "emocion", "tema"], as_index=False).size()
    colores = {**COLORES_POLARIDAD, **COLORES_EMOCION, "Sin clasificar": PALETA["gris"]}
    figura = px.sunburst(resumen, path=["polaridad", "emocion", "tema"], values="size", color="emocion", color_discrete_map=colores)
    figura.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return figura


def figura_temas_negativos(df: pd.DataFrame) -> go.Figure:
    """Grafica los temas negativos más mencionados."""
    negativos = df[df["polaridad"] == "Negativo"]
    temas = negativos["tema"].fillna("Sin clasificar").value_counts().head(10).rename_axis("tema").reset_index(name="total")
    figura = px.bar(temas.sort_values("total"), x="total", y="tema", orientation="h", text="total", color="total", color_continuous_scale=[[0, "#F2BEC0"], [1, PALETA["rojo"]]])
    figura.update_layout(height=420, margin=dict(l=10, r=20, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Menciones negativas", yaxis_title="", coloraxis_showscale=False)
    return figura
