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


def figura_vacia(mensaje: str = "Sin datos para los filtros seleccionados") -> go.Figure:
    """Devuelve una figura vacía con un mensaje estándar."""
    figura = go.Figure()
    figura.add_annotation(
        text=mensaje,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=PALETA["gris"]),
    )
    figura.update_xaxes(visible=False)
    figura.update_yaxes(visible=False)
    figura.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white")
    return figura


def _faltan_columnas(df: pd.DataFrame | None, columnas: tuple[str, ...]) -> bool:
    return df is None or df.empty or not set(columnas).issubset(df.columns)


def _color_nps(valor: float | None) -> str:
    if valor is None or pd.isna(valor):
        return PALETA["gris"]
    if valor >= 70:
        return PALETA["verde"]
    if valor > 50:
        return PALETA["amarillo"]
    return PALETA["rojo"]


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

    if _faltan_columnas(df, ("anio_mes",)):
        return figura_vacia()
    agregados = {col: "mean" for col in ("avg_ins", "avg_ces") if col in df.columns}
    tendencia = df.groupby("anio_mes", as_index=False).agg(agregados) if agregados else pd.DataFrame({"anio_mes": []})
    nps_mes = df.groupby("anio_mes").apply(calcular_nps).reset_index(name="nps")
    if nps_mes.empty and tendencia.empty:
        return figura_vacia()
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
    if _faltan_columnas(df, ("linea", "promotores", "neutros", "detractores")):
        return figura_vacia()
    columnas = [col for col in ("promotores", "neutros", "detractores") if col in df.columns]
    dist = df.groupby("linea", as_index=False)[columnas].sum()
    if dist.empty:
        return figura_vacia()
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
    if _faltan_columnas(df, ("linea", "avg_ins")):
        return figura_vacia()
    radar = df.groupby("linea", as_index=False)["avg_ins"].mean().dropna()
    if radar.empty:
        return figura_vacia()
    figura = px.line_polar(radar, r="avg_ins", theta="linea", line_close=True, range_r=[0, 10])
    figura.update_traces(fill="toself", line_color=PALETA["azul"], fillcolor="rgba(11,74,226,.18)")
    figura.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)")
    return figura


def figura_heatmap_ins(df: pd.DataFrame) -> go.Figure:
    """Crea el mapa de calor INS por línea y tipo de encuesta."""
    if _faltan_columnas(df, ("linea", "tipo_encuesta", "avg_ins")):
        return figura_vacia()
    pivote = df.pivot_table(index="linea", columns="tipo_encuesta", values="avg_ins", aggfunc="mean")
    if pivote.empty:
        return figura_vacia()
    figura = px.imshow(pivote, text_auto=".1f", aspect="auto", zmin=0, zmax=10, color_continuous_scale=[[0, PALETA["rojo"]], [0.7, PALETA["amarillo"]], [1, PALETA["verde"]]])
    figura.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="", coloraxis_showscale=False)
    return figura


def figura_drivers_matriz(rank: pd.DataFrame) -> go.Figure:
    """Crea la matriz de priorización de drivers."""
    if _faltan_columnas(rank, ("categoria", "menciones", "impacto")):
        return figura_vacia()
    mediana_x, mediana_y = rank["menciones"].median(), rank["impacto"].median()
    figura = px.scatter(rank, x="menciones", y="impacto", size="menciones", text="categoria", color="impacto", color_continuous_scale=[[0, PALETA["rojo"]], [0.5, PALETA["amarillo"]], [1, PALETA["verde"]]], size_max=55)
    figura.update_traces(textposition="top center", textfont_size=11)
    figura.add_vline(x=mediana_x, line_dash="dot", line_color=PALETA["gris"])
    figura.add_hline(y=mediana_y, line_dash="dot", line_color=PALETA["gris"])
    figura.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Volumen de menciones →", yaxis_title="← Desempeño (peor abajo)", coloraxis_showscale=False)
    return figura


def figura_drivers_pareto(rank: pd.DataFrame) -> go.Figure:
    """Crea un Pareto de menciones por driver."""
    if _faltan_columnas(rank, ("categoria", "menciones")):
        return figura_vacia()
    pareto = rank.sort_values("menciones", ascending=False).copy()
    if pareto.empty or pareto["menciones"].sum() <= 0:
        return figura_vacia()
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
    if _faltan_columnas(df, ("polaridad",)):
        return figura_vacia()
    dist = df["polaridad"].fillna("Sin clasificar").value_counts().rename_axis("polaridad").reset_index(name="total")
    if dist.empty:
        return figura_vacia()
    mapa = {**COLORES_POLARIDAD, "Sin clasificar": PALETA["gris"]}
    figura = px.pie(dist, values="total", names="polaridad", hole=0.58, color="polaridad", color_discrete_map=mapa)
    figura.update_traces(textinfo="percent+label")
    figura.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return figura


def figura_sunburst_sentimiento(df: pd.DataFrame) -> go.Figure:
    """Crea la jerarquía polaridad → emoción → tema."""
    if _faltan_columnas(df, ("polaridad", "emocion", "tema")):
        return figura_vacia()
    base = df.fillna({"polaridad": "Sin clasificar", "emocion": "Sin clasificar", "tema": "Sin clasificar"})
    resumen = base.groupby(["polaridad", "emocion", "tema"], as_index=False).size()
    if resumen.empty:
        return figura_vacia()
    colores = {**COLORES_POLARIDAD, **COLORES_EMOCION, "Sin clasificar": PALETA["gris"]}
    figura = px.sunburst(resumen, path=["polaridad", "emocion", "tema"], values="size", color="emocion", color_discrete_map=colores)
    figura.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return figura


def figura_temas_negativos(df: pd.DataFrame) -> go.Figure:
    """Grafica los temas negativos más mencionados."""
    if _faltan_columnas(df, ("polaridad", "tema")):
        return figura_vacia()
    negativos = df[df["polaridad"] == "Negativo"]
    temas = negativos["tema"].fillna("Sin clasificar").value_counts().head(10).rename_axis("tema").reset_index(name="total")
    if temas.empty:
        return figura_vacia()
    figura = px.bar(temas.sort_values("total"), x="total", y="tema", orientation="h", text="total", color="total", color_continuous_scale=[[0, "#F2BEC0"], [1, PALETA["rojo"]]])
    figura.update_layout(height=420, margin=dict(l=10, r=20, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Menciones negativas", yaxis_title="", coloraxis_showscale=False)
    return figura


def figura_ranking_lineas(df: pd.DataFrame) -> go.Figure:
    """Ranking horizontal de líneas según NPS."""
    if _faltan_columnas(df, ("linea", "nps")):
        return figura_vacia()
    base = df[df["linea"] != "TOTAL / Promedio compañía"].dropna(subset=["nps"]).sort_values("nps", ascending=True)
    if base.empty:
        return figura_vacia()
    referencia = df.loc[df["linea"] == "TOTAL / Promedio compañía", "nps"]
    promedio = referencia.iloc[0] if not referencia.empty else None
    figura = go.Figure(
        go.Bar(
            x=base["nps"],
            y=base["linea"],
            orientation="h",
            marker_color=[_color_nps(valor) for valor in base["nps"]],
            text=base["nps"].map(lambda valor: f"{valor:.1f}"),
            textposition="outside",
            name="NPS",
        )
    )
    if promedio is not None and pd.notna(promedio):
        figura.add_vline(x=promedio, line_dash="dot", line_color=PALETA["azul"], annotation_text=f"Promedio compañía {promedio:.1f}")
    figura.add_vline(x=70, line_dash="dash", line_color=PALETA["verde"], annotation_text="Meta 70")
    figura.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="NPS", yaxis_title="")
    return figura


def figura_dispersion_lineas(df: pd.DataFrame) -> go.Figure:
    """Dispersión de volumen vs. desempeño por línea."""
    columnas = ("linea", "respuestas", "nps")
    if _faltan_columnas(df, columnas):
        return figura_vacia()
    base = df[df["linea"] != "TOTAL / Promedio compañía"].copy()
    if base.empty:
        return figura_vacia()
    base["verbatims"] = base["verbatims"] if "verbatims" in base.columns else base["respuestas"]
    mediana_x = base["respuestas"].median()
    mediana_y = base["nps"].median()
    figura = px.scatter(
        base,
        x="respuestas",
        y="nps",
        size="verbatims",
        color="nps",
        hover_name="linea",
        text="linea",
        size_max=55,
        color_continuous_scale=[[0, PALETA["rojo"]], [0.5, PALETA["amarillo"]], [1, PALETA["verde"]]],
    )
    figura.update_traces(textposition="top center")
    figura.add_vline(x=mediana_x, line_dash="dot", line_color=PALETA["gris"])
    figura.add_hline(y=mediana_y, line_dash="dot", line_color=PALETA["gris"])
    for x, y, texto in [
        (0.02, 0.95, "Nicho excelente"),
        (0.98, 0.95, "Fortaleza masiva"),
        (0.02, 0.05, "Atención puntual"),
        (0.98, 0.05, "Riesgo crítico"),
    ]:
        figura.add_annotation(xref="paper", yref="paper", x=x, y=y, text=texto, showarrow=False, font=dict(size=11, color=PALETA["gris"]), xanchor="left" if x < 0.5 else "right")
    figura.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Respuestas", yaxis_title="NPS", coloraxis_showscale=False)
    return figura


def figura_delta_lineas(df: pd.DataFrame) -> go.Figure:
    """Brecha de NPS vs. promedio compañía por línea."""
    if _faltan_columnas(df, ("linea", "nps")):
        return figura_vacia()
    referencia = df.loc[df["linea"] == "TOTAL / Promedio compañía", "nps"]
    if referencia.empty or pd.isna(referencia.iloc[0]):
        return figura_vacia()
    promedio = referencia.iloc[0]
    base = df[df["linea"] != "TOTAL / Promedio compañía"].copy()
    if base.empty:
        return figura_vacia()
    base["brecha_compania"] = base["nps"] - promedio
    base = base.sort_values("brecha_compania", ascending=True)
    figura = go.Figure(
        go.Bar(
            x=base["brecha_compania"],
            y=base["linea"],
            orientation="h",
            marker_color=[PALETA["verde"] if valor >= 0 else PALETA["rojo"] for valor in base["brecha_compania"]],
            text=base["brecha_compania"].map(lambda valor: f"{valor:+.1f}"),
            textposition="outside",
        )
    )
    figura.add_vline(x=0, line_color=PALETA["azul"], line_width=2)
    figura.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Brecha vs. promedio compañía (puntos NPS)", yaxis_title="")
    return figura
