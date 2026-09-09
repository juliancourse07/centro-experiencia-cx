"""Estilos y paleta corporativa del tablero CX."""

from __future__ import annotations

import streamlit as st

PALETA = {
    "azul": "#072B7A",
    "azul_claro": "#0B4AE2",
    "naranja": "#F37021",
    "verde": "#00A651",
    "amarillo": "#F2B705",
    "rojo": "#D64045",
    "gris": "#8A94A6",
    "fondo": "#F4F7FB",
    "blanco": "#FFFFFF",
    "borde": "#E3E9F5",
    "texto_suave": "#6B7280",
}

BANDAS = {
    "NPS": {"rango": (-100, 100), "bajo": 50.0, "alto": 70.0, "invertido": False},
    "INS": {"rango": (0, 10), "bajo": 7.0, "alto": 9.0, "invertido": False},
    "CES": {"rango": (0, 5), "bajo": 2.5, "alto": 3.5, "invertido": True},
}


def css_base() -> str:
    """Devuelve los estilos corporativos del tablero."""
    return """
    <style>
    .main .block-container{max-width:1800px;padding-top:1rem;padding-bottom:3rem;}
    .stApp{background:#F4F7FB;}
    #MainMenu, footer {visibility:hidden;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.65rem;}
    .hero{background:linear-gradient(135deg,#072B7A 0%,#0B4AE2 55%,#F37021 160%);border-radius:24px;padding:28px 34px;color:#fff;box-shadow:0 10px 30px rgba(7,43,122,.25);margin-bottom:22px;}
    .hero h1{margin:0;font-size:34px;font-weight:800;letter-spacing:-.5px;}
    .hero .sub{opacity:.9;font-size:15px;margin-top:2px;}
    .hero .big{font-size:22px;font-weight:600;margin-top:14px;}
    .section{font-size:20px;font-weight:700;color:#072B7A;margin:20px 0 10px;padding-bottom:6px;border-bottom:2px solid #E3E9F5;}
    .kpi{background:#fff;border-radius:18px;padding:18px 20px;height:100%;box-shadow:0 3px 16px rgba(7,43,122,.08);border-top:5px solid #F37021;}
    .kpi-title{color:#6B7280;font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;}
    .kpi-value{font-size:38px;font-weight:800;color:#072B7A;line-height:1.15;margin:4px 0;}
    .kpi-delta{font-size:13px;font-weight:600;}
    .kpi-foot{font-size:11.5px;color:#8A94A6;}
    .badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;color:#fff;}
    .insight{background:#fff;border-left:6px solid #072B7A;padding:14px 18px;border-radius:12px;margin-bottom:10px;box-shadow:0 2px 10px rgba(7,43,122,.06);font-size:14.5px;}
    .insight.warn{border-left-color:#F37021;}
    .insight.bad{border-left-color:#D64045;}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.cx-answer){background:#fff;border-left:6px solid #0B4AE2;border-radius:12px;box-shadow:0 2px 10px rgba(7,43,122,.06);}
    .cx-answer{font-size:13px;color:#0B4AE2;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px;}
    .cx-icon-text{display:inline-flex;align-items:center;gap:.5rem;line-height:1.2;}
    .cx-icon-text svg{display:block;flex:0 0 auto;}
    </style>
    """


def aplicar_estilos() -> None:
    """Inyecta el CSS corporativo en la aplicación."""
    st.markdown(css_base(), unsafe_allow_html=True)
