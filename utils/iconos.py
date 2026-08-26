from html import escape

import streamlit as st


ICONOS = {
    "alerta": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='m10.29 3.86-1.82 3.63a2 2 0 0 1-1.23 1l-3.87 1.11a2 2 0 0 0-.52 3.64l2.8 2.17a2 2 0 0 1 .73 2.22l-.8 3.83a2 2 0 0 0 2.9 2.11l3.45-1.78a2 2 0 0 1 1.84 0l3.45 1.78a2 2 0 0 0 2.9-2.11l-.8-3.83a2 2 0 0 1 .73-2.22l2.8-2.17a2 2 0 0 0-.52-3.64l-3.87-1.11a2 2 0 0 1-1.23-1l-1.82-3.63a2 2 0 0 0-3.58 0Z'/><path d='M12 8v4'/><path d='M12 16h.01'/></svg>",
    "bot": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 8V4H8'/><rect width='16' height='12' x='4' y='8' rx='2'/><path d='M2 14h2'/><path d='M20 14h2'/><path d='M15 13v2'/><path d='M9 13v2'/></svg>",
    "cerebro": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 18V5'/><path d='M15 13a3 3 0 1 0-6 0'/><path d='M17.5 6.5a2.5 2.5 0 0 1 0 5'/><path d='M6.5 11.5a2.5 2.5 0 0 1 0-5'/><path d='M18 17a3 3 0 0 0-3-3h-1'/><path d='M6 17a3 3 0 0 1 3-3h1'/><path d='M17.5 17.5a2.5 2.5 0 0 1 0-5'/><path d='M6.5 12.5a2.5 2.5 0 0 0 0 5'/></svg>",
    "check": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M20 6 9 17l-5-5'/></svg>",
    "comentarios": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M7.9 20A9 9 0 1 0 4 16.1L2 22Z'/></svg>",
    "dashboard": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 3h7v9H3z'/><path d='M14 3h7v5h-7z'/><path d='M14 12h7v9h-7z'/><path d='M3 16h7v5H3z'/></svg>",
    "drivers": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 2v20'/><path d='m5 7 7-4 7 4'/><path d='m5 17 7 4 7-4'/><path d='M5 12h14'/></svg>",
    "evolucion": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 3v18h18'/><path d='m19 9-5 5-4-4-3 3'/></svg>",
    "filtros": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 3H2l8 9.46V19l4 2v-8.54z'/></svg>",
    "ins": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 9a2 2 0 1 0-4 0c0 3-3 4-3 7a5 5 0 0 0 10 0c0-3-3-4-3-7z'/></svg>",
    "limpiar": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='m3 2 18 18'/><path d='M21 6H8.7'/><path d='m21 10-5.5 5.5'/><path d='m13.5 7 4-4'/><path d='M8.5 7H3'/><path d='m7 12 5 5'/><path d='M3 18h6'/></svg>",
    "lineas": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polygon points='3 6 9 3 15 6 9 9'/><polygon points='9 9 15 6 21 9 15 12'/><polygon points='3 12 9 9 15 12 9 15'/><path d='M3 6v6'/><path d='M9 15v6'/><path d='M15 12v6'/><path d='M21 9v6'/></svg>",
    "mapa": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M20 10c0 6-8 11-8 11s-8-5-8-11a8 8 0 0 1 16 0Z'/><circle cx='12' cy='10' r='3'/></svg>",
    "nps": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='m16 8-8 8'/><path d='m8 8 8 8'/></svg>",
    "radar": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M19.07 4.93A10 10 0 1 0 21 12'/><path d='M12 2v10'/><path d='m12 12 7-7'/><path d='M12 12H2'/></svg>",
    "recargar": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'/><path d='M3 21v-5h5'/><path d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'/><path d='M16 8h5V3'/></svg>",
    "respuestas": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 2H6a2 2 0 0 0-2 2v16l4-3h10a2 2 0 0 0 2-2V8z'/><path d='M14 2v6h6'/></svg>",
    "termometro": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 14.76V3.5a2 2 0 0 0-4 0v11.26a4 4 0 1 0 4 0Z'/></svg>",
    "verbatims": "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/></svg>",
}


def icono(nombre: str, size: int = 20, color: str = "currentColor") -> str:
    svg = ICONOS.get(nombre) or ICONOS["alerta"]
    return svg.format(size=int(size), color=escape(color, quote=True))


def texto_icono(nombre: str, texto: str, size: int = 20, color: str = "currentColor") -> str:
    return (
        "<span class='cx-icon-text'>"
        f"{icono(nombre, size=size, color=color)}"
        f"<span>{escape(texto)}</span>"
        "</span>"
    )


def titulo_seccion(nombre_icono: str, texto: str) -> None:
    st.markdown(
        f"<div class='section'>{texto_icono(nombre_icono, texto, size=20)}</div>",
        unsafe_allow_html=True,
    )
