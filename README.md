# Centro de Experiencia CX

Tablero de control de experiencia de cliente (CX) para Seguros del Estado,
desplegado como **Databricks App** sobre Streamlit.

---

## Arquitectura de datos

```
SharePoint
    |
    v
01_ingesta_sharepoint          -> clientescx / intermediarioscx (bruto)
    |
    v
02_clientes_transformacion     -> cx_clientes_detalle        (silver)
03_intermediarios_transformacion -> cx_intermediarios_detalle (silver)
    |
    v
04_modelo_gold
    |
    +-- gold_cx_kpis        (1 fila por encuesta - base maestra)
    +-- gold_cx_resumen     (agregado anio_mes x tipo_encuesta x linea)
    +-- gold_cx_drivers     (1 fila por driver mencionado)
    +-- gold_cx_verbatims   (1 fila por comentario)
    |
    v
Databricks App (Streamlit)
```

### Tablas Gold

| Tabla | Grano | Uso en la app |
|---|---|---|
| `gold_cx_kpis` | 1 encuesta | Base maestra; se reagrega para abrir por `cod_suc` |
| `gold_cx_resumen` | anio_mes x tipo_encuesta x linea | KPIs, gauges, evolucion, radar, heatmap |
| `gold_cx_drivers` | 1 driver | Matriz dolor vs. punto de contacto, Pareto |
| `gold_cx_verbatims` | 1 comentario | Tabla, sentimiento, ranking, copiloto |

> **Nota:** `gold_cx_resumen` no expone `cod_suc`. El filtro de sucursal usa
> `resumen_por_sucursal()`, que reagrega `gold_cx_kpis` en vivo.

---

## Metricas y rangos oficiales

| Metrica | Escala | Alto | Medio | Bajo |
|---|---|---|---|---|
| **NPS** (%Promotores - %Detractores) | -100 a 100 | >= 70 | 50,1 - 70 | < 50 |
| **INS** (valoracion media) | 1 a 10 | >= 9 | 7,1 - 8,9 | < 7 |
| **CES** (esfuerzo) | 1 a 5 | Bajo esfuerzo < 2,5 | 2,5 - 3,5 | Alto esfuerzo > 3,5 |

En CES, menor es mejor.

---

## Estructura del proyecto

```
.
|-- app.py                 # Tablero principal
|-- app.yaml               # Configuracion de Databricks App
|-- requirements.txt
|-- utils/
|   |-- warehouse.py       # Conexion al SQL Warehouse (OAuth App Identity)
|   |-- consultas.py       # Consultas a la capa Gold
|   |-- nlp.py             # Sentimiento BERT + sub-sentimiento zero-shot
|   +-- copiloto.py        # Copiloto CX (analitico + LLM opcional)
+-- notebooks/             # Pipeline de ingesta y transformacion
```

---

## Variables de entorno

Se declaran en `app.yaml`.

| Variable | Obligatoria | Descripcion |
|---|---|---|
| `CX_CATALOGO` | Si | Catalogo de Unity Catalog |
| `CX_ESQUEMA` | Si | Schema de las tablas Gold |
| `HF_TOKEN` | No | Token de Hugging Face para el copiloto redactor |
| `CX_MODELO_LLM` | No | Modelo del copiloto |
| `CX_MODELO_SENT` | No | Modelo de sentimiento (no requiere token) |
| `CX_MODELO_ZS` | No | Modelo zero-shot de sub-sentimiento |

**Sin `HF_TOKEN` la app funciona igual:** el sentimiento BERT corre local y el
copiloto responde en modo analitico determinístico.

---

## Seguridad

- La app **no usa PAT ni credenciales embebidas**: autentica con la
  **App Identity** (`app-3ytui2 centro-experiencia-cx`).
- Permisos requeridos sobre el catalogo: `USAGE`, `SELECT`, `READ_METADATA`.
- Los secretos se declaran como **Resources** de la app, nunca en el codigo.
- **No versionar** archivos con datos de clientes: ver `.gitignore`.
