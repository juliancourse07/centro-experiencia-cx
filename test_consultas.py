import unittest
from pathlib import Path

import pandas as pd

from utils.informe import nombre_archivo_base
from utils.nlp import TAXONOMIA, clasificar_tema, validar_emocion
from utils.tablero import construir_opciones_tipo, fuente_datos, ordenar_sucursales, sucursales_disponibles

REPO = Path(__file__).resolve().parent
CONSULTAS = (REPO / "utils" / "consultas.py").read_text(encoding="utf-8")
APP = (REPO / "app.py").read_text(encoding="utf-8")


class SourceRegressionTests(unittest.TestCase):
    def test_consultas_define_expr_fecha_robusta(self):
        self.assertIn("EXPR_FECHA =", CONSULTAS)
        self.assertIn("TRY_TO_DATE(TRIM({col}), 'M/d/yy')", CONSULTAS)
        self.assertIn("TRY_TO_DATE(TRIM({col}), 'yyyy-MM-dd')", CONSULTAS)
        self.assertIn("TRY_CAST({col} AS DATE)", CONSULTAS)

    def test_consultas_expone_tipos_disponibles_desde_kpis(self):
        self.assertIn("def tipos_encuesta_disponibles()", CONSULTAS)
        self.assertIn("SELECT DISTINCT UPPER(TRIM(tipo_encuesta)) AS tipo_encuesta", CONSULTAS)
        self.assertIn("FROM {T_KPIS}", CONSULTAS)

    def test_app_usa_tabs_y_oculta_multipagina(self):
        self.assertIn('tab_tablero, tab_drivers, tab_voz, tab_base, tab_informe, tab_copiloto = st.tabs(', APP)
        self.assertIn('if tipo == "INTERMEDIARIO":', APP)
        self.assertIn('st.session_state.pop("f_suc", None)', APP)


class HelperTests(unittest.TestCase):
    def test_construir_opciones_tipo_normaliza_union(self):
        resumen_df = pd.DataFrame({"tipo_encuesta": ["cliente "]})
        suc_df = pd.DataFrame({"tipo_encuesta": [" intermediario"]})
        verb_df = pd.DataFrame({"tipo_encuesta": ["cliente", "otro"]})
        self.assertEqual(construir_opciones_tipo(resumen_df, suc_df, verb_df, ["cliente"]), ["TODOS", "CLIENTE", "INTERMEDIARIO", "OTRO"])

    def test_fuente_datos_cae_a_resumen_por_sucursal_para_intermediario(self):
        resumen_df = pd.DataFrame({"tipo_encuesta": ["CLIENTE"]})
        suc_df = pd.DataFrame({"tipo_encuesta": ["INTERMEDIARIO"]})
        self.assertIs(fuente_datos("INTERMEDIARIO", "TODAS", resumen_df, suc_df), suc_df)

    def test_orden_sucursales_es_numerico(self):
        self.assertEqual(ordenar_sucursales(["10", "2", "A", "1"]), ["1", "2", "10", "A"])

    def test_sucursales_disponibles_filtra_periodo_linea(self):
        df = pd.DataFrame({"tipo_encuesta": ["INTERMEDIARIO", "INTERMEDIARIO"], "linea": ["VIDA", "AUTO"], "anio_mes": ["2026-08", "2026-08"], "cod_suc": ["2", "1"]})
        self.assertEqual(sucursales_disponibles(df, "VIDA", "2026-08"), ["TODAS", "2"])

    def test_taxonomia_valida_emocion_y_tema(self):
        self.assertEqual(validar_emocion("Negativo", "Satisfacción"), TAXONOMIA["Negativo"]["default"])
        self.assertIn(clasificar_tema("La app demora mucho y el portal falla"), {"demoras", "portal web o app"})

    def test_nombre_archivo_base_es_autodescriptivo(self):
        self.assertEqual(nombre_archivo_base("Verbatims", "2026-08", "Vida", "Intermediario", "12", "csv"), "base_cx_verbatims_2026-08_vida_intermediario_suc12.csv")


if __name__ == "__main__":
    unittest.main()
