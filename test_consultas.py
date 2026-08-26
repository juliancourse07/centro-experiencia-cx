import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent
CONSULTAS = (REPO / "utils" / "consultas.py").read_text(encoding="utf-8")
APP = (REPO / "app.py").read_text(encoding="utf-8")
ICONOS = (REPO / "utils" / "iconos.py").read_text(encoding="utf-8")


class SourceRegressionTests(unittest.TestCase):
    def test_consultas_define_expr_fecha_robusta(self):
        self.assertIn("EXPR_FECHA =", CONSULTAS)
        self.assertIn("TRY_TO_DATE(TRIM({col}), 'M/d/yy')", CONSULTAS)
        self.assertIn("TRY_TO_DATE(TRIM({col}), 'yyyy-MM-dd')", CONSULTAS)
        self.assertIn("TRY_CAST({col} AS DATE)", CONSULTAS)

    def test_drivers_y_verbatims_usan_fecha_parseada(self):
        self.assertIn("WITH base AS (", CONSULTAS)
        self.assertIn("DATE_FORMAT(fecha_parsed, 'yyyy-MM')", CONSULTAS)
        self.assertIn("AND fecha_parsed IS NOT NULL", CONSULTAS)
        self.assertIn("ORDER BY fecha_parsed DESC", CONSULTAS)
        self.assertNotIn("DATE_FORMAT(CAST(fecha AS DATE), 'yyyy-MM')", CONSULTAS)

    def test_diagnostico_reporta_parseabilidad_de_fechas(self):
        self.assertIn("fecha_parseable", CONSULTAS)
        self.assertIn("fecha_no_parseable", CONSULTAS)
        self.assertIn("\"gold_cx_verbatims\": expr_fecha()", CONSULTAS)

    def test_app_habilita_resumen_por_sucursal_y_sucursal_todas(self):
        self.assertIn("from utils.consultas import diagnostico, drivers, resumen, resumen_por_sucursal, verbatims", APP)
        self.assertIn("df_sucursal = cargar_resumen_sucursal()", APP)
        self.assertIn("sucursal = st.selectbox(\"Sucursal\", opciones_suc, key=\"f_suc\")", APP)
        self.assertIn("df = df_sucursal if sucursal != \"TODAS\"", APP)

    def test_app_usa_iconos_inline_en_secciones_y_tarjetas(self):
        self.assertIn("from utils.iconos import icono, texto_icono, titulo_seccion", APP)
        self.assertIn("titulo_seccion(\"drivers\", \"Drivers de experiencia\")", APP)
        self.assertIn("card(\"Verbatims\", f\"{len(verb_f):,.0f}\"", APP)
        self.assertIn("icon_name=\"verbatims\"", APP)
        self.assertNotRegex(APP, re.compile(r"[📊🎯😊⚡🔥🧠💬🤖⚠✅❌😐😠📨📈🧩📍🌡🚨⬇▶🔄↺🔧]"))

    def test_iconos_module_expone_svg_helpers(self):
        self.assertIn("ICONOS = {", ICONOS)
        self.assertIn("def icono(", ICONOS)
        self.assertIn("def texto_icono(", ICONOS)
        self.assertIn("def titulo_seccion(", ICONOS)
        self.assertIn("<svg", ICONOS)


if __name__ == "__main__":
    unittest.main()
