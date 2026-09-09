import unittest
from unittest import mock
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from utils import nlp
from utils.informe import nombre_archivo_base
from utils.nlp import TAXONOMIA, clasificar_tema, validar_emocion
from utils.tablero import (
    comparativo_lineas,
    construir_opciones_tipo,
    deduplicar_columnas,
    es_intermediario,
    fuente_datos,
    ordenar_sucursales,
    sucursales_disponibles,
)

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
        self.assertIn("if es_intermediario(tipo):", APP)
        self.assertIn('with st.expander("Diagnóstico"', APP)
        self.assertIn("def obtener_nlp_vigente(", APP)


class HelperTests(unittest.TestCase):
    def test_deduplicar_columnas_conserva_primera_ocurrencia(self):
        df = pd.DataFrame([[1, 2]], columns=["col", "col"])
        salida = deduplicar_columnas(df)
        self.assertEqual(salida.columns.tolist(), ["col"])
        self.assertEqual(salida.iloc[0, 0], 1)

    def test_es_intermediario_reconoce_variantes(self):
        self.assertTrue(es_intermediario("intermediarios "))
        self.assertTrue(es_intermediario("Corredor"))
        self.assertFalse(es_intermediario("Cliente"))

    def test_construir_opciones_tipo_normaliza_union(self):
        resumen_df = pd.DataFrame({"tipo_encuesta": ["cliente "]})
        suc_df = pd.DataFrame({"tipo_encuesta": [" corredor "]})
        verb_df = pd.DataFrame({"tipo_encuesta": ["cliente", "otro"]})
        self.assertEqual(construir_opciones_tipo(resumen_df, suc_df, verb_df, ["cliente"]), ["TODOS", "CLIENTE", "INTERMEDIARIO", "OTRO"])

    def test_fuente_datos_cae_a_resumen_por_sucursal_para_intermediario(self):
        resumen_df = pd.DataFrame({"tipo_encuesta": ["CLIENTE"]})
        suc_df = pd.DataFrame({"tipo_encuesta": ["CORREDOR"]})
        self.assertIs(fuente_datos("INTERMEDIARIO", "TODAS", resumen_df, suc_df), suc_df)

    def test_orden_sucursales_es_numerico(self):
        self.assertEqual(ordenar_sucursales(["10", "2", "A", "1"]), ["1", "2", "10", "A"])

    def test_sucursales_disponibles_filtra_periodo_linea(self):
        df = pd.DataFrame(
            {
                "tipo_encuesta": ["INTERMEDIARIO", "CORREDOR", "INTERMEDIARIO"],
                "linea": ["VIDA", "AUTO", "VIDA"],
                "anio_mes": ["2026-08", "2026-08", "2026-07"],
                "cod_suc": ["2", "1", "3"],
                "encuestados": [5, 2, 4],
            }
        )
        opciones, conteos, historico = sucursales_disponibles(df, "VIDA", "2026-08")
        self.assertEqual(opciones, ["TODAS", "2"])
        self.assertEqual(conteos["2"], 5)
        self.assertFalse(historico)

    def test_sucursales_disponibles_relaja_periodo(self):
        df = pd.DataFrame(
            {
                "tipo_encuesta": ["INTERMEDIARIO"],
                "linea": ["VIDA"],
                "anio_mes": ["2026-07"],
                "cod_suc": ["2"],
                "encuestados": [4],
            }
        )
        opciones, _, historico = sucursales_disponibles(df, "VIDA", "2026-08")
        self.assertEqual(opciones, ["TODAS", "2"])
        self.assertTrue(historico)

    def test_comparativo_lineas_agrega_total_y_delta(self):
        actual = pd.DataFrame(
            {
                "linea": ["A", "B"],
                "encuestados": [10, 10],
                "promotores": [8, 4],
                "neutros": [1, 2],
                "detractores": [1, 4],
                "avg_ins": [9.0, 7.0],
                "avg_ces": [2.0, 3.0],
            }
        )
        previo = pd.DataFrame(
            {
                "linea": ["A", "B"],
                "encuestados": [10, 10],
                "promotores": [7, 5],
                "neutros": [1, 2],
                "detractores": [2, 3],
                "avg_ins": [8.5, 7.5],
                "avg_ces": [2.3, 2.8],
            }
        )
        salida = comparativo_lineas(actual, previo)
        self.assertIn("TOTAL / Promedio compañía", salida["linea"].tolist())
        fila_a = salida[salida["linea"] == "A"].iloc[0]
        self.assertEqual(fila_a["delta_nps"], 20.0)
        self.assertAlmostEqual(fila_a["pct_promotores"], 80.0)

    def test_taxonomia_valida_emocion_y_tema(self):
        self.assertEqual(validar_emocion("Negativo", "Satisfacción"), TAXONOMIA["Negativo"]["default"])
        self.assertIn(clasificar_tema("La app demora mucho y el portal falla"), {"demoras", "portal web o app"})

    def test_nombre_archivo_base_es_autodescriptivo(self):
        self.assertEqual(nombre_archivo_base("Verbatims", "2026-08", "Vida", "Intermediario", "12", "csv"), "base_cx_verbatims_2026-08_vida_intermediario_suc12.csv")

    @mock.patch("utils.nlp.st.progress")
    @mock.patch("utils.nlp.st.status")
    @mock.patch("utils.nlp.pipe_sentimiento")
    def test_enriquecer_dos_veces_no_duplica_columnas(self, pipe_sentimiento_mock, status_mock, progress_mock):
        class DummyProgress:
            def progress(self, *args, **kwargs):
                return None

        class DummyStatus:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, *args, **kwargs):
                return None

            def update(self, *args, **kwargs):
                return None

        pipe_sentimiento_mock.return_value = lambda textos, **kwargs: [{"label": "POS", "score": 0.99} for _ in textos]
        progress_mock.return_value = DummyProgress()
        status_mock.return_value = DummyStatus()
        nlp.cache_resultados().clear()
        base = pd.DataFrame({"texto": ["Excelente atención", "Muy buen servicio"]})
        torch_stub = mock.Mock()
        torch_stub.inference_mode.return_value = nullcontext()
        with mock.patch.dict("sys.modules", {"torch": torch_stub}):
            una_vez = nlp.enriquecer(base, muestra=2, batch_size=2)
            dos_veces = nlp.enriquecer(una_vez, muestra=2, batch_size=2)
        self.assertFalse(dos_veces.columns.duplicated().any())
        self.assertEqual(dos_veces.columns.tolist().count("_texto"), 1)
        self.assertEqual(dos_veces.columns.tolist().count("polaridad"), 1)


if __name__ == "__main__":
    unittest.main()
