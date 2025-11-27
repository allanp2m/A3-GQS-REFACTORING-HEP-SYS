import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from model.prediction_service import (
    HepatitisPredictor,
    Paths,
    PredictionRepository,
    SQLALCHEMY_AVAILABLE,
)


def make_dataset(tmpdir: Path) -> Path:
    """Cria um CSV sintetico de hepatite para uso nos testes."""
    rows = [
        {"Age": 25, "Sex": "m", "ALB": 30, "ALP": 85, "ALT": 45, "AST": 52, "BIL": 0.7, "CHE": 8.0, "CHOL": 180, "CREA": 0.9, "GGT": 25, "PROT": 70, "Category": "Live"},
        {"Age": 47, "Sex": "f", "ALB": 35, "ALP": 90, "ALT": 60, "AST": 58, "BIL": 1.1, "CHE": 8.5, "CHOL": 195, "CREA": 1.1, "GGT": 32, "PROT": 74, "Category": "Live"},
        {"Age": 52, "Sex": "m", "ALB": 28, "ALP": 102, "ALT": 75, "AST": 68, "BIL": 1.4, "CHE": 7.8, "CHOL": 210, "CREA": 1.3, "GGT": 44, "PROT": 69, "Category": "Die"},
        {"Age": 33, "Sex": "f", "ALB": 33, "ALP": 88, "ALT": 55, "AST": 54, "BIL": 0.9, "CHE": 8.3, "CHOL": 185, "CREA": 1.0, "GGT": 29, "PROT": 72, "Category": "Live"},
        {"Age": 61, "Sex": "m", "ALB": 26, "ALP": 110, "ALT": 80, "AST": 75, "BIL": 1.6, "CHE": 7.4, "CHOL": 220, "CREA": 1.5, "GGT": 50, "PROT": 67, "Category": "Die"},
        {"Age": 40, "Sex": "f", "ALB": 34, "ALP": 95, "ALT": 58, "AST": 56, "BIL": 1.0, "CHE": 8.1, "CHOL": 190, "CREA": 1.0, "GGT": 30, "PROT": 71, "Category": "Live"},
        {"Age": 29, "Sex": "m", "ALB": 29, "ALP": 89, "ALT": 52, "AST": 53, "BIL": 0.8, "CHE": 8.2, "CHOL": 175, "CREA": 0.95, "GGT": 27, "PROT": 70, "Category": "Live"},
        {"Age": 55, "Sex": "f", "ALB": 27, "ALP": 105, "ALT": 78, "AST": 70, "BIL": 1.5, "CHE": 7.6, "CHOL": 205, "CREA": 1.4, "GGT": 46, "PROT": 68, "Category": "Die"},
        {"Age": 43, "Sex": "m", "ALB": 32, "ALP": 92, "ALT": 59, "AST": 57, "BIL": 1.05, "CHE": 8.0, "CHOL": 188, "CREA": 1.05, "GGT": 33, "PROT": 73, "Category": "Live"},
        {"Age": 63, "Sex": "f", "ALB": 25, "ALP": 115, "ALT": 82, "AST": 78, "BIL": 1.7, "CHE": 7.2, "CHOL": 225, "CREA": 1.6, "GGT": 52, "PROT": 66, "Category": "Die"},
        {"Age": 37, "Sex": "m", "ALB": 31, "ALP": 91, "ALT": 57, "AST": 55, "BIL": 0.95, "CHE": 8.1, "CHOL": 182, "CREA": 1.02, "GGT": 31, "PROT": 72, "Category": "Live"},
        {"Age": 50, "Sex": "f", "ALB": 0, "ALP": 100, "ALT": 70, "AST": 65, "BIL": 1.3, "CHE": 7.9, "CHOL": 200, "CREA": 1.2, "GGT": 42, "PROT": 69, "Category": "Die"},
    ]
    df = pd.DataFrame(rows)
    csv_path = tmpdir / "HepatitisCdata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestHepatitisPredictor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        data_csv = make_dataset(self.tmpdir)
        paths = Paths(data_csv=data_csv, model_pkl=self.tmpdir / "knn_model.pkl")
        self.predictor = HepatitisPredictor(paths, random_state=0, n_neighbors=3)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_train_persists_model_and_returns_accuracy(self):
        """Treina o modelo e garante que salva o artefato e retorna metricas."""
        result = self.predictor.train(test_size=0.3)

        self.assertIn("accuracy", result)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertTrue(self.predictor.paths.model_pkl.exists())
        self.assertTrue(result.get("classes"))

    def test_predict_trains_when_model_missing(self):
        """Ao chamar predict sem modelo salvo, ele deve treinar e produzir saida."""
        if self.predictor.paths.model_pkl.exists():
            self.predictor.paths.model_pkl.unlink()

        payload = {"Age": 45, "Sex": "male", "ALB": 31, "ALT": 60}
        result = self.predictor.predict(payload)

        self.assertIn("prediction", result)
        self.assertIn("label", result)
        self.assertTrue(self.predictor.paths.model_pkl.exists())


class TestPredictionRepository(unittest.TestCase):
    def test_repository_disabled_without_url(self):
        """Sem URL de banco, o repositorio permanece desabilitado e nao falha ao logar."""
        repo = PredictionRepository(None)
        repo.log({"Age": 30}, {"prediction": 1, "label": "Live", "confidence": 0.9})
        self.assertFalse(repo.enabled)

    @unittest.skipIf(SQLALCHEMY_AVAILABLE, "Evita tocar em DB real quando SQLAlchemy esta instalado")
    def test_repository_remains_disabled_without_sqlalchemy(self):
        """Mesmo com URL, sem SQLAlchemy ele deve ficar desabilitado."""
        with mock.patch("model.prediction_service.SQLALCHEMY_AVAILABLE", False):
            repo = PredictionRepository("mysql+pymysql://user:pass@localhost/db")
            self.assertFalse(repo.enabled)

    @unittest.skipUnless(SQLALCHEMY_AVAILABLE, "SQLAlchemy nao esta instalado")
    def test_repository_init_invokes_db_setup(self):
        """Quando SQLAlchemy existe, _init_db deve ser invocado na construcao."""
        called = {}

        def fake_init_db(self):
            called["hit"] = True
            self.enabled = True

        with mock.patch.object(PredictionRepository, "_init_db", fake_init_db):
            PredictionRepository("mysql+pymysql://user:pass@localhost/db")

        self.assertTrue(called.get("hit"))


class TestInputNormalization(unittest.TestCase):
    """Casos unitarios focados em normalizacao de payload."""

    def setUp(self):
        paths = Paths(Path("dummy.csv"), Path("dummy.pkl"))
        self.predictor = HepatitisPredictor(paths)

    def test_normalize_sex_variants(self):
        """Varia as representacoes de sexo e espera 'm' ou 'f' normalizado."""
        cases = {
            "m": "m",
            "M": "m",
            "male": "m",
            "MALE": "m",
            "masculino": "m",
            "homem": "m",
            1: "m",
            "f": "f",
            "F": "f",
            "female": "f",
            "FEMALE": "f",
            "feminino": "f",
            "mulher": "f",
            0: "f",
            2: "f",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.predictor._normalize_sex(raw), expected)

    def test_payload_to_frame_orders_columns_and_ignores_extra(self):
        """Garante que colunas seguem ordem esperada e extras sao descartadas."""
        payload = {
            "PROT": 70,
            "Age": 45,
            "Sex": "MALE",
            "GGT": 25.4,
            "ALB": 40.2,
            "ALP": 60.1,
            "ALT": 15.7,
            "AST": 22.3,
            "BIL": 5.1,
            "CHE": 7.2,
            "CHOL": 3.9,
            "CREA": 90,
            "Extra": "ignore-me",
        }
        df = self.predictor._payload_to_frame(payload)
        self.assertEqual(list(df.columns), self.predictor.expected_cols)
        self.assertNotIn("Extra", df.columns)
        self.assertEqual(df.loc[0, "Sex"], "m")

    def test_payload_to_frame_fills_missing_with_nan(self):
        """Campos numericos ausentes devem virar NaN para o pipeline imputar."""
        payload = {"Age": 30, "Sex": "m", "ALB": 40}
        df = self.predictor._payload_to_frame(payload)
        self.assertTrue(pd.isna(df.loc[0, "ALP"]))
        self.assertTrue(pd.isna(df.loc[0, "GGT"]))

    def test_payload_to_frame_converts_numeric_strings(self):
        """Strings numericas sao convertidas para float."""
        payload = {"Age": "45", "Sex": "f", "ALB": "40.2", "ALP": "60"}
        df = self.predictor._payload_to_frame(payload)
        self.assertEqual(df.loc[0, "Age"], 45.0)
        self.assertEqual(df.loc[0, "ALB"], 40.2)

    def test_payload_to_frame_invalid_strings_to_nan(self):
        """Valores nao numericos viram NaN para nao quebrar o pipeline."""
        payload = {"Age": "na", "Sex": "m", "ALB": "oops"}
        df = self.predictor._payload_to_frame(payload)
        self.assertTrue(pd.isna(df.loc[0, "Age"]))
        self.assertTrue(pd.isna(df.loc[0, "ALB"]))


if __name__ == "__main__":
    # Verbosity=2 exibe status individual (ok/fail/skipped) de cada teste no terminal.
    unittest.main(verbosity=2)
