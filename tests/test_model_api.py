import os
import unittest
from unittest.mock import patch, MagicMock
from flask import json

from model import model_api 

app = model_api.app
app.testing = True 

PREDICTOR_PATH = 'model.model_api.HepatitisPredictor'
REPO_PATH = 'model.model_api.PredictionRepository'

class TestModelApiEndpoints(unittest.TestCase):
    
    def setUp(self):
        self.client = app.test_client()
        
        self.test_payload = {
            "Age": 45, "Sex": "m", "ALB": 40.2, "ALP": 60.1, "ALT": 15.7, 
            "AST": 22.3, "BIL": 5.1, "CHE": 7.2, "CHOL": 3.9, "CREA": 90, 
            "GGT": 25.4, "PROT": 70
        }
        
        self.mock_prediction_result = {
            "prediction": 0,
            "label": "0=Blood Donor",
            "confidence": 0.9876
        }
        

        self.mock_train_metrics = {
            "accuracy": 0.95,
            "classes": ["Live", "Die"]
        }
        
        if "DEBUG" in os.environ:
             del os.environ["DEBUG"]
        
       
        with patch.dict('sys.modules', { 'model.model_api': model_api }):
            model_api.DEBUG_MODE = False
            self.client = app.test_client()



    @patch(PREDICTOR_PATH)
    @patch(REPO_PATH)
    def test_predict_success_calls_predictor_and_repo_no_debug(self, MockRepo, MockPredictor):
        """Testa a predição em modo normal: chama preditor/repo e remove 'confidence'."""
        
        
        MockPredictor.return_value.predict.return_value = self.mock_prediction_result.copy()
        
        
        mock_repo_instance = MockRepo.return_value
        
        response = self.client.post('/predict', json=self.test_payload)
        data = response.get_json()

        
        self.assertEqual(response.status_code, 200)
        
        
        MockPredictor.return_value.predict.assert_called_once()
        
        
        mock_repo_instance.log.assert_called_once()
        
        
        self.assertIn("prediction", data)
        self.assertNotIn("confidence", data)
        self.assertEqual(data["label"], self.mock_prediction_result["label"])

    @patch.dict(os.environ, {"DEBUG": "1"})
    @patch(PREDICTOR_PATH)
    def test_predict_success_in_debug_mode_shows_confidence(self, MockPredictor):
        """Testa a predição em modo DEBUG: 'confidence' DEVE ser mantido."""
        
       
        with patch('model.model_api.DEBUG_MODE', True, create=True):
            MockPredictor.return_value.predict.return_value = self.mock_prediction_result.copy()
            response = self.client.post('/predict', json=self.test_payload)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.get_json()
        self.assertIn("confidence", data)
        self.assertEqual(data["confidence"], 0.9876)

    @patch(PREDICTOR_PATH)
    def test_predict_invalid_payload_returns_400(self, MockPredictor):
        """Testa se um payload não-JSON retorna erro 400."""
        response = self.client.post('/predict', data='nao eh json', content_type='text/plain')
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid JSON payload", response.get_json()["error"])
        MockPredictor.return_value.predict.assert_not_called()

    @patch(PREDICTOR_PATH)
    def test_predict_internal_error_returns_500_no_details(self, MockPredictor):
        """Testa se um erro interno retorna 500 sem detalhes em modo normal."""
        MockPredictor.return_value.predict.side_effect = ValueError("Erro no pipeline de ML")
        
        response = self.client.post('/predict', json=self.test_payload)
        data = response.get_json()
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("Prediction failed", data["error"])
        self.assertNotIn("details", data) 
        
    @patch.dict(os.environ, {"DEBUG": "1"})
    @patch(PREDICTOR_PATH)
    def test_predict_internal_error_returns_500_with_details_in_debug(self, MockPredictor):
        """Testa se um erro interno retorna 500 COM detalhes em modo DEBUG."""
        
        error_message = "Erro catastrófico no pré-processamento"
        MockPredictor.return_value.predict.side_effect = Exception(error_message)

        with patch('model.model_api.DEBUG_MODE', True, create=True):
            response = self.client.post('/predict', json=self.test_payload)
        
        data = response.get_json()
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("Prediction failed", data["error"])
        self.assertIn("details", data)
        self.assertIn(error_message, data["details"])



    @patch(PREDICTOR_PATH)
    def test_train_success_returns_metrics(self, MockPredictor):
        """Testa se o treino bem-sucedido retorna 200 e métricas."""
        MockPredictor.return_value.train.return_value = self.mock_train_metrics
        
        response = self.client.post('/train')

        self.assertEqual(response.status_code, 200)
        MockPredictor.return_value.train.assert_called_once()
        
        data = response.get_json()
        self.assertIn("Model re-trained successfully", data["message"])
        self.assertEqual(data["metrics"], self.mock_train_metrics)

    @patch(PREDICTOR_PATH)
    def test_train_internal_error_returns_500(self, MockPredictor):
        """Testa se um erro no treino retorna 500."""
        error_message = "Dados insuficientes para treino"
        MockPredictor.return_value.train.side_effect = Exception(error_message)
        
        response = self.client.post('/train')
        data = response.get_json()
        
        self.assertEqual(response.status_code, 500)
        self.assertIn(error_message, data["error"])


if __name__ == '__main__':
    unittest.main(verbosity=2)