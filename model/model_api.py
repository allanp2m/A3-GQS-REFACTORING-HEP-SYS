from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

# Import flexível: permite executar como módulo ou script direto.
try:  # Tentativa com import relativo
    from .prediction_service import (
        HepatitisPredictor,
        PredictionRepository,
        make_default_paths,
        get_db_url_from_env,
    )
except ImportError:  # Fallback caso executado fora de pacote
    from prediction_service import (
        HepatitisPredictor,
        PredictionRepository,
        make_default_paths,
        get_db_url_from_env,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    # Instancia o preditor e o repositório opcional (MySQL se configurado)
    paths = make_default_paths()
    predictor = HepatitisPredictor(paths)
    repo = PredictionRepository(get_db_url_from_env())

    @app.route("/train", methods=["POST"])
    def train_endpoint():
        # Re-treina o modelo manualmente
        try:
            result = predictor.train()
            return jsonify({"ok": True, **result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        # Realiza predição para um único registro enviado em JSON
        try:
            payload: Dict[str, Any] = request.get_json(force=True) or {}

            # Validação mínima: exige pelo menos um campo chave
            if not any(k in payload for k in ["Age", "ALB", "ALT", "AST"]):
                return jsonify({"error": "Payload vazio ou sem campos esperados."}), 400

            result = predictor.predict(payload)
            repo.log(payload, result)  # Registro opcional (ignorado se sem DB)

            response = {
                "prediction": result.get("prediction"),
                "label": result.get("label"),
            }
            if result.get("confidence") is not None:
                response["accuracy"] = result.get("confidence")
            return jsonify(response)
        except Exception as e:
            # Se DEBUG=1 retorna traceback para facilitar análise
            debug = os.getenv("DEBUG") == "1"
            err_payload = {"error": str(e)}
            if debug:
                err_payload["traceback"] = traceback.format_exc()
            return jsonify(err_payload), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)