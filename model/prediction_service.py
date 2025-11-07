from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Importa SQLAlchemy apenas se disponível para não criar dependência rígida
try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine, MetaData, Table
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


@dataclass
class Paths:
    data_csv: Path
    model_pkl: Path


class HepatitisPredictor:
    """Serviço de predição (POO) para o dataset de Hepatite.

    Responsabilidades:
    - Carregar e dividir o dataset
    - Montar pipeline de pré-processamento + modelo
    - Treinar, salvar, carregar
    - Predizer a partir de um dicionário de entrada
    """

    def __init__(self, paths: Paths, *, random_state: int = 1, n_neighbors: int = 5) -> None:
        self.paths = paths
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        self.pipeline: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None

        # Metadados do dataset
        self.target_col = "Category"
        self.categorical_cols: List[str] = ["Sex"]
        self.numeric_cols: List[str] = [
            "Age", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
        ]
        self.expected_cols: List[str] = ["Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE",
                                         "CHOL", "CREA", "GGT", "PROT"]
        self._debug_enabled = os.getenv("DEBUG") == "1"
        if self._debug_enabled:
            logging.basicConfig(level=logging.INFO, format="[HEP-PREDICT] %(message)s")

    # ---------- Public API ----------
    def train(self, *, test_size: float = 0.2) -> Dict[str, Any]:
        df = self._load_dataset()
        self._debug("Dataset loaded with shape %s" % (df.shape,))

    # Codifica rótulos do alvo
        y = df[self.target_col].astype(str)
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        X = df.drop(columns=[self.target_col])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=self.random_state, stratify=y_enc
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self._debug("Model trained. Classes: %s" % (self.label_encoder.classes_,))

        score = float(self.pipeline.score(X_test, y_test))
        self._save()
        return {"accuracy": round(score, 4), "classes": self.label_encoder.classes_.tolist()}

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self.pipeline is not None and self.label_encoder is not None

    # Monta DataFrame com colunas esperadas e tipos corretos
        X_df = self._payload_to_frame(payload)
        self._debug(f"Payload after normalization: {X_df.to_dict(orient='records')}")

        proba = None
        try:
            proba_vec = self.pipeline.predict_proba(X_df)[0]
            proba = float(np.max(proba_vec))
        except Exception:
            proba = None

        pred_idx = int(self.pipeline.predict(X_df)[0])
        label = str(self.label_encoder.inverse_transform([pred_idx])[0])

        result = {
            "prediction": pred_idx,
            "label": label,
        }
        if proba is not None:
            result["confidence"] = round(proba, 4)
        self._debug(f"Prediction result: {result}")
        return result

    # ---------- Funções internas ----------
    def _build_pipeline(self) -> Pipeline:
        # Numéricos: imputação por média + padronização
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])

        # Categóricos: imputação por mais frequente + one-hot
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])
        return pipe

    def _load_dataset(self) -> pd.DataFrame:
        if not self.paths.data_csv.exists():
            raise FileNotFoundError(f"Dataset not found at {self.paths.data_csv}")
        df = pd.read_csv(self.paths.data_csv)
        # Remove coluna de índice sem nome caso exista
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"]) 
        return df

    def _payload_to_frame(self, payload: Dict[str, Any]) -> pd.DataFrame:
        # Normaliza as chaves do payload
        norm = {k.strip(): v for k, v in payload.items()}

        # Converte Sex para categorias do treino ('m'/'f')
        sex_val = norm.get("Sex")
        norm["Sex"] = self._normalize_sex(sex_val)

        # Garante todos os campos numéricos (usa NaN quando ausente)
        for k in self.numeric_cols:
            if k not in norm:
                norm[k] = np.nan
            else:
                val = self._to_float(norm[k])
                norm[k] = val if val is not None else np.nan

        # Idade como número
        age_val = self._to_float(norm.get("Age"))
        norm["Age"] = age_val if age_val is not None else np.nan

        # Mantém apenas as colunas esperadas, na ordem correta
        row = {col: norm.get(col) for col in self.expected_cols}
        return pd.DataFrame([row])

    @staticmethod
    def _normalize_sex(sex: Any) -> str:
        if sex is None:
            return None 
        if isinstance(sex, (int, float)):
            return "m" if int(sex) == 1 else "f"
        s = str(sex).strip().lower()
        if s in {"m", "male", "masculino", "homem"}:
            return "m"
        if s in {"f", "female", "feminino", "mulher"}:
            return "f"
        return s

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _save(self) -> None:
        assert self.pipeline is not None and self.label_encoder is not None
        self.paths.model_pkl.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
        }, self.paths.model_pkl)
        self._debug(f"Model state saved to {self.paths.model_pkl}")

    def _ensure_loaded(self) -> None:
        if self.pipeline is not None and self.label_encoder is not None:
            return
        if not self.paths.model_pkl.exists():
            
            self.train()
            return
        state = joblib.load(self.paths.model_pkl)
        self.pipeline = state["pipeline"]
        self.label_encoder = state["label_encoder"]
        self._debug("Model state loaded from disk.")

    def _debug(self, msg: str) -> None:
        if self._debug_enabled:
            logging.info(msg)


class PredictionRepository:
    """Optional MySQL-backed repository for logging predictions.

    Uses SQLAlchemy if available and DB_URL env var is set to mysql+pymysql://...
    Schema: predictions(id, created_at, payload_json, prediction, label, confidence)
    """

    def __init__(self, db_url: Optional[str]) -> None:
        self.enabled = bool(db_url and SQLALCHEMY_AVAILABLE)
        self.db_url = db_url
        self.engine = None
        self.table = None
        if self.enabled:
            self._init_db()

    def _init_db(self) -> None:
        assert self.db_url
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            metadata = MetaData()
            self.table = Table(
                "predictions",
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("created_at", DateTime, nullable=False),
                Column("payload_json", String(2048), nullable=True),
                Column("prediction", Integer, nullable=False),
                Column("label", String(128), nullable=False),
                Column("confidence", Float, nullable=True),
            )
            metadata.create_all(self.engine)
        except SQLAlchemyError as e:
            self.enabled = False
            self.engine = None
            self.table = None

    def log(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not self.enabled or not self.engine or not self.table:
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self.table.insert().values(
                        created_at=datetime.utcnow(),
                        payload_json=json.dumps(payload)[:2048],
                        prediction=int(result.get("prediction")),
                        label=str(result.get("label")),
                        confidence=float(result.get("confidence")) if result.get("confidence") is not None else None,
                    )
                )
        except Exception:
            pass


def make_default_paths() -> Paths:
    base = Path(__file__).resolve().parent
    data_csv = base / "HepatitisCdata.csv"
    model_pkl = base / "knn_model.pkl"
    return Paths(data_csv=data_csv, model_pkl=model_pkl)


def get_db_url_from_env() -> Optional[str]:
    # Expect DB_URL like: mysql+pymysql://user:pass@host:3306/dbname
    url = os.getenv("DB_URL")
    return url if url else None
