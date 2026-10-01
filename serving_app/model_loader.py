"""
Day1 -> Day2(MLflow 연동) 확장 파일.

Day1 실습 목표: Lazy Loading vs Eager Loading 두 방식을 직접 구현하고
서버 시작 시간 / 첫 요청 응답 시간을 비교합니다. (44번 슬라이드 결과표 참고)
LSTM은 로컬 pickle 모델보다 로딩 자체가 무거워서, 이 비교가 Day1보다 오히려
더 체감됩니다.

Day2 실습 목표: MODEL_SOURCE=mlflow 로 전환해, 로컬 .keras 파일 대신
MLflow Model Registry의 Production 버전을 로드하도록 확장합니다.
main.py / train_and_register.py 코드는 그대로 두고 이 파일만 손대면 되도록
설계되어 있습니다 - 이것이 "조립 블록" 구조입니다.

스케일러는 물리 범위로 고정되어(data/features.py SensorScaler) Day1~3 모든 모델 버전이
같은 정규화 기준을 씁니다 - 정규화 기준이 바뀌면 이미 학습된 가중치와 어긋나기 때문입니다.

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값, Day1) | mlflow(Day2+)
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""
import os
import time

from data.features import SensorScaler

LOCAL_MODEL_PATH = "serving_app/models/server_room_v1.keras"
MLFLOW_MODEL_NAME = "ServerRoom_Temp"
MLFLOW_ALIAS = "production"  # 이 alias가 가리키는 버전을 로드 (train_and_register.PROD_ALIAS)

_model_cache = None  # Lazy Loading 캐시


class LoadedModel:
    """local .keras와 mlflow 두 소스를 동일한 인터페이스로 감싸는 래퍼."""

    def __init__(self, keras_model, scaler: SensorScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, sequence: list[dict]) -> float:
        """
        sequence: [{"temp": ..., "load_kw": ...}, ...] 길이 SEQ_LEN, 오래된 시간 -> 최근 시간 순서.
        """
        return self.predict_many([sequence])[0]

    def predict_many(self, sequences: list[list[dict]]) -> list[float]:
        """여러 시퀀스를 keras 호출 한 번으로 예측 (batch-test의 슬라이딩 윈도우 24건)."""
        import numpy as np

        x = np.array([[self.scaler.transform_point(p["temp"], p["load_kw"]) for p in s] for s in sequences], dtype="float32")
        return [self.scaler.inverse_temp(float(v)) for v in self._keras_model.predict(x, verbose=0).flatten()]


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    return LoadedModel(keras_model=keras_model, scaler=SensorScaler(), version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """Day2: MLflow Registry의 ServerRoom_Temp @production alias 버전을 로드한다 (train_and_register.py가 등록·승격)."""
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient

    # 버전 번호를 먼저 정하고 그 버전을 로드한다 - 응답의 model_version으로 재배포 여부를 바로 확인할 수 있게
    v = MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_ALIAS).version
    keras_model = mlflow.tensorflow.load_model(f"models:/{MLFLOW_MODEL_NAME}/{v}")
    return LoadedModel(keras_model=keras_model, scaler=SensorScaler(), version=f"v{v}")


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 모델을 로드한다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
