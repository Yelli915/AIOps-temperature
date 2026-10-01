"""
Day2: MLflow로 서버실 온도 LSTM 모델을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격(@production alias).
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

실습 시나리오 (94번 슬라이드를 LSTM 버전으로 재구성):
    1) 서버실 센서 데이터로 base 모델 학습(100 epoch) -> RMSE 확인 (게이트 미달 가능)
    2) 게이트(1.0°C) 통과 시 Production으로 승격
    3) (Day3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 7일 데이터로
       20 epoch만 fine-tuning (처음부터 다시 학습하지 않음 - 168시간으로는 스크래치 학습이 불안정)

실행:
    (대시보드에서 센서 CSV를 먼저 업로드하세요 - data/sample_server_room.csv가 예시입니다)
    python scripts/train_baseline_v1.py     # Day1 로컬 모델 (선택)
    python serving_app/train_and_register.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import load_rows, build_sequences, rmse, train_test_split, SensorScaler
from data.storage import latest_upload
from serving_app.lstm_model import build_model
from serving_app.monitoring.data_validator import check_base_rows
from serving_app.monitoring.drift_detector import RMSE_THRESHOLD as RMSE_GATE, passes_gate

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 크게 흔들려
# 게이트 통과 여부가 운에 좌우됩니다(HAIC 관찰치: 2.22~5.29). numpy/tensorflow/python
# random을 한 번에 고정해 재현 가능한 학습 결과를 보장합니다.
# import 시점이 아니라 학습 함수 안에서 고정합니다 - 서빙 프로세스가 이 모듈을 import만 해도
# 전역 난수 상태가 바뀌지 않게, 그리고 재학습이 몇 번째든 같은 데이터면 같은 결과가 나오게.
SEED = 42

MODEL_NAME = "ServerRoom_Temp"
PROD_ALIAS = "production"  # MLflow stage(deprecated) 대신 alias - model_loader.MLFLOW_ALIAS와 같은 값
BASE_EPOCHS = 100  # 3층 LSTM + 90일(2,160시간) 데이터 기준 Day2 MLflow base RMSE 0.22°C (SEED 고정, Day1 로컬은 0.23°C - 2026-09-30 확인)
FINE_TUNE_EPOCHS = 20  # 서버실: 10 epoch·1e-4로는 설정온도 상향(22→25°C)에 적응하지 못해 게이트 미통과 (변경.md #27 실험)
FINE_TUNE_LR = 3e-4    # base 학습(1e-3)보다 낮은 학습률로 살짝만 갱신


def _prepare(rows: list[dict], scaler: SensorScaler):
    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_temp(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _register_if_gate_passed(model, run_id: str, score: float, current: float | None = None) -> dict:
    """게이트: RMSE ≤ RMSE_GATE, 그리고 current(같은 검증 데이터에서 현재 Production RMSE)가 있으면 그보다 낮아야 승격."""
    result = {"run_id": run_id, "rmse": score, "production_rmse": current, "promoted": False}
    if passes_gate(score, current):
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        # alias는 한 버전만 가리키므로 이전 Production에서 자동으로 옮겨진다 - 롤백은 alias를 이전 버전으로 다시 지정
        MlflowClient().set_registered_model_alias(MODEL_NAME, PROD_ALIAS, v.version)
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.2f} -> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(f"[GATE FAILED] rmse={score:.2f} (gate {RMSE_GATE}, production {current}) -> 배포 차단, 기존 Production 유지")
    return result


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용합니다.

    csv_path를 지정하지 않으면 data/uploads/에 가장 최근 업로드된 CSV를 사용합니다
    (data/storage.py의 latest_upload() - 대시보드에서 업로드한 파일).
    """
    keras.utils.set_random_seed(SEED)
    if rows is None:
        rows = load_rows(csv_path or latest_upload())
    check_base_rows(rows)
    scaler = SensorScaler()
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

        preds = [scaler.inverse_temp(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_metric("rmse", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


def fine_tune(rows: list[dict]) -> dict:
    """
    Day3: 현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 rows(최근 데이터)로
    짧게 fine-tuning합니다. rows가 적을 때(최근 7일 + 선행 24시간 = 192행)도 스크래치 학습보다 훨씬 안정적입니다.
    """
    keras.utils.set_random_seed(SEED)
    scaler = SensorScaler()
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}@{PROD_ALIAS}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    # 학습 전 = 현재 Production. 같은 검증 데이터에서 이보다 나아야 승격한다 (고정 게이트만 보면 더 나쁜 모델로 바뀔 수 있음)
    current = rmse(y_test, [scaler.inverse_temp(p) for p in model.predict(X_test, verbose=0).flatten()])

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)

        preds = [scaler.inverse_temp(p) for p in model.predict(X_test, verbose=0).flatten()]
        score = rmse(y_test, preds)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("production_rmse", current)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, current)


if __name__ == "__main__":
    train_and_register()
