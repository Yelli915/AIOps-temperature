"""
LSTM vs "직전 값 그대로"(1시간 뒤 온도 = 지금 온도) 기준선 RMSE 비교.

모델이 단순 규칙보다 실제로 나은지 확인한다. 서버 없이 MLflow 등록 모델을 직접 불러 계산한다.
    학습 검증 구간: data/sample_server_room.csv의 시간순 뒤 20% (학습과 같은 분할)
    시나리오 배치: simulate_drift.py가 보내는 것과 같은 마지막 48시간 (예측 24건)

실행: python scripts/compare_baseline.py        # 기본 v1 (시연 시작 모델)
      python scripts/compare_baseline.py 6      # 다른 버전
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow.tensorflow
import numpy as np

from data.features import SEQ_LEN, SensorScaler, build_sequences, load_rows, rmse, train_test_split
from data.generate_server_room import scenarios
from serving_app.model_loader import MLFLOW_MODEL_NAME


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "1"
    model = mlflow.tensorflow.load_model(f"models:/{MLFLOW_MODEL_NAME}/{version}")
    scaler = SensorScaler()

    def predict(X) -> list[float]:
        return [scaler.inverse_temp(p) for p in model.predict(np.array(X, dtype="float32"), verbose=0).flatten()]

    X, y = build_sequences(load_rows("data/sample_server_room.csv"), scaler)
    _, _, X_test, y_test = train_test_split(X, y)
    last = [scaler.inverse_temp(x[-1][0]) for x in X_test]  # 시퀀스 마지막 시간의 온도
    print(f"{'구간':<20} {'v' + version:>6} {'직전 값':>8}")
    print(f"{'학습 검증 (' + str(len(y_test)) + '건)':<20} {rmse(y_test, predict(X_test)):>6.2f} {rmse(y_test, last):>8.2f}")

    for name in ("normal", "setpoint", "expansion"):
        rows = scenarios()[name][-2 * SEQ_LEN:]
        temps = [r["Temp"] for r in rows]
        X = [[scaler.transform_point(r["Temp"], r["LoadKW"]) for r in rows[i : i + SEQ_LEN]] for i in range(SEQ_LEN)]
        actual = temps[SEQ_LEN:]
        print(f"{name + ' 배치':<20} {rmse(actual, predict(X)):>6.2f} {rmse(actual, temps[SEQ_LEN - 1 : -1]):>8.2f}")


if __name__ == "__main__":
    main()
