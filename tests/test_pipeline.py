"""
batch-test 상태 로직 확인 - 모델·MLflow 없이 (가짜 모델, 가짜 fine_tune).

served_rows 이어붙이기·새로 시작·상한 192, 고장 배치 차단, 드리프트 → fallback → 재학습 승격 후 초기화,
local 모드 재학습 차단, NaN 설정온도 차단을 순서대로 본다. 배포 게이트(passes_gate)와 편향 드리프트도 확인한다.

실행: python tests/test_pipeline.py
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException

from data.generate_server_room import scenarios
from serving_app import model_loader
from serving_app.monitoring.drift_detector import is_drift, passes_gate
from serving_app.routers import predict as api
from serving_app.schemas import BatchTestRequest, PredictRequest


class FakeModel:
    """직전 온도 + bias를 예측 - bias를 키우면 드리프트가 된다."""
    version = "fake"
    bias = 0.0

    def predict_many(self, windows):
        return [w[-1]["temp"] + self.bias for w in windows]

    def predict_one(self, sequence):
        return self.predict_many([sequence])[0]


fine_tune_rows = []


def fake_fine_tune(rows):
    fine_tune_rows.append(len(rows))
    return {"promoted": True, "rmse": 0.3, "version": "2"}


# retrain_trigger가 함수 안에서 import하는 fine_tune을 가짜로 (tensorflow·mlflow 로드 없이)
sys.modules["serving_app.train_and_register"] = types.ModuleType("serving_app.train_and_register")
sys.modules["serving_app.train_and_register"].fine_tune = fake_fine_tune


def send(rows, setpoints=None):
    req = BatchTestRequest(
        temps=[None if math.isnan(r["Temp"]) else r["Temp"] for r in rows],
        loads=[r["LoadKW"] for r in rows],
        setpoints=setpoints or [r["Setpoint"] for r in rows],
    )
    return api.batch_test(req).drift_check


def test_pipeline():
    os.environ["MODEL_SOURCE"] = "mlflow"
    model = FakeModel()
    model_loader._model_cache = model
    s = scenarios()
    normal = s["normal"]

    # 이어지는 배치는 겹치는 24시간을 빼고 쌓고, 192시간에서 자른다
    assert send(normal[0:48]) == {"status": "ok"} and len(api.served_rows) == 48
    assert send(normal[24:72]) == {"status": "ok"} and len(api.served_rows) == 72
    for i in range(48, 240 - 48 + 1, 24):
        send(normal[i : i + 48])
    assert len(api.served_rows) == 192 and api.served_rows[-1]["Temp"] == normal[-1]["Temp"]

    # 고장 배치: 쌓지도, 예측 기록을 남기지도 않는다
    before = (list(api.served_rows), list(api.recent_predictions))
    assert send(s["zero"][-48:])["status"] == "sensor_fault"
    assert (api.served_rows, api.recent_predictions) == before

    # 이어지지 않는 배치 → 새로 시작. 드리프트인데 192시간 미만 → 재학습 보류, /predict는 직전 값
    model.bias = 3.0
    assert send(normal[100:148]) == {"status": "insufficient_data", "rows": 48}
    assert api.use_fallback
    seq = [{"temp": r["Temp"], "load_kw": r["LoadKW"]} for r in normal[124:148]]
    resp = api.predict(PredictRequest(sequence=seq))
    assert resp.model_version == api.FALLBACK_VERSION and resp.predicted_temp == normal[147]["Temp"]

    # 192시간이 쌓이면 재학습 → 승격 → 오차 기록·fallback·모델 캐시 초기화
    checks = [send(normal[i : i + 48]) for i in range(0, 144 + 1, 24)]  # 48 → 72 → … → 192행 (7번째 배치)
    assert [c["status"] for c in checks[:-1]] == ["insufficient_data"] * 6
    check = checks[-1]
    assert check == {"status": "retrain_triggered", "promoted": True, "rmse": 0.3}
    assert fine_tune_rows == [192]
    assert api.recent_predictions == [] and not api.use_fallback and model_loader._model_cache is None

    # local 모드: 드리프트여도 재학습하지 않는다 (승격해도 v1-local을 다시 로드하므로)
    os.environ["MODEL_SOURCE"] = "local"
    model_loader._model_cache = model
    assert send(normal[0:48]) == {"status": "retrain_disabled"}
    assert fine_tune_rows == [192] and api.use_fallback

    # NaN 설정온도로 잔차 규칙(오프셋 고장)을 우회하지 못한다
    assert send(s["offset"][-48:], setpoints=[float("nan")] * 48)["rule"] == "setpoint_invalid"

    # /predict 고장 입력은 422
    try:
        api.predict(PredictRequest(sequence=[{"temp": 0.0, "load_kw": 400.0}] * 24))
        raise AssertionError("고장 입력인데 예측함")
    except HTTPException as e:
        assert e.status_code == 422 and e.detail["rule"] == "out_of_range"



def test_gate():
    assert passes_gate(0.3) and passes_gate(1.0)        # base 학습: 고정 게이트만
    assert not passes_gate(1.01)
    assert passes_gate(0.8, production_score=2.5)        # 재학습: Production보다 나아야
    assert not passes_gate(0.4, production_score=0.3)    # 게이트는 넘지만 현재보다 나쁨 → 승격 안 함
    assert not passes_gate(0.3, production_score=0.3)    # 같으면 교체하지 않는다
    assert not passes_gate(1.2, production_score=3.0)    # 현재보다 나아도 게이트 초과


def test_bias_drift():
    noise = [0.2 if i % 2 else -0.2 for i in range(24)]  # RMSE 0.2, 평균 0
    window = lambda shift: [{"predicted": 22.0, "actual": 22.0 + n + shift} for n in noise]
    assert not is_drift(window(0.0)) and not is_drift(window(0.3))  # 정상·노이즈 수준 변화
    assert is_drift(window(0.5)) and is_drift(window(-0.5))         # RMSE 0.54 < 1.0이지만 한쪽으로 쏠림
    assert is_drift([{"predicted": 22.0, "actual": 22.0 + 1.1 * (1 if i % 2 else -1)} for i in range(24)])  # RMSE만 큼
    assert not is_drift(window(0.5)[:23])                           # 24건 미만은 판단 안 함


if __name__ == "__main__":
    test_pipeline()
    test_gate()
    test_bias_drift()
    print("ok")
