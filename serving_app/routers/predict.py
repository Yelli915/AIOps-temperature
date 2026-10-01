"""
Day1 -> Day3(시뮬레이션 엔드포인트 추가) 확장 파일.

Day1: POST /predict - 최근 SEQ_LEN(24)시간 시퀀스로 1시간 뒤 온도 예측
Day3: POST /predict/batch-test - 드리프트 감지 시뮬레이션 시작점 (scripts/simulate_drift.py 참고)

두 엔드포인트 모두 예측 전에 monitoring/data_validator.py로 센서 값을 검증한다.
고장 입력으로는 예측도, 오차 누적도, 재학습도 하지 않는다.

재학습 데이터는 batch-test로 들어와 검증을 통과한 연속 관측값(served_rows)이다 - 드리프트를 판정한
데이터와 재학습하는 데이터가 같아야 한다. 배치는 "앞 SEQ_LEN시간 = 직전 배치의 끝"으로 이어 보낸다고 보고,
이어지지 않으면(다른 시계열, 고장으로 빠진 배치) 끊긴 데이터를 붙여 배우지 않도록 버퍼를 새로 시작한다.

드리프트가 감지됐는데 새 모델이 아직 승격되지 않았으면 /predict는 "직전 값 그대로"로 응답한다
(드리프트 중에는 기준선이 LSTM보다 약 10배 정확 - README 기준선 비교, scripts/compare_baseline.py).
"""
import logging
import threading

from fastapi import APIRouter, HTTPException

from data.features import SEQ_LEN
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.retrain_trigger import RETRAIN_HOURS, check_and_trigger
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.monitoring.data_validator import columns, validate

router = APIRouter()
logger = logging.getLogger("aiops")

# ponytail: 아래 상태(recent_predictions·served_rows·use_fallback)는 모듈 전역 - uvicorn 워커 1개 전제.
# 워커·컨테이너가 여럿이면 각자 따로 판정·재학습하므로 운영이면 공유 저장소로 (README 한계)

# Day3: 최근 예측 기록(actual/predicted)을 쌓아두는 슬라이딩 윈도우.
# monitoring/drift_detector.py의 WINDOW_SIZE(24)만큼만 유지한다.
recent_predictions: list[dict] = []
# batch-test는 스레드풀에서 동시에 돌 수 있다 - 두 요청이 같은 윈도우로 재학습을 두 번 돌리고 버전을 두 개 등록하지 않게 한 번에 하나만.
# ponytail: 전역 락이라 재학습(약 10초) 동안 다른 batch-test는 기다린다 - 운영이면 재학습을 작업 큐로 (README 한계)
_batch_lock = threading.Lock()

# 배치에 loads가 없으면(대시보드 배치) IT 부하는 기준값으로 채운다.
SIMULATED_LOAD_KW = 400.0

# 재학습 데이터 버퍼: 서빙에서 검증을 통과한 연속 관측값, 최근 RETRAIN_HOURS + SEQ_LEN(192)시간만 유지.
# ponytail: 메모리 버퍼라 서버 재시작 시 비워진다 - 운영이면 시계열 DB에 Timestamp와 함께 저장
served_rows: list[dict] = []
FALLBACK_VERSION = "fallback-last-value"
use_fallback = False  # 드리프트 감지 후 새 모델 승격 전까지 True


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    sequence = [p.model_dump() for p in req.sequence]
    rule = validate(*columns([{"Temp": p["temp"], "LoadKW": p["load_kw"], "Setpoint": p["setpoint"]} for p in sequence]))
    if rule:  # 고장 입력으로는 예측하지 않는다
        raise HTTPException(422, {"status": "sensor_fault", "rule": rule})
    if use_fallback:
        return PredictResponse(predicted_temp=sequence[-1]["temp"], model_version=FALLBACK_VERSION)
    model = model_loader.get_model()
    predicted_temp = model.predict_one(sequence)
    return PredictResponse(predicted_temp=round(predicted_temp, 2), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    Day3 드리프트 감지 시뮬레이션 엔드포인트.

    SEQ_LEN + N개의 연속 온도를 받아 길이 SEQ_LEN 슬라이딩 윈도우로 예측하고,
    (predicted, actual) 쌍을 recent_predictions에 누적한 뒤 드리프트를 판정한다.
    """
    temps = req.temps
    rule = validate(temps, req.loads, req.setpoints)
    if rule:  # 예측도, 오차 누적도, 재학습도 하지 않는다
        logger.warning(f"[SKIP] sensor fault (rule={rule}) - prediction & retrain blocked")
        return BatchTestResponse(predictions=[], drift_check={"status": "sensor_fault", "rule": rule})

    loads = req.loads or [SIMULATED_LOAD_KW] * len(temps)
    setpoints = req.setpoints or [None] * len(temps)
    rows = [{"Temp": t, "LoadKW": l, "Setpoint": s} for t, l, s in zip(temps, loads, setpoints)]
    global use_fallback
    with _batch_lock:
        if served_rows and served_rows[-SEQ_LEN:] == rows[:SEQ_LEN]:
            served_rows.extend(rows[SEQ_LEN:])  # 앞 SEQ_LEN시간은 직전 배치와 겹치는 문맥
        else:
            served_rows[:] = rows  # 이어지지 않는 배치 - 새 시계열로 시작
        del served_rows[: -(RETRAIN_HOURS + SEQ_LEN)]

        model = model_loader.get_model()
        windows = [[{"temp": r["Temp"], "load_kw": r["LoadKW"]} for r in rows[i : i + SEQ_LEN]] for i in range(len(rows) - SEQ_LEN)]
        predictions = model.predict_many(windows)
        recent_predictions.extend({"predicted": p, "actual": a} for p, a in zip(predictions, temps[SEQ_LEN:]))
        recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]  # WINDOW_SIZE 유지

        drift_check = check_and_trigger(recent_predictions, served_rows)
        use_fallback = drift_check["status"] != "ok" and not drift_check.get("promoted")
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
