"""
Day3: 드리프트 감지 -> 데이터 검증 -> fine-tuning 재학습 -> 재배포를 잇는 파이프라인의 핵심 조립 지점.

흐름: 이상 탐지(RMSE > 1.0°C) -> 알림 -> 서빙된 최근 7일 데이터 -> 데이터 검증(센서 고장이면 중단) ->
      Production 가중치에서 이어서 fine-tuning(warm start) -> 게이트 재검증 ->
      Production 재배포 (통과 못하면 기존 버전 유지)

왜 재학습 전에 검증하는가: 오차가 커진 원인은 모델이 낡았거나(드리프트) 데이터가 망가졌거나(센서 고장)
둘 중 하나입니다. 고장 데이터로 재학습하면 고장 패턴을 배운 모델이 만들어지므로, 데이터부터 확인합니다.

왜 "처음부터 재학습"이 아니라 fine-tuning인가: 최근 7일(168시간)만으로 LSTM을
스크래치로 학습시키기엔 샘플이 적어 불안정합니다. 이미 전체 데이터로 학습된
Production 가중치에서 이어서 짧게 미세조정하는 쪽이 훨씬 안정적입니다.

데이터는 /predict/batch-test로 들어와 검증을 통과한 연속 관측값입니다(routers/predict.py의 served_rows).
드리프트를 판정한 데이터로 재학습해야 하므로 업로드 CSV를 쓰지 않습니다. 192시간이 안 쌓였으면 기다립니다.
"""
import logging

from serving_app.monitoring.drift_detector import is_drift

logger = logging.getLogger("aiops")

RETRAIN_HOURS = 168  # 재학습에 쓰는 최근 데이터 길이 (7일)


def check_and_trigger(recent_predictions: list[dict], served_rows: list[dict]) -> dict:
    if not is_drift(recent_predictions):
        return {"status": "ok"}

    logger.warning("[WARN] drift detected - triggering retrain")

    from data.features import SEQ_LEN
    from serving_app import model_loader
    from serving_app.monitoring.data_validator import columns, validate
    from serving_app.train_and_register import fine_tune

    need = RETRAIN_HOURS + SEQ_LEN
    if len(served_rows) < need:  # 짧은 데이터로 재학습하면 새 상태를 배우지 못한다 - 그동안 /predict는 직전 값으로
        logger.warning(f"[WAIT] served data {len(served_rows)}/{need} hours - retrain postponed, keeping current production")
        return {"status": "insufficient_data", "rows": len(served_rows)}
    rows = list(served_rows)
    rule = validate(*columns(rows))
    if rule:  # 배치 경계를 걸친 고장 등 - 고장이 섞인 데이터로는 재학습하지 않는다
        logger.warning(f"[SKIP] sensor fault (rule={rule}) - retrain blocked")
        return {"status": "sensor_fault", "rule": rule}

    logger.info(f"[INFO] retrain triggered (window=last_{RETRAIN_HOURS}_hours)")
    result = fine_tune(rows)
    if result["promoted"]:
        logger.info(f"[OK] new_rmse={result['rmse']:.2f} - production promoted: ServerRoom_Temp v{result['version']}")
        model_loader._model_cache = None  # 재배포: 다음 요청에서 새 Production 버전을 다시 로드
        recent_predictions.clear()  # 이전 모델의 오차로 다시 드리프트 판정·재학습하지 않도록 비운다
        return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"]}
    logger.warning(f"[FAIL] new_rmse={result['rmse']:.2f} (production_rmse={result['production_rmse']:.2f}) - gate not passed, keeping current production")
    return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"]}
