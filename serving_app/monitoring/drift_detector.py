"""
Day3: RMSE·편향 기반 데이터 드리프트 판정.

판단 기준 - 최근 WINDOW_SIZE(24)건의 (predicted, actual) 쌍으로
    RMSE > RMSE_THRESHOLD(1.0°C)            : 큰 변화 (설정온도 22→25, 증설 400→500kW)
    |평균 오차| > BIAS_THRESHOLD(0.4°C)     : 작은 변화 - 0.5°C 상승은 RMSE 0.5라 RMSE 기준으로는 못 잡는다
너무 짧은 윈도우는 노이즈에 민감하고, 너무 긴 윈도우는 드리프트 반응이 느려진다 - 24는 "최근 하루치 예측"으로 정한 절충점.

편향을 따로 보는 이유: 랜덤 노이즈(σ 0.2)는 24건 평균에서 거의 상쇄되지만(정상 배치 |평균 오차| ≤ 0.05),
드리프트는 오차가 한쪽으로 쏠린다. 실측(v1, 2026-10-01): 설정온도 +0.5°C → 평균 오차 +0.41~0.48,
부하 +30kW(+0.6°C) → +0.52~0.59, +0.3°C → +0.23~0.30 (못 잡음 - 노이즈 수준에 가까운 변화).
"""
import statistics

from data.features import rmse

RMSE_THRESHOLD = 1.0  # °C, 기준 모델(Day2 MLflow base) RMSE 0.22°C(노이즈 σ 0.2)의 약 5배 - 체크포인트 A에서 확정
BIAS_THRESHOLD = 0.4  # °C, 정상 배치 |평균 오차| 최대 0.05의 8배 - 0.5°C 이상 변화부터 잡는다
WINDOW_SIZE = 24      # 최근 24건(하루치 예측) 기준


def is_drift(recent_predictions: list[dict]) -> bool:
    """recent_predictions: [{"predicted": float, "actual": float}, ...]"""
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    actual, predicted = [p["actual"] for p in window], [p["predicted"] for p in window]
    bias = statistics.fmean(a - p for a, p in zip(actual, predicted))
    return rmse(actual, predicted) > RMSE_THRESHOLD or abs(bias) > BIAS_THRESHOLD


def passes_gate(score: float, production_score: float | None = None) -> bool:
    """배포 게이트: RMSE ≤ RMSE_THRESHOLD, 그리고 production_score(같은 검증 데이터에서 현재 Production RMSE)가
    있으면 그보다 낮아야 승격 - 고정 게이트만 보면 더 나쁜 모델로 바뀔 수 있다."""
    return score <= RMSE_THRESHOLD and (production_score is None or score < production_score)
