"""
Day3: RMSE 기반 데이터 드리프트 판정.

판단 기준 - 최근 WINDOW_SIZE(24)건의 (predicted, actual) 쌍으로 RMSE를 계산해
RMSE_THRESHOLD(1.0°C)와 비교한다. 너무 짧은 윈도우는 노이즈에 민감하고,
너무 긴 윈도우는 드리프트 반응이 느려진다 - 24는 "최근 하루치 예측"으로 정한 절충점.
"""
RMSE_THRESHOLD = 1.0  # °C, 기준 모델(Day2 MLflow base) RMSE 0.22°C(노이즈 σ 0.2)의 약 5배 - 체크포인트 A에서 확정
WINDOW_SIZE = 24      # 최근 24건(하루치 예측) 기준


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    recent_predictions: [{"predicted": float, "actual": float}, ...]
    RMSE = sqrt( mean( (actual - predicted) ** 2 ) ), 빈 리스트는 0.0 (드리프트 없음)
    """
    if not recent_predictions:
        return 0.0
    squared = [(p["actual"] - p["predicted"]) ** 2 for p in recent_predictions]
    return (sum(squared) / len(squared)) ** 0.5


def is_drift(recent_predictions: list[dict]) -> bool:
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
