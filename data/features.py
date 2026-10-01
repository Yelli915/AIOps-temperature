"""
서버실 센서 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

Day1 baseline 학습(scripts/train_baseline_v1.py), Day2 MLflow 학습
(serving_app/train_and_register.py), Day3 fine-tuning 재학습
(monitoring/retrain_trigger.py)이 모두 이 모듈을 재사용합니다. 시퀀스 정의를
한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는 실무 사고를
방지할 수 있습니다.

입력 시퀀스: 최근 SEQ_LEN(24)시간의 (온도, IT 부하) - Setpoint(냉방 설정온도) 컬럼은 선택이며 검증에만 쓴다
타깃: 그다음 1시간의 온도
"""
import csv
import math

SEQ_LEN = 24  # LSTM 입력 윈도우 길이 (시간) - 최근 하루
# 물리 범위 - 정규화 기준이자 센서 검증 범위(serving_app/monitoring/data_validator.py가 가져다 쓴다)
TEMP_MIN, TEMP_MAX = 10.0, 40.0  # 서버실 항온 대역. -10~50으로 넓히면 0°C 고장을 놓친다
LOAD_MAX = 2000.0                # kW, 기준 400kW의 5배 - 증설(500kW)은 통과, 부하 센서 튐(1e308·inf)은 차단


def rmse(y_true, y_pred) -> float:
    """학습 검증·드리프트 판정·기준선 비교가 모두 쓰는 RMSE (°C)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def _float_or_nan(value: str) -> float:
    """센서 결측(빈 칸)을 오류 대신 NaN으로 읽는다 - 판정은 data_validator가 한다."""
    return float(value) if value.strip() else float("nan")


def load_rows(csv_path: str = "data/sample_server_room.csv") -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "Timestamp": r["Timestamp"],
                "Temp": _float_or_nan(r["Temp"]),
                "LoadKW": float(r["LoadKW"]),
                "Setpoint": float(r["Setpoint"]) if (r.get("Setpoint") or "").strip() else None,  # 선택 컬럼 - 잔차 규칙용
            }
            for r in reader
        ]
    return rows


class SensorScaler:
    """
    온도/IT 부하를 각각 [0, 1] 범위로 정규화하는 min-max 스케일러.

    LSTM은 스케일에 민감하기 때문에(트리 기반 모델과 달리) 반드시 정규화가 필요합니다.
    범위는 데이터에서 fit하지 않고 data_validator의 물리 범위(온도 10~40°C, 부하 0~2,000kW)로 고정합니다 -
    검증을 통과한 값은 모두 [0, 1] 안에 들어가므로, 설정온도 상향·증설 뒤에도 모델이 범위 밖을 외삽하지 않고,
    모든 모델 버전이 같은 정규화 기준을 공유합니다(파일로 저장할 필요도 없음).
    실험(2026-10-01): base RMSE 0.225 동일, 설정온도 28°C fine-tuning 1.31 → 0.49 (데이터 fit 스케일러 대비).
    """

    temp_min, temp_max = TEMP_MIN, TEMP_MAX
    load_min, load_max = 0.0, LOAD_MAX

    def _scale(self, value: float, lo: float, hi: float) -> float:
        return (value - lo) / (hi - lo)

    def _unscale(self, value: float, lo: float, hi: float) -> float:
        return value * (hi - lo) + lo

    def transform_point(self, temp: float, load_kw: float) -> list[float]:
        return [
            self._scale(temp, self.temp_min, self.temp_max),
            self._scale(load_kw, self.load_min, self.load_max),
        ]

    def scale_temp(self, temp: float) -> float:
        """타깃(1시간 뒤 온도)을 학습용으로 정규화. 입력 시퀀스와 같은 스케일을 써야
        손실(loss)이 과도하게 커지지 않고 학습이 안정적으로 수렴한다."""
        return self._scale(temp, self.temp_min, self.temp_max)

    def inverse_temp(self, scaled_temp: float) -> float:
        """모델이 뱉은 정규화된 예측값을 실제 °C로 되돌린다."""
        return self._unscale(scaled_temp, self.temp_min, self.temp_max)


def build_sequences(rows: list[dict], scaler: SensorScaler, seq_len: int = SEQ_LEN):
    """
    rows(시간순)에서 (SEQ_LEN, 2) 크기의 정규화된 입력 시퀀스와
    1시간 뒤 온도(정규화 전 실값) 타깃을 만든다.

    반환: X (n_samples, seq_len, 2), y (n_samples,) - y는 스케일 안 된 실제 온도
    """
    scaled_points = [scaler.transform_point(r["Temp"], r["LoadKW"]) for r in rows]
    temps = [r["Temp"] for r in rows]

    X, y = [], []
    for i in range(len(rows) - seq_len):
        X.append(scaled_points[i : i + seq_len])
        y.append(temps[i + seq_len])
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2):
    """시간 순서를 유지한 채 앞부분을 train, 뒷부분을 test로 나눈다 (미래 데이터 누수 방지)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
