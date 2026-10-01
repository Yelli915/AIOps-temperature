"""
서버실 1시간 뒤 온도 예측용 LSTM 아키텍처 (Day1 baseline과 Day2 MLflow 학습이 공유).

90일(2,160시간) 데이터 + SEQ_LEN(24)을 적용하면 학습 시퀀스가 약 2,100개로,
파라미터(약 1.6만 개) 대비 샘플 비율이 충분합니다. 구조는 HAIC 실습과 같은 LSTM 3층
(32 -> 32 -> 16, 앞 두 층은 return_sequences=True로 다음 LSTM에 전체 시퀀스를 넘김) +
Dense 1층입니다 - CPU로 100 epoch을 학습해도 1분 내외면 끝납니다.
"""
from tensorflow import keras

from data.features import SEQ_LEN

N_FEATURES = 2  # (온도, IT 부하)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
