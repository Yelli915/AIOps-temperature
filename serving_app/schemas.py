"""
Day1: FastAPI 요청/응답 Pydantic 스키마.

LSTM은 한 시점의 값이 아니라 최근 SEQ_LEN(24)시간의 흐름을 입력받아야 하므로,
/predict는 단일 행이 아니라 "24시간치 시퀀스"를 요청 본문으로 받습니다.
스키마는 형식(길이 SEQ_LEN, 부하 ge=0)만 강제하고, 온도 값이 센서 고장인지는
monitoring/data_validator.py가 판정합니다 - 온도는 0°C 이하도 가능하고, 결측(NaN/null)도
422로 튕기지 않고 "missing" 규칙으로 판정해야 하기 때문입니다.
"""
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from data.features import SEQ_LEN


class HourlyPoint(BaseModel):
    # None 허용: 브라우저의 JSON.stringify는 NaN을 null로 보내므로, 422 대신 validator가 "missing"으로 판정한다.
    temp: float | None = Field(..., description="해당 시간 랙 흡기 온도 (°C)")
    load_kw: float = Field(..., ge=0, description="해당 시간 IT 전력 부하 (kW)")
    setpoint: float | None = Field(None, description="해당 시간 냉방 설정온도 (°C) - 보내면 잔차 규칙으로 오프셋 고장을 가른다")


class PredictRequest(BaseModel):
    sequence: list[HourlyPoint] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 시간 -> 가장 최근 시간 순서의 최근 {SEQ_LEN}시간 시퀀스",
    )


class PredictResponse(BaseModel):
    predicted_temp: float
    model_version: str


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션에서 사용 (scripts/simulate_drift.py 참고)
    # SEQ_LEN + N 개의 연속된 온도를 보내면, 서버가 내부적으로 슬라이딩 윈도우로 잘라
    # 여러 건을 연속 예측한다. loads(IT 부하)를 생략하면 기준값 400kW로 채운다.
    temps: list[float | None] = Field(..., min_length=SEQ_LEN + 1)
    loads: list[Annotated[float, Field(ge=0)]] | None = Field(None, description="temps와 같은 길이의 IT 부하 (kW) - 서버 증설 시나리오용")
    setpoints: list[float] | None = Field(None, description="temps와 같은 길이의 냉방 설정온도 (°C) - 잔차 규칙용")

    @model_validator(mode="after")
    def _same_length(self):
        for name in ("loads", "setpoints"):
            if getattr(self, name) is not None and len(getattr(self, name)) != len(self.temps):
                raise ValueError(f"{name}는 temps와 길이가 같아야 합니다")
        return self


class BatchTestResponse(BaseModel):
    predictions: list[float]
    drift_check: dict
