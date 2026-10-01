"""
서버실 온도·IT 부하 가상 데이터 생성 (도메인제안.md 5장 생성 규칙).

    부하 L(t) = L0 × (1 + 0.15 × sin(2π(t − 9) / 24)) + 노이즈
    온도 T(t) = S + a × (L(t) − 400) + 0.3 × sin(2πt / 24) + 노이즈

실행: python data/generate_server_room.py  → 샘플 CSV + 시나리오별 CSV 생성
"""
import csv
import math
import os
import random
from datetime import datetime, timedelta

SETPOINT = 22.0   # 냉방 설정온도 S (°C)
BASE_LOAD = 400.0  # 기준 IT 부하 L0 (kW)
A = 0.02           # 부하 1kW당 온도 상승 (°C/kW)
START = datetime(2026, 1, 1)


def generate(hours: int, start: int = 0, setpoint: float = SETPOINT, base_load: float = BASE_LOAD, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for t in range(start, start + hours):
        load = base_load * (1 + 0.15 * math.sin(2 * math.pi * (t - 9) / 24)) + rng.gauss(0, 5)
        temp = setpoint + A * (load - BASE_LOAD) + 0.3 * math.sin(2 * math.pi * t / 24) + rng.gauss(0, 0.2)
        rows.append({"t": t, "Temp": round(temp, 2), "LoadKW": round(load, 1), "Setpoint": setpoint})
    return rows


def with_fault(rows: list[dict], kind: str, at: int, hours: int = 6) -> list[dict]:
    """정상 rows의 at번째 행부터 센서 고장을 넣은 복사본. kind = zero | spike | missing | offset"""
    rows = [dict(r) for r in rows]
    if kind == "zero":
        for r in rows[at : at + hours]:
            r["Temp"] = 0.0
    elif kind == "spike":
        rows[at]["Temp"] += 15.0
    elif kind == "missing":
        rows[at]["Temp"] = float("nan")
    elif kind == "offset":  # 센서 교정 오프셋 +3°C - 온도만 보면 설정온도 22→25 변경과 같다 (Setpoint는 22 그대로)
        for r in rows[at : at + hours]:
            r["Temp"] = round(r["Temp"] + 3.0, 2)
    return rows


def with_step(history: list[dict], hours: int, **change) -> list[dict]:
    """history 뒤에 설정온도 상향(setpoint=25) 또는 서버 증설(base_load=500) 이후 데이터를 이어 붙인다."""
    return history + generate(hours, start=history[-1]["t"] + 1, seed=7, **change)


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Temp", "LoadKW", "Setpoint"])
        for r in rows:
            temp = "" if math.isnan(r["Temp"]) else r["Temp"]  # 결측은 빈 칸
            w.writerow([(START + timedelta(hours=r["t"])).strftime("%Y-%m-%d %H:%M"), temp, r["LoadKW"], r["Setpoint"]])


def scenarios() -> dict[str, list[dict]]:
    """시나리오별 240행 (재학습 192행 + 여유).

    센서 고장은 마지막 48행(배치 1개) 안에서 일어난다.
    진짜 드리프트는 72시간 전에 시작된 것으로 둔다 - 변화 후 데이터가 24시간뿐이면 재학습 데이터를
    시간순으로 나눴을 때 새 상태가 전부 검증 쪽으로 들어가, 모델이 배울 수 없어 게이트를 통과하지 못한다.
    """
    history = generate(240, start=2160)
    return {
        "normal": history,
        "zero": with_fault(history, "zero", at=220),
        "spike": with_fault(history, "spike", at=220),
        "missing": with_fault(history, "missing", at=220),
        "offset": with_fault(history, "offset", at=168, hours=72),  # setpoint와 같은 시점·크기, 설정온도만 그대로
        "setpoint": with_step(history[:168], 72, setpoint=25.0),
        "expansion": with_step(history[:168], 72, base_load=500.0),
    }


if __name__ == "__main__":
    write_csv(generate(2160), "data/sample_server_room.csv")  # 90일치
    for name, rows in scenarios().items():
        write_csv(rows, f"data/scenarios/{name}.csv")
    print("ok")
