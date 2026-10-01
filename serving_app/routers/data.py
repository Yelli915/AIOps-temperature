"""
서버실 센서 데이터(CSV) 업로드. 컬럼: Timestamp, Temp, LoadKW, 선택 Setpoint (최소 BASE_MIN_ROWS = 720행, base 학습용).

값(센서 고장)은 여기서 검증하지 않습니다 - 고장 데이터도 "들어온 데이터"로 저장하고,
판정은 base 학습 직전(check_base_rows)과 서빙 입력(routers/predict.py)에서 합니다.
Day3 재학습은 업로드가 아니라 서빙으로 들어온 데이터를 씁니다(routers/predict.py의 served_rows).

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, base 학습(train_baseline_v1.py, train_and_register.py)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import csv
import io
import math
import os
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import load_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.monitoring.data_validator import BASE_MIN_ROWS

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Timestamp", "Temp", "LoadKW"}
MIN_ROWS = BASE_MIN_ROWS  # 업로드는 base 학습용 - 30일(720행) 미만은 어차피 학습이 거부한다


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # 엑셀 "CSV UTF-8" 저장은 BOM이 붙어 첫 컬럼이 "\ufeffTimestamp"로 읽힌다
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")
    # 숫자가 아닌 값은 여기서 거부한다 - 저장되면 최신 업로드가 되어 학습·재학습·/data/status가 전부 500을 낸다.
    # 온도 빈 칸(센서 결측)만 허용 - 고장 판정은 data_validator가 한다.
    for line_no, r in enumerate(rows, start=2):
        try:
            if not math.isfinite(float(r["LoadKW"])):  # "nan"·"inf" 문자열도 float()는 통과한다
                raise ValueError
            if r["Temp"].strip() and math.isinf(float(r["Temp"])):  # inf는 /data/status JSON 변환에서 500 - "nan"은 결측으로 허용
                raise ValueError
            if (r.get("Setpoint") or "").strip() and not math.isfinite(float(r["Setpoint"])):
                raise ValueError
        except (ValueError, TypeError, AttributeError):  # 칸이 모자란 행은 None
            raise HTTPException(400, f"{line_no}행: Temp·LoadKW·Setpoint는 숫자여야 합니다 (Temp·Setpoint만 빈 칸 허용).")
    # 학습·재학습은 행 순서를 시간 순서로 보고 24행씩 자른다 - 뒤집히거나 빠진 시간이 있으면 엉뚱한 시퀀스를 배운다.
    try:
        times = [datetime.fromisoformat(r["Timestamp"]) for r in rows]
    except (ValueError, TypeError):
        raise HTTPException(400, "Timestamp는 '2026-01-01 00:00' 형식이어야 합니다.")
    if len({t.tzinfo is None for t in times}) > 1:  # 시간대 있는 값·없는 값이 섞이면 아래 뺄셈이 TypeError(500)
        raise HTTPException(400, "Timestamp의 시간대(+09:00 등) 표기가 행마다 달라서는 안 됩니다.")
    for line_no, (a, b) in enumerate(zip(times, times[1:]), start=3):
        if b - a != timedelta(hours=1):
            raise HTTPException(400, f"{line_no}행: Timestamp는 1시간 간격으로 오름차순이어야 합니다 ({a} → {b}).")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"server_room_{time.time_ns()}.csv")  # 초 단위면 1초 안에 두 번 올릴 때 덮어쓴다
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    temps = [r["Temp"] for r in rows if not math.isnan(r["Temp"])]  # 결측 제외하고 통계
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start": rows[0]["Timestamp"],
        "end": rows[-1]["Timestamp"],
        "min_temp": min(temps, default=None),  # 온도가 전부 결측이어도 500 대신 null
        "max_temp": max(temps, default=None),
    }
