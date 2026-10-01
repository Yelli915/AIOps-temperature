"""
Day3 서버실 온도 드리프트·센서 고장 시뮬레이션.

각 시나리오(240시간)를 /predict/batch-test로 스트리밍한다: 48시간 배치를 24시간씩 밀어 9번 보낸다
(앞 24시간 = 직전 배치의 끝). 서버는 이어지는 배치를 쌓아 재학습 데이터로 쓴다(routers/predict.py served_rows).

시나리오 (data/generate_server_room.py):
    normal    - 정상
    zero      - 센서 0 고정 (6시간)
    spike     - 센서 값 튐 (1시간 +15°C)
    missing   - 센서 결측 (1시간 NaN)
    offset    - 센서 교정 오프셋 +3°C (72시간) - 온도는 setpoint와 같고 설정온도만 22 그대로
    setpoint  - 진짜 드리프트: 냉방 설정온도 22 → 25°C (72시간)
    expansion - 진짜 드리프트: 서버 증설 400 → 500kW (72시간)

사전 준비: uvicorn serving_app.main:app 서버가 이미 떠 있어야 합니다.
실행: python scripts/simulate_drift.py                    # 기본 순서
      python scripts/simulate_drift.py normal zero missing  # 원하는 시나리오만
"""
import math
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import SEQ_LEN
from data.generate_server_room import scenarios
from serving_app.monitoring.drift_detector import WINDOW_SIZE

API_BASE = os.getenv("API_BASE", "http://localhost:8077")  # Docker는 http://localhost:8000
BATCH_N = SEQ_LEN + WINDOW_SIZE  # 48 - 배치 하나로 판정 윈도우(24건)가 정확히 채워진다
DEFAULT_ORDER = ["normal", "zero", "spike", "missing", "offset", "setpoint"]


def send_batch(rows: list[dict]) -> dict:
    # 표준 JSON에는 NaN이 없어 requests가 전송을 거부한다 - 결측은 브라우저처럼 null로 보낸다
    temps = [None if math.isnan(r["Temp"]) else r["Temp"] for r in rows]
    loads = [r["LoadKW"] for r in rows]  # 부하도 보내야 서버 증설(expansion)이 모델 입력에 반영된다
    setpoints = [r["Setpoint"] for r in rows]  # 설정온도를 보내야 오프셋 고장과 설정온도 변경이 구분된다
    resp = requests.post(f"{API_BASE}/predict/batch-test", json={"temps": temps, "loads": loads, "setpoints": setpoints})
    return resp.json()["drift_check"] if resp.ok else {"status": f"HTTP {resp.status_code}"}


def main():
    all_scenarios = scenarios()
    for name in sys.argv[1:] or DEFAULT_ORDER:
        rows = all_scenarios[name]
        starts = range(0, len(rows) - BATCH_N + 1, WINDOW_SIZE)
        checks = [send_batch(rows[i : i + BATCH_N]) for i in starts]
        for k, check in enumerate(checks):
            if check["status"] != "ok":
                print(f"[{name} #{k}] drift_check = {check}")
        if all(c["status"] == "ok" for c in checks):
            print(f"[{name}] ok ({len(checks)} batches)")
    print("결과 확인: logs/aiops.log 또는 대시보드 '재학습 로그'")


if __name__ == "__main__":
    main()
