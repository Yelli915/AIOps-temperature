"""
센서 온도(·IT 부하) 시퀀스 검증 - 드리프트 판정·재학습 전에 "데이터가 믿을 만한가"를 먼저 본다.

서빙 입력(/predict, /predict/batch-test), 재학습 데이터(서빙된 배치), base 학습 데이터(업로드 CSV) 모두 이 함수를 쓴다.
설정온도(setpoints)를 함께 받으면 잔차 규칙으로 범위 안의 고장(오프셋)까지 가른다.
걸린 규칙 이름을 반환하고, 정상이면 None을 반환한다.
"""
import math

TEMP_MIN, TEMP_MAX = 10.0, 40.0  # 서버실 항온 대역. -10~50으로 넓히면 0°C 고장을 놓친다
STUCK_HOURS = 6                   # 이 시간 동안 값이 완전히 같으면 센서 고정으로 본다
JUMP_C = 10.0                     # 인접 1시간 차이가 이 이상이면 값 튐
BASE_MIN_ROWS = 720               # base 학습 최소 30일 - 짧은 시나리오 CSV(240행)로 base 모델을 다시 만들지 않게
LOAD_MAX = 2000.0                 # kW, 기준 400kW의 5배 - 증설(500kW)은 통과, 부하 센서 튐(1e308·inf)은 차단
# 잔차 규칙: 온도 ≈ 설정온도 + LOAD_GAIN × (부하 − REF_LOAD). 범위 안의 고장(오프셋·작은 튐)을 잡는다.
# ponytail: 선형 물리식 하나 - 실설비는 과거 정상 데이터로 LOAD_GAIN·REF_LOAD를 회귀해 맞출 것 (교정 값)
LOAD_GAIN = 0.02                  # °C/kW, 부하 1kW당 온도 상승
REF_LOAD = 400.0                  # kW, 설정온도에서 운전하는 기준 부하
RESIDUAL_HOURS = 3                # 잔차를 이 시간만큼 평균 - 노이즈(σ 0.2)·하루 주기(±0.3)는 평균 0.7 미만
RESIDUAL_C = 1.5                  # 평균 잔차가 이를 넘으면 설정온도·부하로 설명되지 않는 온도


def validate(temps: list, loads: list | None = None, setpoints: list | None = None) -> str | None:
    if loads and any(l <= 0 for l in loads):  # 가동 중인 서버실의 IT 부하는 0일 수 없다 - 부하 센서 0 고정
        return "load_zero"
    if loads and any(not l <= LOAD_MAX for l in loads):  # inf·NaN도 여기서 걸린다
        return "load_out_of_range"
    if any(t is None or math.isnan(t) for t in temps):
        return "missing"
    if any(not TEMP_MIN <= t <= TEMP_MAX for t in temps):
        return "out_of_range"
    for i in range(len(temps) - STUCK_HOURS + 1):
        if len(set(temps[i : i + STUCK_HOURS])) == 1:
            return "stuck_value"
    if any(abs(b - a) >= JUMP_C for a, b in zip(temps, temps[1:])):
        return "sudden_jump"
    if loads and setpoints:  # 설정온도(BMS 운전값)를 알 때만 - 모르면 오프셋 고장과 설정온도 변경이 구분되지 않는다
        r = [t - s - LOAD_GAIN * (l - REF_LOAD) for t, l, s in zip(temps, loads, setpoints)]
        for i in range(len(r) - RESIDUAL_HOURS + 1):
            if abs(sum(r[i : i + RESIDUAL_HOURS])) / RESIDUAL_HOURS > RESIDUAL_C:
                return "residual"
    return None


def columns(rows: list[dict]) -> tuple[list, list, list | None]:
    """CSV·서빙 rows → validate() 인자. Setpoint가 하나라도 없으면 잔차 규칙은 건너뛴다."""
    setpoints = [r.get("Setpoint") for r in rows]
    return [r["Temp"] for r in rows], [r["LoadKW"] for r in rows], None if None in setpoints else setpoints


def check_base_rows(rows: list[dict]) -> None:
    """base 학습(train_baseline_v1.py, train_and_register.py) 전 확인. base 모델은 모든 fine-tuning 버전의
    출발점이므로, 짧은 시나리오 데이터나 고장 데이터로 다시 만들면 이후 버전 전부가 틀어진다."""
    if len(rows) < BASE_MIN_ROWS:
        raise SystemExit(f"base 학습에는 최소 {BASE_MIN_ROWS}행(30일)이 필요합니다 (최근 업로드 {len(rows)}행) - "
                         "Day3 시나리오 CSV가 최근 업로드라면 data/sample_server_room.csv를 다시 업로드하세요.")
    rule = validate(*columns(rows))
    if rule:
        raise SystemExit(f"최근 업로드에 센서 고장(rule={rule})이 있어 base 학습을 하지 않습니다.")


if __name__ == "__main__":
    normal = [22.0 + 0.1 * (i % 5) for i in range(48)]
    assert validate(normal) is None
    assert validate(normal[:30] + [float("nan")] + normal[31:]) == "missing"
    assert validate(normal[:30] + [None] + normal[31:]) == "missing"      # 브라우저 JSON은 NaN을 null로 보냄
    assert validate(normal[:30] + [0.0] * 6 + normal[36:]) == "out_of_range"
    assert validate(normal[:30] + [23.0] * 6 + normal[36:]) == "stuck_value"
    assert validate(normal[:30] + [37.0] + normal[31:]) == "sudden_jump"  # 22 → 37: 범위 안이지만 급변
    assert validate([25.0 + 0.1 * (i % 5) for i in range(48)]) is None    # 설정온도 25°C는 통과해야 함
    assert validate(normal, [400.0] * 47 + [0.0]) == "load_zero"
    assert validate(normal, [500.0] * 48) is None                          # 서버 증설 500kW는 통과해야 함
    assert validate(normal, [400.0] * 47 + [1e308]) == "load_out_of_range"
    assert validate(normal, [400.0] * 47 + [float("inf")]) == "load_out_of_range"
    # 잔차 규칙: 오프셋 +3°C 고장은 설정온도 변경(22→25)과 온도가 같다 - 설정온도를 알면 구분된다
    temps = [22.0 + 0.3 * math.sin(2 * math.pi * i / 24) + (0.2 if i % 2 else -0.2) for i in range(48)]
    loads = [400.0 + 60 * math.sin(2 * math.pi * (i - 9) / 24) for i in range(48)]
    temps = [t + LOAD_GAIN * (l - REF_LOAD) for t, l in zip(temps, loads)]
    shifted = temps[:24] + [t + 3 for t in temps[24:]]
    assert validate(temps, loads, [22.0] * 48) is None
    assert validate(shifted, loads, [22.0] * 24 + [25.0] * 24) is None          # 설정온도 변경 - 드리프트
    assert validate(shifted, loads, [22.0] * 48) == "residual"                  # 오프셋 고장
    assert validate(shifted, loads) is None                                      # 설정온도 모르면 못 가름
    assert validate(temps[:30] + [temps[30] + 9] + temps[31:], loads, [22.0] * 48) == "residual"  # 범위 안 작은 튐
    assert validate([t + 2 for t in temps], [l + 100 for l in loads], [22.0] * 48) is None        # 증설: 부하로 설명됨
    rows = [{"Temp": t, "LoadKW": 400.0} for t in normal] * 15  # 720행
    check_base_rows(rows)
    for bad in (rows[:240], rows[:-1] + [{"Temp": float("nan"), "LoadKW": 400.0}]):
        try:
            check_base_rows(bad)
            raise AssertionError("통과하면 안 됨")
        except SystemExit:
            pass
    print("ok")
