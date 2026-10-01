"""
/data/upload 입력 검증 확인 - 서버·tensorflow 없이 라우터 함수를 직접 호출한다.
저장 위치는 임시 폴더로 바꿔 data/uploads/의 "최근 업로드"를 건드리지 않는다.

실행: python tests/test_upload.py
"""
import asyncio
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException, UploadFile

from serving_app.routers import data as api

with open("data/sample_server_room.csv", encoding="utf-8") as f:
    HEADER, *ALL_LINES = f.read().splitlines()
LINES = ALL_LINES[:720]


def upload(lines: list[str], header: str = HEADER, encoding: str = "utf-8") -> int:
    body = "\n".join([header, *lines]).encode(encoding)
    try:
        asyncio.run(api.upload(UploadFile(io.BytesIO(body), filename="t.csv")))
        return 200
    except HTTPException as e:
        return e.status_code


def with_cell(line_no: int, col: int, value: str) -> list[str]:
    """LINES의 line_no번째 행의 col번째 칸만 바꾼 복사본 (0: Timestamp, 1: Temp, 2: LoadKW, 3: Setpoint)."""
    lines = list(LINES)
    cells = lines[line_no].split(",")
    cells[col] = value
    lines[line_no] = ",".join(cells)
    return lines


def test_upload():
    with tempfile.TemporaryDirectory() as tmp:
        api.UPLOAD_DIR = tmp
        assert upload(LINES) == 200 and len(os.listdir(tmp)) == 1
        assert upload(LINES, encoding="utf-8-sig") == 200                     # 엑셀 "CSV UTF-8"(BOM)
        assert upload(with_cell(100, 1, "")) == 200                           # 온도 빈 칸 = 결측, 판정은 validator
        assert upload(with_cell(100, 3, "")) == 200                           # Setpoint는 선택

        assert upload(LINES[:-1]) == 400                                      # 720행 미만
        assert upload(LINES, header="Timestamp,Temp,Load,Setpoint") == 400    # 필수 컬럼 없음
        assert upload(LINES, encoding="utf-16") == 400                        # UTF-8 아님
        for col, value in ((2, ""), (2, "abc"), (2, "nan"), (2, "inf"), (1, "abc"), (1, "inf"), (3, "nan")):
            assert upload(with_cell(100, col, value)) == 400, (col, value)
        assert upload(LINES[:100] + LINES[101:] + [LINES[100]]) == 400        # 순서 뒤집힘
        assert upload(with_cell(100, 0, LINES[99].split(",")[0])) == 400      # 중복
        assert upload(ALL_LINES[:100] + ALL_LINES[101:721]) == 400            # 1시간 빠짐 (720행 유지)
        assert upload(with_cell(100, 0, "2026/01/05 04:00")) == 400           # 형식 오류
        assert upload(with_cell(100, 0, LINES[100].split(",")[0] + "+09:00")) == 400  # 시간대 섞임
        assert upload(LINES[:100] + ["2026-01-05 04:00,22.0"] + LINES[101:]) == 400   # 칸 모자란 행
        assert len(os.listdir(tmp)) == 4  # 거부된 파일은 저장되지 않는다


if __name__ == "__main__":
    test_upload()
    print("ok")
