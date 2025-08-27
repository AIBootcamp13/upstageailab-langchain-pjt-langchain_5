# -*- coding: utf-8 -*-
"""
Q-net 단일 파일 다운로드 → ZIP이면 내부 PDF만 추출
- ZIP 내부 한글 파일명(인코딩 깨짐) 복구: CP437→CP949(EUC-KR) 재디코딩
- '병합/합본/통합/merge' 포함 PDF는 제외
- 간단한 PDF 시그니처(%PDF-) 검사
- 저장 경로/결과 목록 출력

사용법:
    uv add requests
    uv run download_qnet_single.py
"""

import os
import io
import re
import zipfile
from pathlib import Path
import requests

# ===== 필요한 곳만 바꾸세요 =====
URL = (
    "https://www.q-net.or.kr/crf011.do?id=crf01106&gSite=Q&gId=&filePath=bbs/Q006/Q006_2220028"
    "&fileName=%EC%A0%9C%EB%B9%B5%EA%B8%B0%EB%8A%A5%EC%82%AC%20%EA%B3%BC%EC%A0%9C(pdf).zip"
)
REFERER = "https://www.q-net.or.kr/crf005.do?id=crf00503&jmCd=7892"
SAVE_DIR = Path("../data/pdf")
TIMEOUT = 30
# ==============================

MERGE_PAT = re.compile(r"(병합|합본|통합|merge)", re.IGNORECASE)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": REFERER,  # 일부 서버는 Referer를 확인함
}


def is_pdf_bytes(b: bytes) -> bool:
    # 간단한 PDF 시그니처 체크
    return b.startswith(b"%PDF-")


def safe_filename(name: str) -> str:
    # 파일명에 쓸 수 없는 문자 제거
    name = name.strip().replace("\n", " ").replace("\r", " ")
    return re.sub(r'[\\/:*?"<>|]+', "_", name) or "file.pdf"


def ensure_unique_path(path: Path) -> Path:
    # 같은 이름이 있으면 (1), (2) … 를 붙여 충돌 방지
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


def _fix_korean_name(name: str) -> str:
    """
    ZIP이 UTF-8 플래그 없이 저장된 경우, zipfile이 CP437로 해석한 문자열을
    다시 바이트로 되돌렸다가 CP949(=EUC-KR) 등으로 재해석해 한글 복구.
    """
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        for enc in ("euc-kr", "ms949", "utf-8"):
            try:
                return name.encode("cp437").decode(enc)
            except Exception:
                pass
    return name  # 못 고치면 원본 반환


def extract_pdfs_from_zip(zip_bytes: bytes, out_dir: Path) -> list[Path]:
    """
    ZIP 바이트에서 PDF만 추출(하위 폴더 구조 무시, 파일명만)
    - 한글 파일명 복구
    - '병합/합본/통합/merge' 포함 PDF 제외
    - PDF 시그니처 확인
    """
    saved = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue

            raw_name = os.path.basename(zi.filename)
            name = _fix_korean_name(raw_name)

            if not name.lower().endswith(".pdf"):
                continue
            if MERGE_PAT.search(name):
                print(f"[SKIP] 병합/합본 PDF 제외: {name}")
                continue

            data = zf.read(zi)
            if not is_pdf_bytes(data):
                print(f"[SKIP] PDF 서명 아님: {name}")
                continue

            out = ensure_unique_path(out_dir / safe_filename(name))
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                f.write(data)
            print(f"[OK] 저장: {out.name}")
            saved.append(out)
    return saved


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[PATH] 저장 폴더: {SAVE_DIR.resolve()}")
    print(f"[GET] {URL}")

    r = requests.get(URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    content = r.content

    # 확장자와 관계없이 ZIP 시도 → 실패하면 PDF 단일 파일로 간주
    saved = []
    try:
        saved = extract_pdfs_from_zip(content, SAVE_DIR)
        if saved:
            print(f"[DONE] ZIP에서 PDF {len(saved)}개 추출 완료.")
        else:
            print("[INFO] ZIP 내부에 저장할 PDF가 없거나 모두 제외됨.")
    except zipfile.BadZipFile:
        # ZIP이 아니면 단일 PDF일 수 있음 → 바로 저장 시도
        if is_pdf_bytes(content):
            out = ensure_unique_path(SAVE_DIR / "downloaded.pdf")
            with open(out, "wb") as f:
                f.write(content)
            print(f"[OK] 단일 PDF 저장: {out.name}")
            saved = [out]
        else:
            print("[FAIL] ZIP도 아니고 PDF도 아닌 응답입니다(확장자/응답 헤더 확인 필요).")

    if saved:
        print("[LIST]")
        for p in saved:
            print(" -", p.resolve())


if __name__ == "__main__":
    main()
