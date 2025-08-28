import requests
import re
import os
import zipfile
import shutil

# ───────── 인코딩 유틸 ─────────
def pick_best_latin1_roundtrip(raw: str) -> str:
    b = raw.encode("latin-1", errors="ignore")
    for enc in ("utf-8", "cp949", "ms949", "euc-kr"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return raw

def fix_zip_internal_name(name: str) -> str:
    """
    zipfile이 cp437로 잘못 해석한 내부 파일명을 복구.
    - UTF-8 플래그가 설정된 항목이면 그대로 사용
    - 그 외에는 "보이는 문자열"을 cp437로 다시 bytes화 후 한글 인코딩 후보로 재해석
    """
    # zipfile은 filename을 이미 str로 줍니다. UTF-8로 온 경우엔 보통 멀쩡.
    # 그래도 안전하게 cp437 round-trip 시도
    try:
        raw = name.encode("cp437", errors="ignore")
        for enc in ("cp949", "ms949", "euc-kr", "utf-8"):
            try:
                return raw.decode(enc)
            except Exception:
                pass
    except Exception:
        pass
    return name

# ───────── 다운로드 ─────────
def download_file_with_fixed_filename(url: str, session=None, dest_folder="downloads"):
    if session is None:
        session = requests.Session()

    resp = session.get(url, stream=True)
    resp.raise_for_status()

    cd = resp.headers.get("content-disposition", "")
    filename = None

    m = re.search(r'filename\*?=([^;]+)', cd, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip().strip('"')
        filename = pick_best_latin1_roundtrip(raw)
    else:
        filename = os.path.basename(url.split("?", 1)[0])

    os.makedirs(dest_folder, exist_ok=True)
    path = os.path.join(dest_folder, filename)

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {path}")
    return path

# ───────── 압축 해제(파일명 복구) ─────────
def unzip_in_place_with_name_fix(folder: str, delete_zip: bool = False):
    """
    folder 안의 zip을 각각 같은 폴더에 풀되,
    내부 파일/폴더명 한글 깨짐을 복구해서 저장.
    상위 이동/폴더 삭제 없음.
    """
    for name in os.listdir(folder):
        if not name.lower().endswith(".zip"):
            continue

        zip_path = os.path.join(folder, name)
        extract_root = os.path.join(folder, os.path.splitext(name)[0])
        os.makedirs(extract_root, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zip_basename = os.path.splitext(name)[0]
            for info in zf.infolist():
                # 원래 상대경로(슬래시 포함) 복구
                fixed_rel = fix_zip_internal_name(info.filename).replace("\\\\", "/") # Fix: Escape backslash for literal interpretation

                # 압축 파일 내의 최상위 폴더명이 압축 파일명과 같을 경우, 해당 폴더 경로를 제거합니다.
                prefix_to_strip = zip_basename + "/"
                if fixed_rel.startswith(prefix_to_strip):
                    fixed_rel = fixed_rel[len(prefix_to_strip):]

                # 경로가 비어있다면(예: 최상위 폴더 자체), 건너뜁니다.
                if not fixed_rel:
                    continue

                target_path = os.path.join(extract_root, fixed_rel)

                if info.is_dir() or fixed_rel.endswith("/"):
                    os.makedirs(target_path, exist_ok=True)
                    continue

                # merged.pdf는 제외
                if os.path.basename(fixed_rel).lower() == "merged.pdf":
                    print(f"Skipped merged.pdf: {fixed_rel}")
                    continue

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                print(f"Extracted: {fixed_rel}")

        if delete_zip:
            try:
                os.remove(zip_path)
                print(f"Deleted zip: {zip_path}")
            except Exception as e:
                print(f"Failed to delete zip ({zip_path}): {e}")


# ───────── 이미 풀어둔 폴더의 파일명 사후 복구(선택) ─────────
def rename_mojibake_in_dir(root_dir: str):
    """
    이미 풀어둔 폴더에서 깨진 파일/폴더명을 cp437→(cp949/ms949/euc-kr/utf-8) 복구로 재명명.
    """
    for cur_root, dirs, files in os.walk(root_dir, topdown=False):
        # 파일
        for fn in files:
            fixed = fix_zip_internal_name(fn)
            if fixed != fn:
                src = os.path.join(cur_root, fn)
                dst = os.path.join(cur_root, fixed)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                    print(f"Renamed file: {fn} -> {fixed}")

        # 디렉터리
        for dn in dirs:
            fixed = fix_zip_internal_name(dn)
            if fixed != dn:
                src = os.path.join(cur_root, dn)
                dst = os.path.join(cur_root, fixed)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                    print(f"Renamed dir: {dn} -> {fixed}")

# ───────── PDF 파일 정리 ─────────
def organize_pdf_files(root_folder: str):
    """
    PDF 파일 중 '변경'이라는 단어가 포함된 파일을 '변경사항' 폴더로 이동합니다.
    """
    change_folder = os.path.join(root_folder, "변경사항")
    os.makedirs(change_folder, exist_ok=True)

    for cur_root, dirs, files in os.walk(root_folder):
        # '변경사항' 폴더 자체는 처리하지 않도록 제외
        if cur_root == change_folder:
            continue
        # 재귀적으로 '변경사항' 폴더로 들어가지 않도록 dirs에서 제거
        if "변경사항" in dirs:
            dirs.remove("변경사항")

        for file_name in files:
            # PDF 파일이고 파일명에 '변경'이 포함되어 있는지 확인
            if file_name.lower().endswith(".pdf") and "변경" in file_name:
                src_path = os.path.join(cur_root, file_name)
                dst_path = os.path.join(change_folder, file_name)
                
                # 대상 폴더에 동일한 파일명이 이미 존재할 경우, 파일명 충돌 방지
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(change_folder, f"{base} ({counter}){ext}")
                    counter += 1

                shutil.move(src_path, dst_path)
                print(f"Moved '{file_name}' to '{change_folder}'")


if __name__ == "__main__":
    # 스크립트의 디렉토리로 작업 디렉토리 변경
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    urls = [
        "https://www.q-net.or.kr/crf011.do?id=crf01106&gSite=Q&gId=&filePath=bbs/Q006/Q006_2220026&fileName=제과제빵기능사 실기시험 변경현황(2025년 적용).pdf",
        "https://www.q-net.or.kr/crf011.do?id=crf01106&gSite=Q&gId=&filePath=bbs/Q006/Q006_2220027&fileName=제과기능사 과제(pdf).zip",
        "https://www.q-net.or.kr/crf011.do?id=crf01106&gSite=Q&gId=&filePath=bbs/Q006/Q006_2220028&fileName=제빵기능사 과제(pdf).zip",
        "https://www.q-net.or.kr/crf011.do?id=crf01106&gSite=Q&gId=&filePath=bbs/Q006/Q006_2208612&fileName=2024년도 실기시험 변경 내역(제과기능사 제빵기능사).pdf",
    ]

    target_folder = "../data/pdf"

    # 1) 다운로드
    for url in urls:
        download_file_with_fixed_filename(url, dest_folder=target_folder)

    # 2) 해당 폴더 안 zip만 해제 (내부 파일명 복구 포함)
    unzip_in_place_with_name_fix(folder=target_folder, delete_zip=True)

    # 3) PDF 파일 정리 (파일명에 '변경' 포함 시 '변경사항' 폴더로 이동)
    organize_pdf_files(target_folder)

    # 4) 이미 풀어둔 폴더에서 이름이 깨져있다면(과거 작업물) 아래 한 줄로 복구 가능
    # rename_mojibake_in_dir(os.path.join(target_folder, "제과기능사 과제(pdf)"))
    # rename_mojibake_in_dir(os.path.join(target_folder, "제빵기능사 과제(pdf)"))