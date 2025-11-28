# RF_app_v10.py
# - CSV/XLSX/XLS 지원 + 시트 선택
# - 한글 폰트: (1) 업로드 TTF → (2) GitHub NotoSansKR 자동 로드 → (3) 로컬 폰트 탐색
# - 테스트 비율 0.8 고정
# - 성능: 설명력(R²) 또는 정확도(Accuracy)
# - PDP: multiselect로 변수 선택, 2개씩 배치
#        + moving-average 스무딩으로 곡선 형태 + 실제 점은 산점도로 표시

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, os, tempfile
from matplotlib import font_manager

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

# ===== 유틸: 간단 스무딩 함수 (moving average) =====
def smooth_1d(y, window=5):
    """1차원 배열 y를 이동평균으로 부드럽게 만듭니다."""
    y = np.asarray(y)
    if len(y) <= window:
        return y
    w = np.ones(window) / window
    return np.convolve(y, w, mode="same")

# ===== 폰트 설정 함수 =====
def set_korean_font(user_font_path=None):
    """
    한글 폰트를 다음 우선순위로 설정:
    1) 사용자가 업로드한 TTF
    2) GitHub의 NotoSansKR 자동 다운로드
    3) 시스템에 설치된 한글 폰트 탐색
    """
    # 1) 사용자가 업로드한 폰트
    if user_font_path and os.path.exists(user_font_path):
        font_manager.fontManager.addfont(user_font_path)
        fname = font_manager.FontProperties(fname=user_font_path).get_name()
        plt.rcParams["font.family"] = fname
        plt.rcParams["axes.unicode_minus"] = False
        return fname

    # 2) GitHub NotoSansKR 자동 로드 시도 (인터넷 연결 시)
    try:
        import requests
        url = (
            "https://github.com/google/fonts/raw/main/ofl/notosanskr/"
            "NotoSansKR-Regular.otf"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".otf") as tmpf:
            tmpf.write(resp.content)
            remote_path = tmpf.name
        font_manager.fontManager.addfont(remote_path)
        fname = font_manager.FontProperties(fname=remote_path).get_name()
        plt.rcParams["font.family"] = fname
        plt.rcParams["axes.unicode_minus"] = False
        return fname
    except Exception:
        # 인터넷이 없거나 requests 미설치 등은 조용히 무시
        pass

    # 3) 로컬 한글 폰트 탐색
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Source Han Sans KR",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    # 그래도 없으면 기본 폰트 (한글이 깨질 수 있음)
    plt.rcParams["axes.unicode_minus"] = False
    return None

# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="랜덤포레스트 기반 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱")

# ===== 사이드바: 폰트 업로드(선택) =====
st.sidebar.header("옵션")
font_file = st.sidebar.file_uploader("한글 폰트 TTF 업로드(선택)", type=["ttf"])
font_path = None
if font_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
        tmp.write(font_file.read())
        font_path = tmp.name

applied_font = set_korean_font(font_path)
if (not applied_font) and (font_file is N
