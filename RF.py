# RF_app_v9.py
# 완전 통합 버전 (2025)
# - CSV/XLSX/XLS 지원 + 시트 선택
# - 한글 폰트 자동탐색 + TTF 업로드
# - 테스트비율 0.8 고정
# - 성능: 설명력(R²) 또는 정확도(Accuracy)
# - PDP: multiselect 선택형

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform, io, os, tempfile
from matplotlib import font_manager

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

# ===== 폰트 설정 함수 =====
def set_korean_font(user_font_path: str | None = None):
    if user_font_path and os.path.exists(user_font_path):
        font_manager.fontManager.addfont(user_font_path)
        fname = font_manager.FontProperties(fname=user_font_path).get_name()
        plt.rcParams["font.family"] = fname
        plt.rcParams["axes.unicode_minus"] = False
        return fname
    candidates = [
        "Malgun Gothic", "AppleGothic", "NanumGothic",
        "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR"
    ]
    available = set(f.name for f in font_manager.fontManager.ttflist)
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="랜덤포레스트 기반 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱")

# ===== 사이드바: 폰트 업로드 =====
st.sidebar.header("옵션")
font_file = st.sidebar.file_uploader("한글 폰트 TTF 업로드(선택)", type=["ttf"])
font_path = None
if font_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
        tmp.write(font_file.read())
        font_path = tmp.name  # 임시 저장

applied_font = set_korean_font(font_path)
if not applied_font_
