# RF_app_v3.py
# (CSV + XLSX + XLS 업로드 지원 / 최신 Streamlit 호환 버전)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, classification_report
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="랜덤포레스트 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱 (CSV + Excel 지원)")

# ----------------- Sidebar -----------------
st.sidebar.header("설정")
test_size = st.sidebar.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
n_estimators = st.sidebar.slider("n_estimators (트리 개수)", 50, 500, 200, 50)
max_depth = st.sidebar.slider("max_depth (None=0)", 0, 50, 0, 1)
max_depth = None if max_depth == 0 else max_depth
pdp_topk = st.sidebar.slider("PDP 대상 변수 수(상위 중요도)", 1, 10, 3, 1)

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("예: 헤더가 포함된 CSV, XLSX, XLS 파일을 업로드하세요.")
    st.stop()

# 파일 확장자 판별
file_name = uploaded_file.name.lower()
read_ok = False
df = None

try:
    if file_name.endswith(".csv"):
        # 여러 인코딩 시도
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                read_ok = True
                break
            except Exception:
                continue
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        read_ok = True
except Exception as e:
    st.error(f"파일 읽기 중 오류 발생: {e}")
    st.stop()

if not read_ok:
    st.error("파일을 불러올 수 없습니다. (인코딩 또는 형식 문제 가능)")
    st.stop()

st.success(f"로드된 데이터 형태: {df.shape}")
st.dataframe(df.head())

# 전처리
df = df.replace(['#DIV/0!', 'NaN', 'nan', ''], np.nan)

# 쉼표가 있는 숫자형 문자열 변환
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

# 타깃 선택
target_col = st.selectbox("예측/분류할 목표 변수(타깃)을 선택하세요", df.columns)
if target_col is None:
    st.stop()

# 결측 제거
df = df.dropna(subset=[target_col])
X = df.drop(columns=[targe]()
