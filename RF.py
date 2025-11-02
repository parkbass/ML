# RF_app_v6.py
# CSV/XLSX/XLS + 시트 선택 + 폰트 수정 + PDP 2개씩 표시 + 지표 단순화

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, classification_report
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

# --------- 폰트 설정 (운영체제별) ---------
plt.rcParams['axes.unicode_minus'] = False
system = platform.system()
if system == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux (Streamlit Cloud 등)
    plt.rc('font', family='NanumGothic')

st.set_page_config(page_title="랜덤포레스트 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱")

# ===== 파일 업로드 =====
uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("CSV, XLSX, XLS 파일을 업로드하세요. (.xls은 xlrd<2.0 필요)")
    st.stop()

file_name = uploaded_file.name.lower()
read_ok, df = False, None

try:
    if file_name.endswith(".csv"):
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                read_ok = True
                break
            except Exception:
                continue
    elif file_name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(uploaded_file)
        except Exception as e:
            st.error(
                "엑셀 파일을 여는 중 오류 발생.\n"
                "만약 .xls 파일이라면 'pip install xlrd<2.0' 설치 후 다시 시도하세요.\n"
                f"오류: {e}"
            )
            st.stop()
        sheet = st.selectbox("불러올 시트를 선택하세요", options=xls.sheet_names, index=0)
        df = xls.parse(sheet_name=sheet)
        read_ok = True
except Exception as e:
    st.error(f"파일 읽기 중 오류: {e}")
    st.stop()

if not read_ok:
    st.error("파일을 불러올 수 없습니다. (인코딩 또는 형식 문제 가능)")
    st.stop()

st.success(f"로드된 데이터 형태: {df.shape}")
st.dataframe(df.head())

# ===== 전처리 =====
