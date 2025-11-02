# RF_app_v4.py
# CSV + XLSX + XLS 업로드 지원 + 엑셀 시트 선택

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
st.title("랜덤포레스트 기반 예측/분류 웹앱 (CSV/XLSX/XLS + 시트 선택)")

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
    st.info("예: 헤더가 포함된 CSV, XLSX, XLS 파일을 업로드하세요. (XLS는 xlrd<2.0 필요)")
    st.stop()

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
        # 엑셀 시트 목록 먼저 얻기
        try:
            xls = pd.ExcelFile(uploaded_file)  # .xls의 경우 xlrd(<2.0) 필요
        except Exception as e:
            st.error(
                "엑셀 파일을 여는 중 오류가 발생했습니다. "
                "만약 .xls 파일이라면 'pip install xlrd<2.0'로 설치 후 다시 시도하세요.\n"
                f"오류 메시지: {e}"
            )
            st.stop()

        # 시트 선택 UI
        sheet = st.selectbox("불러올 시트를 선택하세요", options=xls.sheet_names, index=0)
        df = xls.parse(sheet_name=sheet)
        read_ok = True

except Exception as e:
    st.error(f"파일 읽기 중 오류 발생: {e}")
    st.stop()

if not read_ok:
    st.error("파일을 불러올 수 없습니다. (인코딩 또는 형식 문제 가능)")
    st.stop()

st.success(f"로드된 데이터 형태: {df.shape}")
st.dataframe(df.head())

# -------- 전처리 --------
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

# 결측 정리
df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

X = X.dropna(axis=1, how="all")
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

# 범주형 입력 간단 인코딩
label_encoders = {}
for col in X.columns:
    if X[col].dtype == object:
        if X[col].nunique() <= 50:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            X = X.drop(columns=[col])

# 과제 유형 판별(회귀/분류)
task_type = "regression"
y_encoded = y
if not np.issubdtype(y.dtype, np.number):
    task_type = "classification"
elif y.nunique() <= 10 and y.dtype != float:
    task_type = "classification"

# 타깃 인코딩
y_le = None
if task_type == "classification":
    y = y.astype(str)
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y)
else:
    y_encoded = pd.to_numeric(y, errors="coerce")
    keep_idx = ~y_encoded.isna()
    X, y_encoded = X.loc[keep_idx], y_encoded.loc[keep_idx]

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, random_state=random_state
)

# 모델 학습
if task_type == "regression":
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
else:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
model.fit(X_train, y_train)

# -------- 성능 지표 --------
st.subheader("모델 성능")
if task_type == "regression":
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    st.success(f"회귀 성능: R² = {r2:.3f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}")
else:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    st.success(f"분류 성능: Accuracy = {acc:.3f} | F1(Weighted) = {f1:.3f}")
    if st.checkbox("분류 리포트 보기"):
        st.text(classification_report(y_test, preds, target_names=y_le.classes_))

# -------- 변수 중요도 --------
st.subheader("변수 중요도 (Feature Importance)")
importances = pd.DataFrame({
    "변수": X.columns,
    "중요도": model.feature_importances_
}).sort_values("중요도", ascending=False)
st.dataframe(importances)

fig, ax = plt.subplots(figsize=(6, 4))
top_show = min(15, len(importances))
ax.barh(importances["변수"].head(top_show)[::-1], importances["중요도"].head(top_show)[::-1])
ax.set_xlabel("중요도")
ax.set_ylabel("변수")
ax.set_title("변수 중요도 상위")
st.pyplot(fig)

# -------- PDP --------
st.subheader("부분 의존도(PDP) 그래프")
pdp_candidates = list(importances["변수"])
chosen_features = st.multiselect(
    "PDP를 그릴 변수를 선택하세요 (미선택 시 상위 중요도 k개 자동 선택)",
    pdp_candidates
)
if not chosen_features:
    chosen_features = pdp_candidates[:pdp_topk]

for feat in chosen_features:
    try:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            model, X_test, features=[feat], ax=ax2, kind="average"
        )
        ax2.set_title(f"PDP: {feat}")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"PDP 생성 중 오류({feat}): {e}")

st.caption("※ .xls 파일은 'pip install xlrd<2.0' 설치가 필요합니다. PDP는 변수 수가 많을수록 시간이 오래 걸립니다.")
