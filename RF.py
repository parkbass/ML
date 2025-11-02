# RF_app_v5.py
# CSV/XLSX/XLS + 시트 선택, 사이드바 최소화(테스트 비율=0.8 고정), RMSE 호환 처리

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

# ===== 파일 업로드 =====
uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("예: 헤더가 포함된 CSV, XLSX, XLS 파일을 업로드하세요. (XLS는 xlrd<2.0 필요)")
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
            xls = pd.ExcelFile(uploaded_file)  # .xls는 xlrd<2.0 필요
        except Exception as e:
            st.error(
                "엑셀 파일을 여는 중 오류가 발생했습니다.\n"
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
df = df.replace(['#DIV/0!', 'NaN', 'nan', ''], np.nan)

# 쉼표 포함 숫자형 문자열 -> 숫자
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

# 완전결측 열 제거 + 행 결측 제거
X = X.dropna(axis=1, how="all")
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

# 간단 범주형 인코딩
label_encoders = {}
for col in X.columns:
    if X[col].dtype == object:
        if X[col].nunique() <= 50:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        else:
            X = X.drop(columns=[col])

# 과제 유형 판별
task_type = "regression"
if not np.issubdtype(y.dtype, np.number):
    task_type = "classification"
elif y.nunique() <= 10 and y.dtype != float:
    task_type = "classification"

# 타깃 인코딩
y_le = None
if task_type == "classification":
    y = y.astype(str)
    y_le = LabelEncoder()
    y_enc = y_le.fit_transform(y)
else:
    y_enc = pd.to_numeric(y, errors="coerce")
    keep = ~pd.isna(y_enc)
    X, y_enc = X.loc[keep], y_enc.loc[keep]

# ===== 학습/테스트 분리 (테스트 0.8 고정) =====
TEST_SIZE = 0.8  # 요청사항 반영
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ===== 모델 (고정 하이퍼파라미터; 사이드바 제거) =====
if task_type == "regression":
    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
else:
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )

model.fit(X_train, y_train)

# ===== 성능 =====
st.subheader("모델 성능")
if task_type == "regression":
    preds = model.predict(X_test)
    # R2, MAE, RMSE(버전 호환)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)  # 신버전
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds))        # 구버전 호환
    st.success(f"회귀 성능: R² = {r2:.3f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}")
else:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    st.success(f"분류 성능: Accuracy = {acc:.3f} | F1(Weighted) = {f1:.3f}")
    if st.checkbox("분류 리포트 보기"):
        st.text(classification_report(y_test, preds, target_names=y_le.classes_))

# ===== 변수 중요도 =====
st.subheader("변수 중요도 (Feature Importance)")
importances = pd.DataFrame({
    "변수": X.columns,
    "중요도": model.feature_importances_
}).sort_values("중요도", ascending=False)
st.dataframe(importances)

fig, ax = plt.subplots(figsize=(6, 4))
top_show = min(15, len(importances))
ax.barh(importances["변수"].head(top_show)[::-1], importances["중요도"].head(top_show)[::-1])
ax.set_xlabel("중요도"); ax.set_ylabel("변수"); ax.set_title("변수 중요도 상위")
st.pyplot(fig)

# ===== PDP (필요 시만 그리기) =====
st.subheader("부분 의존도(PDP) 그래프")
pdp_candidates = list(importances["변수"])
chosen = st.multiselect("그릴 변수를 선택하세요 (선택 안 하면 상위 3개)", pdp_candidates, default=[])
if not chosen:
    chosen = pdp_candidates[:3]

for feat in chosen:
    try:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(
            model, X_test, features=[feat], ax=ax2, kind="average"
        )
        ax2.set_title(f"PDP: {feat}")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"PDP 생성 중 오류({feat}): {e}")

st.caption("※ .xls 파일은 'pip install xlrd<2.0' 설치가 필요합니다.")
