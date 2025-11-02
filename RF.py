# RF_app_v6_fix.py
# - 업로드 파일을 bytes로 읽고 BytesIO로 매번 새 스트림 생성 (포인터 이슈 해결)
# - CSV/XLSX/XLS 지원 + 시트 선택
# - 한글 폰트 설정 + PDP 소형 2개씩
# - 지표 단순화(R² 또는 Accuracy)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform, io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

# ---------- 폰트 설정 ----------
plt.rcParams['axes.unicode_minus'] = False
system = platform.system()
if system == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

st.set_page_config(page_title="랜덤포레스트 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱")

# ---------- 파일 업로드 ----------
uploaded = st.file_uploader("CSV / XLSX / XLS 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("파일을 업로드하세요. (.xls는 xlrd<2.0 필요)")
    st.stop()

file_name = uploaded.name.lower()
file_bytes = uploaded.read()  # 핵심: 먼저 bytes로 저장해 두기
df = None

# ---------- 파일 판독 ----------
try:
    if file_name.endswith(".csv"):
        read_ok = False
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                read_ok = True
                break
            except Exception:
                continue
        if not read_ok:
            st.error("CSV 인코딩을 판독할 수 없습니다. (utf-8 / cp949 등 확인)")
            st.stop()

    elif file_name.endswith((".xlsx", ".xls")):
        # 먼저 시트 목록 읽기
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))  # 매번 새 BytesIO!
        except Exception as e:
            st.error(
                "엑셀 파일을 여는 중 오류가 발생했습니다.\n"
                "만약 .xls 파일이라면 'pip install xlrd<2.0' 후 다시 시도하세요.\n"
                f"오류: {e}"
            )
            st.stop()

        sheet = st.selectbox("불러올 시트를 선택하세요", options=xls.sheet_names, index=0)
        # 선택한 시트 파싱 (다시 BytesIO로 열기)
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)

    else:
        st.error("지원하지 않는 파일 형식입니다.")
        st.stop()

except Exception as e:
    st.error(f"파일 읽기 중 오류: {e}")
    st.stop()

# ---------- 미리보기 ----------
st.success(f"로드된 데이터 형태: {df.shape}")
if df.shape[0] == 0 or df.shape[1] == 0:
    st.warning("데이터가 비어 있습니다. 파일 내용을 확인해 주세요.")
    st.stop()

st.dataframe(df.head(30))  # 미리보기 30행

# ---------- 전처리 ----------
df = df.replace(['#DIV/0!', 'NaN', 'nan', ''], np.nan)

for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

# ---------- 타깃 선택 ----------
target_col = st.selectbox("예측/분류할 목표 변수(타깃)을 선택하세요", df.columns)
if not target_col:
    st.stop()

df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

X = X.dropna(axis=1, how="all")
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

# -------- 범주형 간단 인코딩 --------
for col in list(X.columns):
    if X[col].dtype == object:
        if X[col].nunique() <= 50:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = X.drop(columns=[col])

# ---------- 과제 유형(회귀/분류) ----------
task = "regression"
if not np.issubdtype(y.dtype, np.number):
    task = "classification"
elif y.nunique() <= 10 and y.dtype != float:
    task = "classification"

if task == "classification":
    y = LabelEncoder().fit_transform(y.astype(str))
else:
    y = pd.to_numeric(y, errors="coerce")
    keep = ~pd.isna(y)
    X, y = X.loc[keep], y.loc[keep]

# ---------- 학습/테스트 분할 (테스트 0.8 고정) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# ---------- 모델 ----------
if task == "regression":
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)

# ---------- 성능(간단 표시) ----------
st.subheader("모델 성능 결과")
if task == "regression":
    r2 = r2_score(y_test, model.predict(X_test))
    st.success(f"🔹 예측 정확도 (R²): {r2:.3f}")
else:
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"🔹 분류 정확도: {acc:.3f}")

# ---------- 변수 중요도 ----------
st.subheader("변수 중요도 (Feature Importance)")
importances = pd.DataFrame({
    "변수": X.columns,
    "중요도": model.feature_importances_
}).sort_values("중요도", ascending=False)
st.dataframe(importances)

fig, ax = plt.subplots(figsize=(6, 4))
top_n = min(15, len(importances))
ax.barh(importances["변수"].head(top_n)[::-1], importances["중요도"].head(top_n)[::-1])
ax.set_xlabel("중요도"); ax.set_ylabel("변수"); ax.set_title("변수 중요도 상위 항목")
st.pyplot(fig)

# ---------- PDP (작게, 2개씩) ----------
st.subheader("변수별 영향 그래프 (PDP)")
pdp_vars = list(importances["변수"])[:4]  # 상위 4개
cols = 2
rows = int(np.ceil(len(pdp_vars) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
axes = np.atleast_1d(axes).flatten()

for i, feat in enumerate(pdp_vars):
    try:
        PartialDependenceDisplay.from_estimator(model, X_test, features=[feat], ax=axes[i], kind="average")
        axes[i].set_title(str(feat))
    except Exception:
        axes[i].set_visible(False)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.caption("※ 한글이 깨지면 OS에 맞는 폰트(Malgun/AppleGothic/NanumGothic)를 설치해 주세요.")
