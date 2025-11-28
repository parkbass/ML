# RF_app_v19.py
# [수정] st.pyplot(fig) 라인 끝의 불필요한 문자(```) 제거
# [수정] 사용자의 실제 폰트 파일 이름('D2CodingBold-Ver1.3.2-20180524.ttf')을 정확히 인식하도록 수정
# [개선] PDP에 부드러운 곡선(스무딩)과 원본 산점도를 함께 표시하여 경향성 파악 용이하게 변경

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

# ===== 유틸: 간단 스무딩 함수 =====
def smooth_1d(y, window=5):
    y = np.asarray(y)
    if len(y) <= window: return y
    w = np.ones(window) / window
    return np.convolve(y, w, mode="same")

# ===== 폰트 설정 함수 =====
def set_korean_font(user_font_path=None):
    try:
        script_dir = os.path.dirname(__file__)
        local_font_filename = 'D2CodingBold-Ver1.3.2-20180524.ttf'
        font_path = os.path.join(script_dir, local_font_filename)

        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            fname = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = fname
            plt.rcParams["axes.unicode_minus"] = False
            return fname
    except NameError:
        pass

    if user_font_path and os.path.exists(user_font_path):
        font_manager.fontManager.addfont(user_font_path)
        fname = font_manager.FontProperties(fname=user_font_path).get_name()
        plt.rcParams["font.family"] = fname
        plt.rcParams["axes.unicode_minus"] = False
        return fname
        
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    plt.rcParams["axes.unicode_minus"] = False
    return None

# ===== Streamlit 기본 설정 및 사이드바 =====
st.set_page_config(page_title="랜덤포레스트 기반 예측/분류 웹앱", layout="wide")
st.title("랜덤포레스트 기반 예측/분류 웹앱")
st.sidebar.header("옵션")

font_file = st.sidebar.file_uploader("한글 폰트 TTF 업로드(선택)", type=["ttf"])
font_path = None
if font_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
        tmp.write(font_file.read())
        font_path = tmp.name

applied_font = set_korean_font(font_path)
if not applied_font:
    st.sidebar.warning("한글 폰트를 찾지 못했습니다. 폰트 파일을 앱 폴더에 추가하거나, 직접 업로드 해주세요.")

test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.8, 0.2, 0.05)
st.sidebar.caption(f"현재 설정: 학습 데이터 {100 - test_size*100:.0f}% / 테스트 데이터 {test_size*100:.0f}%")

# ===== 파일 업로드 및 이하 로직은 이전과 동일 =====
uploaded = st.file_uploader("CSV / XLSX / XLS 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("CSV, XLSX, XLS 파일을 업로드하세요. (.xls은 xlrd<2.0 필요)")
    st.stop()

file_name = uploaded.name.lower()
file_bytes = uploaded.read()
df = None
try:
    if file_name.endswith(".csv"):
        read_ok = False
        for enc in ["utf-8-sig", "utf-8", "cp949"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                read_ok = True; break
            except Exception: continue
        if not read_ok: st.error("CSV 인코딩을 판독할 수 없습니다."); st.stop()
    elif file_name.endswith((".xlsx", ".xls")):
        try: xls = pd.ExcelFile(io.BytesIO(file_bytes))
        except Exception as e: st.error(f"엑셀 파일 열기 오류: {e}\n.xls 파일은 'pip install xlrd<2.0' 필요"); st.stop()
        sheet = st.selectbox("불러올 시트를 선택하세요", options=xls.sheet_names, index=0)
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    else: st.error("지원하지 않는 파일 형식입니다."); st.stop()
except Exception as e: st.error(f"파일 읽기 중 오류: {e}"); st.stop()

st.success(f"로드된 데이터 형태: {df.shape}")
if df.shape == 0 or df.shape == 0: st.warning("데이터가 비어 있습니다."); st.stop()
st.dataframe(df.head(30))

df = df.replace(["#DIV/0!", "NaN", "nan", ""], np.nan)
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception: pass

target_col = st.selectbox("예측/분류할 목표 변수(타깃)을 선택하세요", df.columns)
if not target_col: st.stop()

df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]
X = X.dropna(axis=1, how="all")
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

dropped_cols = []
for col in list(X.columns):
    if X[col].dtype == object:
        if X[col].nunique() <= 50:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            dropped_cols.append(col)
            X = X.drop(columns=[col])
if dropped_cols: st.info(f"ℹ️ 고유값이 50개를 초과하여 다음 변수는 분석에서 제외되었습니다: **{', '.join(dropped_cols)}**")

task = "regression"
if not np.issubdtype(y.dtype, np.number) or (y.nunique() <= 10 and y.dtype != float):
    task = "classification"
if task == "classification":
    y = LabelEncoder().fit_transform(y.astype(str))
else:
    y = pd.to_numeric(y, errors="coerce")
    keep = ~pd.isna(y)
    X, y = X.loc[keep], y.loc[keep]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) if task == "regression" else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

st.subheader("모델 성능 결과")
if task == "regression":
    r2 = r2_score(y_test, model.predict(X_test))
    st.success(f"🔹 설명력 (R²): {r2:.3f}")
else:
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"🔹 정확도 (Accuracy): {acc:.3f}")

st.subheader("변수 중요도 (Feature Importance)")
importances = pd.DataFrame({"변수": X.columns.astype(str), "중요도": model.feature_importances_}).sort_values("중요도", ascending=False)
st.dataframe(importances)
fig, ax = plt.subplots(figsize=(6, 4))
top_n = min(15, len(importances))
ax.barh(importances["변수"].head(top_n)[::-1], importances["중요도"].head(top_n)[::-1])
ax.set_xlabel("중요도"); ax.set_ylabel("변수"); ax.set_title("변수 중요도 상위 항목")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontfamily(plt.rcParams["font.family"])
st.pyplot(fig)

# ===== PDP (스무딩 기능 포함) =====
st.subheader("변수별 영향 그래프 (PDP)")
pdp_candidates = importances["변수"].tolist()
default_vars = pdp_candidates[:4]
selected_vars = st.multiselect("PDP로 확인할 변수를 선택하세요", pdp_candidates, default=default_vars)

if not selected_vars:
    st.info("변수를 선택하면 개별 의존도 그래프가 표시됩니다.")
else:
    cols = 2
    rows = int(np.ceil(len(selected_vars) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(selected_vars):
        ax_i = axes[i]
        try:
            display = PartialDependenceDisplay.from_estimator(
                model, X_test, features=[feat], kind="average", ax=ax_i
            )
            
            if ax_i.lines:
                line = ax_i.lines
                x_data, y_data = line.get_data()
                y_smooth = smooth_1d(y_data)

                ax_i.cla() 
                ax_i.plot(x_data, y_smooth, "-", linewidth=2, label="Trend")
                ax_i.scatter(x_data, y_data, s=10, color="gray", alpha=0.5, label="Raw PDP")
                
                from sklearn.inspection._plot.partial_dependence import _get_deciles
                deciles = _get_deciles(X_test[feat])
                ax_i.plot(deciles, [ax_i.get_ylim()] * len(deciles), "|", color="k")

                ax_i.set_title(str(feat))
                ax_i.set_xlabel(str(feat))
                ax_i.set_ylabel("Partial dependence")
            
            for item in ([ax_i.title, ax.xaxis.label, ax.yaxis.label] + ax_i.get_xticklabels() + ax_i.get_yticklabels()):
                item.set_fontfamily(plt.rcParams["font.family"])
        except Exception as e:
            ax_i.set_visible(False)
            st.warning(f"PDP 생성 중 오류({feat}): {e}")

    for j in range(len(selected_vars), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # [수정] 아래 라인에서 불필요한 ``` 제거
    st.pyplot(fig)
