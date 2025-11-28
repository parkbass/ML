네, 물론입니다. 제안 드린 개선 사항들을 모두 반영하여 완성된 전체 코드를 정리해 드리겠습니다.

주요 변경 사항은 다음과 같습니다.

(UX) 전처리에서 제외되는 변수 알림 기능 추가: 고유값(unique value)이 50개를 초과하여 분석에서 제외되는 범주형 변수가 있을 경우, 사용자에게 어떤 변수가 제외되었는지 st.info 메시지로 알려줍니다.

(코드 안정성) PDP 데이터 생성 로직 개선: PartialDependenceDisplay로 그래프를 그린 후 데이터를 추출하는 대신, partial_dependence 함수를 사용하여 데이터를 먼저 계산하고 그 결과를 바탕으로 직접 그래프를 그리는 방식으로 변경했습니다. 이는 라이브러리 업데이트에 더 강건한 코드입니다.

(UX) 테스트 비율 슬라이더 설명 문구 개선: 사용자가 슬라이더 값을 변경할 때마다 학습 데이터와 테스트 데이터의 비율이 몇 %인지 직관적으로 보여주도록 수정했습니다.

아래는 완성된 전체 코드입니다. 그대로 복사하여 .py 파일로 저장하고 실행하시면 됩니다.

최종 수정 코드 (RF_app_v13.py)
code
Python
download
content_copy
expand_less
# RF_app_v13.py
# - CSV/XLSX/XLS 지원 + 시트 선택
# - 한글 폰트: (1) 업로드 TTF → (2) GitHub NotoSansKR 자동 로드 → (3) 로컬 폰트 탐색
# - 테스트 비율 슬라이더 (기본 0.8)
# - 성능: 설명력(R²) 또는 정확도(Accuracy)
# - PDP: multiselect로 변수 선택, 2개씩 배치
#        + moving-average 스무딩으로 곡선 + 원래 PDP 점도 같이 표시
# - [개선] 전처리 시 제외된 변수 사용자에게 알림
# - [개선] PDP 생성 로직을 데이터 계산 후 그리도록 변경 (안정성 강화)
# - [개선] 테스트 비율 슬라이더 설명 문구 직관적으로 변경

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, os, tempfile
from matplotlib import font_manager

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.inspection import partial_dependence # PDP 데이터 계산을 위해 직접 임포트
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

    # 2) GitHub NotoSansKR 자동 로드 시도 (인터넷 연결 필요)
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
        pass

    # 3) 로컬 한글 폰트 탐색
    candidates = [
        "Malgun Gothic", "AppleGothic", "NanumGothic",
        "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
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

# ===== 사이드바: 폰트 업로드 + 테스트 비율 설정 =====
st.sidebar.header("옵션")

font_file = st.sidebar.file_uploader("한글 폰트 TTF 업로드(선택)", type=["ttf"])
font_path = None
if font_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
        tmp.write(font_file.read())
        font_path = tmp.name

applied_font = set_korean_font(font_path)
if (not applied_font) and (font_file is None):
    st.sidebar.warning(
        "시스템에서 한글 폰트를 찾지 못했습니다. "
        "그래프 한글이 깨지면 TTF를 업로드하거나 인터넷 연결을 확인해 주세요."
    )

test_size = st.sidebar.slider(
    "테스트 데이터 비율", min_value=0.1, max_value=0.8, value=0.2, step=0.05
)
st.sidebar.caption(f"현재 설정: 학습 데이터 {100 - test_size*100:.0f}% / 테스트 데이터 {test_size*100:.0f}%")

# ===== 파일 업로드 =====
uploaded = st.file_uploader("CSV / XLSX / XLS 파일 업로드", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("CSV, XLSX, XLS 파일을 업로드하세요. (.xls은 xlrd<2.0 필요)")
    st.stop()

file_name = uploaded.name.lower()
file_bytes = uploaded.read()
df = None

# ===== 파일 읽기 =====
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
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(
                "엑셀 파일을 여는 중 오류가 발생했습니다.\n"
                "만약 .xls 파일이라면 'pip install xlrd<2.0' 설치 후 다시 시도하세요.\n"
                f"오류: {e}"
            )
            st.stop()
        sheet = st.selectbox("불러올 시트를 선택하세요", options=xls.sheet_names, index=0)
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        st.stop()
except Exception as e:
    st.error(f"파일 읽기 중 오류: {e}")
    st.stop()

# ===== 데이터 미리보기 =====
st.success(f"로드된 데이터 형태: {df.shape}")
if df.shape[0] == 0 or df.shape[1] == 0:
    st.warning("데이터가 비어 있습니다. 파일 내용을 확인해 주세요.")
    st.stop()
st.dataframe(df.head(30))

# ===== 전처리 =====
df = df.replace(["#DIV/0!", "NaN", "nan", ""], np.nan)
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass

# ===== 타깃 선택 =====
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

# ===== 범주형 간단 인코딩 & 제외 변수 알림 [개선] =====
dropped_cols = []
for col in list(X.columns):
    if X[col].dtype == object:
        if X[col].nunique() <= 50:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            dropped_cols.append(col)
            X = X.drop(columns=[col])

if dropped_cols:
    st.info(f"ℹ️ 고유값이 50개를 초과하여 다음 변수는 분석에서 제외되었습니다: **{', '.join(dropped_cols)}**")

# ===== 과제 유형 판별 =====
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

# ===== 학습/테스트 분할 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ===== 모델 =====
if task == "regression":
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ===== 성능 표시 =====
st.subheader("모델 성능 결과")
if task == "regression":
    r2 = r2_score(y_test, model.predict(X_test))
    st.success(f"🔹 설명력 (R²): {r2:.3f}")
else:
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"🔹 정확도 (Accuracy): {acc:.3f}")

# ===== 변수 중요도 =====
st.subheader("변수 중요도 (Feature Importance)")
importances = pd.DataFrame(
    {"변수": X.columns.astype(str), "중요도": model.feature_importances_}
).sort_values("중요도", ascending=False)
st.dataframe(importances)

fig, ax = plt.subplots(figsize=(6, 4))
top_n = min(15, len(importances))
ax.barh(
    importances["변수"].head(top_n)[::-1],
    importances["중요도"].head(top_n)[::-1],
)
ax.set_xlabel("중요도")
ax.set_ylabel("변수")
ax.set_title("변수 중요도 상위 항목")
# 폰트 적용 (축, 제목, 라벨 모두)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontfamily(plt.rcParams["font.family"])
st.pyplot(fig)

# ===== PDP (안정성 개선) =====
st.subheader("변수별 영향 그래프 (PDP)")
pdp_candidates = importances["변수"].tolist()
default_vars = pdp_candidates[:4]
selected_vars = st.multiselect(
    "PDP로 확인할 변수를 선택하세요 (여러 개 선택 가능)",
    pdp_candidates,
    default=default_vars,
)

if len(selected_vars) == 0:
    st.info("변수를 선택하면 개별 의존도 그래프가 표시됩니다.")
else:
    cols = 2
    rows = int(np.ceil(len(selected_vars) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(selected_vars):
        ax_i = axes[i]
        try:
            # 1) PDP 데이터 직접 계산
            pdp_result = partial_dependence(model, X_test, features=feat, kind="average")
            x_values = pdp_result['values'][0]
            y_values = pdp_result['average'][0]

            # 2) 스무딩 적용
            y_smooth = smooth_1d(y_values, window=5)
            
            # 3) 계산된 데이터로 직접 그리기
            ax_i.plot(x_values, y_smooth, "-", linewidth=2)
            ax_i.scatter(x_values, y_values, s=10, color="gray", alpha=0.5)
            ax_i.set_title(str(feat), fontfamily=plt.rcParams["font.family"])
            ax_i.set_xlabel(str(feat), fontfamily=plt.rcParams["font.family"])
            ax_i.set_ylabel("Partial dependence", fontfamily=plt.rcParams["font.family"])
            
            # 폰트 적용 (축, 제목, 라벨 모두)
            for item in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label] + ax_i.get_xticklabels() + ax_i.get_yticklabels()):
                item.set_fontfamily(plt.rcParams["font.family"])

        except Exception as e:
            ax_i.set_visible(False)
            st.warning(f"PDP 생성 중 오류({feat}): {e}")

    # 남는 축 숨기기
    last_index = len(selected_vars) - 1
    for j in range(last_index + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

st.caption(
    "※ 그래프 한글이 깨지면 사이드바에서 한글 TTF를 업로드하거나, "
    "인터넷 연결 상태를 확인해 주세요 (NotoSansKR 자동 로드)."
)
