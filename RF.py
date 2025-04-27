import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence

# --- 웹앱 기본 설정 ---
st.set_page_config(page_title="랜덤 포레스트 분석기", layout="wide")

st.title("🎯 종속변수(예측하고 싶은 목표 변수)를 선택하세요")

# --- 파일 업로드 ---
uploaded_file = st.file_uploader("파일 업로드", type=["csv"])

if uploaded_file is not None:
    # 데이터 불러오기
    df = pd.read_csv(uploaded_file)

    # 문자열로 된 숫자 쉼표 제거 및 변환
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    st.success("✅ 데이터 업로드 및 전처리 완료!")

    # --- 종속변수 선택 ---
    target = st.selectbox("종속변수 선택", options=df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # --- 데이터 분할 ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 모델 학습 ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- 성능 평가 ---
        score = model.score(X_test, y_test)
        st.success(f"모델의 R²(테스트 데이터 기준): {score:.3f}")

        # --- 변수 중요도 출력 ---
        st.header("📝 변수 중요도 (Feature Importance)")
        feature_importance = pd.DataFrame({
            "변수": X.columns,
            "중요도": model.feature_importances_
        }).sort_values("중요도", ascending=False)

        st.dataframe(feature_importance, use_container_width=True)

        # --- 부분 의존도 그래프 (PDP) ---
        st.header("📊 변수별 부분 의존도 그래프 (PDP)")

        features_to_plot = st.multiselect(
            "PDP를 그리고 싶은 변수를 선택하세요",
            options=X.columns.tolist()
        )

        if features_to_plot:
            fig, axs = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 4))
            if len(features_to_plot) == 1:
                axs = [axs]

            for i, feature in enumerate(features_to_plot):
                try:
                    pd_result = partial_dependence(model, X_test, [feature])
                    grid_values = pd_result['values'][0]
                    averages = pd_result['average'][0]

                    axs[i].plot(grid_values, averages, marker='o')
                    axs[i].set_title(f"{feature}")
                    axs[i].set_xlabel(feature)
                    axs[i].set_ylabel("Partial Dependence")
                except Exception as e:
                    st.error(f"PDP를 그리는데 오류가 발생했습니다: {e}")

            st.pyplot(fig)

else:
    st.info("⬆️ 왼쪽에서 CSV 파일을 업로드해주세요!")
