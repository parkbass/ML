import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

# ----- 기본 세팅 -----
st.set_page_config(page_title="랜덤포레스트 분석기", page_icon="🌳", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>확률분류 방식(Random Forest) 데이터 분석 웹앱</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----- 파일 업로드 -----
uploaded_file = st.file_uploader("파일 업로드 (CSV 형식)", type=['csv'])

if uploaded_file is not None:
    try:
        # 데이터 불러오기
        df = pd.read_csv(uploaded_file)

        st.success("✅ 파일 업로드 성공!")
        st.write("**업로드된 데이터 미리보기:**")
        st.dataframe(df.head())

        # ----- 종속 변수 선택 -----
        st.markdown("### 🎯 종속변수(예측하고 싶은 목표 변수)를 선택하세요")
        target_column = st.selectbox("종속변수 선택", options=df.columns.tolist())

        # ----- 모델링 -----
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # 데이터 전처리: 숫자형만 사용
            X = X.select_dtypes(include=[np.number])
            y = pd.to_numeric(y, errors='coerce')

            # 결측치 제거
            valid_idx = y.notna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

            # 학습용/테스트용 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 모델 성능
            score = model.score(X_test, y_test)
            st.success(f"📈 모델의 R²(테스트 데이터 기준) : **{score:.3f}**")

            # ----- 변수 중요도 표시 -----
            st.markdown("### 📝 변수 중요도 (Feature Importance)")

            importance_df = pd.DataFrame({
                '변수': X.columns,
                '중요도': model.feature_importances_
            }).sort_values(by='중요도', ascending=False)

            st.dataframe(importance_df)

            # ----- 부분 의존도 플롯 표시 -----
            st.markdown("### 📊 변수별 부분 의존도 그래프 (PDP)")

            selected_features = st.multiselect(
                "PDP를 그리고 싶은 변수를 선택하세요",
                options=X.columns.tolist(),
                default=X.columns.tolist()[:3]
            )

            if selected_features:
                fig, axes = plt.subplots(1, len(selected_features), figsize=(5 * len(selected_features), 4))

                if len(selected_features) == 1:
                    axes = [axes]

                for idx, feature in enumerate(selected_features):
                    pd_result = partial_dependence(model, X_test, [feature])
                    grid_values = pd_result['features'][0]
                    averages = pd_result['average'][0]

                    axes[idx].plot(grid_values, averages, marker='o')
                    axes[idx].set_xlabel(feature)
                    axes[idx].set_ylabel('Partial Dependence')
                    axes[idx].set_title(feature)

                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")

else:
    st.info("👈 왼쪽에서 CSV 파일을 업로드 해주세요!")
