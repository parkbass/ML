# RF.py (최신 streamlit용 버전)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# 1. 제목
st.title("확률분류 방식(Random Forest) 자료 분석 웹앱")

# 2. 파일 업로드
uploaded_file = st.file_uploader("파일 업로드", type=['csv'])

if uploaded_file:
    try:
        # 데이터 읽기
        df = pd.read_csv(uploaded_file)
        st.success(f"로드된 데이터형태: {df.shape}")

        # 문자열 데이터가 있으면 삭제 또는 처리
        df = df.replace(['#DIV/0!', 'NaN', 'nan'], np.nan)
        df = df.dropna()

        # 쉼표(,)가 들어간 숫자형 문자열이 있으면 변환
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.replace(",", "").astype(float)
                except:
                    pass

        # 종속변수 선택
        target_col = st.selectbox("확장할 목표 변수를 선택하세요", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # 학습/테스트 분리
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # R2 출력
            r2 = model.score(X_test, y_test)
            st.success(f"평가 결과 (테스트 데이터 R²): {r2:.3f}")

            # 변수 중요도 출력
            st.subheader("확장된 변수 중요도 (Feature Importance)")
            feature_importance = pd.DataFrame({
                '변수': X.columns,
                '중요도': model.feature_importances_
            }).sort_values(by='중요도', ascending=False)
            st.dataframe(feature_importance)

            # PDP 그래프
            st.subheader("확장된 변수별 부분 의존도 (PDP) 그래프")
            for feature in X.columns:
                fig, ax = plt.subplots(figsize=(6, 4))  # 여기 크기 고정 (적당히 보기 좋은 크기)
                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_test,
                    features=[feature],
                    ax=ax,
                    kind='average'
                )
                plt.tight_layout()  # 여백 조정
                st.pyplot(fig)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
