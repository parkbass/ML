# RF.py (최신 streamlit용 버전)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

# 1. 제목
st.title("\ud655\ub960\ubd84\ub958 \ubc29\uc2dd(Random Forest) \uc790\ub8cc \ubd84\uc11d \uc6f9\uc571")

# 2. 파일 업로드
uploaded_file = st.file_uploader("\ud30c\uc77c \uc5c5\ub85c\ub4dc", type=['csv'])

if uploaded_file:
    try:
        # 데이터 읽기
        df = pd.read_csv(uploaded_file)
        st.success(f"\ub85c\ub4dc\ub41c \ub370\uc774\ud130\ud615\ud0dc: {df.shape}")

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
        target_col = st.selectbox("\ud655\uc7a5\ud560 \ubaa9\ud45c \ubcc0\uc218\ub97c \uc120\ud0dd\ud558\uc138\uc694", df.columns)

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
            st.success(f"\ud3c9\uac00 \uacb0\uacfc (\ud14c\uc2a4\ud2b8 \ub370\uc774\ud130 R\xb2): {r2:.3f}")

            # 변수 중요도 출력
            st.subheader("\ud655\uc7a5\ub41c \ubcc0\uc218 \uc911\uc694\ub3c4 (Feature Importance)")
            feature_importance = pd.DataFrame({
                '변수': X.columns,
                '중요도': model.feature_importances_
            }).sort_values(by='중요도', ascending=False)
            st.dataframe(feature_importance)

            # PDP 그래프
            st.subheader("\ud655\uc7a5\ub41c \ubcc0\uc218\ubcc4 \ubd84\ub958 \uc758\uc870\ub3c4 (PDP)\uadf8\ub798\ud504")
            for feature in X.columns:
                fig, ax = plt.subplots(figsize=(6,4))
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_test,
                    features=[feature],
                    ax=ax,
                    kind='average'
                )
                st.pyplot(fig)

    except Exception as e:
        st.error(f"\uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4: {e}")
