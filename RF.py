import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from fpdf import FPDF
import io

# 제목
st.title("확률분류 방식(Random Forest) 자료 분석 웹앱")

# 파일 업로드
uploaded_file = st.file_uploader("파일 업로드", type=["csv"])

if uploaded_file:
    # 데이터 로딩
    df = pd.read_csv(uploaded_file)
    st.success("파일이 업로드되었습니다!")
    st.dataframe(df)

    # 종속변수 선택
    target = st.selectbox("종속변수(예측하고 싶은 목표 변수)를 선택하세요", df.columns)

    # 모델 학습
    if target:
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 성능 출력
        score = model.score(X_test, y_test)
        st.success(f"\ud3ec함 R^2(테스트 데이터 기준): {score:.3f}")

        # Feature Importance 출력
        st.subheader("\ud658상 \ubcc0수 중요도 (Feature Importance)")
        importance_df = pd.DataFrame({
            '변수': X.columns,
            '중요도': model.feature_importances_
        }).sort_values('중요도', ascending=False)

        st.dataframe(importance_df)

        # ✅ PDF 저장 버튼 추가
        buffer = io.BytesIO()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "변수 중요도 (Feature Importance)", ln=True, align='C')

        pdf.ln(10)
        for index, row in importance_df.iterrows():
            pdf.cell(0, 10, f"{row['변수']}: {row['중요도']:.4f}", ln=True)

        pdf.output(buffer)

        st.download_button(
            label="📄 변수 중요도 결과를 PDF로 저장하기",
            data=buffer.getvalue(),
            file_name="feature_importance.pdf",
            mime="application/pdf"
        )

        # PDP (부분의존도 그래프)
        st.subheader("\ubcc0수별 \ubd80분의여도 그래프 (PDP)")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            PartialDependenceDisplay.from_estimator(model, X_test, X.columns, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"PDP \uadf8래프 \uad6c현에 \ec5c5\uc73c\ub85c \ec7a5\uc0b0: {e}")
