import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="AI Alpha Trader", layout="wide")

# --- 분석 함수 (캐싱 적용으로 속도 향상) ---
@st.cache_data
def get_stock_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    
    # 지표 계산
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # 볼린저 밴드 (50, 0.5)
    bb = ta.bbands(df['Close'], length=50, std=0.5)
    df = pd.concat([df, bb], axis=1)
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Snow'] = ta.ema(df['Close'], length=20).diff()
    
    # 피보나치 (최근 1년)
    high_1y = df['High'].iloc[-252:].max()
    low_1y = df['Low'].iloc[-252:].min()
    df['Fib_618'] = high_1y - ((high_1y - low_1y) * 0.618)
    
    return df.dropna()

# --- 사이드바 설정 ---
st.sidebar.title("🤖 AI 분석 설정")
ticker = st.sidebar.text_input("종목 티커 입력 (예: 005930.KS, AAPL)", "005930.KS")
analyze_btn = st.sidebar.button("분석 시작")

# --- 메인 대시보드 ---
st.title(f"📈 {ticker} AI 전략 분석 리포트")

if analyze_btn:
    df = get_stock_data(ticker)
    
    if df is not None:
        # 1. AI 학습 및 예측
        df_ml = df.copy()
        df_ml['Target'] = (df_ml['Close'].shift(-5) > df_ml['Close'] * 1.03).astype(int)
        features = ['RSI', 'MACD_12_26_9', 'Snow', 'ATR']
        
        X = df_ml[features][:-5]
        y = df_ml['Target'][:-5]
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        prob = model.predict_proba(df_ml[features].iloc[[-1]])[0][1]

        # 2. 상단 요약 지표 (Metrics)
        curr_price = float(df['Close'].iloc[-1])
        atr = float(df['ATR'].iloc[-1])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("현재가", f"{curr_price:,.0f}원")
        col2.metric("AI 상승 확률", f"{prob*100:.1f}%")
        col3.metric("변동성(ATR)", f"{atr:,.1f}")

        # 3. 차트 시각화 (Plotly)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name="캔들"))
        fig.add_trace(go.Scatter(x=df.index, y=df['BBM_50_0.5'], line=dict(color='orange'), name="BB 중심"))
        fig.update_layout(title=f"{ticker} 주가 차트", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # 4. 분석 결과 레이아웃
        st.subheader("🛠 매매 전략 가이드")
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.info("🎯 **매수 및 물타기 타점 (ATR 기반)**")
            st.write(f"- **1차 진입:** {curr_price:,.0f}")
            st.write(f"- **2차 물타기:** {curr_price - (atr * 1.5):,.0f}")
            st.write(f"- **3차 물타기:** {curr_price - (atr * 3):,.0f}")
            
        with right_col:
            st.warning("🔥 **매도 및 불타기 타점**")
            st.write(f"- **1차 불타기:** {curr_price + (atr * 2):,.0f}")
            st.write(f"- **최종 익절:** {curr_price + (atr * 4):,.0f}")
            st.write(f"- **피보나치 지지(0.618):** {df['Fib_618'].iloc[-1]:,.0f}")

    else:
        st.error("데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")

# 하단 정보
st.markdown("---")
st.caption("주의: 본 데이터는 AI 학습 결과이며 실제 투자 수익을 보장하지 않습니다.")
