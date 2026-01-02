import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests
import numpy as np
import pandas_ta as ta  # 보조지표 계산 최적화용
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed

# [기존 설정 유지...]
st.set_page_config(page_title="주식 비서 V63.0 AI Pro", page_icon="🤖", layout="wide")

# ==========================================
# 🧠 1. 고도화된 AI & 지표 엔진 (V63 업그레이드)
# ==========================================

def get_hybrid_indicators(df):
    if df is None or len(df) < 150: return None
    df = df.copy()
    
    # 1. 요청하신 보조지표군 통합 (pandas_ta 활용)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # 볼린저 밴드 (50, 0.5)
    bb = ta.bbands(df['Close'], length=50, std=0.5)
    df = pd.concat([df, bb], axis=1)
    
    # 스토캐스틱 (K, D)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, stoch], axis=1)
    
    # ATR 및 스노우(EMA 기울기 가속도)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Snow'] = ta.ema(df['Close'], length=20).diff()
    
    # 2. 오더블록 (OB) 정밀 계산
    avg_vol = df['Volume'].rolling(20).mean()
    df['OB_Zone'] = np.nan
    for i in range(len(df)-100, len(df)):
        # 급등/급락 직전의 캔들 포착
        if abs(df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Open'].iloc[i] > 0.03:
            if df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
                df.iloc[i, df.columns.get_loc('OB_Zone')] = df['Low'].iloc[i]

    # 3. 피보나치 되돌림 (5년/1년 통합 최저/최고)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fib_618'] = hi_1y - (hi_1y - lo_1y) * 0.618
    df['Fib_500'] = hi_1y - (hi_1y - lo_1y) * 0.5
    
    # 4. AI 학습 모듈 (습득형 로직)
    # 5일 후 종가가 현재보다 3% 이상 상승하면 1, 아니면 0
    df['Target'] = (df['Close'].shift(-5) > df['Close'] * 1.03).astype(int)
    features = ['RSI', 'MACD_12_26_9', 'Snow', 'ATR', f'BBP_50_0.5']
    
    # 결측치 제거 후 학습
    train_df = df.dropna(subset=features + ['Target'])
    if len(train_df) > 100:
        X = train_df[features][:-5]
        y = train_df['Target'][:-5]
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        # 최신 데이터로 확률 예측
        latest_X = df[features].iloc[[-1]]
        df['AI_Prob'] = model.predict_proba(latest_X)[0][1]
    else:
        df['AI_Prob'] = 0.5

    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr = curr['Close'], curr['ATR']
    ai_prob = curr['AI_Prob']
    
    # 호가 단위 조정 (기존 로직 유지)
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    # 3분할 매수/매도 로직 고도화
    # AI 확률이 높으면 공격적(타점 높임), 낮으면 보수적(타점 낮춤)
    if ai_prob > 0.6:
        buy = [adj(cp), adj(curr['Fib_500']), adj(curr['Fib_618'])]
        sell = [adj(cp + atr*2), adj(cp + atr*4), adj(cp + atr*6)]
    else:
        buy = [adj(curr['Fib_618']), adj(curr['Fib_618'] - atr), adj(curr['Fib_618'] - atr*2)]
        sell = [adj(cp + atr*1.5), adj(curr['Fib_500']), adj(df['High'].max())]

    # 물타기/불타기 로직
    pyramiding = {"type": "💤 관망", "msg": "현재 분석 대기 중입니다.", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            pyramiding = {"type": "💧 물타기", "msg": f"{yield_pct:.1f}% 손실 중. {buy[1]:,}원 부근 비중 확대", "color": "#FF4B4B"}
        elif yield_pct > 7 and ai_prob > 0.65:
            pyramiding = {"type": "🔥 불타기", "msg": f"{yield_pct:.1f}% 수익 중. 추세 강화 구간 추가 매수 가능", "color": "#4FACFE"}

    return {
        "buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93),
        "ai_prob": ai_prob, "rsi": curr['RSI'], "pyramiding": pyramiding,
        "ob": curr['OB_Zone'] if not np.isnan(curr['OB_Zone']) else curr['Fib_618']
    }

# ==========================================
# 🖥️ 2. UI 레이아웃 (탭 1: AI 리포트 집중 수정)
# ==========================================

# [보유 종목 불러오기 로직...]

with tabs[1]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        selected = st.selectbox("진단할 종목 선택", portfolio['Name'].unique(), key="ai_report_select")
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        
        with st.spinner('AI가 과거 데이터를 학습하여 패턴을 분석 중입니다...'):
            df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
            
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=s_info['Buy_Price'])
            
            # 메트릭 섹션
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AI 예측 승률", f"{strat['ai_prob']*100:.1f}%")
            c2.metric("RSI (14)", f"{strat['rsi']:.1f}")
            c3.metric("주요 지지(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"AI 권장 손절: {strat['stop']:,}원")
            
            # 가이드 섹션 (물타기/불타기)
            py = strat['pyramiding']
            st.markdown(f"""<div style="background-color:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid {py['color']};">
                <h3 style="margin:0; color:{py['color']};">{py['type']} 전략</h3>
                <p style="font-size:1.2em; margin-top:10px;">{py['msg']}</p></div>""", unsafe_allow_html=True)
            
            # 3분할 가격 섹션
            st.write("")
            col_b, col_s = st.columns(2)
            with col_b:
                st.info(f"🔵 **AI 선정 3분할 매수/물타기**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원\n\n3차: {strat['buy'][2]:,}원")
            with col_s:
                st.success(f"🔴 **AI 선정 3분할 매도/불타기**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원\n\n3차: {strat['sell'][2]:,}원")

            # 차트 시각화 (볼린저 밴드 포함)
            fig = go.Figure()
            df_p = df_detail.tail(150)
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BBU_50_0.5'], line=dict(color='rgba(200,200,200,0.5)'), name="BB Upper"))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BBL_50_0.5'], line=dict(color='rgba(200,200,200,0.5)'), name="BB Lower", fill='tonexty'))
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
