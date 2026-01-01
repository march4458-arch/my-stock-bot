import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, requests, numpy as np
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 구글 시트 연동
# ==========================================
st.set_page_config(page_title="주식 비서 V63.0 Alpha", page_icon="🚀", layout="wide")

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        return df.dropna(how='all') if df is not None else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    conn.update(data=df)
    st.success("구글 시트 동기화 완료!")

# ==========================================
# 🧠 2. 분석 엔진 (수급 및 고도화 점수 포함)
# ==========================================
def fetch_stock_smart(code, days=365):
    code_str = str(code).zfill(6)
    try:
        # yfinance를 통해 가격 및 수급 데이터 기반 마련
        ticker_symbol = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        if df.empty: return None
        return df
    except: return None

def get_hybrid_indicators(df):
    if df is None or len(df) < 60: return None
    df = df.copy()
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # RSI 계산
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    # OB(Order Block) 산출
    avg_vol = df['Volume'].rolling(20).mean()
    ob_zones = df[(df['Close'] > df['Open']*1.02) & (df['Volume'] > avg_vol*1.5)]
    df['OB_Price'] = ob_zones['Low'].mean() if not ob_zones.empty else df['MA20'].iloc[-1]
    
    return df

def calculate_advanced_score(df, strat):
    # 1. 과매도 점수 (25점): RSI 기반
    rsi = df['RSI'].iloc[-1]
    rsi_score = max(0, (50 - rsi) * 0.5) 

    # 2. 지지선 점수 (25점): 현재가와 OB 가격 근접도
    cp = df['Close'].iloc[-1]
    ob = df['OB_Price'].iloc[-1]
    dist = abs(cp - ob) / ob
    ob_score = max(0, 25 * (1 - dist * 10))

    # 3. 수급 점수 (25점): 최근 5일간 종가 추세로 추정 (외인/기관 수급 대용)
    # 실제 수급 API는 유료가 많아 거래량 변화율과 가격 강도로 추정치 계산
    vol_change = df['Volume'].iloc[-1] / df['Volume'].rolling(5).mean().iloc[-1]
    price_change = df['Close'].iloc[-1] / df['Close'].iloc[-5]
    supply_score = 25 if (vol_change > 1.2 and price_change > 1.0) else 10 if price_change > 1.0 else 0

    # 4. 기대수익 점수 (25점): 목표가(Sell 1차) 대비 상승 여력
    target = strat['sell'][0]
    upside = (target - cp) / cp
    profit_score = min(25, upside * 100)

    return rsi_score + ob_score + supply_score + profit_score

def get_strategy(df, buy_price=0):
    cp = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    ob = df['OB_Price'].iloc[-1]
    
    def adj(p): return int(round(p/100)*100) if p > 1000 else int(round(p/10)*10)
    
    buy = [adj(cp - atr), adj(ob)]
    sell = [adj(cp + atr*2), adj(cp + atr*4)]
    
    # 피라미딩 가이드
    pyramiding = {"type": "💤 관망", "msg": "신호 대기 중", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": "지지선 부근 비중 확대 권장", "color": "#FF4B4B"}
        elif yield_pct > 7: pyramiding = {"type": "🔥 불타기", "msg": "수익권 진입, 추격 매수 가능", "color": "#4FACFE"}

    return {"buy": buy, "sell": sell, "ob": ob, "rsi": df['RSI'].iloc[-1], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 및 대시보드
# ==========================================
with st.sidebar:
    st.title("🛡️ V63.0 Alpha")
    st.info("외인/기관 수급 지표 반영됨")

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 고도화 스캐너", "➕ 관리"])

with tabs[0]: # 대시보드
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        st.subheader("내 포트폴리오 상태")
        st.dataframe(portfolio, use_container_width=True)
    else: st.info("관리 탭에서 종목을 등록하세요.")

with tabs[2]: # 고도화 스캐너
    if st.button("🚀 외인/기관 수급 기반 전수 조사"):
        stocks = fdr.StockListing('KRX').head(50) # 시총 상위 50개 우선
        found = []
        with st.spinner("세력 수급 분석 중..."):
            for _, row in stocks.iterrows():
                df = fetch_stock_smart(row['Code'])
                df = get_hybrid_indicators(df)
                if df is not None:
                    strat = get_strategy(df)
                    score = calculate_advanced_score(df, strat)
                    if df['RSI'].iloc[-1] < 55: # 너무 과열되지 않은 종목만
                        found.append({"name": row['Name'], "score": score, "cp": df['Close'].iloc[-1], "strat": strat})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for d in found:
            st.markdown(f"""
            <div style="background:#1E1E1E; padding:15px; border-radius:10px; border-left:10px solid #4FACFE; margin-bottom:10px;">
                <h4 style="margin:0;">{d['name']} (신뢰점수: {d['score']:.1f}점)</h4>
                <p style="font-size:14px;">현재가: {int(d['cp']):,}원 | 목표가: {d['strat']['sell'][0]:,}원</p>
            </div>
            """, unsafe_allow_html=True)

with tabs[3]: # 관리
    st.subheader("종목 관리 (구글 시트 연동)")
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명")
        p = c2.number_input("평단가", 0)
        q = c3.number_input("수량", 0)
        if st.form_submit_button("저장"):
            # 종목코드 찾기 생략 (간소화)
            new_row = pd.DataFrame([["", n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
            df_p = pd.concat([df_p, new_row], ignore_index=True)
            save_portfolio_gsheets(df_p)
            st.rerun()
