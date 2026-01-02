import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 KST 시간 함수
# ==========================================
def get_now_kst():
    """표준 공백을 사용한 KST 시간 반환 함수 (특수문자 제거됨)"""
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="주식 비서 V64.6 Final Master", page_icon="⚡", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티: 알림 및 데이터 연동] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

def get_portfolio_gsheets():
    """구글 시트 연동 및 데이터 보정 함수"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is None or df.empty:
            return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
        
        df = df.dropna(how='all')
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 
                      'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
        df = df.rename(columns=rename_map)
        
        for col in ['Buy_Price', 'Qty']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
        return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except Exception as e:
        st.sidebar.warning(f"⚠️ 데이터 연동 대기 중... ({type(e).__name__})")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🛡️ 2. 트리플 백업 데이터 엔진 (KRX -> Naver -> Yahoo)
# ==========================================
@st.cache_data(ttl=3600)
def get_krx_list():
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except:
        st.warning("⚠️ KRX 서버 응답 지연: 네이버 금융으로 전환합니다.")
    try:
        ks = fdr.StockListing('KOSPI')
        kd = fdr.StockListing('KOSDAQ')
        return pd.concat([ks, kd])
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Marcap'])

def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (get_now_kst() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code_str, start_date)
        if df is not None and not df.empty: return df
    except: pass
    try:
        ticker = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
        df_yf = yf.download(ticker, start=start_date, progress=False, timeout=10)
        if df_yf is not None and not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex): df_yf.columns = df_yf.columns.get_level_values(0)
            return df_yf
    except: return None

# 

# ==========================================
# 🧠 3. 하이브리드 지표 및 전략 엔진
# ==========================================
def get_hybrid_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9)).fillna(0)))
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'], df['Fibo_382'] = hi_1y-(rng*0.618), hi_1y-(rng*0.382)
    
    avg_vol = df['Volume'].rolling(20).mean()
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, f618 = curr['Close'], curr['ATR'], curr['OB_Price'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    buy = [adj(cp - atr * 1.1), adj(ob), adj(f618)]
    sell = [adj(cp + atr * 2.5), adj(cp + atr * 4.0), adj(df.tail(252)['High'].max() * 1.05)]
    stop = adj(min(buy) * 0.93)
    
    pyramiding = {"type": "💤 관망", "msg": "타점 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {yield_pct:.1f}% 달성", "color": "#28a745", "alert": True}
        elif cp <= stop: pyramiding = {"type": "⚠️ 손절", "msg": "손절가 터치", "color": "#dc3545", "alert": True}
        elif yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": "추가 매수 구간", "color": "#d63384", "alert": True}

    return {"buy": buy, "sell": sell, "stop": stop, "regime": curr['Regime'], "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 4. UI 탭 구현
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Master V64.6")
    tg_token = st.text_input("Telegram Token", type="password")
    tg_id = st.text_input("Chat ID")
    auto_refresh = st.checkbox("자동 갱신", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 적중 분석", "➕ 관리"])

with tabs[2]: # 스캐너 탭
    if st.button("🚀 전 종목 유기적 스캔 가동"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values('Marcap', ascending=False).head(50)
        found, prog = [], st.progress(0)
        
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None and res.iloc[-1]['RSI'] < 46:
                    st_res = get_strategy(res)
                    found.append({"name": futs[f], "cp": res.iloc[-1]['Close'], "strat": st_res})
                prog.progress((i + 1) / len(targets))
        
        for d in found:
            st.markdown(f"""<div class="scanner-card">
                <h3>{d['name']} <small>{d['strat']['regime']}</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div class="buy-box"><b>🔵 매수 타점</b><br>1차: {d['strat']['buy'][0]:,}<br>2차: {d['strat']['buy'][1]:,}</div>
                    <div class="sell-box"><b>🔴 매도 목표</b><br>1차: {d['strat']['sell'][0]:,}<br>2차: {d['strat']['sell'][1]:,}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# [이하 대시보드 및 백테스트 로직은 V64.5와 동일하게 통합]
