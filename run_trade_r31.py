import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests
import numpy as np
import pytz
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 타임존 정의
# ==========================================
st.set_page_config(page_title="주식 비서 V62.2 Full Spec", page_icon="⚡", layout="wide")

KST = pytz.timezone('Asia/Seoul')

def get_now_kst():
    return datetime.datetime.now(KST)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    .guide-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; background-color: #ffffff; border: 1px solid #dee2e6; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- [시간 및 시장 상태] ---
def get_market_status():
    now = get_now_kst()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if start <= now <= end: return True, "정규장 운영 중 🚀"
    return False, "장외 시간 🌙"

def is_report_time():
    now = get_now_kst()
    return now.hour == 18 and 0 <= now.minute <= 15

# --- [데이터 연동] ---
@st.cache_data(ttl=3600)
def get_krx_list():
    return fdr.StockListing('KRX')

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트 저장 완료!")
    except Exception as e: st.error(f"저장 실패: {e}")

# ==========================================
# 🧠 2. 분석 엔진 (복구된 정밀 로직)
# ==========================================
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (get_now_kst() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code_str, start_date)
        if df is not None and not df.empty: return df
    except:
        try:
            ticker = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
            df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except: return None

def get_hybrid_indicators(df):
    if df is None or len(df) < 150: return None
    df = df.copy()
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    # Z-Score (거래량)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)
    
    # OB Price (강한 돌파 지점)
    ob_zones = []
    avg_vol = df['Volume'].rolling(20).mean()
    for i in range(len(df)-60, len(df)):
        if df['Close'].iloc[i] > df['Open'].iloc[i]*1.03 and df['Volume'].iloc[i] > avg_vol.iloc[i]*1.8:
            ob_zones.append(df['Low'].iloc[i])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    # Fibo
    hi, lo = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_500'] = hi - (hi - lo) * 0.5
    df['Fibo_618'] = hi - (hi - lo) * 0.618
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_advanced_score(df, strat):
    """AI 투자 점수 산출 로직 복구"""
    score = 50
    curr = df.iloc[-1]
    # RSI 점수 (과매도 가점)
    if curr['RSI'] < 35: score += 20
    elif curr['RSI'] > 75: score -= 15
    # 추세 점수
    if strat['regime'] == "🚀 상승": score += 15
    elif strat['regime'] == "📉 하락": score -= 10
    # 가격 위치 점수
    if curr['Close'] < strat['ob']: score += 10
    # 거래량 점수
    if curr['Vol_Z'] > 2: score += 5
    return max(0, min(100, score))

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    # 3분할 가격 산정 로직 복구
    if curr['Regime'] == "🚀 상승":
        buy = [adj(cp - atr), adj(ob), adj(curr['Fibo_500'])]
        sell = [adj(cp + atr*2.5), adj(cp + atr*4), adj(cp * 1.25)]
    else:
        buy = [adj(curr['Fibo_618']), adj(ob), adj(df.tail(252)['Low'].min())]
        sell = [adj(curr['Fibo_500']), adj(ob), adj(cp + atr*3)]
    
    stop_loss = adj(min(buy) * 0.93)
    
    # 알림 로직 복구
    pyramiding = {"type": "💤 관망", "msg": "대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"1차 목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "위험 구간 이탈", "color": "#dc3545", "alert": True}
        elif y < -7: pyramiding = {"type": "💧 물타기", "msg": "추가 매수 고려", "color": "#d63384", "alert": True}
    
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": curr['Regime'], "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 및 탭별 기능
# ==========================================
with st.sidebar:
    st.title("⚡ 주식 비서 Full Spec")
    market_on, market_msg = get_market_status()
    st.write(f"🇰🇷 KST: {get_now_kst().strftime('%H:%M:%S')}")
    st.info(market_msg)
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    auto_refresh = st.checkbox("자동 갱신", value=False)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, alert_msg = 0, 0, ""
        for _, row in portfolio.iterrows():
            df = fetch_stock_smart(row['Code'], days=150)
            if df is not None:
                idx = get_hybrid_indicators(df)
                st_res = calculate_organic_strategy(idx, row['Buy_Price'])
                cp = idx.iloc[-1]['Close']
                total_buy += (row['Buy_Price'] * row['Qty'])
                total_eval += (cp * row['Qty'])
                if st_res['pyramiding']['alert'] and market_on:
                    alert_msg += f"[{st_res['pyramiding']['type']}] {row['Name']}\n"
        
        if alert_msg: send_telegram_msg(tg_token, tg_id, "🚨 실시간 알림\n" + alert_msg)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수액", f"{int(total_buy):,}원")
        y_total = ((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0)
        c2.metric("총 평가액", f"{int(total_eval):,}원", f"{y_total:+.2f}%")
        c3.metric("손익", f"{int(total_eval-total_buy):,}원")

# --- [AI 리포트] ---
with tabs[1]:
    if not portfolio.empty:
        sel = st.selectbox("종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_raw = fetch_stock_smart(row['Code'])
        df_idx = get_hybrid_indicators(df_raw)
        if df_idx is not None:
            st_res = calculate_organic_strategy(df_idx, row['Buy_Price'])
            score = calculate_advanced_score(df_idx, st_res)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("AI 점수", f"{score}점")
            c2.metric("추세", st_res['regime'])
            c3.error(f"손절가: {st_res['stop']:,}원")
            
            st.markdown(f'<div class="guide-box" style="border-left:10px solid {st_res["pyramiding"]["color"]};"><h3>{st_res["pyramiding"]["type"]}</h3>{st_res["pyramiding"]["msg"]}</div>', unsafe_allow_html=True)
            
            b_col, s_col = st.columns(2)
            b_col.info(f"🔵 **3분할 매수**\n1차: {st_res['buy'][0]:,}원\n2차: {st_res['buy'][1]:,}원\n3차: {st_res['buy'][2]:,}원")
            s_col.success(f"🔴 **3분할 매도**\n1차: {st_res['sell'][0]:,}원\n2차: {st_res['sell'][1]:,}원\n3차: {st_res['sell'][2]:,}원")

# --- [스캐너] ---
with tabs[2]:
    if st.button("🚀 KOSPI 50 전수 스캐닝"):
        stocks = get_krx_list().sort_values(by='Marcap', ascending=False).head(50)
        results = []
        for _, r in stocks.iterrows():
            d = get_hybrid_indicators(fetch_stock_smart(r['Code']))
            if d is not None:
                s = calculate_organic_strategy(d)
                sc = calculate_advanced_score(d, s)
                results.append({"점수": sc, "종목명": r['Name'], "추세": s['regime'], "RSI": round(d.iloc[-1]['RSI'],1)})
        st.table(pd.DataFrame(results).sort_values(by="점수", ascending=False))

# --- [백테스트] ---
with tabs[3]:
    target = st.text_input("백테스트 종목명", "에코프로비엠")
    if st.button("백테스트 시작"):
        m = get_krx_list()[get_krx_list()['Name'] == target]
        if not m.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(m.iloc[0]['Code'], days=365))
            # 단순 수익률 시뮬레이션 로직 복구
            df_bt['Return'] = df_bt['Close'].pct_change()
            cum_ret = (1 + df_bt['Return']).prod() - 1
            st.write(f"최근 1년 홀딩 수익률: {cum_ret*100:.2f}%")
            st.line_chart(df_bt['Close'])

# --- [관리] ---
with tabs[4]:
    st.write("구글 시트 연동 데이터")
    st.dataframe(portfolio, use_container_width=True)

# ==========================================
# ⏳ 4. 자동 갱신 및 마감 리포트
# ==========================================
if is_report_time() and "report_done" not in st.session_state:
    # 18시 마감 리포트 전송 로직
    send_telegram_msg(tg_token, tg_id, "📝 오늘 장 마감 리포트 전송됨")
    st.session_state.report_done = True

if auto_refresh:
    time.sleep(600)
    st.rerun()
