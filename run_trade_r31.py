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
# ⚙️ 1. 시스템 설정 및 KST 타임존
# ==========================================
st.set_page_config(page_title="주식 비서 V62.5 Full Spec", page_icon="⚡", layout="wide")
KST = pytz.timezone('Asia/Seoul')

def get_now_kst():
    return datetime.datetime.now(KST)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; border: 1px solid #e9ecef; }
    .inner-box b { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티] ---
@st.cache_data(ttl=3600)
def get_krx_list():
    return fdr.StockListing('KRX')

def get_market_status():
    now = get_now_kst()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return (start <= now <= end), ("정규장 운영 중 🚀" if start <= now <= end else "장외 시간 🌙")

def is_report_time():
    now = get_now_kst()
    return now.hour == 18 and 0 <= now.minute <= 10

# --- [데이터 연동] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            for col in ['Code', 'Name', 'Buy_Price', 'Qty']:
                if col not in df.columns: df[col] = 0
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 복구된 정밀 분석 엔진
# ==========================================
@st.cache_data(ttl=300)
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (get_now_kst() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code_str, start_date)
        if df is not None and not df.empty: return df
    except:
        try:
            ticker = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
            df = yf.download(ticker, start=start_date, progress=False, timeout=5)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except: return None

def get_hybrid_indicators(df):
    if df is None or len(df) < 150: return None
    df = df.copy()
    close = df['Close']
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    # OB(Order Block) 계산 (초기 정밀 로직)
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    # 피보나치 복구
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (rng * 0.382)
    df['Fibo_500'] = hi_1y - (rng * 0.500)
    df['Fibo_618'] = hi_1y - (rng * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_advanced_score(df, strat):
    """초기 점수 산출 로직 100% 복구"""
    curr = df.iloc[-1]
    rsi_score = max(0, (75 - curr['RSI']) * 0.4)
    vol_score = min(25, max(0, curr['Vol_Zscore'] * 10)) if curr['Close'] > curr['Open'] else 0
    dist_ob = abs(curr['Close'] - curr['OB_Price']) / (curr['OB_Price'] + 1e-9)
    ob_score = max(0, 25 * (1 - dist_ob * 10))
    upside = (strat['sell'][0] - curr['Close']) / (curr['Close'] + 1e-9)
    profit_score = min(20, upside * 100)
    return float(rsi_score + vol_score + ob_score + profit_score)

def calculate_organic_strategy(df, buy_price=0):
    """초기 3분할 전략 로직 100% 복구"""
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f500, f618 = curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy, sell = [adj(cp - atr*1.1), adj(ob), adj(f500)], [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(cp * 1.2)]
    elif regime == "📉 하락":
        buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)], [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절 알림", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절 알림", "msg": "위험 지지선 이탈", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {y:.1f}%. 추가 매수 권장", "color": "#d63384", "alert": True}
        elif y > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {y:.1f}%. 비중 확대", "color": "#0d6efd", "alert": True}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 및 탭별 로직
# ==========================================
with st.sidebar:
    st.title("⚡ 주식 비서 Full Spec")
    market_on, market_msg = get_market_status()
    st.write(f"🇰🇷 KST: {get_now_kst().strftime('%H:%M:%S')}")
    st.info(f"**상태: {market_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    auto_refresh = st.checkbox("자동 갱신", value=False)
    refresh_interval = st.slider("주기 (분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드 - 데이터 타입 오류 수정본] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list, alert_needed, alert_msg = 0.0, 0.0, [], False, "🚨 <b>실시간 포트폴리오 감시</b>\n\n"
        with st.spinner('실시간 분석 중...'):
            for _, row in portfolio.iterrows():
                try:
                    b_p = float(row['Buy_Price'])
                    qty = float(row['Qty'])
                    if qty <= 0: continue
                    df = fetch_stock_smart(row['Code'], days=150)
                    if df is not None:
                        idx = get_hybrid_indicators(df)
                        st_res = calculate_organic_strategy(idx, b_p)
                        cp = float(idx.iloc[-1]['Close'])
                        total_buy += (b_p * qty)
                        total_eval += (cp * qty)
                        dash_list.append({"종목": row['Name'], "수익": (cp-b_p)*qty, "평가액": cp*qty})
                        if market_on and st_res['pyramiding']['alert']:
                            alert_needed = True
                            alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n- 가이드: {st_res['pyramiding']['msg']}\n\n"
                except: continue
        
        if alert_needed: send_telegram_msg(tg_token, tg_id, alert_msg)
        if dash_list:
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            y_total = ((total_eval - total_buy) / total_buy * 100 if total_buy > 0 else 0)
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{y_total:+.2f}%")
            c3.metric("평가손익", f"{int(total_eval - total_buy):,}원")
            st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='수익', template="plotly_white"), use_container_width=True)
            st.dataframe(pd.DataFrame(dash_list), use_container_width=True)

# --- [💼 탭 1: AI 리포트 - 상세 분석 복구] ---
with tabs[1]:
    if not portfolio.empty:
        selected = st.selectbox("리포트 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            score = calculate_advanced_score(df_detail, strat)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AI 점수", f"{score:.1f}점"); c2.metric("국면", strat['regime']); c3.metric("RSI", f"{strat['rsi']:.1f}"); c4.error(f"손절가: {strat['stop']:,}원")
            st.markdown(f'<div class="guide-box" style="border-left:10px solid {strat["pyramiding"]["color"]};"><h3>{strat["pyramiding"]["type"]} 가이드</h3><p>{strat["pyramiding"]["msg"]}</p></div>', unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n2차: {strat['buy'][1]:,}원\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n2차: {strat['sell'][1]:,}원\n3차: {strat['sell'][2]:,}원")
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(120).index, open=df_detail.tail(120)['Open'], high=df_detail.tail(120)['High'], low=df_detail.tail(120)['Low'], close=df_detail.tail(120)['Close'])])
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너 - 초기 카드 디자인 복구] ---
with tabs[2]:
    if st.button("🚀 AI 시장 전수 조사 시작"):
        stocks = get_krx_list().sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with ThreadPoolExecutor(max_workers=8) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in stocks.iterrows()}
            for f in as_completed(futures):
                name, df_scan = futures[f], f.result()
                if df_scan is not None:
                    strat_tmp = calculate_organic_strategy(df_scan)
                    score = calculate_advanced_score(df_scan, strat_tmp)
                    if df_scan.iloc[-1]['RSI'] < 65:
                        found.append({"name": name, "score": score, "strat": strat_tmp})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found[:10]):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between;">
                        <h2 style="margin:0;">{icon} {d['name']}</h2>
                        <span style="color:#007bff; font-weight:bold; font-size:1.5em;">{d['score']:.1f}점</span>
                    </div>
                    <hr>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:25px;">
                        <div class="inner-box" style="border-top:5px solid #007bff;">
                            <b>🔵 3분할 매수 가격</b><br>
                            1차: <b style="float:right;">{d['strat']['buy'][0]:,}원</b><br>
                            2차: <b style="float:right;">{d['strat']['buy'][1]:,}원</b><br>
                            3차: <b style="float:right;">{d['strat']['buy'][2]:,}원</b>
                        </div>
                        <div class="inner-box" style="border-top:5px solid #dc3545;">
                            <b>🔴 3분할 매도 가격</b><br>
                            1차: <b style="float:right;">{d['strat']['sell'][0]:,}원</b><br>
                            2차: <b style="float:right;">{d['strat']['sell'][1]:,}원</b><br>
                            3차: <b style="float:right;">{d['strat']['sell'][2]:,}원</b>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    target = st.text_input("분석 종목명", "에코프로비엠")
    if st.button("전략 백테스트"):
        m = get_krx_list()[get_krx_list()['Name'] == target]
        if not m.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(m.iloc[0]['Code'], days=365))
            if df_bt is not None: st.line_chart(df_bt['Close'])

# --- [➕ 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_final"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가", 0), c3.number_input("수량", 0)
        if st.form_submit_button("시트 저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=df_p.columns)
                conn = st.connection("gsheets", type=GSheetsConnection)
                conn.update(data=pd.concat([df_p, new_row], ignore_index=True))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

# --- [갱신] ---
if auto_refresh:
    time.sleep(refresh_interval * 60)
    st.rerun()
