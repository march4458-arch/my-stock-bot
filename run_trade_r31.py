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
# ⚙️ 1. 시스템 설정 및 라이트 테마 CSS
# ==========================================
st.set_page_config(page_title="주식 비서 V62.3 Hybrid Full Spec", page_icon="⚡", layout="wide")

# 타임존 설정 (한국 시간 고정)
KST = pytz.timezone('Asia/Seoul')

def get_now_kst():
    return datetime.datetime.now(KST)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #666666 !important; }
    .guide-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; background-color: #ffffff; border: 1px solid #dee2e6; }
    .guide-box p { color: #212529 !important; font-size: 1.1rem; margin: 0; }
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; border: 1px solid #e9ecef; }
    .inner-box b { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티 함수] ---
@st.cache_data(ttl=3600)
def get_krx_list():
    return fdr.StockListing('KRX')

def get_market_status():
    now = get_now_kst()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if start <= now <= end: return True, "정규장 운영 중 🚀"
    return False, "장외 시간 🌙"

def is_report_time():
    now = get_now_kst()
    return now.hour == 18 and 0 <= now.minute <= 10

# --- [데이터 연동 함수] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            cols = ['Code', 'Name', 'Buy_Price', 'Qty']
            for col in cols:
                if col not in df.columns: df[col] = 0
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: 
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트 동기화 완료!")
    except Exception as e: st.error(f"저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 🧠 2. 고도화된 분석 엔진
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
    
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_advanced_score(df, strat):
    curr = df.iloc[-1]
    rsi_score = max(0, (75 - curr['RSI']) * 0.4)
    vol_score = min(25, max(0, curr['Vol_Zscore'] * 10)) if curr['Close'] > curr['Open'] else 0
    dist_ob = abs(curr['Close'] - curr['OB_Price']) / (curr['OB_Price'] + 1e-9)
    ob_score = max(0, 25 * (1 - dist_ob * 10))
    upside = (strat['sell'][0] - curr['Close']) / (curr['Close'] + 1e-9)
    profit_score = min(20, upside * 100)
    return float(rsi_score + vol_score + ob_score + profit_score)

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy, sell = [adj(cp - atr*1.1), adj(ob), adj(curr['Fibo_500'])], [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(cp * 1.2)]
    elif regime == "📉 하락":
        buy, sell = [adj(curr['Fibo_618']), adj(df.tail(252)['Low'].min())], [adj(curr['Fibo_500']), adj(ob)]
    else:
        buy, sell = [adj(curr['Fibo_500']), adj(ob)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max())]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": "목표가 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "손절가 이탈!", "color": "#dc3545", "alert": True}
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 로직
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Final KST")
    market_on, market_msg = get_market_status()
    st.write(f"🇰🇷 한국 시간: {get_now_kst().strftime('%H:%M:%S')}")
    st.info(f"**시장 상태: {market_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_portfolio = st.checkbox("보유종목 실시간 알림", value=True)
    auto_refresh = st.checkbox("자동 갱신 활성화", value=False)
    refresh_interval = st.slider("갱신 주기 (분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드 (오류 수정 반영)] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list, alert_needed, alert_msg = 0.0, 0.0, [], False, "🚨 <b>실시간 시장 보고</b>\n\n"
        with st.spinner('포트폴리오 분석 중...'):
            for _, row in portfolio.iterrows():
                try:
                    b_price = float(row['Buy_Price'])
                    qty = float(row['Qty'])
                    if qty <= 0: continue

                    df = fetch_stock_smart(row['Code'], days=150)
                    if df is not None:
                        idx = get_hybrid_indicators(df)
                        st_res = calculate_organic_strategy(idx, b_price)
                        cp = float(idx.iloc[-1]['Close'])
                        
                        buy_sum = b_price * qty
                        eval_sum = cp * qty
                        profit = eval_sum - buy_sum
                        
                        total_buy += buy_sum
                        total_eval += eval_sum
                        dash_list.append({"종목": row['Name'], "수익": profit, "평가액": eval_sum})

                        if alert_portfolio and market_on and st_res['pyramiding']['alert']:
                            alert_needed = True
                            alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']} ({int(cp):,}원)\n"
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
    else: st.info("종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    if not portfolio.empty:
        selected = st.selectbox("종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime']); c2.metric("RSI", f"{strat['rsi']:.1f}"); c3.metric("세력지지", f"{int(strat['ob']):,}원"); c4.error(f"손절가: {strat['stop']:,}원")
            st.markdown(f'<div class="guide-box" style="border-left:8px solid {strat["pyramiding"]["color"]};"><h3>{strat["pyramiding"]["type"]}</h3><p>{strat["pyramiding"]["msg"]}</p></div>', unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n2차: {strat['buy'][1]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n2차: {strat['sell'][1]:,}원")
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'])])
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 시장 전수 스캔"):
        stocks = get_krx_list().sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with ThreadPoolExecutor(max_workers=5) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in stocks.iterrows()}
            for f in as_completed(futures):
                name, df_scan = futures[f], f.result()
                if df_scan is not None:
                    st_tmp = calculate_organic_strategy(df_scan)
                    score = calculate_advanced_score(df_scan, st_tmp)
                    found.append({"name": name, "score": score, "buy": st_tmp['buy'][0]})
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for d in found[:10]:
            st.write(f"**{d['name']}**: {d['score']:.1f}점 (1차 매수가: {d['buy']:,}원)")

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    bt_name = st.text_input("종목명", "에코프로비엠")
    if st.button("백테스트 시작"):
        krx = get_krx_list()
        match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=365))
            if df_bt is not None:
                st.line_chart(df_bt['Close'])

# --- [➕ 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_gs"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가", 0), c3.number_input("수량", 0)
        if st.form_submit_button("저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=df_p.columns)
                save_portfolio_gsheets(pd.concat([df_p, new_row], ignore_index=True))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

# --- [갱신] ---
if auto_refresh:
    time.sleep(refresh_interval * 60)
    st.rerun()
