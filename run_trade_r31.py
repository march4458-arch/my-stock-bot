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
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="주식 비서 V62.5 Final Stable", page_icon="⚡", layout="wide")

# 라이트 테마 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티 함수] ---
@st.cache_data(ttl=3600)
def get_krx_list(): return fdr.StockListing('KRX')

def get_market_status():
    now = get_now_kst()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return (True, "정규장 운영 중 🚀") if start <= now <= end else (False, "장외 시간 🌙")

# --- [데이터 연동 및 AttributeError 방어] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            # 필수 컬럼 강제 생성 및 형식 지정
            for col in ['Code', 'Name', 'Buy_Price', 'Qty']:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            
            # 타입 변환 (AttributeError 방지 핵심)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (모든 수식 유지)
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
    
    # Order Block (OB) 계산
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    # Fibonacci Levels
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (rng * 0.382)
    df['Fibo_500'] = hi_1y - (rng * 0.500)
    df['Fibo_618'] = hi_1y - (rng * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f500, f618 = curr['Fibo_382'], curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = curr['Regime']
    if regime == "🚀 상승":
        buy, sell = [adj(cp - atr*1.1), adj(ob), adj(f500)], [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(cp * 1.2)]
    elif regime == "📉 하락":
        buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)], [adj(f382), adj(f500), adj(ob)]
    else:
        buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절 알림", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절 알림", "msg": f"손절가 {stop_loss:,}원 하회!", "color": "#dc3545", "alert": True}
        elif yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. 추가 매입 고려", "color": "#d63384", "alert": True}
        elif yield_pct > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 추격 확대", "color": "#0d6efd", "alert": True}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding, "fibo": [f382, f500, f618]}

# ==========================================
# 🖥️ 3. UI 구현
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Final Spec")
    m_on, m_msg = get_market_status()
    st.info(f"**KST: {get_now_kst().strftime('%H:%M')} | {m_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_on = st.checkbox("알림 활성화", value=True)
    auto_refresh = st.checkbox("자동 갱신", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_data, alert_needed, alert_msg = 0.0, 0.0, [], False, "🚨 <b>시장 보고</b>\n"
        with st.spinner('분석 중...'):
            for idx, row in portfolio.iterrows():
                try:
                    raw_df = fetch_stock_smart(row['Code'], days=150)
                    if raw_df is not None:
                        idx_df = get_hybrid_indicators(raw_df)
                        if idx_df is not None:
                            st_dict = calculate_organic_strategy(idx_df, row['Buy_Price'])
                            cp = float(idx_df['Close'].iloc[-1])
                            qty, bp = float(row['Qty']), float(row['Buy_Price'])
                            
                            t_buy += (bp * qty)
                            t_eval += (cp * qty)
                            dash_data.append({"종목": row['Name'], "수익": (cp-bp)*qty, "평가액": cp*qty})
                            
                            if alert_on and m_on and st_dict['pyramiding']['alert']:
                                alert_needed = True
                                alert_msg += f"- {row['Name']}: {st_dict['pyramiding']['type']}\n"
                except: continue
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        
        if dash_data:
            st.plotly_chart(px.bar(pd.DataFrame(dash_data), x='종목', y='수익', color='수익', template="plotly_white"), use_container_width=True)
        if alert_needed: send_telegram_msg(tg_token, tg_id, alert_msg)
    else: st.info("등록된 종목이 없습니다.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        raw_df = fetch_stock_smart(row['Code'])
        idx_df = get_hybrid_indicators(raw_df)
        if idx_df is not None:
            st_res = calculate_organic_strategy(idx_df, row['Buy_Price'])
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("국면", st_res['regime'])
            m2.metric("RSI", f"{st_res['rsi']:.1f}")
            m3.metric("세력지지(OB)", f"{int(st_res['ob']):,}원")
            m4.error(f"손절가: {st_res['stop']:,}원")
            
            st.markdown(f"""<div class="guide-box" style="border-left:8px solid {st_res['pyramiding']['color']};">
                <h3>{st_res['pyramiding']['type']}</h3><p>{st_res['pyramiding']['msg']}</p></div>""", unsafe_allow_html=True)
            
            st.info(f"🔵 **매수**: {st_res['buy'][0]:,} | {st_res['buy'][1]:,} | {st_res['buy'][2]:,}")
            st.success(f"🔴 **매도**: {st_res['sell'][0]:,} | {st_res['sell'][1]:,} | {st_res['sell'][2]:,}")
            
            fig = go.Figure(data=[go.Candlestick(x=idx_df.index[-120:], open=idx_df['Open'][-120:], high=idx_df['High'][-120:], low=idx_df['Low'][-120:], close=idx_df['Close'][-120:])])
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue")
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 시장 스캔"):
        krx = get_krx_list().sort_values('Marcap', ascending=False).head(50)
        found = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in krx.iterrows()}
            for f in as_completed(futs):
                res = f.result()
                if res is not None:
                    sc = (70 - res['RSI'].iloc[-1]) * 0.5 + (res['Vol_Zscore'].iloc[-1] * 5)
                    found.append({"name": futs[f], "score": sc, "strat": calculate_organic_strategy(res)})
        
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:5]:
            st.markdown(f"""<div class="scanner-card"><h3>{d['name']} ({d['score']:.1f}점)</h3>
                <p>1차 매수: {d['strat']['buy'][0]:,}원 | 1차 매도: {d['strat']['sell'][0]:,}원</p></div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    name = st.text_input("종목명", "에코프로비엠")
    if st.button("분석"):
        krx = get_krx_list()
        match = krx[krx['Name']==name]
        if not match.empty:
            df = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df is not None:
                trades, in_pos = [], False
                for i in range(150, len(df)):
                    curr = df.iloc[i]
                    s = calculate_organic_strategy(df.iloc[:i])
                    if not in_pos and curr['Low'] <= s['buy'][0]:
                        entry, in_pos = s['buy'][0], True
                    elif in_pos:
                        if curr['High'] >= entry * 1.1: trades.append(10); in_pos = False
                        elif curr['Low'] <= entry * 0.93: trades.append(-7); in_pos = False
                if trades:
                    st.metric("승률", f"{sum(1 for t in trades if t>0)/len(trades)*100:.1f}%")
                    st.line_chart(np.cumsum(trades))

# --- [➕ 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단"), c3.number_input("수량")
        if st.form_submit_button("저장"):
            krx = get_krx_list()
            match = krx[krx['Name']==n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=df_p.columns)
                conn = st.connection("gsheets", type=GSheetsConnection)
                conn.update(data=pd.concat([df_p, new_row]))
                st.rerun()
    st.dataframe(df_p)

if auto_refresh:
    time.sleep(interval * 60)
    st.rerun()
