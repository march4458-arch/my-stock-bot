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

st.set_page_config(page_title="주식 비서 V64.4 Dynamic Master", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티 및 데이터 연동] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
            df = df.rename(columns=rename_map)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0).astype(float)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(float)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df[['Code', 'Name', 'Buy_Price', 'Qty']]
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (Dynamic Priority Logic)
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
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    low_min, high_max = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['Stoch_K'] = ((close - low_min) / (high_max - low_min + 1e-9)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    std = close.rolling(20).std()
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'], df['Fibo_500'], df['Fibo_382'] = hi_1y-(rng*0.618), hi_1y-(rng*0.5), hi_1y-(rng*0.382)
    
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]

    hist_df = df.tail(20)
    counts, edges = np.histogram(hist_df['Close'], bins=10, weights=hist_df['Volume'])
    df['POC_Price'] = edges[np.argmax(counts)]
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, poc = curr['Close'], curr['ATR'], curr['OB_Price'], curr['POC_Price']
    f618, bbl = curr['Fibo_618'], curr['BB_Lower']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    # [V64.4] 유기적 타점 평가 엔진 (Dynamic priority)
    candidates = [
        {"name": "매물대(POC)", "price": poc, "score": 0},
        {"name": "피보나치(618)", "price": f618, "score": 0},
        {"name": "세력선(OB)", "price": ob, "score": 0},
        {"name": "밴드하단(BB)", "price": bbl, "score": 0}
    ]

    for cand in candidates:
        p = cand['price']
        if curr['RSI'] < 30 and p < cp * 0.95: cand['score'] += 20
        if curr['Stoch_K'] < 20: cand['score'] += 15
        dist = abs(cp - p) / (cp + 1e-9)
        if dist < 0.03: cand['score'] += 30
        if abs(p - bbl) / (bbl + 1e-9) < 0.01: cand['score'] += 25

    sorted_cand = sorted(candidates, key=lambda x: x['score'], reverse=True)
    buy = [adj(sorted_cand[0]['price']), adj(sorted_cand[1]['price']), adj(sorted_cand[2]['price'])]
    buy_names = [sorted_cand[0]['name'], sorted_cand[1]['name'], sorted_cand[2]['name']]
    
    sell = [adj(cp + atr*2.0), adj(cp + atr*3.5), adj(cp + atr*5.0)]
    stop_loss = adj(min(buy) * 0.93)
    
    pyramiding = {"type": "💤 관망", "msg": f"{buy_names[0]} 대기", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / (buy_price + 1e-9) * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": "수익 실현 구간", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "리스크 관리 필요", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"{buy_names[0]} 지점 추매", "color": "#d63384", "alert": True}

    return {"buy": buy, "buy_names": buy_names, "sell": sell, "stop": stop_loss, "regime": curr['Regime'], 
            "rsi": curr['RSI'], "pyramiding": pyramiding, "poc": poc, "ob": ob, "fibo": f618, "bb_l": bbl}

# ==========================================
# 🖥️ 3. 메인 인터페이스 (Tabs & Alerts)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Master V64.4")
    now_kst = get_now_kst()
    st.info(f"**KST: {now_kst.strftime('%H:%M')}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    st.markdown("---")
    min_marcap_input = st.number_input("최소 시총 (억)", value=5000)
    min_marcap = min_marcap_input * 100000000
    alert_scanner = st.checkbox("상세 텔레그램 알림", value=True)
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 전략 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list = 0.0, 0.0, []
        for _, row in portfolio.iterrows():
            idx_df = get_hybrid_indicators(fetch_stock_smart(row['Code'], days=200))
            if idx_df is not None:
                st_res = get_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['pyramiding']['type']})
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_dark"), use_container_width=True)

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = get_strategy(df_ai, row['Buy_Price'])
            py = st_res['pyramiding']
            st.markdown(f'<div class="guide-box" style="border-left:10px solid {py["color"]};"><h2>{py["type"]}</h2><p>{py["msg"]}</p></div>', unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            with col_b: st.info(f"🔵 **유기적 매수타점**\n1차({st_res['buy_names'][0]}): {st_res['buy'][0]:,}원\n2차({st_res['buy_names'][1]}): {st_res['buy'][1]:,}원")
            with col_s: st.error(f"🔴 **익절 목표가**\n1차: {st_res['sell'][0]:,}원\n2차: {st_res['sell'][1]:,}원")
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:])])
            fig.add_hline(y=st_res['poc'], line_color="green", annotation_text="POC")
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB")
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 전략 스캐너 (Speed Engine + Dynamic UI)] ---
with tabs[2]:
    if st.button(f"🚀 초고속 유기적 전수조사 (Top 100)"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, has_scan, scan_msg = [], False, "🔍 **V64.4 발굴 종목**\n\n"
        prog_bar = st.progress(0); status_txt = st.empty()

        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'], days=300)): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    curr = res.iloc[-1]; st_res = get_strategy(res)
                    sc = curr['Vol_Zscore'] * 15 + (25 if curr['RSI'] < 35 else 0) + (25 if abs(curr['Close']-curr['POC_Price'])/curr['POC_Price'] < 0.02 else 0)
                    found.append({"name": futs[f], "score": sc, "rsi": curr['RSI'], "regime": curr['Regime'], "strat": st_res, "cp": curr['Close']})
                prog_bar.progress((i + 1) / len(targets)); status_txt.text(f"분석 중: {futs[f]}")

        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        for idx, d in enumerate(found):
            acc_c = "#007bff" if d['regime'] == "🚀 상승" else "#dc3545"
            st.markdown(f"""<div class="scanner-card" style="border-left: 8px solid {acc_c};">
                <h3 style="margin:0;">{d['name']} <small>Score: {d['score']:.1f}</small></h3>
                <p>현재가: <b>{int(d['cp']):,}원</b> | 최우선 타점: <b>{d['strat']['buy_names'][0]}</b></p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div style="background:#f0f7ff; padding:10px; border-radius:10px;">
                        <b>🔵 유기적 매수</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원
                    </div>
                    <div style="background:#fff5f5; padding:10px; border-radius:10px;">
                        <b>🔴 목표가</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            if alert_scanner and idx < 3:
                has_scan = True; scan_msg += f"🔥 **{d['name']}**\n타점: {d['strat']['buy'][0]:,}원({d['strat']['buy_names'][0]})\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    st.header("📈 과거 성과 검증")
    bt_name = st.text_input("검증 종목명", "삼성전자")
    if st.button("📊 시뮬레이션 시작"):
        krx = fdr.StockListing('KRX'); match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                trades = []
                for i in range(120, len(df_bt)-5):
                    strat = get_strategy(df_bt.iloc[:i])
                    if df_bt['Low'].iloc[i] <= strat['buy'][0]:
                        profit = 10.0 if df_bt['High'].iloc[i+1:i+6].max() >= strat['buy'][0] * 1.1 else -7.0
                        trades.append({'date': df_bt.index[i], 'profit': profit})
                if trades:
                    tdf = pd.DataFrame(trades)
                    st.metric("예상 승률", f"{(tdf['profit'] > 0).mean()*100:.1f}%")
                    st.plotly_chart(px.line(tdf, x='date', y=tdf['profit'].cumsum(), title="누적 수익곡선"))

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            match = fdr.StockListing('KRX')[fdr.StockListing('KRX')['Name']==n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
