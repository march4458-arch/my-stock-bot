import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, time, requests
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

st.set_page_config(page_title="주식 비서 V64.7 Snow Master", page_icon="❄️", layout="wide")

# UI 디자인 CSS (기존 스타일 복구)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; background-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티 함수] ---
@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except: pass
    fallback = [['005930', '삼성전자'], ['000660', 'SK하이닉스'], ['005380', '현대차']]
    return pd.DataFrame(fallback, columns=['Code', 'Name']).assign(Marcap=10**14)

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
            df = df.rename(columns=rename_map); df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except: pass
    return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (Snow 파동 통합)
# ==========================================
def calc_stoch(df, n, m, t):
    low_min, high_max = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    k = ((df['Close'] - low_min) / (high_max - low_min + 1e-9)) * 100
    return k.rolling(m).mean().rolling(t).mean()

def get_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy(); close = df['Close']
    df['MA20'], df['MA120'] = close.rolling(20).mean(), close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    # Snow Waves
    df['SNOW_S'], df['SNOW_M'], df['SNOW_L'] = calc_stoch(df, 5, 3, 3), calc_stoch(df, 10, 6, 6), calc_stoch(df, 20, 12, 12)
    # Fibo & POC
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    # RSI & BB
    delta = close.diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (g / (l + 1e-9)))); df['BB_L'] = df['MA20'] - (close.rolling(20).std() * 2)
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp, atr = curr['Close'], curr['ATR']
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)
    
    # 유기적 3분할 타점
    buy_pts = sorted([adj(curr['POC']), adj(curr['Fibo_618']), adj(curr['BB_L'])], reverse=True)
    sell_pts = [adj(cp + atr*2), adj(cp + atr*3.5), adj(cp + atr*5)]
    
    # Snow Score
    score = (25 if curr['SNOW_L'] < 25 else 0) + (15 if curr['SNOW_M'] < 25 else 0) + (20 if curr['RSI'] < 35 else 0)
    if abs(cp - curr['POC'])/cp < 0.02: score += 20
    
    status = {"type": "💤 관망", "color": "#6c757d", "msg": "타점 대기", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0]: status = {"type": "💰 익절", "color": "#28a745", "msg": f"{y:.1f}% 수익", "alert": True}
        elif cp <= buy_pts[2] * 0.93: status = {"type": "⚠️ 손절", "color": "#dc3545", "msg": "리스크 관리", "alert": True}
        elif score >= 50: status = {"type": "❄️ 스노우", "color": "#00d2ff", "msg": "강력 추매 구간", "alert": True}
    return {"buy": buy_pts, "sell": sell_pts, "score": score, "status": status, "poc": curr['POC'], "fibo": curr['Fibo_618']}

# ==========================================
# 🖥️ 3. 사이드바 (모든 설정 복구)
# ==========================================
with st.sidebar:
    st.title("🛡️ Snow Master V64.7")
    st.info(f"**KST: {get_now_kst().strftime('%H:%M')}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    st.divider()
    min_marcap = st.number_input("최소 시총 (억)", value=5000) * 100000000
    alert_portfolio = st.checkbox("보유종목 실시간 감시", value=True)
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

# ==========================================
# 🖥️ 4. 메인 탭 구현 (모든 탭 복구)
# ==========================================
tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스노우 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, alert_msg = 0, 0, [], ""
        for _, row in portfolio.iterrows():
            df_raw = fdr.DataReader(row['Code'], (get_now_kst()-timedelta(days=200)).strftime('%Y-%m-%d'))
            idx_df = get_indicators(df_raw)
            if idx_df is not None:
                res = get_strategy(idx_df, row['Buy_Price'])
                cp = idx_df['Close'].iloc[-1]
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
                if alert_portfolio and res['status']['alert']:
                    alert_msg += f"[{res['status']['type']}] {row['Name']}: {res['status']['msg']}\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)
        if alert_msg: send_telegram_msg(tg_token, tg_id, f"❄️ <b>실시간 포트폴리오 신호</b>\n\n{alert_msg}")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    if not portfolio.empty:
        sel = st.selectbox("진단 종목", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_indicators(fdr.DataReader(row['Code'], (get_now_kst()-timedelta(days=300)).strftime('%Y-%m-%d')))
        if df_ai is not None:
            res = get_strategy(df_ai, row['Buy_Price'])
            st.markdown(f'<div class="metric-card" style="border-left:10px solid {res["status"]["color"]};"><h2>{res["status"]["type"]} <small>(Score: {res["score"]})</small></h2><p>{res["status"]["msg"]}</p></div>', unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            with col_b: st.markdown(f'<div class="buy-box"><b>🔵 유기적 3분할 매수</b><br>1차: {res["buy"][0]:,}원<br>2차: {res["buy"][1]:,}원<br>3차: {res["buy"][2]:,}원</div>', unsafe_allow_html=True)
            with col_s: st.markdown(f'<div class="sell-box"><b>🔴 유기적 3분할 매도</b><br>1차: {res["sell"][0]:,}원<br>2차: {res["sell"][1]:,}원<br>3차: {res["sell"][2]:,}원</div>', unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:], name="주가")])
            fig.add_hline(y=res['poc'], line_color="orange", annotation_text="POC")
            fig.add_hline(y=res['fibo'], line_color="green", line_dash="dot", annotation_text="Fibo 618")
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스노우 스캐너] ---
with tabs[2]:
    if st.button("🚀 스노우 파동 유기적 전수조사 (상위 100)"):
        krx = get_safe_stock_listing()
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(get_indicators, fdr.DataReader(r['Code'], (get_now_kst()-timedelta(days=300)).strftime('%Y-%m-%d'))): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    s_res = get_strategy(res)
                    found.append({"name": futs[f], "score": s_res['score'], "strat": s_res})
                prog.progress((i + 1) / len(targets))
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:15]:
            st.markdown(f"""<div class="scanner-card" style="border-left:8px solid #00d2ff;">
                <h3>{d['name']} <span style="font-size:0.6em; color:#007bff;">Snow Score: {d['score']}</span></h3>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                    <div class="buy-box"><b>매수: {d['strat']['buy'][0]:,}원</b></div>
                    <div class="sell-box"><b>목표: {d['strat']['sell'][0]:,}원</b></div>
                </div></div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    bt_name = st.text_input("검증 종목", "삼성전자")
    if st.button("📊 Snow 전략 백테스트"):
        krx = get_safe_stock_listing(); match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_indicators(fdr.DataReader(match.iloc[0]['Code'], (get_now_kst()-timedelta(days=730)).strftime('%Y-%m-%d')))
            if df_bt is not None:
                cash, stocks, equity = 10000000, 0, []
                for i in range(120, len(df_bt)):
                    curr = df_bt.iloc[i]; strat = get_strategy(df_bt.iloc[:i]); cp = curr['Close']
                    if stocks == 0 and strat['score'] >= 50 and curr['Low'] <= strat['buy'][0]:
                        stocks = cash // cp; cash -= (stocks * cp)
                    elif stocks > 0 and curr['High'] >= strat['sell'][0]:
                        cash += (stocks * cp); stocks = 0
                    equity.append(cash + (stocks * cp))
                st.plotly_chart(px.line(pd.DataFrame(equity, columns=['total']), y='total', title=f"{bt_name} Snow 전략 자산곡선"))

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            krx = get_safe_stock_listing(); m = krx[krx['Name']==n]
            if not m.empty:
                new = pd.DataFrame([[m.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh:
    time.sleep(interval * 60)
    st.rerun()
