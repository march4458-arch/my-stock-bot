import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 구글 시트 연동
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Hybrid Pro Final", page_icon="⚡", layout="wide")

# 가독성을 위한 전역 CSS 추가
st.markdown("""
    <style>
    .reportview-container .main .block-container { color: white; }
    div[data-testid="stMetricValue"] { color: #4FACFE !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; }
    </style>
    """, unsafe_allow_html=True)

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            cols = ['Code', 'Name', 'Buy_Price', 'Qty']
            for col in cols:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트 동기화 완료!")
    except: st.error("저장 실패")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except: pass

@st.cache_data(ttl=3600)
def get_krx_list(): return fdr.StockListing('KRX')

@st.cache_data(ttl=600)
def get_fear_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static/data"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            return data['now']['value'], data['now']['value_text']
        return 50, "Neutral"
    except: return 50, "Neutral"

# ==========================================
# 🧠 2. 분석 엔진
# ==========================================
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
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
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    avg_vol = df['Volume'].rolling(20).mean()
    std_vol = df['Volume'].rolling(20).std()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (std_vol + 1e-9)
    
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
    f500, f618 = curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy = [adj(cp - atr*1.1), adj(ob), adj(f500)]
        sell = [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(df.tail(252)['High'].max() * 1.1)]
    elif regime == "📉 하락":
        buy = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)]
        sell = [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy = [adj(f500), adj(ob), adj(f618)]
        sell = [adj(curr['Fibo_382']), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]

    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#777", "alert": False}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. 추가 매수 권장", "color": "#FF4B4B", "alert": True}
        elif yield_pct > 7 and regime == "🚀 상승":
            pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 비중 확대 가능", "color": "#4FACFE", "alert": True}

    return {"buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93), "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 구성
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Turbo Final")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("CNN Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_on = st.checkbox("물타기/불타기 텔레그램 알림")
    auto_refresh = st.checkbox("자동 갱신 활성화")
    refresh_interval = st.slider("주기 (분)", 1, 60, 5)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0.0, 0.0, []
        alert_msg = "🚨 <b>보유종목 실시간 대응 알림</b>\n\n"
        alert_needed = False
        
        with st.spinner('실시간 자산 동기화 중...'):
            for _, row in portfolio.iterrows():
                try:
                    df = fetch_stock_smart(row['Code'], days=10)
                    if df is not None and not df.empty:
                        cp = float(df.iloc[-1]['Close'])
                        b_price = float(row['Buy_Price'])
                        qty = float(row['Qty'])
                        total_buy += b_price * qty
                        total_eval += cp * qty
                        dash_list.append({"종목": str(row['Name']), "수익": (cp - b_price) * qty, "평가액": cp * qty})
                        if alert_on:
                            full_df = fetch_stock_smart(row['Code'], days=150)
                            df_idx = get_hybrid_indicators(full_df)
                            if df_idx is not None:
                                strat = calculate_organic_strategy(df_idx, buy_price=b_price)
                                if strat['pyramiding']['alert']:
                                    alert_needed = True
                                    alert_msg += f"📌 <b>{row['Name']}</b>\n- {strat['pyramiding']['type']}: {strat['pyramiding']['msg']}\n\n"
                except: continue

        if alert_on and alert_needed:
            send_telegram_msg(tg_token, tg_id, alert_msg)
            st.toast("대응 타점 알림 발송 완료!")

        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            yield_pct = ((total_eval - total_buy) / total_buy * 100 if total_buy > 0 else 0)
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{yield_pct:+.2f}%")
            c3.metric("평가손익", f"{int(total_eval - total_buy):,}원")
            st.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', template="plotly_dark"), use_container_width=True)
            st.plotly_chart(px.pie(df_dash, values='평가액', names='종목', hole=0.3, template="plotly_dark"), use_container_width=True)

# --- [💼 탭 1: AI 리포트 (가독성 수정)] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("진단 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime'])
            c2.metric("RSI", f"{strat['rsi']:.1f}")
            c3.metric("세력지지(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"손절가: {strat['stop']:,}원")
            
            # 가이드 박스 가독성 수정 (배경색과 대비되는 흰색 글씨 강제)
            py = strat['pyramiding']
            st.markdown(f"""
                <div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border-left:8px solid {py['color']}; margin-bottom:20px;">
                    <h3 style="margin:0; color:{py['color']};">{py['type']} 가이드</h3>
                    <p style="margin:5px 0; color:#FFFFFF; font-size:1.1em;">{py['msg']}</p>
                </div>
                """, unsafe_allow_html=True)

            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원\n\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원\n\n3차: {strat['sell'][2]:,}원")
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'])])
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너 (이미지 UI 재현)] ---
with tabs[2]:
    st.header("🔍 확률 기반 타점 스캐너")
    if st.button("🚀 AI 전수 조사 시작"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=8) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futures)):
                name = futures[f]; df_scan = f.result()
                if df_scan is not None and df_scan.iloc[-1]['RSI'] < 55:
                    s_scan = calculate_organic_strategy(df_scan)
                    score = calculate_advanced_score(df_scan, s_scan)
                    found.append({"name": name, "cp": df_scan.iloc[-1]['Close'], "strat": s_scan, "score": score})
                progress.progress((i + 1) / len(targets))
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""
            <div style="background-color:#1E1E1E; padding:25px; border-radius:15px; margin-bottom:25px; border-left:8px solid #4FACFE; border-top:1px solid #333;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="margin:0; font-size:1.8em; color:white;">{icon} {d['name']}</h2>
                    <span style="color:#FFD700; font-weight:bold; font-size:1.3em;">신뢰 점수: {d['score']:.1f}점</span>
                </div>
                <hr style="border:0.1px solid #444; margin:15px 0;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:25px;">
                    <div style="background:#121212; padding:20px; border-radius:12px; border-top:4px solid #4FACFE;">
                        <h4 style="margin:0 0 15px 0; color:#4FACFE;">🔵 3분할 매수</h4>
                        <div style="font-family: monospace; line-height:2.2; color:#FFFFFF;">
                            1차 : <b style="float:right;">{d['strat']['buy'][0]:,}원</b><br>
                            2차 : <b style="float:right;">{d['strat']['buy'][1]:,}원</b><br>
                            3차 : <b style="float:right;">{d['strat']['buy'][2]:,}원</b>
                        </div>
                    </div>
                    <div style="background:#121212; padding:20px; border-radius:12px; border-top:4px solid #FF4B4B;">
                        <h4 style="margin:0 0 15px 0; color:#FF4B4B;">🔴 3분할 매도</h4>
                        <div style="font-family: monospace; line-height:2.2; color:#FFFFFF;">
                            1차 : <b style="float:right;">{d['strat']['sell'][0]:,}원</b><br>
                            2차 : <b style="float:right;">{d['strat']['sell'][1]:,}원</b><br>
                            3차 : <b style="float:right;">{d['strat']['sell'][2]:,}원</b>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    st.subheader("📌 구글 시트 관리")
    df_p = get_portfolio_gsheets()
    with st.form("add_gsheet"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명"); p = c2.number_input("평단가", 0); q = c3.number_input("수량", 0)
        if st.form_submit_button("추가 및 저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new_row]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_interval * 60); st.rerun()
