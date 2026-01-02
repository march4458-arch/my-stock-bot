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
# ⚙️ 1. 시스템 설정 및 라이트 테마 CSS 적용
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Hybrid Light Pro", page_icon="⚡", layout="wide")

# 라이트 모드 전용 가독성 강화 CSS
st.markdown("""
    <style>
    /* 전체 배경 및 기본 텍스트 색상 */
    .stApp { background-color: #f8f9fa; color: #333333; }
    
    /* 메트릭 스타일 보정 */
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #666666 !important; }
    
    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #ddd; }
    
    /* 카드 및 가이드 박스 가독성 (이미지 에러 해결) */
    .guide-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; color: #000000 !important; }
    .guide-box h3, .guide-box p { color: #000000 !important; font-weight: 500; }
    
    /* 스캐너 카드 스타일 */
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; }
    .inner-box b { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 구글 시트 데이터 로드 ---
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

# --- 유틸리티 함수 ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except: pass

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
# 🧠 2. 분석 엔진 (이전 로직 유지)
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
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. 추가 매수 권장", "color": "#d63384", "alert": True}
        elif yield_pct > 7 and regime == "🚀 상승":
            pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 비중 확대 가능", "color": "#0d6efd", "alert": True}
    return {"buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93), "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 구성 (라이트 모드 테마)
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Light Pro")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_on = st.checkbox("물타기/불타기 알림")
    auto_refresh = st.checkbox("자동 갱신 활성화")
    refresh_interval = st.slider("주기 (분)", 1, 60, 5)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0.0, 0.0, []
        alert_msg = "🚨 <b>대응 알림</b>\n\n"
        alert_needed = False
        with st.spinner('실시간 동기화 중...'):
            for _, row in portfolio.iterrows():
                try:
                    df = fetch_stock_smart(row['Code'], days=10)
                    if df is not None and not df.empty:
                        cp = float(df.iloc[-1]['Close'])
                        total_buy += float(row['Buy_Price'] * row['Qty'])
                        total_eval += float(cp * row['Qty'])
                        dash_list.append({"종목": row['Name'], "수익": (cp - row['Buy_Price']) * row['Qty'], "평가액": cp * row['Qty']})
                        if alert_on:
                            full_df = fetch_stock_smart(row['Code'], days=150)
                            df_idx = get_hybrid_indicators(full_df)
                            if df_idx is not None:
                                strat = calculate_organic_strategy(df_idx, buy_price=row['Buy_Price'])
                                if strat['pyramiding']['alert']:
                                    alert_needed = True
                                    alert_msg += f"📌 <b>{row['Name']}</b>\n- {strat['pyramiding']['type']}: {strat['pyramiding']['msg']}\n\n"
                except: continue
        if alert_on and alert_needed: send_telegram_msg(tg_token, tg_id, alert_msg)
        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            yield_p = ((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0)
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{yield_p:+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            st.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', template="plotly_white"), use_container_width=True)
    else: st.info("종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime'])
            c2.metric("RSI", f"{strat['rsi']:.1f}")
            c3.metric("세력지지(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"손절가: {strat['stop']:,}원")
            py = strat['pyramiding']
            st.markdown(f"""
                <div class="guide-box" style="border-left:8px solid {py['color']}; background-color: #f8f9fa; border: 1px solid #ddd;">
                    <h3>{py['type']} 가이드</h3>
                    <p>{py['msg']}</p>
                </div>
                """, unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원\n\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원\n\n3차: {strat['sell'][2]:,}원")
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'])])
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 AI 분석 시작"):
        stocks = fdr.StockListing('KRX')
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with ThreadPoolExecutor(max_workers=8) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futures):
                name = futures[f]; df_scan = f.result()
                if df_scan is not None and df_scan.iloc[-1]['RSI'] < 55:
                    s_scan = calculate_organic_strategy(df_scan)
                    score = calculate_advanced_score(df_scan, s_scan)
                    found.append({"name": name, "cp": df_scan.iloc[-1]['Close'], "strat": s_scan, "score": score})
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""
            <div class="scanner-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                    <h2 style="margin:0; color:#212529;">{icon} {d['name']}</h2>
                    <span style="color:#007bff; font-weight:bold; font-size:1.3em;">신뢰 점수: {d['score']:.1f}점</span>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:25px;">
                    <div class="inner-box" style="border-top:4px solid #007bff;">
                        <h4 style="margin:0 0 15px 0; color:#007bff;">🔵 3분할 매수</h4>
                        <div style="line-height:2.2;">1차 : <b style="float:right;">{d['strat']['buy'][0]:,}원</b><br>2차 : <b style="float:right;">{d['strat']['buy'][1]:,}원</b><br>3차 : <b style="float:right;">{d['strat']['buy'][2]:,}원</b></div>
                    </div>
                    <div class="inner-box" style="border-top:4px solid #dc3545;">
                        <h4 style="margin:0 0 15px 0; color:#dc3545;">🔴 3분할 매도</h4>
                        <div style="line-height:2.2;">1차 : <b style="float:right;">{d['strat']['sell'][0]:,}원</b><br>2차 : <b style="float:right;">{d['strat']['sell'][1]:,}원</b><br>3차 : <b style="float:right;">{d['strat']['sell'][2]:,}원</b></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_gs"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명"); p = c2.number_input("평단가", 0); q = c3.number_input("수량", 0)
        if st.form_submit_button("저장"):
            match = fdr.StockListing('KRX')[fdr.StockListing('KRX')['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new_row]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_interval * 60); st.rerun()
