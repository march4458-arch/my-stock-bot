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
# ⚙️ 1. 시스템 설정 및 라이트 테마 CSS
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Hybrid Full Spec", page_icon="⚡", layout="wide")

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

# --- [시간 관련 함수] ---
def get_market_status():
    now = datetime.datetime.now()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0)
    end = now.replace(hour=15, minute=30, second=0)
    if start <= now <= end: return True, "정규장 운영 중 🚀"
    return False, "장외 시간 🌙"

def is_report_time():
    now = datetime.datetime.now()
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
    except Exception as e: st.error(f"저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

@st.cache_data(ttl=600)
def get_fear_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static/data"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        return r.json()['now']['value'], r.json()['now']['value_text']
    except: return 50, "Neutral"

# ==========================================
# 🧠 2. 고도화된 분석 엔진
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
    # OB 계산
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    # 피보나치
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    # 추세
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
        buy, sell = [adj(cp - atr*1.1), adj(ob), adj(f500)], [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(cp * 1.2)]
    elif regime == "📉 하락":
        buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)], [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절 알림", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절 알림", "msg": f"손절가 {stop_loss:,}원 하회!", "color": "#dc3545", "alert": True}
        elif yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. 추가 매수 권장", "color": "#d63384", "alert": True}
        elif yield_pct > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 추격 비중 확대", "color": "#0d6efd", "alert": True}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 및 자동 갱신 로직 (장 시간 & 마감 리포트)
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Light Final")
    market_on, market_msg = get_market_status()
    st.info(f"**현재 시장: {market_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_portfolio = st.checkbox("보유종목 실시간 알림", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    auto_refresh = st.checkbox("자동 갱신 활성화", value=True)
    refresh_interval = st.slider("정규장 갱신 주기 (분)", 1, 60, 10)

# --- [18시 마감 리포트 로직] ---
if daily_report_on and is_report_time():
    if "report_sent" not in st.session_state or st.session_state.report_sent != datetime.date.today():
        portfolio = get_portfolio_gsheets()
        report_msg = f"📝 <b>오늘의 마감 리포트 ({datetime.date.today()})</b>\n\n💼 <b>보유 종목 현황</b>\n"
        for _, row in portfolio.iterrows():
            df = fetch_stock_smart(row['Code'], days=10)
            if df is not None:
                cp = df.iloc[-1]['Close']
                yield_p = (cp - row['Buy_Price']) / row['Buy_Price'] * 100
                report_msg += f"- {row['Name']}: {yield_p:+.2f}% ({int(cp):,}원)\n"
        send_telegram_msg(tg_token, tg_id, report_msg + "\n내일의 대응 준비를 마치세요! 🌙")
        st.session_state.report_sent = datetime.date.today()

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드 & 실시간 알림] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list, alert_needed, alert_msg = 0.0, 0.0, [], False, "🚨 <b>실시간 시장 감시 보고</b>\n\n"
        with st.spinner('실시간 분석 중...'):
            for _, row in portfolio.iterrows():
                try:
                    df = fetch_stock_smart(row['Code'], days=150)
                    if df is not None:
                        idx = get_hybrid_indicators(df)
                        strat = calculate_organic_strategy(idx, row['Buy_Price'])
                        cp = float(idx.iloc[-1]['Close'])
                        total_buy += float(row['Buy_Price'] * row['Qty'])
                        total_eval += cp * row['Qty']
                        dash_list.append({"종목": row['Name'], "수익": (cp - row['Buy_Price']) * row['Qty'], "평가액": cp * row['Qty']})
                        if alert_portfolio and market_on and strat['pyramiding']['alert']:
                            alert_needed = True
                            alert_msg += f"<b>[{strat['pyramiding']['type']}]</b> {row['Name']}\n- 현재가: {int(cp):,}원\n- 안내: {strat['pyramiding']['msg']}\n\n"
                except: continue
        if alert_needed: send_telegram_msg(tg_token, tg_id, alert_msg)
        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            yield_total = ((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0)
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{yield_total:+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            st.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', template="plotly_white", title="종목별 평가손익"), use_container_width=True)
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
            c1.metric("국면", strat['regime']); c2.metric("RSI", f"{strat['rsi']:.1f}"); c3.metric("세력지지(OB)", f"{int(strat['ob']):,}원"); c4.error(f"손절가: {strat['stop']:,}원")
            st.markdown(f'<div class="guide-box" style="border-left:8px solid {strat["pyramiding"]["color"]};"><h3 style="color:{strat["pyramiding"]["color"]};">{strat["pyramiding"]["type"]} 가이드</h3><p>{strat["pyramiding"]["msg"]}</p></div>', unsafe_allow_html=True)
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n2차: {strat['buy'][1]:,}원\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n2차: {strat['sell'][1]:,}원\n3차: {strat['sell'][2]:,}원")
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'])])
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 AI 시장 전수 조사 시작"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found, sc_alert_msg = [], "🔍 <b>고득점 발굴 종목</b>\n\n"
        with ThreadPoolExecutor(max_workers=8) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futures):
                name, df_scan = futures[f], f.result()
                if df_scan is not None:
                    strat_tmp = calculate_organic_strategy(df_scan)
                    score = calculate_advanced_score(df_scan, strat_tmp)
                    if df_scan.iloc[-1]['RSI'] < 55:
                        found.append({"name": name, "cp": df_scan.iloc[-1]['Close'], "strat": strat_tmp, "score": score})
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            if alert_scanner and idx < 3 and market_on: sc_alert_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n- 1차매수: {d['strat']['buy'][0]:,}원\n\n"
            st.markdown(f"""<div class="scanner-card"><div style="display:flex; justify-content:space-between;"><h2>{icon} {d['name']}</h2><span style="color:#007bff; font-weight:bold; font-size:1.3em;">{d['score']:.1f}점</span></div><hr><div style="display:grid; grid-template-columns: 1fr 1fr; gap:25px;"><div class="inner-box" style="border-top:4px solid #007bff;"><b>🔵 3분할 매수</b><br>1차: <b style="float:right;">{d['strat']['buy'][0]:,}원</b><br>2차: <b style="float:right;">{d['strat']['buy'][1]:,}원</b><br>3차: <b style="float:right;">{d['strat']['buy'][2]:,}원</b></div><div class="inner-box" style="border-top:4px solid #dc3545;"><b>🔴 3분할 매도</b><br>1차: <b style="float:right;">{d['strat']['sell'][0]:,}원</b><br>2차: <b style="float:right;">{d['strat']['sell'][1]:,}원</b><br>3차: <b style="float:right;">{d['strat']['sell'][2]:,}원</b></div></div></div>""", unsafe_allow_html=True)
        if alert_scanner and found and market_on: send_telegram_msg(tg_token, tg_id, sc_alert_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    st.header("📈 전략 백테스트")
    bt_name = st.text_input("종목명 입력", "에코프로비엠")
    c1, c2 = st.columns(2)
    tp_p, sl_p = c1.slider("익절 목표 %", 3.0, 30.0, 10.0), c2.slider("손절 제한 %", 3.0, 30.0, 7.0)
    if st.button("📊 분석 실행"):
        krx = get_krx_list()
        match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                trades, in_pos, entry_p = [], False, 0
                for i in range(150, len(df_bt)):
                    sub, today = df_bt.iloc[:i], df_bt.iloc[i]
                    strat = calculate_organic_strategy(sub)
                    if not in_pos:
                        if today['Low'] <= strat['buy'][0]: entry_p, in_pos = strat['buy'][0], True
                    else:
                        if today['High'] >= entry_p * (1+tp_p/100): trades.append({'profit': tp_p, 'type': '익절'}); in_pos = False
                        elif today['Low'] <= entry_p * (1-sl_p/100): trades.append({'profit': -sl_p, 'type': '손절'}); in_pos = False
                if trades:
                    tdf = pd.DataFrame(trades)
                    st.metric("승률", f"{(tdf['type']=='익절').sum()/len(tdf)*100:.1f}%")
                    st.plotly_chart(px.line(tdf['profit'].cumsum(), title="누적 수익", template="plotly_white"), use_container_width=True)
        else: st.error("종목명을 확인하세요.")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_gs"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가", 0), c3.number_input("수량", 0)
        if st.form_submit_button("저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                save_portfolio_gsheets(pd.concat([df_p, pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=df_p.columns)]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

# ==========================================
# ⏳ 4. 지능형 자동 갱신
# ==========================================
if auto_refresh:
    now = datetime.datetime.now()
    if market_on or now.hour == 18: time.sleep(refresh_interval * 60)
    else: time.sleep(3600)
    st.rerun()
