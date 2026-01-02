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

st.set_page_config(page_title="주식 비서 V62.7 Full Alarm", page_icon="⚡", layout="wide")

# 라이트 테마 CSS 유지
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; }
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [텔레그램 발송 함수] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# --- [데이터 연동 및 AttributeError 방어] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            for col in ['Code', 'Name', 'Buy_Price', 'Qty']:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 분석 엔진 (시총 필터 포함)
# ==========================================
@st.cache_data(ttl=3600)
def get_krx_filtered():
    """시가총액 5000억 원 이상 종목만 필터링"""
    df = fdr.StockListing('KRX')
    # Marcap 단위: 원 (5000억 = 500,000,000,000)
    return df[df['Marcap'] >= 500000000000]

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
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_382'], df['Fibo_500'], df['Fibo_618'] = hi_1y-(rng*0.382), hi_1y-(rng*0.5), hi_1y-(rng*0.618)
    
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
    if regime == "🚀 상승": buy, sell = [adj(cp-atr*1.1), adj(ob), adj(f500)], [adj(cp+atr*2.5), adj(cp+atr*4.5), adj(cp*1.2)]
    elif regime == "📉 하락": buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min()-atr)], [adj(f382), adj(f500), adj(ob)]
    else: buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max()+atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": f"손절가 {stop_loss:,}원 하회!", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {y:.1f}%. 추가 매입", "color": "#d63384", "alert": True}
        elif y > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {y:.1f}%. 비중 확대", "color": "#0d6efd", "alert": True}
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding, "fibo": [f382, f500, f618]}

# ==========================================
# 🖥️ 3. 메인 UI 및 알림 로직
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid 500B Spec")
    now_kst = get_now_kst()
    m_on, m_msg = (True, "정규장 운영 중 🚀") if now_kst.weekday() < 5 and 900 <= now_kst.hour*100+now_kst.minute <= 1530 else (False, "장외 시간 🌙")
    st.info(f"**KST: {now_kst.strftime('%H:%M')} | {m_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_portfolio = st.checkbox("보유종목 실시간 알림", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    auto_refresh = st.checkbox("자동 갱신", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

# --- [🔔 알림 로직: 18시 마감 리포트] ---
if daily_report_on and now_kst.hour == 18 and 0 <= now_kst.minute <= 10:
    today_str = now_kst.strftime('%Y-%m-%d')
    if "last_report_date" not in st.session_state or st.session_state.last_report_date != today_str:
        portfolio = get_portfolio_gsheets()
        if not portfolio.empty:
            report_msg = f"📝 <b>마감 리포트 ({today_str})</b>\n\n"
            for _, r in portfolio.iterrows():
                df_r = fetch_stock_smart(r['Code'], days=10)
                if df_r is not None:
                    cp_r = df_r['Close'].iloc[-1]
                    y_r = (cp_r - r['Buy_Price']) / r['Buy_Price'] * 100
                    report_msg += f"- {r['Name']}: {y_r:+.2f}% ({int(cp_r):,}원)\n"
            send_telegram_msg(tg_token, tg_id, report_msg + "\n오늘도 수고하셨습니다! 🌙")
            st.session_state.last_report_date = today_str

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_port_alert = 0.0, 0.0, [], "🚨 <b>보유종목 실시간 감시</b>\n\n", False
        with st.spinner('포트폴리오 분석 중...'):
            for _, row in portfolio.iterrows():
                raw_df = fetch_stock_smart(row['Code'], days=150)
                idx_df = get_hybrid_indicators(raw_df)
                if idx_df is not None:
                    st_res = calculate_organic_strategy(idx_df, row['Buy_Price'])
                    cp = float(idx_df['Close'].iloc[-1])
                    t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                    dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "평가액": cp*row['Qty']})
                    if alert_portfolio and m_on and st_res['pyramiding']['alert']:
                        has_port_alert = True
                        port_alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n{st_res['pyramiding']['msg']}\n현재가: {int(cp):,}원\n\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원"); c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%"); c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='수익', template="plotly_white"), use_container_width=True)
        if has_port_alert: send_telegram_msg(tg_token, tg_id, port_alert_msg)
    else: st.info("관리 탭에서 종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("리포트 종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = calculate_organic_strategy(df_ai, row['Buy_Price'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("국면", st_res['regime']); m2.metric("RSI", f"{st_res['rsi']:.1f}"); m3.metric("평단가", f"{int(row['Buy_Price']):,}원"); m4.error(f"손절가: {st_res['stop']:,}원")
            st.markdown(f"""<div class="guide-box" style="border-left:8px solid {st_res['pyramiding']['color']};"><h3>{st_res['pyramiding']['type']}</h3><p>{st_res['pyramiding']['msg']}</p></div>""", unsafe_allow_html=True)
            st.info(f"🔵 매수: {st_res['buy']} | 🔴 매도: {st_res['sell']}")
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:])])
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB Line")
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너 (시총 5000억 이상 필터 적용)] ---
with tabs[2]:
    if st.button("🚀 우량주 전수 조사 (시총 5000억↑)"):
        all_stocks = get_krx_filtered()
        # 시총 순 정렬 후 상위 100개 집중 스캔 (속도 최적화)
        targets = all_stocks.sort_values(by='Marcap', ascending=False).head(100)
        found, scan_alert_msg, has_scan_alert = [], "🔍 <b>우량주 발굴 알림</b>\n\n", False
        
        with st.spinner(f'시총 5000억 이상 {len(targets)}개 종목 분석 중...'):
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futs):
                    res = f.result()
                    if res is not None:
                        # 스코어링: 낮은 RSI(과매도) + 높은 거래량 점수
                        sc = (70 - res['RSI'].iloc[-1]) * 0.5 + (res['Vol_Zscore'].iloc[-1] * 5)
                        if res['Regime'].iloc[-1] != "📉 하락": # 하락 국면 제외
                            found.append({"name": futs[f], "score": sc, "strat": calculate_organic_strategy(res)})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div class="scanner-card"><h3>{icon} {d['name']} ({d['score']:.1f}점)</h3>
                <p>매수타점: {d['strat']['buy'][0]:,}원 | 목표가: {d['strat']['sell'][0]:,}원</p></div>""", unsafe_allow_html=True)
            if alert_scanner and m_on and idx < 3:
                has_scan_alert = True
                scan_alert_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n- 신호: {d['strat']['regime']}\n- 매수: {d['strat']['buy'][0]:,}원\n\n"
        if has_scan_alert: send_telegram_msg(tg_token, tg_id, scan_alert_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    bt_name = st.text_input("종목명", "삼성전자")
    if st.button("백테스트 실행"):
        krx = fdr.StockListing('KRX')
        match = krx[krx['Name']==bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                trades, in_pos = [], False
                for i in range(150, len(df_bt)):
                    curr_bt = df_bt.iloc[i]
                    s_bt = calculate_organic_strategy(df_bt.iloc[:i])
                    if not in_pos and curr_bt['Low'] <= s_bt['buy'][0]:
                        entry_bt, in_pos = s_bt['buy'][0], True
                    elif in_pos:
                        if curr_bt['High'] >= entry_bt * 1.1: trades.append(10); in_pos = False
                        elif curr_bt['Low'] <= entry_bt * 0.93: trades.append(-7); in_pos = False
                if trades:
                    st.metric("승률", f"{sum(1 for t in trades if t>0)/len(trades)*100:.1f}%")
                    st.line_chart(np.cumsum(trades))

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_p"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("저장"):
            krx_all = fdr.StockListing('KRX')
            match_p = krx_all[krx_all['Name']==n]
            if not match_p.empty:
                new_p = pd.DataFrame([[match_p.iloc[0]['Code'], n, p, q]], columns=df_p.columns)
                conn_p = st.connection("gsheets", type=GSheetsConnection)
                conn_p.update(data=pd.concat([df_p, new_p]))
                st.rerun()
    st.dataframe(df_p)

if auto_refresh: time.sleep(interval * 60); st.rerun()
