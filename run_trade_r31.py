import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests, random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 구글 시트 연동
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Hybrid Cloud", page_icon="⚡", layout="wide")

# 구글 시트 연결 및 데이터 로드 함수
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            # 데이터 타입 강제 변환 (계산 오류 방지)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
        st.error(f"시트 로드 실패: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트에 성공적으로 동기화되었습니다!")
    except Exception as e:
        st.error(f"구글 시트 저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except: pass

@st.cache_data(ttl=3600)
def get_krx_list(): 
    return fdr.StockListing('KRX')

@st.cache_data(ttl=600)
def get_fear_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static/data"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        return r.json()['now']['value'], r.json()['now']['value_text']
    except: return 50, "Neutral"

# ==========================================
# 🧠 2. 고도화된 분석 엔진 (피보나치 + OB + 국면분석)
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
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # RSI 계산
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    # OB(Order Block) 분석
    ob_zones = []
    avg_vol = df['Volume'].rolling(20).mean()
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    # 피보나치 되돌림 (1년 기준)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    # 추세 국면 판단
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
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

    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy = [adj(cp - atr*1.1), adj(ob), adj(f500)]
        sell = [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(df.tail(252)['High'].max() * 1.1)]
    elif regime == "📉 하락":
        buy = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)]
        sell = [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy = [adj(f500), adj(ob), adj(f618)]
        sell = [adj(f382), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]

    pyramiding = {"type": "💤 관망", "msg": "현재 대응 구간 대기 중", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. {min(buy):,}원 부근 비중 확대 권장", "color": "#FF4B4B"}
        elif yield_pct > 7 and regime == "🚀 상승":
            pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 상단 돌파 시 추격 가능", "color": "#4FACFE"}

    return {"buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93), "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 레이아웃 (Turbo UI 유지 + GSheets 연동)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Final V62.1")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("CNN Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    st.subheader("🔔 알림 및 갱신")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    auto_refresh = st.checkbox("자동 갱신 활성화")
    refresh_interval = st.slider("주기 (분)", 1, 60, 5)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0, 0, []
        with st.spinner('구글 시트 동기화 중...'):
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=10)
                if df is not None and not df.empty:
                    cp = float(df.iloc[-1]['Close'])
                    b_total = float(row['Buy_Price']) * float(row['Qty'])
                    e_total = cp * float(row['Qty'])
                    total_buy += b_total; total_eval += e_total
                    dash_list.append({"종목": row['Name'], "수익": e_total - b_total, "평가액": e_total})
        
        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0):+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', title="종목별 손익", template="plotly_dark"), use_container_width=True)
            col2.plotly_chart(px.pie(df_dash, values='평가액', names='종목', hole=0.3, title="자산 비중", template="plotly_dark"), use_container_width=True)
    else: st.info("관리 탭에서 구글 시트에 종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("진단 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime']); c2.metric("RSI", f"{strat['rsi']:.1f}"); c3.metric("OB(지지)", f"{int(strat['ob']):,}원"); c4.error(f"손절: {strat['stop']:,}원")
            
            py = strat['pyramiding']
            st.markdown(f'<div style="background:#1E1E1E; padding:20px; border-radius:10px; border-left:8px solid {py["color"]};"><h3>{py["type"]} 가이드</h3><p>{py["msg"]}</p></div>', unsafe_allow_html=True)

            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원\n\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원\n\n3차: {strat['sell'][2]:,}원")
            
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'])])
            fig.add_hline(y=strat['ob'], line_color="yellow", annotation_text="OB")
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 전수 조사 시작"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with st.spinner("AI 분석 중..."):
            with ThreadPoolExecutor(max_workers=10) as exec:
                futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futures):
                    name = futures[f]; df_s = f.result()
                    if df_s is not None and df_s.iloc[-1]['RSI'] < 50:
                        s = calculate_organic_strategy(df_s)
                        upside = ((s['sell'][0] - df_s.iloc[-1]['Close']) / df_s.iloc[-1]['Close']) * 100
                        score = (100 - s['rsi']) + (upside * 1.5)
                        found.append({"name": name, "cp": df_s.iloc[-1]['Close'], "strat": s, "score": score})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid #4FACFE; margin-bottom:15px;">
                <h3 style="margin:0;">{icon} {d['name']} <small>(신뢰점수: {d['score']:.1f})</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:10px;">
                    <div style="background:#111; padding:10px; border-radius:5px;"><b>🔵 매수 타점</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원</div>
                    <div style="background:#111; padding:10px; border-radius:5px;"><b>🔴 매도 목표</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원</div>
                </div></div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    t_name = st.text_input("백테스트 종목", "삼성전자")
    c1, c2 = st.columns(2)
    tp, sl = c1.slider("익절 %", 3, 20, 7), c2.slider("손절 %", 3, 20, 8)
    if st.button("시뮬레이션 가동"):
        match = get_krx_list()[get_krx_list()['Name'] == t_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code']))
            if df_bt is not None:
                trades, in_pos = [], False
                for i in range(150, len(df_bt)-1):
                    strat = calculate_organic_strategy(df_bt.iloc[:i])
                    low, high = df_bt['Low'].iloc[i], df_bt['High'].iloc[i]
                    if not in_pos and low <= strat['buy'][0]:
                        entry_p = strat['buy'][0]; exit_tp, exit_sl = entry_p*(1+tp/100), entry_p*(1-sl/100); in_pos = True
                    elif in_pos:
                        if high >= exit_tp: trades.append({"ret": tp, "res": "익절"}); in_pos = False
                        elif low <= exit_sl: trades.append({"ret": -sl, "res": "손절"}); in_pos = False
                if trades:
                    tdf = pd.DataFrame(trades)
                    st.metric("승률", f"{(tdf['res']=='익절').sum()/len(tdf)*100:.1f}%")
                    st.plotly_chart(px.line(tdf, y=tdf['ret'].cumsum(), title="수익 곡선", template="plotly_dark"), use_container_width=True)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    st.subheader("📌 구글 시트 포트폴리오 관리")
    df_p = get_portfolio_gsheets()
    with st.form("add_form"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명"); p = c2.number_input("평단가", 0); q = c3.number_input("수량", 0)
        if st.form_submit_button("시트 추가 및 저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)
    if st.button("시트 전체 초기화"):
        save_portfolio_gsheets(pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty']))
        st.rerun()

if auto_refresh:
    time.sleep(refresh_interval * 60); st.rerun()
