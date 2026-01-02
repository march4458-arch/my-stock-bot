import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests, random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚙️ 1. 시스템 설정 및 기초 함수
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Tracking Spec", page_icon="⚡", layout="wide")

@st.cache_data(ttl=600)
def get_fear_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static/data"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        return r.json()['now']['value'], r.json()['now']['value_text']
    except: return 50, "Neutral"

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except: pass

BASE_DIR = os.path.join(os.getcwd(), 'Stock_System')
if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)
PORTFOLIO_FILE = os.path.join(BASE_DIR, 'my_portfolio.csv')

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try: 
            df = pd.read_csv(PORTFOLIO_FILE, dtype={'Code': str})
            return df if not df.empty else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
        except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

@st.cache_data(ttl=3600)
def get_krx_list(): return fdr.StockListing('KRX')

# ==========================================
# 🧠 2. 고도화된 유기적 분석 엔진 (OB + Fibo + ATR)
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
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    ob_zones = []
    avg_vol = df['Volume'].rolling(20).mean()
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20']
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f618 = curr['Fibo_382'], curr['Fibo_618']
    f500 = (f382 + f618) / 2
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
    return {"buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93), "regime": regime, "ob": ob, "rsi": curr['RSI']}

# ==========================================
# 🖥️ 3. UI 레이아웃 및 탭 구성
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Turbo V62.1")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("CNN Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    st.subheader("🔔 텔레그램 설정")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    auto_refresh = st.checkbox("자동 갱신")
    refresh_interval = st.slider("갱신 주기 (분)", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 실전 추적 분석", "➕ 관리"])

# --- [📊 탭 0/1: 대시보드 & 리포트] ---
with tabs[0]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0, 0, []
        with st.spinner('자산 데이터 동기화 중...'):
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=10)
                if df is not None and not df.empty:
                    cp = float(df.iloc[-1]['Close'])
                    total_buy += row['Buy_Price'] * row['Qty']; total_eval += cp * row['Qty']
                    dash_list.append({"종목": str(row['Name']), "수익": float(cp*row['Qty'] - row['Buy_Price']*row['Qty']), "평가액": float(cp*row['Qty'])})
        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0):+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', title="종목별 손익", color_continuous_scale='RdBu_r'), use_container_width=True)
            col2.plotly_chart(px.pie(df_dash, values='평가액', names='종목', hole=0.3, title="자산 비중"), use_container_width=True)

with tabs[1]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        selected = st.selectbox("진단 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail)
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수 타점**\n\n1차: {strat['buy'][0]:,} | 2차(OB): {strat['buy'][1]:,} | 3차: {strat['buy'][2]:,}")
            col_s.success(f"🔴 **3분할 매도 목표**\n\n1차: {strat['sell'][0]:,} | 2차: {strat['sell'][1]:,} | 3차: {strat['sell'][2]:,}")
            fig = go.Figure()
            df_p = df_detail.tail(200)
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
            fig.add_hline(y=df_detail['Fibo_382'].iloc[-1], line_dash="dash", line_color="white", opacity=0.3, annotation_text="Fibo 0.382")
            fig.add_hline(y=strat['ob'], line_color="yellow", annotation_text="OB 세력선")
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너 (V62 HTML 카드 UI + Telegram)] ---
with tabs[2]:
    st.header("🔍 유기적 타점 발굴 스캐너")
    if st.button("🚀 AI 분석팀 가동"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        progress = st.progress(0)
        with ThreadPoolExecutor(max_workers=25) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futures)):
                name = futures[f]; df_scan = f.result()
                if df_scan is not None and df_scan.iloc[-1]['RSI'] < 46:
                    found.append({"name": name, "cp": df_scan.iloc[-1]['Close'], "strat": calculate_organic_strategy(df_scan)})
                progress.progress((i + 1) / len(targets))
        if found:
            tg_msg = "🔍 <b>V62.1 스캔 리포트</b>\n\n"
            for d in found:
                st.markdown(f"""
                <div style="background-color:#1E1E1E; padding:20px; border-radius:15px; margin-bottom:20px; border-left:10px solid #4FACFE; box-shadow: 0px 4px 10px rgba(0,0,0,0.5);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="margin:0; color:#4FACFE;">{d['name']}</h2>
                        <span style="background-color:#333; padding:5px 15px; border-radius:20px; color:#FFD700; font-weight:bold;">{d['strat']['regime']} 국면</span>
                    </div>
                    <hr style="border:0.5px solid #444; margin:15px 0;">
                    <p style="font-size:1.1em;">현재가: <b>{int(d['cp']):,}원</b> | RSI: <span style="color:#FF4B4B;">{d['strat']['rsi']:.1f}</span></p>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px;">
                        <div style="background:#121212; padding:15px; border-radius:10px; border:1px solid #2E5A88;">
                            <h4 style="margin-top:0; color:#4FACFE;">🔵 3분할 매수</h4>
                            1차: {d['strat']['buy'][0]:,} | 2차: {d['strat']['buy'][1]:,} | 3차: {d['strat']['buy'][2]:,}
                        </div>
                        <div style="background:#121212; padding:15px; border-radius:10px; border:1px solid #882E2E;">
                            <h4 style="margin-top:0; color:#FF4B4B;">🔴 3분할 매도</h4>
                            1차: {d['strat']['sell'][0]:,} | 2차: {d['strat']['sell'][1]:,} | 3차: {d['strat']['sell'][2]:,}
                        </div>
                    </div>
                    <div style="margin-top:15px; padding:10px; background:#262626; border-radius:8px; display:flex; justify-content:space-between;">
                        <span style="color:#FFA500;">🚩 OB: {int(d['strat']['ob']):,}원</span>
                        <span style="color:#FF4B4B;">⚠️ 손절: {d['strat']['stop']:,}원</span>
                    </div>
                </div>""", unsafe_allow_html=True)
                tg_msg += f"📌 <b>{d['name']}</b> ({d['strat']['regime']})\n- 현재가: {int(d['cp']):,}\n- 타점: {d['strat']['buy'][0]:,}\n\n"
            if tg_token and tg_id: send_telegram_msg(tg_token, tg_id, tg_msg)

# --- [📈 탭 3: 실전 추적 분석 (백테스트 대대적 수정)] ---
with tabs[3]:
    st.header("📈 로직 실전 적중 추적기")
    st.info("스캐너 로직(RSI 저가 + 국면별 타점 도달)이 과거에 발생했을 때, 실제로 수익을 냈는지 추적합니다.")
    t_name = st.text_input("분석 종목명", "삼성전자")
    lookback_m = st.slider("추적 기간 (개월)", 3, 24, 12)
    
    if st.button("📊 실전 추적 시뮬레이션 가동"):
        match = get_krx_list()[get_krx_list()['Name'] == t_name]
        if not match.empty:
            with st.spinner('데이터 추적 중...'):
                df_bt = fetch_stock_smart(match.iloc[0]['Code'], days=lookback_m*30+150)
                if df_bt is not None:
                    hits = []
                    for i in range(150, len(df_bt)-5):
                        sub = df_bt.iloc[:i]; ind = get_hybrid_indicators(sub)
                        if ind is not None and ind.iloc[-1]['RSI'] < 46:
                            strat = calculate_organic_strategy(ind)
                            # 실제 그날의 저가가 1차 타점 이하로 내려갔을 때 "포착"
                            if df_bt['Low'].iloc[i] <= strat['buy'][0]:
                                post = df_bt.loc[df_bt.index[i]:].head(22) # 이후 약 한달간 추적
                                res = "진행중"
                                if post['High'].max() >= strat['sell'][0]: res = "익절성공"
                                elif post['Low'].min() <= strat['stop']: res = "손절발생"
                                hits.append({"date": df_bt.index[i], "p": strat['buy'][0], "res": res})
                    
                    if hits:
                        hdf = pd.DataFrame(hits)
                        wr = (hdf['res']=="익절성공").sum() / len(hdf) * 100
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="주가", line=dict(color='gray', width=1), opacity=0.4))
                            for h in hits:
                                color = "lime" if h['res']=="익절성공" else "red" if h['res']=="손절발생" else "yellow"
                                fig_t.add_trace(go.Scatter(x=[h['date']], y=[h['p']], mode='markers', 
                                                         marker=dict(color=color, size=10, symbol='triangle-up'), 
                                                         name=h['res']))
                            fig_t.update_layout(title=f"{t_name} 로직 적중 시각화", template="plotly_dark", height=500)
                            st.plotly_chart(fig_t, use_container_width=True)
                        with c2:
                            st.metric("로직 승률", f"{wr:.1f}%")
                            st.subheader("최근 적중 내역")
                            st.dataframe(hdf.tail(15), use_container_width=True)
                    else: st.warning("해당 기간 동안 스캐너 조건에 부합하는 타점이 없었습니다.")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 종목 추가")
        n_add = st.text_input("추가 종목명"); p_add = st.number_input("평단가", 0); q_add = st.number_input("수량", 0)
        if st.button("저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n_add]
            if not match.empty:
                df_p = load_portfolio()
                pd.concat([df_p, pd.DataFrame([[match.iloc[0]['Code'], n_add, p_add, q_add]], columns=['Code','Name','Buy_Price','Qty'])]).to_csv(PORTFOLIO_FILE, index=False); st.rerun()
    with c2:
        st.subheader("🗑️ 종목 삭제")
        df_p = load_portfolio()
        if not df_p.empty:
            del_n = st.selectbox("삭제 종목 선택", df_p['Name'].tolist())
            if st.button("삭제 실행"):
                df_p[df_p['Name']!=del_n].to_csv(PORTFOLIO_FILE, index=False); st.rerun()

if auto_refresh: time.sleep(refresh_interval*60); st.rerun()
