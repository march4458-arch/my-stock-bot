import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, time, requests, os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 구글 시트 연동
# ==========================================
st.set_page_config(page_title="주식 비서 V63 Cloud Pro", page_icon="🌐", layout="wide")

# 구글 시트 커넥션 (secrets.toml 설정 필요)
def get_db():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn

def load_portfolio():
    conn = get_db()
    try:
        # 데이터가 없을 경우를 대비해 기본 스키마 유지
        df = conn.read(ttl=5) # 5초 캐시로 실시간성 확보
        return df.dropna(subset=['Code'])
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio(df):
    conn = get_db()
    conn.update(data=df)
    st.cache_data.clear()

# ==========================================
# 🧠 2. 분석 엔진 (핵심 로직)
# ==========================================
@st.cache_data(ttl=3600)
def get_krx_list(): 
    return fdr.StockListing('KRX')

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
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
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

    pyramiding = {"type": "💤 관망", "msg": "신규 진입 구간 대기", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            pyramiding = {"type": "💧 물타기", "msg": f"{yield_pct:.1f}% 손실. {min(buy):,}원 비중 확대", "color": "#FF4B4B"}
        elif yield_pct > 7 and regime == "🚀 상승":
            pyramiding = {"type": "🔥 불타기", "msg": f"수익률 {yield_pct:.1f}%. 추세 강화 중", "color": "#4FACFE"}

    return {"buy": buy, "sell": sell, "stop": adj(min(buy)*0.93), "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 레이아웃
# ==========================================
with st.sidebar:
    st.title("🛡️ Cloud V63 Pro")
    st.caption("Google Sheets 연동 모드")
    st.divider()
    tg_token = st.text_input("Telegram Token", type="password")
    tg_id = st.text_input("Telegram ID")
    auto_refresh = st.checkbox("자동 갱신")
    refresh_int = st.slider("분 단위", 1, 60, 10)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "⚙️ 데이터 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        total_buy, total_eval, dash_data = 0, 0, []
        with st.spinner('실시간 데이터 동기화...'):
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=10)
                if df is not None:
                    cp = float(df.iloc[-1]['Close'])
                    b_total = row['Buy_Price'] * row['Qty']; e_total = cp * row['Qty']
                    total_buy += b_total; total_eval += e_total
                    dash_data.append({"종목": row['Name'], "수익": e_total - b_total, "평가액": e_total})
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(total_buy):,}원")
        c2.metric("총 평가", f"{int(total_eval):,}원", f"{(total_eval-total_buy)/total_buy*100 if total_buy>0 else 0:+.2f}%")
        c3.metric("실손익", f"{int(total_eval-total_buy):,}원")
        
        col1, col2 = st.columns(2)
        df_plot = pd.DataFrame(dash_data)
        col1.plotly_chart(px.bar(df_plot, x='종목', y='수익', color='수익', title="종목별 성과"), use_container_width=True)
        col2.plotly_chart(px.pie(df_plot, values='평가액', names='종목', hole=0.4, title="자산 분배"), use_container_width=True)
    else:
        st.info("데이터 관리 탭에서 종목을 추가하세요.")

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    st.header("🔍 마켓 스캔 (시총 상위 50)")
    if st.button("AI 분석 시작"):
        krx = get_krx_list()
        targets = krx[krx['Marcap'] >= 500000000000].sort_values('Marcap', ascending=False).head(50)
        found = []
        bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                name = futs[f]; df_s = f.result()
                if df_s is not None and df_s.iloc[-1]['RSI'] < 50:
                    res = calculate_organic_strategy(df_s)
                    upside = (res['sell'][0] - df_s.iloc[-1]['Close']) / df_s.iloc[-1]['Close'] * 100
                    found.append({"name": name, "cp": df_s.iloc[-1]['Close'], "strat": res, "score": (100-res['rsi']) + upside})
                bar.progress((i+1)/50)
        
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:10]:
            st.markdown(f"""<div style="background:#1E1E1E; padding:15px; border-radius:10px; border-left:5px solid #4FACFE; margin-bottom:10px;">
                <h4>{d['name']} (점수: {d['score']:.1f})</h4>
                <p>현재가: {int(d['cp']):,}원 | RSI: {d['strat']['rsi']:.1f} | 국면: {d['strat']['regime']}</p>
                <b>매수: {d['strat']['buy'][0]:,}원 / 매도: {d['strat']['sell'][0]:,}원</b></div>""", unsafe_allow_html=True)

# --- [⚙️ 탭 4: 데이터 관리 (핵심)] ---
with tabs[4]:
    st.header("⚙️ 구글 시트 데이터 제어")
    portfolio = load_portfolio()
    
    col_add, col_del = st.columns(2)
    with col_add:
        st.subheader("➕ 종목 추가")
        with st.form("add_stock", clear_on_submit=True):
            name = st.text_input("종목명")
            price = st.number_input("평단가", min_value=0)
            qty = st.number_input("수량", min_value=0)
            if st.form_submit_button("시트에 저장"):
                krx = get_krx_list()
                match = krx[krx['Name'] == name]
                if not match.empty:
                    new_row = pd.DataFrame([{'Code': match.iloc[0]['Code'], 'Name': name, 'Buy_Price': price, 'Qty': qty}])
                    save_portfolio(pd.concat([portfolio, new_row], ignore_index=True))
                    st.success(f"{name} 추가 완료!"); st.rerun()
    
    with col_del:
        st.subheader("🗑️ 종목 삭제")
        if not portfolio.empty:
            target = st.selectbox("삭제 대상", portfolio['Name'].tolist())
            if st.button("즉시 삭제"):
                save_portfolio(portfolio[portfolio['Name'] != target])
                st.warning(f"{target} 삭제됨"); st.rerun()

    st.divider()
    st.subheader("📋 현재 구글 시트 원본 데이터")
    st.dataframe(portfolio, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_int * 60); st.rerun()
