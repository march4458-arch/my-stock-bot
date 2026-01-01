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
st.set_page_config(page_title="주식 비서 V62.1 Full Spec Pro", page_icon="⚡", layout="wide")

# 함수 이름을 하나로 통일하여 NameError 방지
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        return df.dropna(how='all') if df is not None else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트 동기화 완료!")
    except:
        st.error("구글 시트 저장 실패")

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
# 🧠 2. 고도화 분석 엔진 (수급 로직 포함)
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
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_advanced_score(df, strat):
    rsi = df['RSI'].iloc[-1]
    cp = df['Close'].iloc[-1]
    ob = df['OB_Price'].iloc[-1]
    vol_avg = df['Volume'].rolling(10).mean().iloc[-1]
    supply_boost = 25 if (df['Volume'].iloc[-1] > vol_avg * 1.3 and df['Close'].iloc[-1] > df['Open'].iloc[-1]) else 0
    rsi_score = max(0, (60 - rsi) * 0.41)
    ob_score = max(0, 25 * (1 - (abs(cp-ob)/ob) * 10))
    upside = (strat['sell'][0] - cp) / cp
    return float(rsi_score + ob_score + supply_boost + min(25, upside * 100))

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = df['Regime'].iloc[-1]
    buy = [adj(cp - atr*1.2), adj(ob)]
    sell = [adj(cp + atr*2.5), adj(cp + atr*4.5)]
    
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": f"평단 대비 {yield_pct:.1f}% 손실. {buy[1]:,}원 지점에서 비중 확대 권장", "color": "#FF4B4B"}
        elif yield_pct > 7: pyramiding = {"type": "🔥 불타기", "msg": f"수익권 진입. 추가 매수 시나리오 가동", "color": "#4FACFE"}

    return {"buy": buy, "sell": sell, "ob": ob, "rsi": curr['RSI'], "regime": regime, "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 구성 (V62.1 Full Spec Pro UI 유지)
# ==========================================
with st.sidebar:
    st.title("🛡️ V62.1 Full Spec Pro")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("Fear & Greed", f"{fg_val}pts", fg_txt)
    st.info("💡 수급 분석 엔진 가동 중")

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "➕ 관리"])

# --- [📊 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets() # load_portfolio 대신 통일된 이름 사용
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0, 0, []
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
            st.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', template="plotly_dark"), use_container_width=True)
    else: st.info("관리 탭에서 종목을 먼저 등록하세요.")

# --- [💼 AI 리포트] (기존 가로형 요약 UI) ---
with tabs[1]:
    portfolio = get_portfolio_gsheets() # load_portfolio 대신 통일된 이름 사용
    if not portfolio.empty:
        selected = st.selectbox("진단할 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = get_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            
            # 가로형 상단 요약 바
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            c1.metric("국면", strat['regime'])
            c2.metric("RSI", f"{strat['rsi']:.1f}")
            c3.metric("세력방어(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"손절가: {int(strat['buy'][1] * 0.93):,}원")
            
            py = strat['pyramiding']
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:10px; border-left:8px solid {py['color']}; margin-top:10px;">
                <h3 style="margin:0; color:{py['color']};">{py['type']} 가이드</h3><p>{py['msg']}</p></div>""", unsafe_allow_html=True)
            
            # 2단 타점 레이아웃
            col_buy, col_sell = st.columns(2)
            with col_buy:
                st.markdown(f"""<div style="background:#1B2635; padding:20px; border-radius:10px; height:160px;">
                    <h4 style="color:#4FACFE; margin-top:0;">🔵 매수 타점</h4>
                    <p style="font-size:18px;">1차: {strat['buy'][0]:,}원<br>2차: {strat['buy'][1]:,}원</p></div>""", unsafe_allow_html=True)
            with col_sell:
                st.markdown(f"""<div style="background:#2D1B1B; padding:20px; border-radius:10px; height:160px;">
                    <h4 style="color:#FF4B4B; margin-top:0;">🔴 매도 목표</h4>
                    <p style="font-size:18px;">1차: {strat['sell'][0]:,}원<br>2차: {strat['sell'][1]:,}원</p></div>""", unsafe_allow_html=True)
            
            fig = px.line(df_detail.tail(100), y='Close', title=f"{selected} 추세 분석")
            fig.add_hline(y=strat['ob'], line_dash="dash", line_color="yellow", annotation_text="산부인과(OB)")
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 스캐너] (기존 카드 디자인 유지) ---
with tabs[2]:
    if st.button("🚀 신뢰도순 전수 조사 시작"):
        stocks = get_krx_list().head(50)
        found = []
        with st.spinner("수급 및 신뢰도 분석 중..."):
            for _, r in stocks.iterrows():
                df_s = get_hybrid_indicators(fetch_stock_smart(r['Code']))
                if df_s is not None and df_s.iloc[-1]['RSI'] < 55:
                    s = get_strategy(df_s)
                    score = calculate_advanced_score(df_s, s)
                    found.append({"name": r['Name'], "score": score, "cp": df_s.iloc[-1]['Close'], "strat": s})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid #4FACFE; margin-bottom:15px;">
                <h3>{icon} {d['name']} <small>(점수: {d['score']:.1f})</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; font-family:monospace;">
                    <div><b>🔵 매수타점</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원</div>
                    <div><b>🔴 매도목표</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원</div>
                </div></div>""", unsafe_allow_html=True)

# --- [➕ 관리] ---
with tabs[3]:
    st.subheader("📌 구글 시트 데이터 관리")
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명")
        p = c2.number_input("평단가", 0)
        q = c3.number_input("수량", 0)
        if st.form_submit_button("저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)
