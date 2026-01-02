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

st.set_page_config(page_title="주식 비서 V64.7.1 AI Fixed", page_icon="⚡", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
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
        if not os.path.exists(".streamlit/secrets.toml"):
            return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is None or df.empty: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
        df = df.dropna(how='all')
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 
                      'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
        df = df.rename(columns=rename_map)
        for col in ['Buy_Price', 'Qty']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
        return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 분석 엔진 (Triple & Dynamic)
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
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9)).fillna(0)))
    low_min, high_max = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['Stoch_K'] = ((close - low_min) / (high_max - low_min + 1e-9)) * 100
    df['BB_Lower'] = df['MA20'] - (close.rolling(20).std() * 2)
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'] = hi_1y-(rng*0.618)
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    counts, edges = np.histogram(df.tail(20)['Close'], bins=10, weights=df.tail(20)['Volume'])
    df['POC_Price'] = edges[np.argmax(counts)]
    df['Regime'] = "🚀 상승" if (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) > 0.004 else "📉 하락"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, poc, f618, bbl = curr['Close'], curr['ATR'], curr['OB_Price'], curr['POC_Price'], curr['Fibo_618'], curr['BB_Lower']
    hi_120 = df.tail(120)['High'].max()
    def adj(p): return int(round(p/10)*10) if p<100000 else int(round(p/100)*100)
    
    candidates = [{"name": "매물대(POC)", "price": poc, "score": 0}, {"name": "피보나치(618)", "price": f618, "score": 0},
                  {"name": "세력선(OB)", "price": ob, "score": 0}, {"name": "밴드하단(BB)", "price": bbl, "score": 0}]
    for cand in candidates:
        if curr['RSI'] < 35: cand['score'] += 20
        if abs(cp - cand['price']) / (cp + 1e-9) < 0.03: cand['score'] += 30
    
    sorted_cand = sorted(candidates, key=lambda x: x['score'], reverse=True)
    buy = [adj(sorted_cand[0]['price']), adj(sorted_cand[1]['price']), adj(sorted_cand[2]['price'])]
    buy_names = [sorted_cand[0]['name'], sorted_cand[1]['name'], sorted_cand[2]['name']]
    sell = [adj(cp + atr * 2.0), adj(max(cp + atr * 3.5, hi_120)), adj(max(cp + atr * 5.0, hi_120 + atr * 2.0))]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": f"{buy_names[0]} 대기", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / (buy_price + 1e-9) * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y:.1f}% 달성!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "리스크 관리", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"{buy_names[0]} 추매", "color": "#d63384", "alert": True}
    return {"buy": buy, "buy_names": buy_names, "sell": sell, "stop": stop_loss, "regime": curr['Regime'], "pyramiding": pyramiding, "poc": poc, "ob": ob}

# ==========================================
# 🖥️ 3. 메인 인터페이스
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid V64.7.1")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    min_marcap = st.number_input("최소 시총(억)", value=5000) * 100000000
    alert_portfolio = st.checkbox("보유종목 감시", value=True)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 전략 스캐너", "📈 백테스트", "➕ 관리"])

# 공통 포트폴리오 로드
global_portfolio = get_portfolio_gsheets()

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    if not global_portfolio.empty:
        t_buy, t_eval, dash_list = 0.0, 0.0, []
        for _, row in global_portfolio.iterrows():
            idx_df = get_hybrid_indicators(fetch_stock_smart(row['Code'], days=200))
            if idx_df is not None:
                st_res = get_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['pyramiding']['type']})
        st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_dark"), use_container_width=True)
    else: st.info("➕ 관리 탭에서 종목을 먼저 등록하세요.")

# --- [💼 탭 1: AI 리포트 (누락 보정 완료)] ---
with tabs[1]:
    if not global_portfolio.empty:
        st.subheader("🤖 보유 종목 하이브리드 진단")
        # 종목 선택 selectbox 누락 복구
        selected_name = st.selectbox("진단할 종목을 선택하세요", global_portfolio['Name'].tolist())
        row = global_portfolio[global_portfolio['Name'] == selected_name].iloc[0]
        
        with st.spinner(f"{selected_name} 분석 중..."):
            df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
            if df_ai is not None:
                st_res = get_strategy(df_ai, row['Buy_Price'])
                py = st_res['pyramiding']
                
                st.markdown(f'<div class="guide-box" style="border-left:10px solid {py["color"]};"><h2>{py["type"]}</h2><p>{py["msg"]}</p></div>', unsafe_allow_html=True)
                
                col_b, col_s = st.columns(2)
                with col_b: st.markdown(f'<div class="buy-box"><b>🔵 유기적 3분할 매수</b><br>1차({st_res["buy_names"][0]}): {st_res["buy"][0]:,}원<br>2차: {st_res["buy"][1]:,}원<br>3차: {st_res["buy"][2]:,}원</div>', unsafe_allow_html=True)
                with col_s: st.markdown(f'<div class="sell-box"><b>🔴 전략적 3분할 매도</b><br>1차: {st_res["sell"][0]:,}원<br>2차: {st_res["sell"][1]:,}원<br>3차: {st_res["sell"][2]:,}원</div>', unsafe_allow_html=True)
                
                
                fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:])])
                fig.add_hline(y=st_res['poc'], line_color="green", annotation_text="POC")
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 포트폴리오에 등록된 종목이 없습니다. '➕ 관리' 탭에서 종목을 추가해주세요.")

# --- [🔍 탭 2: 전략 스캐너] ---
with tabs[2]:
    if st.button("🚀 유기적 전수조사 (Top 100)"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found = []
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'], days=300)): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futs):
                res = f.result()
                if res is not None:
                    st_res = get_strategy(res)
                    found.append({"name": futs[f], "score": res.iloc[-1]['Vol_Zscore']*20, "strat": st_res, "regime": st_res['regime']})
        
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:10]:
            st.markdown(f"""<div class="scanner-card">
                <h3>{d['name']} <small>Score: {d['score']:.1f}</small></h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div class="buy-box"><b>🔵 매수 타점</b><br>1차: {d['strat']['buy'][0]:,}원</div>
                    <div class="sell-box"><b>🔴 목표가</b><br>1차: {d['strat']['sell'][0]:,}원</div>
                </div>
            </div>""", unsafe_allow_html=True)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    st.subheader("📂 종목 관리")
    with st.form("stock_form"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명")
        p = c2.number_input("평단가", value=0)
        q = c3.number_input("수량", value=0)
        if st.form_submit_button("등록"):
            krx_list = fdr.StockListing('KRX')
            match = krx_list[krx_list['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([global_portfolio, new_row]))
                st.success(f"{n} 등록 완료! 앱을 새로고침 하세요.")
                st.rerun()
    st.dataframe(global_portfolio, use_container_width=True)
