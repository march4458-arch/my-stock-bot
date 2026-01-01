import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests, random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection  # 구글 시트 연결 추가

# ==========================================
# ⚙️ 1. 시스템 설정 및 구글 시트 연동
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Full Spec Pro", page_icon="⚡", layout="wide")

# 구글 시트 연결 함수 (데이터 로드 및 저장)
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # 캐시를 사용하지 않고 실시간으로 읽기 (ttl=0)
        df = conn.read(ttl=0)
        return df.dropna(how='all') if df is not None else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
        st.error(f"구글 시트 연결 오류: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트에 동기화되었습니다!")
    except Exception as e:
        st.error(f"구글 시트 저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            st.error(f"텔레그램 전송 실패: {e}")

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
# 🧠 2. 분석 엔진 (기존 로직 유지)
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
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f618 = curr['Fibo_382'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy = [adj(cp - atr*1.1), adj(ob), adj(f382)]
        sell = [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(df.tail(252)['High'].max() * 1.1)]
    elif regime == "📉 하락":
        buy = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)]
        sell = [adj(ob), adj(df['MA120'].iloc[-1]), adj(f382)]
    else:
        buy = [adj(ob), adj(f618), adj(f382)]
        sell = [adj(f382), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]

    pyramiding = {"type": "💤 관망", "msg": "현재 대응 구간이 아닙니다.", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            target = min(buy)
            pyramiding = {"type": "💧 물타기", "msg": f"{yield_pct:.1f}% 손실. {target:,}원 부근 비중 확대 권장", "color": "#FF4B4B"}
        elif yield_pct > 7 and regime == "🚀 상승":
            target = adj(cp + atr * 0.5)
            pyramiding = {"type": "🔥 불타기", "msg": f"{yield_pct:.1f}% 수익. {target:,}원 돌파 시 추가 매수 가능", "color": "#4FACFE"}

    return {
        "buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93),
        "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding
    }

# ==========================================
# 🖥️ 3. UI 구성 (구글 시트 적용)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Pro V62.1")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
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
    else: st.info("관리 탭에서 구글 시트에 종목을 먼저 등록하세요.")

# --- [💼 AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("진단 종목", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            py = strat['pyramiding']
            st.markdown(f"""<div style="background:#1E1E1E; padding:15px; border-radius:10px; border-left:8px solid {py['color']};">
                <h3 style="margin:0; color:{py['color']};">{py['type']} 가이드</h3>
                <p>{py['msg']}</p></div>""", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.info(f"🔵 **매수 타점**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원")
            c2.success(f"🔴 **매도 목표**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원")

# --- [🔍 스캐너] ---
with tabs[2]:
    if st.button("🚀 신뢰도순 전수 조사"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with ThreadPoolExecutor(max_workers=5) as exec:
            futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futures):
                name = futures[f]; df_scan = f.result()
                if df_scan is not None and df_scan.iloc[-1]['RSI'] < 48:
                    s = calculate_organic_strategy(df_scan)
                    cp = df_scan.iloc[-1]['Close']
                    score = (100 - s['rsi']) + (((s['sell'][0]-cp)/cp)*150)
                    found.append({"name": name, "cp": cp, "strat": s, "score": score})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid #4FACFE; margin-bottom:15px;">
                <h3>{icon} {d['name']} <small>(점수: {d['score']:.1f})</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; font-family:monospace;">
                    <div><b>🔵 매수타점</b><br>1차: {d['strat']['buy'][0]:>8,}원<br>2차: {d['strat']['buy'][1]:>8,}원</div>
                    <div><b>🔴 매도목표</b><br>1차: {d['strat']['sell'][0]:>8,}원<br>2차: {d['strat']['sell'][1]:>8,}원</div>
                </div></div>""", unsafe_allow_html=True)

# --- [➕ 관리] ---
with tabs[4]:
    st.subheader("📌 구글 시트 데이터 관리")
    df_p = get_portfolio_gsheets()
    
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명")
        p = c2.number_input("평단가", 0)
        q = c3.number_input("수량", 0)
        if st.form_submit_button("시트에 추가 및 저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                df_p = pd.concat([df_p, new_row], ignore_index=True)
                save_portfolio_gsheets(df_p)
                st.rerun()
    
    if not df_p.empty:
        st.write("현재 구글 시트 저장 데이터")
        st.dataframe(df_p, use_container_width=True)
        if st.button("시트 전체 초기화"):
            save_portfolio_gsheets(pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty']))
            st.rerun()
