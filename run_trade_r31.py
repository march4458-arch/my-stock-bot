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

st.set_page_config(page_title="주식 비서 V64.7 Snow Master", page_icon="❄️", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
    .snow-badge { background-color: #e3f2fd; color: #0d47a1; padding: 2px 8px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
            df = df.rename(columns=rename_map)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0).astype(float)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(float)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df[['Code', 'Name', 'Buy_Price', 'Qty']]
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (Snow 파동 통합)
# ==========================================
def calc_stoch(df, n, m, t):
    low_min = df['Low'].rolling(n).min()
    high_max = df['High'].rolling(n).max()
    k = ((df['Close'] - low_min) / (high_max - low_min + 1e-9)) * 100
    d = k.rolling(m).mean()
    slow_d = d.rolling(t).mean()
    return slow_d

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
            df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except: return None

def get_hybrid_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']
    
    # 기본 이동평균 및 변동성
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # RSI & 볼린저밴드
    delta = close.diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['BB_L'] = df['MA20'] - (close.rolling(20).std() * 2)
    
    # ❄️ 스토캐스틱 스노우 (Snow Snow)
    df['SNOW_S'] = calc_stoch(df, 5, 3, 3)
    df['SNOW_M'] = calc_stoch(df, 10, 6, 6)
    df['SNOW_L'] = calc_stoch(df, 20, 12, 12)
    
    # 매물대(POC) & 세력선(OB)
    hist = df.tail(20)
    counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Z'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    ob_zones = [df['Low'].iloc[i] for i in range(len(df)-40, len(df)) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr = curr['Close'], curr['ATR']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)

    # 매수 타점 후보 및 점수 계산
    snow_score = 0
    if curr['SNOW_L'] < 20: snow_score += 25
    if curr['SNOW_M'] < 20: snow_score += 15
    if curr['SNOW_S'] < curr['SNOW_M']: snow_score += 10
    if curr['RSI'] < 30: snow_score += 20
    if abs(cp - curr['POC'])/cp < 0.02: snow_score += 15

    # 유기적 3분할 매수 타점
    buy_pts = sorted([adj(curr['POC']), adj(curr['OB']), adj(curr['BB_L'])], reverse=True)
    # 트리플 3분할 매도 타점 (30:30:40)
    sell_pts = [adj(cp + atr*2.0), adj(cp + atr*3.5), adj(cp + atr*5.0)]
    
    # 상태 진단
    pyramiding = {"type": "💤 관망", "msg": "타점 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y:.1f}% 달성! 분할 익절", "color": "#28a745", "alert": True}
        elif cp <= buy_pts[2] * 0.93: pyramiding = {"type": "⚠️ 손절", "msg": "리스크 관리 가동", "color": "#dc3545", "alert": True}
        elif snow_score >= 45: pyramiding = {"type": "❄️ 스노우", "msg": "파동 바닥 수렴! 강력 추매", "color": "#00d2ff", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": "매물대 근처 지지 확인", "color": "#d63384", "alert": True}

    return {"buy": buy_pts, "sell": sell_pts, "status": pyramiding, "snow_score": snow_score, "regime": curr['Regime'], "poc": curr['POC'], "ob": curr['OB']}

# ==========================================
# 🖥️ 3. 사이드바 및 메인 탭 구현
# ==========================================
with st.sidebar:
    st.title("❄️ Snow Master V64.7")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    min_marcap = st.number_input("최소 시총(억)", value=5000) * 100000000
    alert_on = st.checkbox("실시간 알림 활성", value=True)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스노우 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, alert_msg = 0, 0, [], ""
        for _, row in portfolio.iterrows():
            df = get_hybrid_indicators(fetch_stock_smart(row['Code'], days=200))
            if df is not None:
                st_res = get_strategy(df, row['Buy_Price'])
                cp = df['Close'].iloc[-1]
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['status']['type']})
                if alert_on and st_res['status']['alert']:
                    alert_msg += f"[{st_res['status']['type']}] {row['Name']}: {st_res['status']['msg']}\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)
        if alert_msg: send_telegram_msg(tg_token, tg_id, f"❄️ <b>실시간 신호</b>\n\n{alert_msg}")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    if not portfolio.empty:
        sel = st.selectbox("종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            res = get_strategy(df_ai, row['Buy_Price'])
            st.markdown(f'<div class="guide-box" style="border-left:10px solid {res["status"]["color"]};"><h2>{res["status"]["type"]} <small>(Snow: {res["snow_score"]})</small></h2><p>{res["status"]["msg"]}</p></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.info(f"🔵 매수타점: {res['buy']}")
            col2.success(f"🔴 매도타점: {res['sell']}")
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:], name="주가")])
            fig.add_hline(y=res['poc'], line_color="orange", annotation_text="POC")
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스노우 스캐너] ---
with tabs[2]:
    if st.button("🚀 스노우 파동 전수조사 (상위 100)"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found = []
        prog = st.progress(0)
        
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'], days=300)): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    st_res = get_strategy(res)
                    found.append({"name": futs[f], "score": st_res['snow_score'], "strat": st_res, "regime": st_res['regime']})
                prog.progress((i + 1) / 100)

        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:10]:
            st.markdown(f"""<div class="scanner-card">
                <h3>{d['name']} <span class="snow-badge">Snow: {d['score']}</span></h3>
                <p>국면: {d['regime']} | 추천매수: {d['strat']['buy'][0]:,}원</p>
                </div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    bt_name = st.text_input("백테스트 종목", "에코프로")
    if st.button("📊 테스트 시작"):
        krx = fdr.StockListing('KRX'); match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                cash, stocks, equity = 10000000, 0, []
                for i in range(100, len(df_bt)):
                    curr = df_bt.iloc[i]; strat = get_strategy(df_bt.iloc[:i]); cp = curr['Close']
                    if stocks == 0 and curr['Low'] <= strat['buy'][0]:
                        stocks = cash // strat['buy'][0]; cash -= (stocks * strat['buy'][0])
                    elif stocks > 0 and curr['High'] >= strat['sell'][0]:
                        cash += (stocks * strat['sell'][0]); stocks = 0
                    equity.append(cash + (stocks * cp))
                
                edf = pd.DataFrame(equity, columns=['total'])
                st.metric("최종 수익률", f"{(edf['total'].iloc[-1]-10000000)/100000:+.2f}%")
                st.line_chart(edf)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("추가"):
            krx = fdr.StockListing('KRX'); match = krx[krx['Name']==n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)
