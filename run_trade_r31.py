import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, time, requests
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# ⚙️ 1. 시스템 설정 및 KST 시간 함수
# ==========================================
def get_now_kst():
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="AI-Ultimate Snow Master V64.8", page_icon="🧠", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; background-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
    .ai-label { color: #7b1fa2; font-weight: bold; font-size: 0.85em; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티: 안전한 데이터 로딩] ---
@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except: pass
    fallback = [['005930', '삼성전자'], ['000660', 'SK하이닉스'], ['005380', '현대차']]
    return pd.DataFrame(fallback, columns=['Code', 'Name']).assign(Marcap=10**14)

def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 🧠 2. AI 학습형 엔진 (ML Learning)
# ==========================================
def run_ai_prediction(df):
    """현재 종목의 패턴을 학습하여 상승 확률 예측"""
    if len(df) < 150: return 50
    data = df.copy()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = ['RSI', 'SNOW_S', 'SNOW_M', 'SNOW_L', 'Vol_Z']
    
    train_df = data.dropna().tail(200)
    if len(train_df) < 50: return 50
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    prob = model.predict_proba(data[features].iloc[-1:])[0][1]
    return int(prob * 100)

# ==========================================
# 📊 3. 기술적 지표 및 전략 엔진 (Snow + Ultimate)
# ==========================================
def calc_stoch(df, n, m, t):
    low_min, high_max = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    k = ((df['Close'] - low_min) / (high_max - low_min + 1e-9)) * 100
    return k.rolling(m).mean().rolling(t).mean()

def get_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy(); close = df['Close']
    df['MA20'], df['MA120'] = close.rolling(20).mean(), close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['SNOW_S'], df['SNOW_M'], df['SNOW_L'] = calc_stoch(df, 5, 3, 3), calc_stoch(df, 10, 6, 6), calc_stoch(df, 20, 12, 12)
    delta = close.diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (g / (l + 1e-9)))); df['BB_L'] = df['MA20'] - (close.rolling(20).std() * 2)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_ultimate_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp, atr = curr['Close'], curr['ATR']
    ai_prob = run_ai_prediction(df) # AI 습득형 확률 추출
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)
    
    buy_pts = sorted([adj(curr['POC']), adj(curr['Fibo_618']), adj(curr['BB_L'])], reverse=True)
    sell_pts = [adj(cp + atr*2), adj(cp + atr*3.5), adj(cp + atr*5)]
    
    # Snow Score + AI 확률 결합
    snow_score = (25 if curr['SNOW_L'] < 25 else 0) + (15 if curr['SNOW_M'] < 25 else 0) + (20 if curr['RSI'] < 35 else 0)
    total_score = (snow_score * 0.6) + (ai_prob * 0.4) 

    status = {"type": "💤 관망", "color": "#6c757d", "msg": f"상승확률 {ai_prob}% | 분석 중", "alert": False}
    
    if buy_price > 0:
        yield_rate = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0]: status = {"type": "💰 익절", "color": "#28a745", "msg": f"{yield_rate:.1f}% 수익! AI확률 {ai_prob}%", "alert": True}
        elif cp <= buy_pts[2] * 0.93: status = {"type": "⚠️ 손절", "color": "#dc3545", "msg": "리스크 관리 구간", "alert": True}
        # V64.7 Ultimate 물타기: 손실 중 + 기술적 바닥 + AI 긍정
        elif yield_rate < -3 and total_score >= 55:
            status = {"type": "❄️ 스노우", "color": "#00d2ff", "msg": "AI 습득 바닥! 지능형 물타기", "alert": True}
        # V64.7 Ultimate 불타기: 수익 중 + 정배열 + AI 강력 추천
        elif yield_rate > 2 and ai_prob > 65 and curr['Regime'] == "🚀 상승":
            status = {"type": "🔥 불타기", "color": "#ff4b4b", "msg": "추세 가속! 수익 극대화 불타기", "alert": True}
        elif yield_rate < -5:
            status = {"type": "💧 손실", "color": "#d63384", "msg": "반등 파동 대기", "alert": False}

    return {"buy": buy_pts, "sell": sell_pts, "score": int(total_score), "ai_prob": ai_prob, "status": status, "regime": curr['Regime']}

# ==========================================
# 🖥️ 4. UI 탭 구현 (사이드바 및 메인 화면)
# ==========================================
with st.sidebar:
    st.title("🧠 AI-Ultimate V64.8")
    st.info(f"KST: {get_now_kst().strftime('%H:%M')}")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    st.divider()
    min_marcap = st.number_input("최소 시총 (억)", value=5000) * 100000000
    alert_on = st.checkbox("실시간 신호 감시", value=True)
    auto_refresh = st.checkbox("자동 새로고침", value=False)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_safe_stock_listing() # 실제 사용 시 get_portfolio_gsheets() 호출
    # (포트폴리오 수익 계산 및 텔레그램 발송 로직 - 이전 버전과 동일)
    st.write("구글 시트 연동 후 포트폴리오 상태가 여기에 표시됩니다.")

# --- [🔍 탭 2: AI 스캐너] ---
with tabs[2]:
    if st.button("🚀 AI-Ultimate 전수조사 (상위 100)"):
        krx = get_safe_stock_listing()
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(get_indicators, fdr.DataReader(r['Code'], (get_now_kst()-timedelta(days=365)).strftime('%Y-%m-%d'))): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    s_res = get_ultimate_strategy(res)
                    found.append({"name": futs[f], "score": s_res['score'], "strat": s_res})
                prog.progress((i + 1) / len(targets))
        
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:15]:
            st.markdown(f"""
                <div class="scanner-card" style="border-left:8px solid #7b1fa2;">
                    <h3 style="margin:0;">{d['name']} <span style="font-size:0.7em; color:#7b1fa2;">Total Score: {d['score']}</span></h3>
                    <p class="ai-label">AI 상승 확률: {d['strat']['ai_prob']}% | 추세: {d['strat']['regime']}</p>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                        <div class="buy-box"><b>매수: {d['strat']['buy'][0]:,}원</b></div>
                        <div class="sell-box"><b>익절: {d['strat']['sell'][0]:,}원</b></div>
                    </div>
                </div>""", unsafe_allow_html=True)
