import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, time, requests
from datetime import timezone, timedelta
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection

# ==========================================
# ⚙️ 1. 시스템 설정 및 유틸리티
# ==========================================
def get_now_kst():
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="Snow Master V64.7 Organic", page_icon="❄️", layout="wide")

@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except:
        st.warning("KRX 서버 지연 - 백업 리스트 사용")
    fallback = [['005930', '삼성전자'], ['000660', 'SK하이닉스'], ['005380', '현대차'], ['035420', 'NAVER']]
    return pd.DataFrame(fallback, columns=['Code', 'Name']).assign(Marcap=10**14)

# ==========================================
# 🧠 2. 분석 엔진: 유기적 지표 및 3분할 로직
# ==========================================
def calc_stoch(df, n, m, t):
    low_min, high_max = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    k = ((df['Close'] - low_min) / (high_max - low_min + 1e-9)) * 100
    return k.rolling(m).mean().rolling(t).mean()

def get_hybrid_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']
    
    # 변동성 및 지지선 계산
    df['MA20'], df['MA120'] = close.rolling(20).mean(), close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # ❄️ 스노우 파동
    df['SNOW_S'], df['SNOW_M'], df['SNOW_L'] = calc_stoch(df, 5, 3, 3), calc_stoch(df, 10, 6, 6), calc_stoch(df, 20, 12, 12)
    
    # 피보나치 & 매물대(POC)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    
    # RSI & 볼린저밴드 하단
    delta = close.diff(); g = delta.where(delta > 0, 0).rolling(14).mean(); l = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (g / (l + 1e-9))))
    df['BB_L'] = df['MA20'] - (close.rolling(20).std() * 2)
    
    return df

def get_organic_strategy(df):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr = curr['Close'], curr['ATR']
    
    # 호가 단위 조정 함수
    def adj(p):
        if p < 2000: t = 1
        elif p < 5000: t = 5
        elif p < 20000: t = 10
        elif p < 50000: t = 50
        elif p < 200000: t = 100
        else: t = 500
        return int(round(p/t)*t)

    # 1️⃣ 유기적 3분할 매수 타점 (지지 강도순)
    # 1차: 가장 가까운 주요 지지선 (POC)
    # 2차: 중기 지지선 (Fibo 618)
    # 3차: 강력 지지선 (BB 하단 혹은 ATR 2배 하단 중 낮은 값)
    buy_1 = adj(curr['POC'])
    buy_2 = adj(curr['Fibo_618'])
    buy_3 = adj(min(curr['BB_L'], cp - (atr * 2.5)))
    buy_points = sorted([buy_1, buy_2, buy_3], reverse=True)

    # 2️⃣ 유기적 3분할 매도 타점 (변동성 ATR 기반)
    # 1차: 보수적 익절 (ATR 1.5배)
    # 2차: 추세 익절 (ATR 3.0배)
    # 3차: 극대화 익절 (ATR 5.0배)
    sell_1 = adj(cp + (atr * 1.5))
    sell_2 = adj(cp + (atr * 3.0))
    sell_3 = adj(cp + (atr * 5.0))
    sell_points = [sell_1, sell_2, sell_3]

    # 3️⃣ Snow Score 계산
    score = 0
    if curr['SNOW_L'] < 25: score += 30 # 대파동 바닥
    if curr['SNOW_M'] < 25: score += 20 # 중파동 바닥
    if curr['RSI'] < 35: score += 20    # 심리적 바닥
    if cp <= buy_points[0]: score += 30 # 가격적 타점 도달
    
    return {
        "buy": buy_points,
        "sell": sell_points,
        "score": score,
        "rsi": curr['RSI'],
        "cp": cp
    }

# ==========================================
# 🖥️ 3. 스노우 스캐너 화면 구성
# ==========================================
with st.sidebar:
    st.title("❄️ Snow Master")
    min_m = st.number_input("최소 시총(억)", value=5000) * 100000000

if st.button("🚀 유기적 3분할 스캔 시작 (상위 100)"):
    krx = get_safe_stock_listing()
    targets = krx[krx['Marcap'] >= min_m].sort_values('Marcap', ascending=False).head(100)
    found, prog = [], st.progress(0)

    with ThreadPoolExecutor(max_workers=10) as ex:
        # 데이터 수집 및 분석 병렬 처리
        futures = {ex.submit(get_hybrid_indicators, fdr.DataReader(r['Code'], (get_now_kst()-timedelta(days=300)).strftime('%Y-%m-%d'))): r['Name'] for _, r in targets.iterrows()}
        for i, f in enumerate(as_completed(futures)):
            name = futures[f]
            try:
                res_df = f.result()
                if res_df is not None:
                    strat = get_organic_strategy(res_df)
                    found.append({"name": name, "strat": strat})
            except: continue
            prog.progress((i + 1) / len(targets))

    # 결과 출력: Snow Score 높은 순
    found = sorted(found, key=lambda x: x['strat']['score'], reverse=True)
    
    for item in found[:15]:
        s = item['strat']
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 15px; border-left: 10px solid #00d2ff; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin:0; color:#333;">{item['name']} <span style="font-size:0.6em; color:#666;">현재가: {int(s['cp']):,}원</span></h3>
                <span style="background:#e3f2fd; color:#0d47a1; padding:5px 12px; border-radius:20px; font-weight:bold;">Snow Score: {s['score']}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div style="background:#f0f7ff; padding:15px; border-radius:10px;">
                    <b style="color:#007bff;">🔵 유기적 3분할 매수</b><br>
                    <span style="font-size:0.9em;">
                        1차(POC): <b>{s['buy'][0]:,}원</b> (40%)<br>
                        2차(Fibo): <b>{s['buy'][1]:,}원</b> (30%)<br>
                        3차(Strong): <b>{s['buy'][2]:,}원</b> (30%)
                    </span>
                </div>
                <div style="background:#fff5f5; padding:15px; border-radius:10px;">
                    <b style="color:#dc3545;">🔴 유기적 3분할 매도</b><br>
                    <span style="font-size:0.9em;">
                        1차(보수): <b>{s['sell'][0]:,}원</b> (30%)<br>
                        2차(추세): <b>{s['sell'][1]:,}원</b> (30%)<br>
                        3차(목표): <b>{s['sell'][2]:,}원</b> (40%)
                    </span>
                </div>
            </div>
            <p style="font-size:0.8em; color:#999; margin-top:10px;">*본 가이드는 ATR 변동성과 매물대 밀집도를 계산하여 종목별로 다르게 산출되었습니다.</p>
        </div>
        """, unsafe_allow_html=True)
