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

# 구글 시트 연결 함수 (NameError 방지를 위해 명칭 통일)
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        return df.dropna(how='all') if df is not None else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
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
# 🧠 2. 고도화된 분석 엔진 (수급 및 신뢰 점수)
# ==========================================
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        # FDR 우선 시도 후 실패 시 yfinance 보완
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
    
    # OB(Order Block) 세력 지지선 산출
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
    # 100점 만점 고도화 점수 체계
    rsi = df['RSI'].iloc[-1]
    cp = df['Close'].iloc[-1]
    ob = df['OB_Price'].iloc[-1]
    
    # [수급 점수] 거래량 실린 양봉 분석 (외인/기관 개입 추정)
    vol_avg = df['Volume'].rolling(10).mean().iloc[-1]
    supply_score = 25 if (df['Volume'].iloc[-1] > vol_avg * 1.3 and df['Close'].iloc[-1] > df['Open'].iloc[-1]) else 10 if df['Close'].iloc[-1] > df['Open'].iloc[-1] else 0
    
    # [과매도 점수] RSI가 60 이하일 때 역순으로 점수 부여
    rsi_score = max(0, (60 - rsi) * 0.41)
    
    # [지지선 점수] 현재가가 OB 세력선에 얼마나 근접했는지
    ob_dist = abs(cp - ob) / ob
    ob_score = max(0, 25 * (1 - ob_dist * 10))
    
    # [익절 여력 점수] 1차 목표가까지의 상승폭
    upside = (strat['sell'][0] - cp) / cp
    profit_score = min(25, upside * 100)
    
    return float(rsi_score + ob_score + supply_score + profit_score)

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
        elif yield_pct > 7: pyramiding = {"type": "🔥 불타기", "msg": f"수익권 진입. {cp+atr*0.5:,}원 돌파 시 추가 매수 가능", "color": "#4FACFE"}

    return {"buy": buy, "sell": sell, "ob": ob, "rsi": curr['RSI'], "regime": regime, "pyramiding": pyramiding}

# ==========================================
# 🖥️ 3. UI 구성 (V62.1 Full Spec UI 유지)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Pro V62.1")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("Fear & Greed", f"{fg_val}pts", fg_txt)
    st.info("💡 외인/기관 수급 분석 엔진 가동 중")
    st.divider()
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0, 0, []
        with st.spinner('실시간 자산 동기화 중...'):
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
    else: st.info("관리 탭에서 구글 시트에 종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트 (V62 가로 요약 UI)] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("진단할 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = get_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            
            # 상단 가로 요약 바 (V62.1 스타일)
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            c1.metric("국면", strat['regime'])
            c2.metric("RSI", f"{strat['rsi']:.1f}")
            c3.metric("세력방어(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"손절가: {int(strat['buy'][1] * 0.93):,}원")
            
            py = strat['pyramiding']
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:10px; border-left:8px solid {py['color']}; margin-top:10px;">
                <h3 style="margin:0; color:{py['color']};">{py['type']} 가이드</h3><p>{py['msg']}</p></div>""", unsafe_allow_html=True)
            
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

# --- [🔍 탭 2: 스캐너 (카드 UI + 고도화 점수)] ---
with tabs[2]:
    if st.button("🚀 수급/신뢰도순 전수 조사 시작"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        with st.spinner("외인/기관 수급 분석 중..."):
            with ThreadPoolExecutor(max_workers=5) as exec:
                futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futures):
                    name = futures[f]; df_s = f.result()
                    if df_s is not None and df_s.iloc[-1]['RSI'] < 55:
                        s = get_strategy(df_s)
                        score = calculate_advanced_score(df_s, s)
                        found.append({"name": name, "score": score, "cp": df_s.iloc[-1]['Close'], "strat": s})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        tg_msg = "🔍 <b>수급 고도화 스캔 결과</b>\n\n"
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div style="background:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid #4FACFE; margin-bottom:15px;">
                <h3>{icon} {d['name']} <small>(신뢰점수: {d['score']:.1f}점)</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; font-family:monospace;">
                    <div><b>🔵 매수타점</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원</div>
                    <div><b>🔴 매도목표</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원</div>
                </div></div>""", unsafe_allow_html=True)
            tg_msg += f"📌 {d['name']} ({d['score']:.1f}점)\n현재가: {int(d['cp']):,}원\n\n"
        
        if tg_token and tg_id and found:
            send_telegram_msg(tg_token, tg_id, tg_msg)
            st.toast("텔레그램 전송 완료!")

# --- [➕ 탭 3: 관리] ---
with tabs[3]:
    st.subheader("📌 구글 시트 종목 관리")
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명")
        p = c2.number_input("평단가", 0)
        q = c3.number_input("수량", 0)
        if st.form_submit_button("시트에 추가 및 저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)
    if st.button("시트 전체 초기화"):
        save_portfolio_gsheets(pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty']))
        st.rerun()
