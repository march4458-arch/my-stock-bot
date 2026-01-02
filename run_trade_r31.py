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

st.set_page_config(page_title="주식 비서 V64.4 Dynamic Final", page_icon="⚡", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .inner-box { background-color: #f1f3f5; padding: 15px; border-radius: 12px; color: #333333 !important; }
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
# 🧠 2. 하이브리드 분석 엔진 (유기적 타점 로직)
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
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    low_min, high_max = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['Stoch_K'] = ((close - low_min) / (high_max - low_min + 1e-9)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    std = close.rolling(20).std()
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'], df['Fibo_500'], df['Fibo_382'] = hi_1y-(rng*0.618), hi_1y-(rng*0.5), hi_1y-(rng*0.382)
    
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]

    hist_df = df.tail(20)
    counts, edges = np.histogram(hist_df['Close'], bins=10, weights=hist_df['Volume'])
    df['POC_Price'] = edges[np.argmax(counts)]
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, poc = curr['Close'], curr['ATR'], curr['OB_Price'], curr['POC_Price']
    f618, bbl = curr['Fibo_618'], curr['BB_Lower']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    # [V64.4] 유기적 타점 재배치 (Confluence Scoring)
    candidates = [
        {"name": "매물대(POC)", "price": poc, "score": 0},
        {"name": "피보나치(618)", "price": f618, "score": 0},
        {"name": "세력선(OB)", "price": ob, "score": 0},
        {"name": "밴드하단(BB)", "price": bbl, "score": 0}
    ]

    for cand in candidates:
        p = cand['price']
        if curr['RSI'] < 30 and p < cp * 0.95: cand['score'] += 20
        if curr['Stoch_K'] < 20: cand['score'] += 15
        dist = abs(cp - p) / (cp + 1e-9)
        if dist < 0.03: cand['score'] += 30 # 현재가 근접 가점
        if abs(p - bbl) / (bbl + 1e-9) < 0.01: cand['score'] += 25 # BB하단 중첩 가점

    sorted_cand = sorted(candidates, key=lambda x: x['score'], reverse=True)
    buy = [adj(sorted_cand[0]['price']), adj(sorted_cand[1]['price']), adj(sorted_cand[2]['price'])]
    buy_names = [sorted_cand[0]['name'], sorted_cand[1]['name'], sorted_cand[2]['name']]
    
    sell = [adj(cp + atr*2.0), adj(cp + atr*3.5), adj(cp + atr*5.0)]
    sell_names = ["1차 목표(30%)", "2차 목표(30%)", "최종 목표(40%)"]
    stop_loss = adj(min(buy) * 0.93)
    
    pyramiding = {"type": "💤 관망", "msg": f"{buy_names[0]} 타점 근접 대기", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / (buy_price + 1e-9) * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y:.1f}% 달성! 분할 익절 권장", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "손절선 하회. 비중 축소 필요", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {y:.1f}%. {buy_names[0]}에서 추매 대응", "color": "#d63384", "alert": True}
        elif y > 7 and curr['Regime'] == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": "추세 강화. 비중 확대 구간", "color": "#0d6efd", "alert": True}

    return {"buy": buy, "buy_names": buy_names, "sell": sell, "sell_names": sell_names, "stop": stop_loss, 
            "regime": curr['Regime'], "rsi": curr['RSI'], "pyramiding": pyramiding, 
            "poc": poc, "ob": ob, "fibo": f618, "bb_l": bbl, "stoch": curr['Stoch_K']}

# ==========================================
# 🖥️ 3. 사이드바 (텔레그램 및 알림 설정 복구)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Master V64.4")
    now_kst = get_now_kst()
    m_on, m_msg = (True, "정규장 운영 중 🚀") if now_kst.weekday() < 5 and 900 <= now_kst.hour*100+now_kst.minute <= 1530 else (False, "장외 시간 🌙")
    st.info(f"**KST: {now_kst.strftime('%H:%M')} | {m_msg}**")
    
    st.subheader("🔔 알림 설정")
    tg_token = st.text_input("Bot Token", type="password", help="텔레그램 봇 토큰")
    tg_id = st.text_input("Chat ID", help="텔레그램 채팅 ID")
    
    st.markdown("---")
    st.subheader("🔍 스캐너 설정")
    min_marcap_input = st.number_input("최소 시가총액 (억 원)", min_value=100, value=5000, step=500)
    min_marcap = min_marcap_input * 100000000
    
    alert_portfolio = st.checkbox("보유종목 실시간 감시", value=True)
    alert_scanner = st.checkbox("스캐너 상세 정보 발송", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    
    st.markdown("---")
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

# ==========================================
# 🖥️ 4. 메인 탭 구현
# ==========================================
tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 전략 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>실시간 포트폴리오 신호</b>\n\n", False
        for _, row in portfolio.iterrows():
            idx_df = get_hybrid_indicators(fetch_stock_smart(row['Code'], days=200))
            if idx_df is not None:
                st_res = get_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['pyramiding']['type']})
                if alert_portfolio and m_on and st_res['pyramiding']['alert']:
                    has_alert = True
                    port_alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n{st_res['pyramiding']['msg']}\n\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_dark"), use_container_width=True)
        if has_alert: send_telegram_msg(tg_token, tg_id, port_alert_msg)
    else: st.info("종목을 등록해주세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("진단 종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = get_strategy(df_ai, row['Buy_Price'])
            py = st_res['pyramiding']
            st.markdown(f'<div class="guide-box" style="border-left:10px solid {py["color"]};"><h2 style="color:{py["color"]}; margin:0;">{py["type"]}</h2><p>{py["msg"]}</p></div>', unsafe_allow_html=True)
            
            col_b, col_s = st.columns(2)
            with col_b: 
                st.markdown(f"""
                <div style="background:#e7f3ff; padding:20px; border-radius:15px; border:1px solid #b3d7ff;">
                    <h4 style="color:#0056b3; margin-top:0;">🔵 유기적 3분할 매수</h4>
                    1차({st_res['buy_names'][0]}): {st_res['buy'][0]:,}원 (30%)<br>
                    2차({st_res['buy_names'][1]}): {st_res['buy'][1]:,}원 (30%)<br>
                    3차({st_res['buy_names'][2]}): {st_res['buy'][2]:,}원 (40%)
                </div>
                """, unsafe_allow_html=True)
            with col_s:
                st.markdown(f"""
                <div style="background:#fff2f2; padding:20px; border-radius:15px; border:1px solid #ffcccc;">
                    <h4 style="color:#c82333; margin-top:0;">🔴 3분할 매도 목표</h4>
                    1차: {st_res['sell'][0]:,}원 (30%)<br>
                    2차: {st_res['sell'][1]:,}원 (30%)<br>
                    3차: {st_res['sell'][2]:,}원 (40%)
                </div>
                """, unsafe_allow_html=True)
            
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:], name="주가")])
            fig.add_hline(y=st_res['poc'], line_width=2, line_color="green", annotation_text="POC")
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB")
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 전략 스캐너 (3분할 수치 노출 복구)] ---
with tabs[2]:
    if st.button(f"🚀 초고속 유기적 전수조사 (Top 100)"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, has_scan, scan_msg = [], False, "🔍 <b>V64.4 발굴 종목</b>\n\n"
        prog_bar = st.progress(0); status_txt = st.empty()

        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'], days=300)): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    curr = res.iloc[-1]; st_res = get_strategy(res)
                    # 수급 + 지표 중첩 스코어링
                    sc = curr['Vol_Zscore'] * 15 + (25 if curr['RSI'] < 35 else 0) + (25 if abs(curr['Close']-curr['POC_Price'])/curr['POC_Price'] < 0.02 else 0)
                    found.append({"name": futs[f], "score": sc, "rsi": curr['RSI'], "regime": curr['Regime'], "strat": st_res, "cp": curr['Close']})
                prog_bar.progress((i + 1) / len(targets)); status_txt.text(f"분석 중: {futs[f]} ({i+1}/100)")

        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        status_txt.success("✅ 분석 완료!")
        for idx, d in enumerate(found):
            acc_c = "#007bff" if d['regime'] == "🚀 상승" else "#dc3545"
            st.markdown(f"""
            <div class="scanner-card" style="border-left: 8px solid {acc_c};">
                <h3 style="margin:0; color:{acc_c};">{d['name']} <small style="color:gray;">Score: {d['score']:.1f}</small></h3>
                <p style="margin:5px 0;">현재가: <b>{int(d['cp']):,}원</b> | 최우선 타점: <b>{d['strat']['buy_names'][0]}</b></p>
                <hr style="margin:10px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div style="background:#f0f7ff; padding:12px; border-radius:10px;">
                        <b style="color:#0056b3;">🔵 유기적 3분할 매수</b><br>
                        1차({d['strat']['buy_names'][0]}): {d['strat']['buy'][0]:,}원<br>
                        2차({d['strat']['buy_names'][1]}): {d['strat']['buy'][1]:,}원<br>
                        3차({d['strat']['buy_names'][2]}): {d['strat']['buy'][2]:,}원
                    </div>
                    <div style="background:#fff5f5; padding:12px; border-radius:10px;">
                        <b style="color:#c82333;">🔴 목표가 (익절)</b><br>
                        1차 목표: {d['strat']['sell'][0]:,}원<br>
                        2차 목표: {d['strat']['sell'][1]:,}원
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if alert_scanner and idx < 3:
                has_scan = True
                scan_msg += f"🔥 <b>{d['name']}</b> ({d['score']:.1f}점)\n최우선: {d['strat']['buy'][0]:,}원({d['strat']['buy_names'][0]})\n목표: {d['strat']['sell'][0]:,}원\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    st.header("📈 전략 과거 성과 검증 (최근 2년)")
    bt_name = st.text_input("검증할 종목명", "삼성전자")
    c1, c2 = st.columns(2); tp_p = c1.slider("익절 목표 (%)", 3.0, 30.0, 10.0); sl_p = c2.slider("손절 제한 (%)", 3.0, 20.0, 7.0)
    if st.button("📊 시뮬레이션 시작"):
        krx = fdr.StockListing('KRX'); match = krx[krx['Name'] == bt_name]
        if not match.empty:
            with st.spinner('과거 데이터 분석 중...'):
                df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
                if df_bt is not None:
                    trades, in_pos, entry_p = [], False, 0
                    for i in range(120, len(df_bt)):
                        curr_day = df_bt.iloc[i]; strat = get_strategy(df_bt.iloc[:i])
                        if not in_pos:
                            if curr_day['Low'] <= strat['buy'][0]: entry_p, in_pos = strat['buy'][0], True
                        else:
                            if curr_day['High'] >= entry_p * (1 + tp_p/100): trades.append({'profit': tp_p, 'type': '익절', 'date': df_bt.index[i]}); in_pos = False
                            elif curr_day['Low'] <= entry_p * (1 - sl_p/100): trades.append({'profit': -sl_p, 'type': '손절', 'date': df_bt.index[i]}); in_pos = False
                    if trades:
                        tdf = pd.DataFrame(trades); m1, m2, m3 = st.columns(3)
                        m1.metric("총 거래", f"{len(tdf)}회"); m2.metric("승률", f"{(tdf['type'] == '익절').sum()/len(tdf)*100:.1f}%"); m3.metric("누적 수익률", f"{tdf['profit'].sum():+.1f}%")
                        st.plotly_chart(px.line(tdf, x='date', y=tdf['profit'].cumsum(), title="누적 수익 곡선"), use_container_width=True)
                    else: st.warning("매수 타점 도달 기록 없음.")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            match = fdr.StockListing('KRX')[fdr.StockListing('KRX')['Name']==n]
            if not match.empty:
                new = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
