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
    """서버 위치와 상관없이 항상 한국 표준시(KST) 반환"""
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="주식 비서 V63.6 Master", page_icon="⚡", layout="wide")

# 라이트 테마 기반 Pro 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [텔레그램 발송 함수] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# --- [데이터 연동 및 KeyError 방어] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            df.columns = [str(c).strip().capitalize() for c in df.columns]
            rename_map = {
                'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name',
                'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'
            }
            df = df.rename(columns=rename_map)
            for col in ['Code', 'Name', 'Buy_Price', 'Qty']:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0).astype(float)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(float)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df[['Code', 'Name', 'Buy_Price', 'Qty']]
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (기술적 반등 & 수급 로직)
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
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'] = hi_1y - (rng * 0.618)
    df['Fibo_500'] = hi_1y - (rng * 0.500)
    df['Fibo_382'] = hi_1y - (rng * 0.382)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f500, f618 = curr['Fibo_382'], curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = curr['Regime']
    if regime == "🚀 상승":
        buy, sell = [adj(cp-atr*1.2), adj(ob)], [adj(cp+atr*2.5), adj(cp+atr*4.5)]
    elif regime == "📉 하락":
        lo_1y = df.tail(252)['Low'].min()
        buy, sell = [adj(f618), adj(lo_1y)], [adj(ob), adj(df['MA20'].iloc[-1])]
    else:
        buy, sell = [adj(ob), adj(f618)], [adj(cp+atr*2.0), adj(cp+atr*4.0)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y:.1f}% 목표가 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": f"손절선 하회({y:.1f}%). 비중 축소!", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {y:.1f}%. {buy[1]:,}원 지지 확인 후 추매", "color": "#d63384", "alert": True}
        elif y > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {y:.1f}% 추세 강화. 비중 확대", "color": "#0d6efd", "alert": True}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding, "fibo": [f382, f500, f618], "vol_z": curr['Vol_Zscore']}

# ==========================================
# 🖥️ 3. 사이드바 및 실시간 알림 로직
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Master V63.6")
    now_kst = get_now_kst()
    m_on, m_msg = (True, "정규장 운영 중 🚀") if now_kst.weekday() < 5 and 900 <= now_kst.hour*100+now_kst.minute <= 1530 else (False, "장외 시간 🌙")
    st.info(f"**KST: {now_kst.strftime('%H:%M')} | {m_msg}**")
    
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    
    st.markdown("---")
    min_marcap_input = st.number_input("최소 시가총액 (억 원)", min_value=100, value=5000, step=500)
    min_marcap = min_marcap_input * 100000000
    
    st.subheader("🔔 알림 설정")
    alert_portfolio = st.checkbox("보유종목 실시간 감시", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

# --- [🔔 알림 로직: 18시 마감 리포트] ---
if daily_report_on and now_kst.hour == 18 and 0 <= now_kst.minute <= 10:
    today_str = now_kst.strftime('%Y-%m-%d')
    if "last_report_date" not in st.session_state or st.session_state.last_report_date != today_str:
        portfolio = get_portfolio_gsheets()
        if not portfolio.empty:
            msg = f"📝 <b>마감 리포트 ({today_str})</b>\n"
            for _, r in portfolio.iterrows():
                df_r = fetch_stock_smart(r['Code'], days=10)
                if df_r is not None:
                    cp_r = df_r['Close'].iloc[-1]
                    y_r = (cp_r - r['Buy_Price']) / r['Buy_Price'] * 100
                    msg += f"- {r['Name']}: {y_r:+.2f}% ({int(cp_r):,}원)\n"
            send_telegram_msg(tg_token, tg_id, msg + "\n오늘도 수고하셨습니다! 🌙")
            st.session_state.last_report_date = today_str

# ==========================================
# 🖥️ 4. 메인 탭 구현
# ==========================================
tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 전략 스캐너", "➕ 관리"])

with tabs[0]: # 대시보드
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>보유종목 실시간 감시</b>\n\n", False
        with st.spinner('자산 분석 중...'):
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=200)
                idx_df = get_hybrid_indicators(df)
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
    else: st.info("관리 탭에서 종목을 등록해주세요.")

with tabs[1]: # AI 리포트 (차트 복구)
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("진단할 종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = get_strategy(df_ai, row['Buy_Price'])
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("현재 국면", st_res['regime']); m2.metric("RSI", f"{st_res['rsi']:.1f}"); m3.metric("세력방어(OB)", f"{int(st_res['ob']):,}원"); m4.error(f"손절가: {st_res['stop']:,}원")
            
            py = st_res['pyramiding']
            st.markdown(f'<div class="guide-box" style="border-left:8px solid {py["color"]};"><h3>{py["type"]} 가이드</h3><p>{py["msg"]}</p></div>', unsafe_allow_html=True)
            
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:], close=df_ai['Close'][-100:], name="주가")])
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB Support")
            fig.add_hline(y=st_res['fibo'][1], line_dash="dash", line_color="orange", annotation_text="Fibo 0.5")
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, yaxis_title="가격 (원)")
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]: # 전략 스캐너
    if st.button(f"🚀 시총 {min_marcap_input}억↑ 우량/반등주 스캔"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, scan_msg, has_scan = [], "🔍 <b>전략 스캐너 발굴</b>\n\n", False
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futs):
                res = f.result()
                if res is not None:
                    curr_rsi = res['RSI'].iloc[-1]; curr_vol_z = res['Vol_Zscore'].iloc[-1]
                    sc = curr_vol_z * 15 # 수급 가점
                    if res['Regime'].iloc[-1] == "📉 하락":
                        if curr_rsi < 35: sc += 40 # 낙폭과대 반등
                        if curr_rsi > res['RSI'].iloc[-2]: sc += 20
                    else:
                        if 45 <= curr_rsi <= 65: sc += 30 # 추세 지속
                    found.append({"name": futs[f], "score": sc, "strat": get_strategy(res)})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        for idx, d in enumerate(found):
            icon = "🔥" if d['strat']['regime'] == "🚀 상승" else "⚡"
            bg = "#fdfdfe" if d['strat']['regime'] == "🚀 상승" else "#fff9f9"
            st.markdown(f"""<div class="scanner-card" style="background-color:{bg}; border-left:5px solid {'#007bff' if d['strat']['regime'] == '🚀 상승' else '#dc3545'};">
                <h4 style="margin:0;">{icon} {d['name']} ({d['score']:.1f}점)</h4>
                <p>국면: {d['strat']['regime']} | <b>RSI: {d['strat']['rsi']:.1f}</b><br>매수: {d['strat']['buy'][0]:,}원 | 매도: {d['strat']['sell'][0]:,}원</p></div>""", unsafe_allow_html=True)
            if alert_scanner and m_on and idx < 3:
                has_scan = True
                scan_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n매수: {d['strat']['buy'][0]:,}원\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

with tabs[3]: # 관리
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("저장"):
            krx_list = fdr.StockListing('KRX'); match = krx_list[krx_list['Name']==n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new_row], ignore_index=True))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
