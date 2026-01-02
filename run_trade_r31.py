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

st.set_page_config(page_title="주식 비서 V63.0 Master", page_icon="⚡", layout="wide")

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
                'Code': 'Code', '코드': 'Code', '종목코드': 'Code',
                'Name': 'Name', '종목명': 'Name', '종목': 'Name',
                'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', '매수가': 'Buy_Price',
                'Qty': 'Qty', '수량': 'Qty', '보유수량': 'Qty'
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
# 🧠 2. 분석 엔진 (수식 및 지표 완벽 유지)
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
    if df is None or len(df) < 30: return None
    df = df.copy()
    close = df['Close']
    df['MA120'] = close.rolling(min(len(df), 120)).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(min(len(df), 14)).mean()
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    avg_vol = df['Volume'].rolling(min(len(df), 20)).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(min(len(df), 20)).std() + 1e-9)
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-20, len(df)-1) if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.02]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_382'], df['Fibo_500'], df['Fibo_618'] = hi_1y-(rng*0.382), hi_1y-(rng*0.5), hi_1y-(rng*0.618)
    df['Regime'] = "🚀 상승" if close.iloc[-1] > df['MA120'].iloc[-1] else "📉 하락"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f500, f618 = curr['Fibo_382'], curr['Fibo_500'], curr['Fibo_618']
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    regime = curr['Regime']
    if regime == "🚀 상승": buy, sell = [adj(cp-atr*1.1), adj(ob), adj(f500)], [adj(cp+atr*2.5), adj(cp+atr*4.5), adj(cp*1.2)]
    elif regime == "📉 하락": buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min()-atr)], [adj(f382), adj(f500), adj(ob)]
    else: buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max()+atr)]
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": f"손절가 {stop_loss:,}원 하회!", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {y:.1f}%. 추가 매입", "color": "#d63384", "alert": True}
        elif y > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {y:.1f}%. 비중 확대", "color": "#0d6efd", "alert": True}
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding, "fibo": [f382, f500, f618]}

# ==========================================
# 🖥️ 3. 사이드바 및 알림 설정 (복구 완료)
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Master")
    now_kst = get_now_kst()
    m_on, m_msg = (True, "정규장 운영 중 🚀") if now_kst.weekday() < 5 and 900 <= now_kst.hour*100+now_kst.minute <= 1530 else (False, "장외 시간 🌙")
    st.info(f"**KST: {now_kst.strftime('%H:%M')} | {m_msg}**")
    
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    
    st.markdown("---")
    st.subheader("🔔 알림 설정")
    alert_portfolio = st.checkbox("보유종목 실시간 감시", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    
    st.markdown("---")
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

# --- [🔔 알림 로직 1: 18시 마감 리포트] ---
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
tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 우량주 스캐너", "➕ 관리"])

with tabs[0]: # 대시보드 + 🔔 실시간 알림
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>보유종목 신호</b>\n\n", False
        for _, row in portfolio.iterrows():
            raw_df = fetch_stock_smart(row['Code'], days=200)
            idx_df = get_hybrid_indicators(raw_df)
            if idx_df is not None:
                st_res = calculate_organic_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "평가액": cp*row['Qty'], "수익률": (cp-row['Buy_Price'])/row['Buy_Price']*100 if row['Buy_Price']>0 else 0})
                if alert_portfolio and m_on and st_res['pyramiding']['alert']:
                    has_alert = True
                    port_alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n{st_res['pyramiding']['msg']}\n\n"
        if dash_list:
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수", f"{int(t_buy):,}원")
            c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
            c3.metric("평가손익", f"{int(t_eval-t_buy):,}원")
            st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='수익', template="plotly_white"), use_container_width=True)
        if has_alert: send_telegram_msg(tg_token, tg_id, port_alert_msg)
    else: st.info("종목을 등록해주세요.")

with tabs[1]: # AI 리포트
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("리포트 대상", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = calculate_organic_strategy(df_ai, row['Buy_Price'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("국면", st_res['regime']); m2.metric("RSI", f"{st_res['rsi']:.1f}"); m3.metric("OB지지", f"{int(st_res['ob']):,}원"); m4.error(f"손절: {st_res['stop']:,}원")
            st.markdown(f'<div style="padding:20px; border-left:5px solid {st_res["pyramiding"]["color"]}; background:#f8f9fa;"><h3>{st_res["pyramiding"]["type"]}</h3><p>{st_res["pyramiding"]["msg"]}</p></div>', unsafe_allow_html=True)
            st.info(f"🔵 매수: {st_res['buy']} | 🔴 매도: {st_res['sell']}")

with tabs[2]: # 스캐너 + 🔔 스캐너 알림
    if st.button("🚀 시총 5000억↑ 전수조사"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= 500000000000].sort_values('Marcap', ascending=False).head(50)
        found, scan_msg, has_scan = [], "🔍 <b>스캐너 발굴</b>\n\n", False
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for f in as_completed(futs):
                res = f.result()
                if res is not None:
                    sc = (70 - res['RSI'].iloc[-1]) * 0.5 + (res['Vol_Zscore'].iloc[-1] * 5)
                    found.append({"name": futs[f], "score": sc, "strat": calculate_organic_strategy(res)})
        found = sorted(found, key=lambda x: x['score'], reverse=True)[:5]
        for idx, d in enumerate(found):
            st.success(f"**{d['name']}** ({d['score']:.1f}점) / 타점: {d['strat']['buy'][0]:,}원")
            if alert_scanner and m_on and idx < 3:
                has_scan = True
                scan_msg += f"🔥 <b>{d['name']}</b> ({d['score']:.1f}점)\n매수추천: {d['strat']['buy'][0]:,}원\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

with tabs[3]: # 관리
    df_p = get_portfolio_gsheets()
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            krx_list = fdr.StockListing('KRX'); match = krx_list[krx_list['Name']==n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new_row], ignore_index=True))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
