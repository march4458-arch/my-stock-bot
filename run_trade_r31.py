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

st.set_page_config(page_title="주식 비서 V63.5 Custom Master", page_icon="⚡", layout="wide")

# 라이트 테마 CSS 및 UI 레이아웃 설정
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 12px; border: 1px solid #ddd; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
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
            # 컬럼명 표준화 (KeyError 방지)
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
# 🧠 2. 하이브리드 분석 엔진 (기술적 반등 로직)
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
    
    # OB(Order Block) 및 피보나치 수식 (누락 없이 유지)
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
    if regime == "🚀 상승":
        buy, sell = [adj(cp-atr*1.1), adj(ob), adj(f500)], [adj(cp+atr*2.5), adj(cp+atr*4.5), adj(cp*1.2)]
    elif regime == "📉 하락":
        lo_1y = df.tail(252)['Low'].min()
        buy, sell = [adj(f618), adj(lo_1y), adj(lo_1y-atr)], [adj(f500), adj(f382), adj(ob)]
    else:
        buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max()+atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    y = (cp - buy_price) / buy_price * 100 if buy_price > 0 else 0
    pyramiding = {"type": "💰 익절" if y > 10 else "⚠️ 손절" if y < -7 else "💤 관망", "msg": "원칙 대응", "color": "#28a745" if y > 10 else "#dc3545" if y < -7 else "#6c757d", "alert": (y > 10 or y < -7)}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding, "fibo": [f382, f500, f618]}

# ==========================================
# 🖥️ 3. 사이드바 설정 (Custom Cap 반영)
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Master V63.5")
    now_kst = get_now_kst()
    m_on, m_msg = (True, "정규장 운영 중 🚀") if now_kst.weekday() < 5 and 900 <= now_kst.hour*100+now_kst.minute <= 1530 else (False, "장외 시간 🌙")
    st.info(f"**현재 시간(KST): {now_kst.strftime('%H:%M')} | {m_msg}**")
    
    tg_token = st.text_input("텔레그램 Bot Token", type="password")
    tg_id = st.text_input("텔레그램 Chat ID")
    
    st.markdown("---")
    st.subheader("🔍 스캐너 설정")
    # 사용자가 직접 최소 시가총액 설정 가능 (억 원 단위)
    min_marcap_input = st.number_input("최소 시가총액 기준 (억 원)", min_value=100, value=5000, step=500)
    min_marcap = min_marcap_input * 100000000 # 원 단위 변환
    
    st.subheader("🔔 알림 설정")
    alert_portfolio = st.checkbox("보유종목 실시간 감시 알림", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 종목 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 자동발송", value=True)
    
    st.markdown("---")
    auto_refresh = st.checkbox("화면 자동 새로고침", value=False)
    interval = st.slider("갱신 주기 (분)", 1, 60, 10)

# --- [🔔 알림 로직: 18시 마감 리포트] ---
if daily_report_on and now_kst.hour == 18 and 0 <= now_kst.minute <= 10:
    today_str = now_kst.strftime('%Y-%m-%d')
    if "last_report_date" not in st.session_state or st.session_state.last_report_date != today_str:
        portfolio = get_portfolio_gsheets()
        if not portfolio.empty:
            msg = f"📝 <b>오늘의 마감 리포트 ({today_str})</b>\n"
            for _, r in portfolio.iterrows():
                df_r = fetch_stock_smart(r['Code'], days=10)
                if df_r is not None:
                    cp_r = df_r['Close'].iloc[-1]
                    yield_r = (cp_r - r['Buy_Price']) / r['Buy_Price'] * 100
                    msg += f"- {r['Name']}: {yield_r:+.2f}% ({int(cp_r):,}원)\n"
            send_telegram_msg(tg_token, tg_id, msg + "\n오늘도 수고하셨습니다! 🌙")
            st.session_state.last_report_date = today_str

# ==========================================
# 🖥️ 4. 메인 탭 구현
# ==========================================
tabs = st.tabs(["📊 실시간 대시보드", "💼 AI 리포트 & 차트", "🔍 전략 스캐너", "➕ 종목 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>보유종목 신호 감지</b>\n\n", False
        with st.spinner('포트폴리오 분석 중...'):
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
                        port_alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n신호 발생 - 현재가: {int(cp):,}원\n\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매입금액", f"{int(t_buy):,}원")
        c2.metric("총 평가금액", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("총 평가손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='수익', template="plotly_white", title="종목별 수익 금액"), use_container_width=True)
        if has_alert: send_telegram_msg(tg_token, tg_id, port_alert_msg)
    else: st.info("종목 관리 탭에서 종목을 등록해주세요.")

# --- [💼 탭 1: AI 리포트 & 기술적 분석 차트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("리포트 및 차트 분석 대상", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        
        with st.spinner('데이터 로딩 및 차트 생성 중...'):
            raw_df = fetch_stock_smart(row['Code'])
            df_ai = get_hybrid_indicators(raw_df)
            
            if df_ai is not None:
                st_res = calculate_organic_strategy(df_ai, row['Buy_Price'])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("현재 국면", st_res['regime']); m2.metric("RSI (14)", f"{st_res['rsi']:.1f}"); m3.metric("세력지지(OB)", f"{int(st_res['ob']):,}원"); m4.error(f"손절가: {st_res['stop']:,}원")
                
                st.markdown(f'<div class="guide-box" style="border-left:8px solid {st_res["pyramiding"]["color"]};"><h3>{st_res["pyramiding"]["type"]} 신호</h3><p>전략 가이드에 따라 대응하십시오.</p></div>', unsafe_allow_html=True)
                
                
                fig = go.Figure(data=[go.Candlestick(
                    x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:], name="주가"
                )])
                fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB Support")
                fig.add_hline(y=st_res['fibo'][1], line_dash="dash", line_color="orange", annotation_text="Fibo 0.5")
                fig.add_hline(y=st_res['stop'], line_dash="solid", line_color="red", annotation_text="Stop Loss")
                
                fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False, title=f"{sel} ({row['Code']}) 기술적 분석")
                st.plotly_chart(fig, use_container_width=True)
                
                col_b, col_s = st.columns(2)
                col_b.info(f"🔵 **분할 매수 타점**\n\n1차: {st_res['buy'][0]:,}원\n2차: {st_res['buy'][1]:,}원\n3차: {st_res['buy'][2]:,}원")
                col_s.success(f"🔴 **분할 매도 타점**\n\n1차: {st_res['sell'][0]:,}원\n2차: {st_res['sell'][1]:,}원\n3차: {st_res['sell'][2]:,}원")

# --- [🔍 탭 2: 전략 스캐너 (사용자 설정 시총 반영)] ---
with tabs[2]:
    if st.button(f"🚀 시총 {min_marcap_input}억↑ 우량주/반등주 스캔 시작"):
        krx = fdr.StockListing('KRX')
        # 사이드바에서 설정한 최소 시가총액 필터 적용
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, scan_msg, has_scan = [], "🔍 <b>전략 스캐너 발굴 알림</b>\n\n", False
        
        with st.spinner(f'시총 {min_marcap_input}억 이상 종목 전수조사 중...'):
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futs):
                    res = f.result()
                    if res is not None:
                        curr_rsi = res['RSI'].iloc[-1]
                        curr_vol_z = res['Vol_Zscore'].iloc[-1]
                        # 스코어링 로직: 기술적 반등 가중치 반영
                        sc = curr_vol_z * 15
                        if res['Regime'].iloc[-1] == "📉 하락":
                            if curr_rsi < 35: sc += 40
                            if curr_rsi > res['RSI'].iloc[-2]: sc += 20
                        else:
                            if 45 <= curr_rsi <= 65: sc += 30
                        found.append({"name": futs[f], "score": sc, "strat": calculate_organic_strategy(res)})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        for idx, d in enumerate(found):
            icon = "🔥" if d['strat']['regime'] == "🚀 상승" else "⚡"
            bg = "#f8f9fa" if d['strat']['regime'] == "🚀 상승" else "#fff5f5"
            st.markdown(f"""<div class="scanner-card" style="background-color:{bg};">
                <h4 style="margin:0;">{icon} {d['name']} ({d['score']:.1f}점)</h4>
                <p>국면: {d['strat']['regime']} | 1차 매수: {d['strat']['buy'][0]:,}원 | 1차 매도: {d['strat']['sell'][0]:,}원</p></div>""", unsafe_allow_html=True)
            if alert_scanner and m_on and idx < 3:
                has_scan = True
                scan_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n매수타점: {d['strat']['buy'][0]:,}원 발굴\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

# --- [➕ 탭 3: 종목 관리] ---
with tabs[3]:
    df_p = get_portfolio_gsheets()
    with st.form("add_stock_gs"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명 (정확히 입력)")
        p = c2.number_input("평단가 (원)", min_value=0, step=100)
        q = c3.number_input("보유수량", min_value=0, step=1)
        if st.form_submit_button("구글 시트에 저장"):
            krx_list = fdr.StockListing('KRX'); match = krx_list[krx_list['Name']==n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new_row], ignore_index=True))
                st.success(f"{n} 저장 완료!")
                st.rerun()
            else: st.error("해당 종목명을 찾을 수 없습니다.")
    st.divider()
    st.subheader("📋 현재 등록된 포트폴리오")
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
