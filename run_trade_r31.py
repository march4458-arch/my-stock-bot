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

st.set_page_config(page_title="주식 비서 V62.3 KST-Hybrid Full", page_icon="⚡", layout="wide")

# 라이트 테마 및 사용자 커스텀 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #666666 !important; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .guide-box h4 { color: #007bff; margin-top: 0; }
    .guide-box p { color: #495057 !important; font-size: 1rem; margin-bottom: 8px; }
    .scanner-card { background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .inner-box { background-color: #f1f3f5; padding: 20px; border-radius: 12px; color: #333333 !important; border: 1px solid #e9ecef; }
    .inner-box b { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [기본 유틸리티 함수] ---
@st.cache_data(ttl=3600)
def get_krx_list():
    return fdr.StockListing('KRX')

def get_market_status():
    now = get_now_kst()
    if now.weekday() >= 5: return False, "주말 휴장 😴"
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if start <= now <= end: return True, "정규장 운영 중 🚀"
    return False, "장외 시간 🌙"

def is_report_time():
    now = get_now_kst()
    return now.hour == 18 and 0 <= now.minute <= 10

# --- [데이터 연동 함수] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            cols = ['Code', 'Name', 'Buy_Price', 'Qty']
            for col in cols:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            # 데이터 클리닝
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
        st.error(f"시트 연결 오류: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트 동기화 완료!")
    except Exception as e: st.error(f"저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 🧠 2. 고도화된 분석 엔진
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
    if df is None or len(df) < 150: return None
    df = df.copy()
    close = df['Close']
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    ob_zones = []
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA120'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f500, f618 = curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    
    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy, sell = [adj(cp - atr*1.1), adj(ob), adj(f500)], [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(cp * 1.2)]
    elif regime == "📉 하락":
        buy, sell = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)], [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy, sell = [adj(f500), adj(ob), adj(f618)], [adj(df.tail(252)['High'].max()*0.95), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "대응 구간 대기 중", "color": "#6c757d", "alert": False}
    
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절 알림", "msg": f"목표가 {sell[0]:,}원 도달!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절 알림", "msg": f"손절가 {stop_loss:,}원 하회!", "color": "#dc3545", "alert": True}
        elif yield_pct < -5: pyramiding = {"type": "💧 물타기", "msg": f"손실 {yield_pct:.1f}%. 추가 매수 권장", "color": "#d63384", "alert": True}
        elif yield_pct > 7 and regime == "🚀 상승": pyramiding = {"type": "🔥 불타기", "msg": f"수익 {yield_pct:.1f}%. 추격 비중 확대", "color": "#0d6efd", "alert": True}
            
    return {"buy": buy, "sell": sell, "stop": stop_loss, "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding}

def calculate_advanced_score(df, strat):
    curr = df.iloc[-1]
    rsi_score = max(0, (75 - curr['RSI']) * 0.4)
    vol_score = min(25, max(0, curr['Vol_Zscore'] * 10)) if curr['Close'] > curr['Open'] else 0
    dist_ob = abs(curr['Close'] - curr['OB_Price']) / (curr['OB_Price'] + 1e-9)
    ob_score = max(0, 25 * (1 - dist_ob * 10))
    upside = (strat['sell'][0] - curr['Close']) / (curr['Close'] + 1e-9)
    profit_score = min(20, upside * 100)
    return float(rsi_score + vol_score + ob_score + profit_score)

# ==========================================
# 🖥️ 3. UI 로직 및 통합
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid KST V62.3")
    market_on, market_msg = get_market_status()
    st.info(f"**현재 시간(KST): {get_now_kst().strftime('%H:%M:%S')}**\n**시장 상태: {market_msg}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_portfolio = st.checkbox("보유종목 실시간 알림", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림", value=True)
    daily_report_on = st.checkbox("18시 마감 리포트 수신", value=True)
    auto_refresh = st.checkbox("자동 갱신 활성화", value=False)
    refresh_interval = st.slider("정규장 갱신 주기 (분)", 1, 60, 10)

# 마감 리포트 (한국 시간 기준)
if daily_report_on and is_report_time():
    today_kst = get_now_kst().date()
    if "report_sent" not in st.session_state or st.session_state.report_sent != today_kst:
        portfolio = get_portfolio_gsheets()
        if not portfolio.empty:
            report_msg = f"📝 <b>오늘의 마감 리포트 ({today_kst})</b>\n\n💼 <b>보유 종목 현황</b>\n"
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=10)
                if df is not None:
                    cp = df.iloc[-1]['Close']
                    yield_p = (cp - row['Buy_Price']) / row['Buy_Price'] * 100
                    report_msg += f"- {row['Name']}: {yield_p:+.2f}% ({int(cp):,}원)\n"
            send_telegram_msg(tg_token, tg_id, report_msg + "\n오늘도 고생하셨습니다! 🌙")
            st.session_state.report_sent = today_kst

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드 (수정 보완 버전)] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if portfolio is not None and not portfolio.empty:
        total_buy, total_eval, dash_list = 0.0, 0.0, []
        alert_needed, alert_msg = False, "🚨 <b>실시간 시장 감시 보고</b>\n\n"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner('실시간 분석 중...'):
            for idx, row in portfolio.iterrows():
                try:
                    status_text.text(f"분석 중: {row['Name']}")
                    df = fetch_stock_smart(row['Code'], days=150)
                    if df is not None:
                        idx_df = get_hybrid_indicators(df)
                        strat = calculate_organic_strategy(idx_df, float(row['Buy_Price']))
                        cp = float(idx_df.iloc[-1]['Close'])
                        qty = float(row['Qty'])
                        bp = float(row['Buy_Price'])
                        
                        total_buy += bp * qty
                        total_eval += cp * qty
                        profit = (cp - bp) * qty
                        
                        dash_list.append({
                            "종목": row['Name'], 
                            "수익": profit, 
                            "평가액": cp * qty,
                            "수익률": ((cp-bp)/bp*100) if bp>0 else 0
                        })
                        
                        if alert_portfolio and market_on and strat['pyramiding']['alert']:
                            alert_needed = True
                            alert_msg += f"<b>[{strat['pyramiding']['type']}]</b> {row['Name']}\n- 현재가: {int(cp):,}원\n- 안내: {strat['pyramiding']['msg']}\n\n"
                    progress_bar.progress((idx + 1) / len(portfolio))
                except: continue
        
        progress_bar.empty()
        status_text.empty()

        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            yield_total = ((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{yield_total:+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            
            col_a, col_b = st.columns(2)
            col_a.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', color_continuous_scale='RdYlGn', title="종목별 평가손익"), use_container_width=True)
            col_b.plotly_chart(px.pie(df_dash, values='평가액', names='종목', title="보유 비중", hole=0.3), use_container_width=True)
            
            if alert_needed: send_telegram_msg(tg_token, tg_id, alert_msg)
    else:
        st.info("현재 등록된 종목이 없습니다. [➕ 관리] 탭에서 종목을 추가하세요.")
        st.markdown("""<div class="guide-box"><h4>💡 시작 가이드</h4><p>1. 구글 시트에 <b>Code, Name, Buy_Price, Qty</b> 컬럼이 있는지 확인하세요.</p><p>2. <b>[➕ 관리]</b> 탭에서 첫 종목을 입력하면 대시보드가 활성화됩니다.</p></div>""", unsafe_allow_html=True)

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        selected = st.selectbox("분석 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=float(s_info['Buy_Price']))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime'])
            c2.metric("RSI", f"{strat['rsi']:.1f}")
            c3.metric("세력지지(OB)", f"{int(strat['ob']):,}원")
            c4.error(f"손절가: {strat['stop']:,}원")
            
            st.markdown(f'<div class="guide-box" style="border-left:8px solid {strat["pyramiding"]["color"]};"><h3 style="color:{strat["pyramiding"]["color"]};">{strat["pyramiding"]["type"]} 가이드</h3><p>{strat["pyramiding"]["msg"]}</p></div>', unsafe_allow_html=True)
            
            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **권장 매수 구간**\n\n1차: {strat['buy'][0]:,}원\n2차: {strat['buy'][1]:,}원\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **권장 매도 구간**\n\n1차: {strat['sell'][0]:,}원\n2차: {strat['sell'][1]:,}원\n3차: {strat['sell'][2]:,}원")
            
            fig = go.Figure(data=[go.Candlestick(x=df_detail.tail(150).index, open=df_detail.tail(150)['Open'], high=df_detail.tail(150)['High'], low=df_detail.tail(150)['Low'], close=df_detail.tail(150)['Close'], name='Candle')])
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    if st.button("🚀 AI 시장 전수 조사 시작 (시총 상위 50)"):
        stocks = get_krx_list()
        targets = stocks.sort_values(by='Marcap', ascending=False).head(50)
        found, sc_alert_msg = [], "🔍 <b>고득점 발굴 종목</b>\n\n"
        
        with st.spinner('시장 데이터를 스캔 중...'):
            with ThreadPoolExecutor(max_workers=8) as exec:
                futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futures):
                    name, df_scan = futures[f], f.result()
                    if df_scan is not None:
                        strat_tmp = calculate_organic_strategy(df_scan)
                        score = calculate_advanced_score(df_scan, strat_tmp)
                        if df_scan.iloc[-1]['RSI'] < 65:
                            found.append({"name": name, "score": score, "strat": strat_tmp})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        for idx, d in enumerate(found[:10]): # 상위 10개만 표시
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            if alert_scanner and idx < 3 and market_on: 
                sc_alert_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n- 1차매수: {d['strat']['buy'][0]:,}원\n\n"
            
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{icon} {d['name']}</h3>
                        <span style="background-color:#007bff; color:white; padding:5px 15px; border-radius:20px; font-weight:bold;">{d['score']:.1f}점</span>
                    </div>
                    <hr>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                        <div class="inner-box" style="border-top:4px solid #007bff;">
                            <b>🔵 분할 매수 구간</b><br>
                            1차: {d['strat']['buy'][0]:,}원 / 2차: {d['strat']['buy'][1]:,}원
                        </div>
                        <div class="inner-box" style="border-top:4px solid #dc3545;">
                            <b>🔴 목표 매도 구간</b><br>
                            1차: {d['strat']['sell'][0]:,}원 / 2차: {d['strat']['sell'][1]:,}원
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        if alert_scanner and found and market_on: send_telegram_msg(tg_token, tg_id, sc_alert_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    st.header("📈 전략 백테스트")
    bt_name = st.text_input("분석할 종목명", "삼성전자")
    c1, c2 = st.columns(2)
    tp_p = c1.slider("익절 목표 (%)", 3.0, 30.0, 10.0)
    sl_p = c2.slider("손절 제한 (%)", 3.0, 30.0, 7.0)
    
    if st.button("📊 과거 수익률 분석 실행"):
        krx = get_krx_list()
        match = krx[krx['Name'] == bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                trades, in_pos, entry_p = [], False, 0
                for i in range(150, len(df_bt)):
                    sub, today = df_bt.iloc[:i], df_bt.iloc[i]
                    strat = calculate_organic_strategy(sub)
                    if not in_pos:
                        if today['Low'] <= strat['buy'][0]: 
                            entry_p, in_pos = strat['buy'][0], True
                    else:
                        if today['High'] >= entry_p * (1+tp_p/100): 
                            trades.append({'profit': tp_p, 'type': '익절'})
                            in_pos = False
                        elif today['Low'] <= entry_p * (1-sl_p/100): 
                            trades.append({'profit': -sl_p, 'type': '손절'})
                            in_pos = False
                if trades:
                    tdf = pd.DataFrame(trades)
                    win_rate = (tdf['type']=='익절').sum()/len(tdf)*100
                    st.metric("테스트 승률", f"{win_rate:.1f}%")
                    st.plotly_chart(px.line(tdf['profit'].cumsum(), title="2년간 누적 수익률 추이 (%)", template="plotly_white"), use_container_width=True)
                else: st.warning("입력한 매수 조건에 부합하는 과거 거래 내역이 없습니다.")
        else: st.error("종목명을 정확히 입력해주세요.")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    st.subheader("➕ 새 종목 등록")
    with st.form("add_stock_form"):
        c1, c2, c3 = st.columns(3)
        n = c1.text_input("종목명 (정확히 입력)")
        p = c2.number_input("평균 단가 (원)", min_value=0, step=100)
        q = c3.number_input("보유 수량", min_value=0, step=1)
        if st.form_submit_button("포트폴리오에 저장"):
            krx = get_krx_list()
            match = krx[krx['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                save_portfolio_gsheets(pd.concat([df_p, new_row], ignore_index=True))
                st.rerun()
            else: st.error("종목명을 찾을 수 없습니다.")
    
    st.subheader("📋 현재 포트폴리오 리스트")
    st.dataframe(df_p, use_container_width=True)
    if st.button("🗑️ 전체 데이터 초기화 (주의)"):
        save_portfolio_gsheets(pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty']))
        st.rerun()

# ==========================================
# ⏳ 4. 지능형 자동 갱신
# ==========================================
if auto_refresh:
    time.sleep(refresh_interval * 60)
    st.rerun()
