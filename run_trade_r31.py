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

st.set_page_config(page_title="주식 비서 V64.6 Final Master", page_icon="⚡", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티 함수] ---
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
        if df is None or df.empty:
            return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
        df = df.dropna(how='all')
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 
                      'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
        df = df.rename(columns=rename_map)
        for col in ['Buy_Price', 'Qty']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
        return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except Exception as e:
        st.sidebar.warning(f"⚠️ 데이터 연동 대기 중... ({type(e).__name__})")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🛡️ 2. 데이터 엔진 우선순위 (Naver -> KRX -> Yahoo)
# ==========================================
@st.cache_data(ttl=3600)
def get_krx_list():
    try:
        ks = fdr.StockListing('KOSPI')
        kd = fdr.StockListing('KOSDAQ')
        df = pd.concat([ks, kd])
        if df is not None and not df.empty: return df
    except:
        st.warning("⚠️ 네이버 금융 응답 지연: KRX 서버로 전환합니다.")
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Marcap'])

def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (get_now_kst() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code_str, start_date)
        if df is not None and not df.empty: return df
    except: pass
    try:
        ticker = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
        df_yf = yf.download(ticker, start=start_date, progress=False, timeout=10)
        if df_yf is not None and not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex): df_yf.columns = df_yf.columns.get_level_values(0)
            return df_yf
    except: return None

# ==========================================
# 🧠 3. 하이브리드 지표 및 전략 엔진
# ==========================================
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
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9)).fillna(0)))
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_618'], df['Fibo_382'] = hi_1y-(rng*0.618), hi_1y-(rng*0.382)
    avg_vol = df['Volume'].rolling(20).mean()
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, f618 = curr['Close'], curr['ATR'], curr['OB_Price'], curr['Fibo_618']
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)
    buy = [adj(cp - atr * 1.1), adj(ob), adj(f618)]
    sell = [adj(cp + atr * 2.5), adj(cp + atr * 4.0), adj(df.tail(252)['High'].max() * 1.05)]
    stop = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": "타점 대기", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y_pct = (cp - buy_price) / buy_price * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y_pct:.1f}%!", "color": "#28a745", "alert": True}
        elif cp <= stop: pyramiding = {"type": "⚠️ 손절", "msg": "손절가 터치", "color": "#dc3545", "alert": True}
        elif y_pct < -5: pyramiding = {"type": "💧 물타기", "msg": "추매 구간", "color": "#d63384", "alert": True}
    return {"buy": buy, "sell": sell, "stop": stop, "regime": curr['Regime'], "rsi": curr['RSI'], "pyramiding": pyramiding}

# ==========================================
# 🖥️ 4. UI 구성 (사이드바 및 탭 전체)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Master V64.6")
    st.subheader("🔔 알림 설정")
    tg_token = st.text_input("Telegram Bot Token", type="password")
    tg_id = st.text_input("Telegram Chat ID")
    st.divider()
    st.subheader("⚙️ 스캔 설정")
    min_marcap_input = st.number_input("최소 시가총액 (억 원)", value=5000)
    min_marcap = min_marcap_input * 100000000
    st.divider()
    auto_refresh = st.checkbox("실시간 자동 갱신", value=False)
    interval = st.slider("갱신 주기(분)", 1, 60, 10)
    if st.button("🔄 캐시 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 적중 분석", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>실시간 포트폴리오 알림</b>\n\n", False
        for _, row in portfolio.iterrows():
            df = fetch_stock_smart(row['Code'], days=200)
            idx_df = get_hybrid_indicators(df)
            if idx_df is not None:
                st_res = get_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['pyramiding']['type']})
                if st_res['pyramiding']['alert']:
                    has_alert = True
                    alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n{st_res['pyramiding']['msg']}\n\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("평가손익", f"{int(t_eval-t_buy):,}원")
        if dash_list:
            st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', title="포트폴리오 수익 현황", template="plotly_dark"), use_container_width=True)
        if has_alert: send_telegram_msg(tg_token, tg_id, alert_msg)
    else: st.info("관리 탭에서 구글 시트를 연결하거나 종목을 추가하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel_stock = st.selectbox("진단할 종목 선택", portfolio['Name'].unique())
        code = portfolio[portfolio['Name']==sel_stock]['Code'].iloc[0]
        buy_p = portfolio[portfolio['Name']==sel_stock]['Buy_Price'].iloc[0]
        df_rep = get_hybrid_indicators(fetch_stock_smart(code))
        if df_rep is not None:
            strat = get_strategy(df_rep, buy_p)
            st.subheader(f"💼 {sel_stock} AI 진단 리포트")
            c1, c2, c3 = st.columns(3)
            c1.metric("시장 국면", strat['regime'])
            c2.metric("현재 RSI", f"{strat['rsi']:.1f}")
            c3.metric("대응 상태", strat['pyramiding']['type'])
            
            st.markdown(f"""<div style="background:#f0f7ff; padding:15px; border-radius:10px; border-left:5px solid #007bff;">
                <b>🔵 AI 추천 매수 타점:</b> 1차 {strat['buy'][0]:,}원 | 2차 {strat['buy'][1]:,}원 | 3차 {strat['buy'][2]:,}원<br>
                <b>🔴 AI 목표 매도 전술:</b> 1차 {strat['sell'][0]:,}원 | 2차 {strat['sell'][1]:,}원 | 3차 {strat['sell'][2]:,}원
            </div>""", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=df_rep.index, open=df_rep['Open'], high=df_rep['High'], low=df_rep['Low'], close=df_rep['Close'], name="주가")])
            fig.add_trace(go.Scatter(x=df_rep.index, y=df_rep['MA20'], name="20일선", line=dict(color='yellow', width=1)))
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    else: st.warning("분석할 종목이 없습니다.")

# --- [🔍 탭 2: 스캐너] ---
with tabs[2]:
    st.header("🔍 유기적 타점 발굴 스캐너")
    if st.button("🚀 네이버/KRX 통합 전수조사 가동"):
        stocks = get_krx_list()
        targets = stocks[stocks['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(50)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None and res.iloc[-1]['RSI'] < 46:
                    st_res = get_strategy(res)
                    found.append({"name": futs[f], "cp": res.iloc[-1]['Close'], "strat": st_res})
                prog.progress((i + 1) / len(targets))
        
        for d in found:
            acc_c = "#007bff" if d['strat']['regime'] == "🚀 상승" else "#dc3545"
            st.markdown(f"""<div class="scanner-card" style="border-left: 8px solid {acc_c};">
                <h3 style="margin:0; color:{acc_c};">{d['name']} <small>{d['strat']['regime']}</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px;">
                    <div class="buy-box"><b>🔵 3분할 매수</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원</div>
                    <div class="sell-box"><b>🔴 3분할 매도</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원</div>
                </div>
            </div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 적중 분석] ---
with tabs[3]:
    st.header("📈 로직 실전 적중 추적기")
    bt_name = st.text_input("분석 종목명", "삼성전자")
    if st.button("📊 시뮬레이션 시작"):
        stocks = get_krx_list()
        match = stocks[stocks['Name'] == bt_name]
        if not match.empty:
            df_bt = fetch_stock_smart(match.iloc[0]['Code'], days=500)
            if df_bt is not None:
                hits = []
                for i in range(150, len(df_bt)-5):
                    sub = df_bt.iloc[:i]; ind = get_hybrid_indicators(sub)
                    if ind is not None and ind.iloc[-1]['RSI'] < 46:
                        strat = get_strategy(ind)
                        if df_bt['Low'].iloc[i] <= strat['buy'][0]:
                            post = df_bt.loc[df_bt.index[i]:].head(22)
                            res = "익절성공" if post['High'].max() >= strat['sell'][0] else "손절발생" if post['Low'].min() <= strat['stop'] else "진행중"
                            hits.append({"날짜": df_bt.index[i], "타점": strat['buy'][0], "결과": res})
                if hits:
                    hdf = pd.DataFrame(hits)
                    st.metric("로직 승률", f"{(hdf['결과']=='익절성공').sum()/len(hdf)*100:.1f}%")
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="주가", line=dict(color='gray', width=1), opacity=0.4))
                    for h in hits:
                        color = "lime" if h['결과']=="익절성공" else "red" if h['결과']=="손절발생" else "yellow"
                        fig_t.add_trace(go.Scatter(x=[h['날짜']], y=[h['타점']], mode='markers', marker=dict(color=color, size=10, symbol='triangle-up'), name=h['결과']))
                    st.plotly_chart(fig_t, use_container_width=True)
                else: st.warning("타점 포착 데이터가 없습니다.")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    st.header("➕ 포트폴리오 관리")
    df_p = get_portfolio_gsheets()
    st.subheader("현재 등록된 종목")
    st.dataframe(df_p, use_container_width=True)
    with st.form("add_stock_form"):
        st.write("새 종목 수동 추가 (GSheets 자동 동기화)")
        c1, c2, c3 = st.columns(3)
        n_add = c1.text_input("종목명")
        p_add = c2.number_input("평단가", value=0)
        q_add = c3.number_input("수량", value=0)
        if st.form_submit_button("등록 및 업데이트"):
            krx_list = get_krx_list()
            match = krx_list[krx_list['Name']==n_add]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n_add, p_add, q_add]], columns=['Code','Name','Buy_Price','Qty'])
                # GSheets 업데이트 로직 (st-gsheets-connection 설정 필요)
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    updated_df = pd.concat([df_p, new_row], ignore_index=True)
                    conn.update(data=updated_df)
                    st.success(f"{n_add} 등록 완료!")
                    st.rerun()
                except: st.error("구글 시트 쓰기 권한이 없습니다.")
            else: st.error("종목명을 찾을 수 없습니다.")

if auto_refresh:
    time.sleep(interval * 60)
    st.rerun()
