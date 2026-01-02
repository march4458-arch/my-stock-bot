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

st.set_page_config(page_title="AI Ultimate Master V64.9.6", page_icon="🛡️", layout="wide")

# UI 전문 디자인 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 5px solid #7b1fa2; }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; background-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; font-size: 0.9em; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; font-size: 0.9em; }
    .ai-label { background-color: #f3e5f5; color: #7b1fa2; padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8em; }
    .status-badge { padding: 3px 8px; border-radius: 5px; color: white; font-weight: bold; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# --- [🛠️ 유틸리티: 데이터 연동 및 알림] ---
@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    """KRX 서버 에러 방지용 안전 리스팅"""
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty: return df
    except: pass
    # 백업용 우량주 리스트
    fallback = [['005930', '삼성전자'], ['000660', 'SK하이닉스'], ['005380', '현대차'], 
                ['005490', 'POSCO홀딩스'], ['035420', 'NAVER'], ['000270', '기아']]
    return pd.DataFrame(fallback, columns=['Code', 'Name']).assign(Marcap=10**14)

def get_portfolio_gsheets():
    """구글 시트 컬럼 자동 매핑 및 보정"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            rename_map = {
                '코드': 'Code', '종목코드': 'Code', 'Code': 'Code', 'code': 'Code',
                '종목명': 'Name', '종목': 'Name', 'Name': 'Name', 'name': 'Name',
                '평단가': 'Buy_Price', '매수가': 'Buy_Price', 'Buy_Price': 'Buy_Price', 'buy_price': 'Buy_Price',
                '수량': 'Qty', '보유수량': 'Qty', 'Qty': 'Qty', 'qty': 'Qty'
            }
            df = df.rename(columns=rename_map)
            if 'Code' in df.columns:
                df = df.dropna(subset=['Code'])
                df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
                df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0)
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except: pass
    return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 📊 2. 분석 엔진 (지표 계산 + AI 학습 + 전략)
# ==========================================
def calc_stoch(df, n, m, t):
    l, h = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    return ((df['Close'] - l) / (h - l + 1e-9) * 100).rolling(m).mean().rolling(t).mean()

def get_all_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy(); close = df['Close']
    
    # 기본 지표
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['MA20'], df['MA120'] = close.rolling(20).mean(), close.rolling(120).mean()
    
    # ❄️ Snow 파동 (Stochastic)
    df['SNOW_S'], df['SNOW_M'], df['SNOW_L'] = calc_stoch(df,5,3,3), calc_stoch(df,10,6,6), calc_stoch(df,20,12,12)
    
    # RSI & MACD (가짜신호 보정용)
    delta = close.diff(); g = delta.where(delta>0,0).rolling(14).mean(); l = -delta.where(delta<0,0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/(l+1e-9))))
    exp1 = close.ewm(span=12, adjust=False).mean(); exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2; df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Osc'] = df['MACD'] - df['MACD_Sig']
    
    # 지지선: POC(매물대), Fibo 618, OB(세력선)
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    ob_zones = [df['Low'].iloc[i] for i in range(len(df)-40, len(df)) if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025]
    df['OB'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    df['Vol_Z'] = (df['Volume']-df['Volume'].rolling(20).mean())/df['Volume'].rolling(20).std()
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr, prev = df.iloc[-1], df.iloc[-2]; cp, atr = curr['Close'], curr['ATR']
    
    # 🧠 AI ML 예측 (Random Forest)
    data_ml = df.copy().dropna()
    features = ['RSI', 'SNOW_S', 'SNOW_M', 'SNOW_L', 'Vol_Z', 'MACD_Osc']
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    # Target: 내일 종가가 오늘보다 높은가?
    data_ml['Target'] = (data_ml['Close'].shift(-1) > data_ml['Close']).astype(int)
    # 최근 150일 데이터 학습
    train = data_ml.tail(150)
    model.fit(train[features], train['Target'])
    ai_prob = int(model.predict_proba(df[features].iloc[-1:])[0][1] * 100)
    
    # ⚙️ Self-Tuning (변동성 대응)
    vol = atr / cp
    tune = {'rsi': 28, 'snow': 25, 'mode': '🛡️ 보수'} if vol > 0.04 else {'rsi': 45, 'snow': 40, 'mode': '⚡ 공격'} if vol < 0.015 else {'rsi': 35, 'snow': 30, 'mode': '⚖️ 균형'}

    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)
    
    # 유기적 3분할 타점
    buy_pts = sorted([adj(curr['POC']), adj(curr['Fibo_618']), adj(curr['OB'])], reverse=True)
    sell_pts = [adj(cp + atr*2.2), adj(cp + atr*3.8), adj(cp + atr*5.5)]
    
    # 최종 점수 산출
    score = (20 if curr['SNOW_L'] < tune['snow'] else 0) + \
            (20 if curr['RSI'] < tune['rsi'] else 0) + \
            (20 if curr['MACD_Osc'] > prev['MACD_Osc'] else 0) + \
            (ai_prob * 0.4)
    
    status = {"type": "💤 관망", "color": "#6c757d", "msg": "진입 대기", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0]: status = {"type": "💰 익절", "color": "#28a745", "msg": f"{y:.1f}% 수익권", "alert": True}
        elif cp <= buy_pts[2] * 0.93: status = {"type": "⚠️ 손절", "color": "#dc3545", "msg": "리스크 관리", "alert": True}
        elif y < -3 and score >= 50: status = {"type": "❄️ 스노우", "color": "#00d2ff", "msg": "지능 물타기", "alert": True}
        elif y > 2 and ai_prob > 65: status = {"type": "🔥 불타기", "color": "#ff4b4b", "msg": "추세 가속 불타기", "alert": True}
    
    return {"buy": buy_pts, "sell": sell_pts, "score": int(score), "status": status, "ai": ai_prob, "tune": tune, "poc": curr['POC'], "fibo": curr['Fibo_618'], "ob": curr['OB']}

# ==========================================
# 🖥️ 3. 메인 UI (사이드바 및 탭 구성)
# ==========================================
with st.sidebar:
    st.title("🛡️ Ultimate V64.9.6")
    now = get_now_kst()
    st.info(f"KST: {now.strftime('%H:%M:%S')}")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    st.divider()
    min_m = st.number_input("최소 시총(억)", value=5000) * 100000000
    auto_report = st.checkbox("16시 마감 리포트 발송", value=True)
    
    # [16시 자동 알림 로직]
    if auto_report and now.hour == 16 and now.minute == 0:
        st.toast("16시 마감 리포트 생성 중...")
        # (실제 배포 시 중복 발송 방지를 위해 세션 스테이트 활용 권장)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        t_buy, t_eval, dash_list, alert_msg = 0, 0, [], ""
        for _, row in portfolio.iterrows():
            df = get_all_indicators(fdr.DataReader(row['Code'], (now-timedelta(days=200)).strftime('%Y-%m-%d')))
            if df is not None:
                res = get_strategy(df, row['Buy_Price'])
                cp = df['Close'].iloc[-1]; t_buy += (row['Buy_Price']*row['Qty']); t_eval += (cp*row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
                if res['status']['alert']: alert_msg += f"[{res['status']['type']}] {row['Name']}: {res['status']['msg']}\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)
        if alert_msg: send_telegram_msg(tg_token, tg_id, f"❄️ <b>실시간 포트폴리오</b>\n\n{alert_msg}")

# --- [💼 탭 1: AI 리포트 (완벽 복구)] ---
with tabs[1]:
    if not portfolio.empty:
        sel_stock = st.selectbox("정밀 진단할 종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel_stock].iloc[0]
        df_ai = get_all_indicators(fdr.DataReader(row['Code'], (now-timedelta(days=365)).strftime('%Y-%m-%d')))
        
        if df_ai is not None:
            res = get_strategy(df_ai, row['Buy_Price'])
            
            # 헤더 정보
            st.markdown(f"""
                <div class="metric-card" style="border-left:10px solid {res['status']['color']};">
                    <h2 style="margin:0;">{sel_stock} <span class="ai-label">AI 신뢰도: {res['ai']}%</span></h2>
                    <p style="font-size:1.1em; color:#555; margin-top:8px;">
                        <span class="status-badge" style="background-color:{res['status']['color']};">{res['status']['type']}</span>
                        {res['status']['msg']} (모드: {res['tune']['mode']})
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # 3분할 가격표
            c_b, c_s = st.columns(2)
            with c_b: st.markdown(f'<div class="buy-box"><b>🔵 유기적 3분할 매수</b><br>1차(POC): {res["buy"][0]:,}원<br>2차(Fibo): {res["buy"][1]:,}원<br>3차(OB): {res["buy"][2]:,}원</div>', unsafe_allow_html=True)
            with c_s: st.markdown(f'<div class="sell-box"><b>🔴 유기적 3분할 매도</b><br>1차: {res["sell"][0]:,}원<br>2차: {res["sell"][1]:,}원<br>3차: {res["sell"][2]:,}원</div>', unsafe_allow_html=True)
            
            # 메인 차트 (지지선 포함)
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:], close=df_ai['Close'][-100:], name="Candle")])
            fig.add_hline(y=res['poc'], line_color="orange", annotation_text="POC(매물대)")
            fig.add_hline(y=res['fibo'], line_color="green", line_dash="dot", annotation_text="Fibo 618")
            fig.add_hline(y=res['ob'], line_color="purple", line_dash="dash", annotation_text="OB(세력선)")
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False, title=f"{sel_stock} 기술적 지지선 분석")
            st.plotly_chart(fig, use_container_width=True)

            
            
            # 서브 차트 (Snow + MACD)
            st.subheader("❄️ Snow Waves & MACD Momentum")
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Scatter(x=df_ai.index[-60:], y=df_ai['SNOW_L'][-60:], name="대파동(20-12-12)", line=dict(color='blue', width=2)))
            fig_sub.add_trace(go.Scatter(x=df_ai.index[-60:], y=df_ai['SNOW_M'][-60:], name="중파동(10-6-6)", line=dict(color='orange', width=1.5)))
            fig_sub.add_trace(go.Bar(x=df_ai.index[-60:], y=df_ai['MACD_Osc'][-60:], name="MACD Osc", marker_color='red', opacity=0.3))
            fig_sub.add_hline(y=20, line_dash="dot", line_color="gray")
            fig_sub.add_hline(y=80, line_dash="dot", line_color="gray")
            fig_sub.update_layout(height=300, template="plotly_white", margin=dict(t=30, b=20))
            st.plotly_chart(fig_sub, use_container_width=True)

# --- [🔍 탭 2: 스캐너 (3분할 가격 노출)] ---
with tabs[2]:
    if st.button("🚀 AI-Self Tuning 전수조사 (상위 50)"):
        krx = get_safe_stock_listing()
        targets = krx[krx['Marcap'] >= min_m].sort_values('Marcap', ascending=False).head(50)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(get_all_indicators, fdr.DataReader(r['Code'], (now-timedelta(days=200)).strftime('%Y-%m-%d'))): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    s = get_strategy(res)
                    found.append({"name": futs[f], "score": s['score'], "strat": s})
                prog.progress((i+1)/len(targets))
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:15]:
            st.markdown(f"""<div class="scanner-card">
                <h3>{d['name']} <span class="ai-label">Total Score: {d['score']}</span></h3>
                <p>AI확률: {d['strat']['ai']}% | 모드: {d['strat']['tune']['mode']}</p>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div class="buy-box"><b>🔵 매수 타점</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원<br>3차: {d['strat']['buy'][2]:,}원</div>
                    <div class="sell-box"><b>🔴 매도 타점</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원<br>3차: {d['strat']['sell'][2]:,}원</div>
                </div></div>""", unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트 (복구)] ---
with tabs[3]:
    st.subheader("📊 전략 과거 검증")
    bt_name = st.text_input("백테스트 종목명", "삼성전자")
    if st.button("검증 실행"):
        krx = get_safe_stock_listing(); m = krx[krx['Name'] == bt_name]
        if not m.empty:
            df_bt = get_all_indicators(fdr.DataReader(m.iloc[0]['Code'], (now-timedelta(days=730)).strftime('%Y-%m-%d')))
            if df_bt is not None:
                cash, stocks, equity = 10000000, 0, []
                for i in range(120, len(df_bt)):
                    curr_bt = df_bt.iloc[:i+1]; s_res = get_strategy(curr_bt); cp = df_bt.iloc[i]['Close']
                    if stocks == 0 and s_res['score'] >= 55: # 매수 조건
                        stocks = cash // cp; cash -= (stocks * cp)
                    elif stocks > 0 and cp >= s_res['sell'][0]: # 매도 조건
                        cash += (stocks * cp); stocks = 0
                    equity.append(cash + (stocks * cp))
                st.plotly_chart(px.line(pd.DataFrame(equity, columns=['total']), y='total', title=f"{bt_name} 자산 성장"))

# --- [➕ 탭 4: 관리 (복구)] ---
with tabs[4]:
    st.subheader("➕ 포트폴리오 관리 (구글 시트)")
    df_p = get_portfolio_gsheets()
    with st.form("add_p"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            krx = get_safe_stock_listing(); m = krx[krx['Name']==n]
            if not m.empty:
                new_row = pd.DataFrame([[m.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new_row], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)
