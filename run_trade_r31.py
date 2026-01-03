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
# ⚙️ 1. 시스템 설정
# ==========================================
def get_now_kst():
    return datetime.datetime.now(timezone(timedelta(hours=9)))

st.set_page_config(page_title="AI Master V65.3.2", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #2962ff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .buy-box { background-color: #e3f2fd; padding: 10px; border-radius: 8px; border: 1px solid #90caf9; font-size: 0.85em; color: #1565c0; }
    .sell-box { background-color: #ffebee; padding: 10px; border-radius: 8px; border: 1px solid #ef9a9a; font-size: 0.85em; color: #c62828; }
    .ob-badge { background-color: #f3e5f5; color: #7b1fa2; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8em; }
    .fibo-badge { background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8em; }
    .ai-badge { background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8em; }
    .score-badge { background-color: #263238; color: white; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티] ---
@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')
        df = pd.concat([kospi, kosdaq])
        if not df.empty: return df
    except: pass
    
    fallback_data = [
        ['005930', '삼성전자'], ['000660', 'SK하이닉스'], ['373220', 'LG에너지솔루션'],
        ['207940', '삼성바이오로직스'], ['005380', '현대차'], ['000270', '기아'],
        ['005490', 'POSCO홀딩스'], ['035420', 'NAVER'], ['068270', '셀트리온'],
        ['006400', '삼성SDI'], ['051910', 'LG화학'], ['035720', '카카오'],
        ['028260', '삼성물산'], ['105560', 'KB금융'], ['012330', '현대모비스'],
        ['055550', '신한지주'], ['003670', '포스코퓨처엠'], ['032830', '삼성생명'],
        ['086790', '하나금융지주'], ['000810', '삼성화재'], ['015760', '한국전력'],
        ['034020', '두산에너빌리티'], ['017670', 'SK텔레콤'], ['018260', '삼성에스디에스'],
        ['042660', '한화오션'], ['323410', '카카오뱅크'], ['316140', '우리금융지주'],
        ['009150', '삼성전기'], ['010130', '고려아연'], ['259960', '크래프톤'],
        ['011200', 'HMM'], ['003490', '대한항공'], ['010950', 'S-Oil'],
        ['030200', 'KT'], ['009540', 'HD한국조선해양'], ['033780', 'KT&G'],
        ['012450', '한화에어로스페이스'], ['024110', '기업은행'], ['009830', '한화솔루션'],
        ['247540', '에코프로비엠'], ['086520', '에코프로'], ['028300', 'HLB'],
        ['403870', 'HPSP'], ['022100', '포스코DX'], ['005070', '코스모신소재'],
        ['035900', 'JYP Ent.'], ['041510', '에스엠'], ['196170', '알테오젠'],
        ['066970', '엘앤에프'], ['277810', '천보']
    ]
    df_fb = pd.DataFrame(fallback_data, columns=['Code', 'Name'])
    df_fb['Marcap'] = 10**15 
    return df_fb

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            rename_map = {'코드':'Code','종목코드':'Code','Code':'Code','종목명':'Name','종목':'Name','Name':'Name','평단가':'Buy_Price','매수가':'Buy_Price','Buy_Price':'Buy_Price','수량':'Qty','보유수량':'Qty','Qty':'Qty'}
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
        try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# ==========================================
# 📊 2. 지표 엔진 (CCI 수정됨)
# ==========================================
def calc_stoch(df, n, m, t):
    l, h = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    return ((df['Close'] - l) / (h - l + 1e-9) * 100).rolling(m).mean().rolling(t).mean()

def get_all_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy(); close = df['Close']
    
    # [Fix] 기본 지표 우선 계산
    df['MA20'] = close.rolling(20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # 1. 🏛️ Order Block (SMC)
    df['Is_Impulse'] = (df['Close'] > df['Open'] * 1.03) & (df['Volume'] > df['Volume'].rolling(20).mean())
    ob_price = 0
    for i in range(len(df)-2, len(df)-60, -1):
        if df['Is_Impulse'].iloc[i]:
            if df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
                ob_price = (df['Open'].iloc[i-1] + df['Low'].iloc[i-1]) / 2
                break
    df['OB'] = ob_price if ob_price > 0 else df['MA20'].iloc[-1]

    # 2. 🧬 Fibonacci
    hi_1y = df.tail(252)['High'].max()
    lo_1y = df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)

    # 3. 추가 지표들
    ma_bb1 = close.rolling(50).mean(); std_bb1 = close.rolling(50).std()
    df['BB1_Up'] = ma_bb1 + (std_bb1 * 0.5); df['BB1_Lo'] = ma_bb1 - (std_bb1 * 0.5)
    
    df['SNOW_L'] = calc_stoch(df, 20, 12, 12)
    delta = close.diff(); g = delta.where(delta>0,0).rolling(14).mean(); l = -delta.where(delta<0,0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/(l+1e-9))))
    
    exp1 = close.ewm(span=12, adjust=False).mean(); exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD_Osc'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    # [FIX] CCI Calculation (mad() 함수 제거 및 대체)
    tp = (df['High'] + df['Low'] + close) / 3
    # mad() 대신 (x - x.mean()).abs().mean() 사용
    mad = tp.rolling(14).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (tp - tp.rolling(14).mean()) / (0.015 * mad + 1e-9)
    
    raw_mf = tp * df['Volume']
    pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-9))))
    
    tr = df['ATR']; dm_pos = df['High'].diff().clip(lower=0); dm_neg = -df['Low'].diff().clip(upper=0)
    di_pos = 100 * (dm_pos.ewm(alpha=1/14).mean() / tr); di_neg = 100 * (dm_neg.ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (100 * abs(di_pos - di_neg) / (di_pos + di_neg + 1e-9)).rolling(14).mean()

    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-9)

    return df

# ==========================================
# 🧠 3. 전략 엔진
# ==========================================
def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp = curr['Close']; atr = curr['ATR']
    
    # AI ML
    data_ml = df.copy()[['RSI','SNOW_L','CCI','MFI','ADX','Vol_Z']].dropna()
    ai_prob = 50
    if len(data_ml) > 60:
        try:
            model = RandomForestClassifier(n_estimators=40, random_state=42)
            train_df = data_ml.iloc[:-1]; train_df['Target'] = (data_ml['Close'].shift(-1).iloc[:-1] > train_df['Close']).astype(int)
            model.fit(train_df.tail(150), train_df['Target'])
            ai_prob = int(model.predict_proba(data_ml.iloc[-1:])[0][1] * 100)
        except: pass

    # Tuning
    vol = atr / cp if cp > 0 else 0
    tune = {'rsi': 30, 'snow': 28, 'mode': '🛡️ 보수'} if vol > 0.04 else {'rsi': 50, 'snow': 45, 'mode': '⚡ 공격'} if vol < 0.015 else {'rsi': 40, 'snow': 35, 'mode': '⚖️ 균형'}

    def adj(p):
        if np.isnan(p) or p <= 0: return 0
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)

    # 3분할 타점
    candidates = [
        (adj(curr['POC']), "POC"),
        (adj(curr['OB']), "OB"),
        (adj(curr['Fibo_618']), "Fibo"),
        (adj(curr['BB1_Lo']), "BB")
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    valid_buys = [x for x in candidates if x[0] <= cp]
    
    final_buys = []
    if not valid_buys: final_buys = [adj(cp), adj(cp*0.95), adj(cp*0.90)]
    elif len(valid_buys) == 1: final_buys = [valid_buys[0][0], adj(valid_buys[0][0]*0.95), adj(valid_buys[0][0]*0.90)]
    elif len(valid_buys) == 2: final_buys = [valid_buys[0][0], valid_buys[1][0], adj(valid_buys[1][0]*0.95)]
    else: final_buys = [valid_buys[0][0], valid_buys[1][0], valid_buys[2][0]]

    sell_pts = [adj(curr['BB1_Up']), adj(cp + atr*3), adj(cp + atr*5)]
    
    # 점수
    score = 0
    if curr['SNOW_L'] < tune['snow']: score += 15
    if curr['RSI'] < tune['rsi']: score += 10
    if curr['MFI'] < 20: score += 15
    if cp <= curr['OB'] * 1.05: score += 15 
    if cp <= curr['Fibo_618'] * 1.05: score += 15 
    score += (ai_prob * 0.4)

    status = {"type": "💤 관망", "color": "#6c757d", "msg": "대기", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0]: status = {"type": "💰 익절", "color": "#28a745", "msg": "수익권", "alert": True}
        elif y < -3 and score >= 45: status = {"type": "❄️ 스노우", "color": "#00d2ff", "msg": "추매(SMC)", "alert": True}
        elif y > 2 and ai_prob > 60: status = {"type": "🔥 불타기", "color": "#ff4b4b", "msg": "추세가속", "alert": True}

    return {"buy": final_buys, "sell": sell_pts, "score": int(score), "status": status, "ai": ai_prob, "tune": tune, "ob": curr['OB'], "fibo": curr['Fibo_618'], "poc": curr['POC']}

# ==========================================
# 🖥️ 4. 메인 UI
# ==========================================
with st.sidebar:
    st.title("🏛️ V65.3.2 Fix")
    now = get_now_kst()
    st.info(f"KST: {now.strftime('%H:%M:%S')}")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    min_m = st.number_input("최소 시총(억)", value=3000) * 100000000
    auto_report = st.checkbox("16시 마감 리포트", value=True)
    
    if auto_report and now.hour == 16 and now.minute == 0:
        pf_rep = get_portfolio_gsheets()
        if not pf_rep.empty:
            msg = "🔔 <b>[16시 마감 리포트]</b>\n"
            for _, r in pf_rep.iterrows():
                try:
                    d = fdr.DataReader(r['Code'], (now-timedelta(days=5)).strftime('%Y-%m-%d'))
                    p = d['Close'].iloc[-1]; pct = (p-r['Buy_Price'])/r['Buy_Price']*100
                    msg += f"{r['Name']}: {pct:+.2f}%\n"
                except: pass
            send_telegram_msg(tg_token, tg_id, msg)

tabs = st.tabs(["📊 대시보드", "🔍 스캐너", "💼 AI 리포트", "📈 백테스트", "➕ 관리"])

with tabs[0]: # 대시보드
    pf = get_portfolio_gsheets()
    if not pf.empty:
        t_buy, t_eval, dash_list = 0, 0, []
        for _, row in pf.iterrows():
            df = get_all_indicators(fdr.DataReader(row['Code'], (get_now_kst()-timedelta(days=200)).strftime('%Y-%m-%d')))
            if df is not None:
                res = get_strategy(df, row['Buy_Price'])
                cp = df['Close'].iloc[-1]; t_buy += (row['Buy_Price']*row['Qty']); t_eval += (cp*row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)

with tabs[1]: # 스캐너
    if st.button("🚀 SMC + Fibo 전수조사"):
        krx = get_safe_stock_listing(); targets = krx[krx['Marcap'] >= min_m].sort_values('Marcap', ascending=False).head(50)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(get_all_indicators, fdr.DataReader(r['Code'], (get_now_kst()-timedelta(days=250)).strftime('%Y-%m-%d'))): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    s = get_strategy(res)
                    found.append({"name": futs[f], "score": s['score'], "strat": s})
                prog.progress((i+1)/len(targets))
        
        for d in sorted(found, key=lambda x: x['score'], reverse=True)[:15]:
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0;">{d['name']}</h3>
                        <div>
                            <span class="ob-badge">OB: {d['strat']['ob']:,}</span>
                            <span class="fibo-badge">Fibo: {d['strat']['fibo']:,}</span>
                            <span class="ai-badge">AI: {d['strat']['ai']}%</span>
                        </div>
                    </div>
                    <p style="font-size:0.8em; color:#555; margin:5px 0;">Total Score: <span class="score-badge">{d['score']}</span></p>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div class="buy-box"><b>🔵 3분할 매수</b><br>1차: {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원<br>3차: {d['strat']['buy'][2]:,}원</div>
                        <div class="sell-box"><b>🔴 3분할 매도</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원</div>
                    </div>
                </div>""", unsafe_allow_html=True)

with tabs[2]: # AI 리포트
    if not pf.empty:
        sel = st.selectbox("종목 선택", pf['Name'].unique())
        row = pf[pf['Name'] == sel].iloc[0]
        df_ai = get_all_indicators(fdr.DataReader(row['Code'], (get_now_kst()-timedelta(days=365)).strftime('%Y-%m-%d')))
        if df_ai is not None:
            res = get_strategy(df_ai, row['Buy_Price'])
            st.markdown(f"""<div class="metric-card" style="border-left:10px solid {res['status']['color']};">
                <h2>{sel} <span class="ai-badge">AI승률: {res['ai']}%</span></h2>
                <p>{res['status']['msg']} (OB: {res['ob']:,}원 / Fibo: {res['fibo']:,}원)</p></div>""", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], close=df_ai['Close'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:])])
            fig.add_hline(y=res['ob'], line_color="purple", line_width=2, line_dash="dash", annotation_text="Order Block")
            fig.add_hline(y=res['fibo'], line_color="green", line_width=2, line_dash="dot", annotation_text="Fibo 0.618")
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]: # 백테스트
    bt_name = st.text_input("백테스트 종목", "삼성전자")
    if st.button("검증 실행"):
        krx = get_safe_stock_listing(); m = krx[krx['Name'] == bt_name]
        if not m.empty:
            df_bt = get_all_indicators(fdr.DataReader(m.iloc[0]['Code'], (get_now_kst()-timedelta(days=730)).strftime('%Y-%m-%d')))
            if df_bt is not None:
                cash, stocks, equity = 10000000, 0, []
                for i in range(120, len(df_bt)):
                    curr = df_bt.iloc[:i+1]; s_res = get_strategy(curr); cp = df_bt.iloc[i]['Close']
                    if stocks == 0 and s_res['score'] >= 50: stocks = cash // cp; cash -= (stocks * cp)
                    elif stocks > 0 and cp >= s_res['sell'][0]: cash += (stocks * cp); stocks = 0
                    equity.append(cash + (stocks * cp))
                st.plotly_chart(px.line(pd.DataFrame(equity, columns=['total']), y='total', title=f"{bt_name} 자산 성장"))

with tabs[4]: # 관리
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            krx = get_safe_stock_listing(); m = krx[krx['Name']==n]
            if not m.empty:
                new = pd.DataFrame([[m.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)
