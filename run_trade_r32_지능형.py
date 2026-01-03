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

st.set_page_config(page_title="AI Master V67.2 Ultimate", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #6200ea; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    
    .buy-box { background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #90caf9; color: #0d47a1; margin-bottom: 10px; }
    .sell-box { background-color: #ffebee; padding: 15px; border-radius: 10px; border: 1px solid #ef9a9a; color: #b71c1c; margin-bottom: 10px; }
    .avg-text { font-weight: bold; color: #4a148c; text-align: center; background-color: #f3e5f5; padding: 5px; border-radius: 5px; margin-top: 5px; }
    
    .price-tag { font-weight: bold; font-size: 1.1em; }
    .current-price { font-size: 1.5em; font-weight: bold; color: #333; }
    .logic-tag { font-size: 0.8em; color: #555; background-color: rgba(255,255,255,0.7); padding: 2px 5px; border-radius: 4px; margin-left: 5px; }
    .mode-badge { background-color: #263238; color: #00e676; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .ai-badge { background-color: #6200ea; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .win-badge { background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 6px; font-weight: bold; }
    
    /* 실시간 시계 스타일 */
    .clock-box { font-size: 1.2em; font-weight: bold; color: #333; text-align: center; margin-bottom: 10px; padding: 10px; background: #e0f7fa; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- [Data Loader] ---
@st.cache_data(ttl=3600)
def get_data_safe(code, days=365):
    start_date = (get_now_kst() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty: return df
    except: pass
    try:
        df = yf.download(f"{code}.KS", start=start_date, progress=False)
        if not df.empty: return df
        df = yf.download(f"{code}.KQ", start=start_date, progress=False)
        if not df.empty: return df
    except: pass
    return None

@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        kospi = fdr.StockListing('KOSPI'); kosdaq = fdr.StockListing('KOSDAQ')
        df = pd.concat([kospi, kosdaq])
        if not df.empty: return df
    except: pass
    fb = [['005930','삼성전자'],['000660','SK하이닉스'],['373220','LG에너지솔루션'],['207940','삼성바이오로직스'],['005380','현대차'],['000270','기아'],['005490','POSCO홀딩스'],['035420','NAVER'],['035720','카카오'],['006400','삼성SDI'],['051910','LG화학'],['003670','포스코퓨처엠'],['028260','삼성물산'],['105560','KB금융'],['055550','신한지주'],['086520','에코프로'],['247540','에코프로비엠'],['042660','한화오션'],['010130','고려아연'],['034020','두산에너빌리티']]
    return pd.DataFrame(fb, columns=['Code','Name']).assign(Marcap=10**15)

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            rename_map = {'코드':'Code','종목코드':'Code','Code':'Code','종목명':'Name','Name':'Name','평단가':'Buy_Price','Buy_Price':'Buy_Price','수량':'Qty','Qty':'Qty'}
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
# 📊 2. 지표 엔진
# ==========================================
def calc_stoch(df, n, m, t):
    l, h = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    return ((df['Close'] - l) / (h - l + 1e-9) * 100).rolling(m).mean().rolling(t).mean()

def get_all_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    close = df['Close']
    
    df['MA20'] = close.rolling(20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    df['Is_Impulse'] = (df['Close'] > df['Open'] * 1.03) & (df['Volume'] > df['Volume'].rolling(20).mean())
    ob_price = 0
    for i in range(len(df)-2, len(df)-60, -1):
        if df['Is_Impulse'].iloc[i]:
            if df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
                ob_price = (df['Open'].iloc[i-1] + df['Low'].iloc[i-1]) / 2
                break
    df['OB'] = ob_price if ob_price > 0 else df['MA20'].iloc[-1]

    hi_1y = df.tail(252)['High'].max(); lo_1y = df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)

    ma_bb1 = close.rolling(50).mean(); std_bb1 = close.rolling(50).std()
    df['BB1_Up'] = ma_bb1 + (std_bb1 * 0.5); df['BB1_Lo'] = ma_bb1 - (std_bb1 * 0.5)
    
    df['SNOW_L'] = calc_stoch(df, 20, 12, 12)
    delta = close.diff(); g = delta.where(delta>0,0).rolling(14).mean(); l = -delta.where(delta<0,0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/(l+1e-9))))
    
    exp1 = close.ewm(span=12, adjust=False).mean(); exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD_Osc'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

    tp = (df['High'] + df['Low'] + close) / 3
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
# 🧠 3. Darwin 전략
# ==========================================
def get_darwin_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp = curr['Close']; atr = curr['ATR']
    
    df['BB_Dist'] = (curr['BB1_Lo'] - cp) / cp 
    data_ml = df.copy()[['RSI','SNOW_L','CCI','MFI','ADX','Vol_Z', 'BB_Dist']].dropna()
    features = ['RSI','SNOW_L','CCI','MFI','ADX','Vol_Z', 'BB_Dist']
    
    ai_prob = 50; logic_mode = "⚖️ Balanced"; top_feature = "None"
    if len(data_ml) > 60:
        try:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            train_df = data_ml.iloc[:-1]; train_df['Target'] = (data_ml['Close'].shift(-1).iloc[:-1] > train_df['Close']).astype(int)
            model.fit(train_df.tail(150)[features], train_df.tail(150)['Target'])
            top_feature = features[np.argmax(model.feature_importances_)]
            
            if top_feature == 'ADX': logic_mode = "🔥 Trend Mode"
            elif top_feature in ['CCI', 'RSI', 'SNOW_L']: logic_mode = "🌊 Cycle Mode"
            elif top_feature in ['Vol_Z', 'MFI']: logic_mode = "🏛️ Whale Mode"
            elif top_feature == 'BB_Dist': logic_mode = "🛡️ Defense Mode"
            ai_prob = int(model.predict_proba(data_ml[features].iloc[-1:])[0][1] * 100)
        except: pass

    score = 0
    if (curr['SNOW_L'] < 30) or (curr['RSI'] < 35): score += 10
    
    if logic_mode == "🔥 Trend Mode":
        if curr['ADX'] > 20: score += 10
        if cp <= curr['BB1_Lo'] * 1.02: score += 30
        if cp <= curr['Fibo_618'] * 1.02: score += 20
    elif logic_mode == "🌊 Cycle Mode":
        if curr['CCI'] < -100: score += 30
        if curr['RSI'] < 30: score += 20
    elif logic_mode == "🏛️ Whale Mode":
        if cp <= curr['OB'] * 1.05: score += 40
        if curr['MFI'] < 20: score += 10
    elif logic_mode == "🛡️ Defense Mode":
        if cp <= curr['BB1_Lo']: score += 30
        if cp <= curr['POC']: score += 20
    else:
        if cp <= curr['OB']: score += 15
        if curr['CCI'] < -100: score += 15
    score += (ai_prob * 0.4)

    def adj(p):
        if np.isnan(p) or p <= 0: return 0
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)
    
    candidates = [
        (adj(curr['POC']), "POC"),
        (adj(curr['OB']), "OB"),
        (adj(curr['Fibo_618']), "Fibo"),
        (adj(curr['BB1_Lo']), "BB")
    ]
    
    if logic_mode == "🔥 Trend Mode": candidates.sort(key=lambda x: (x[1] != 'BB', x[1] != 'Fibo', -x[0]))
    elif logic_mode == "🏛️ Whale Mode": candidates.sort(key=lambda x: (x[1] != 'OB', x[1] != 'POC', -x[0]))
    else: candidates.sort(key=lambda x: x[0], reverse=True)
    
    valid_buys = [x for x in candidates if x[0] <= cp]
    
    final_buys = []
    if not valid_buys: 
        final_buys = [(adj(cp), "현재가"), (adj(cp*0.95), "-5%"), (adj(cp*0.90), "-10%")]
    elif len(valid_buys) == 1:
        final_buys = [valid_buys[0], (adj(valid_buys[0][0]*0.95), "Supp-5%"), (adj(valid_buys[0][0]*0.90), "Supp-10%")]
    elif len(valid_buys) == 2:
        final_buys = [valid_buys[0], valid_buys[1], (adj(valid_buys[1][0]*0.95), "2nd-5%")]
    else:
        final_buys = valid_buys[:3]

    est_avg_price = int(sum([p[0] for p in final_buys]) / 3)
    sell_pts = [(adj(curr['BB1_Up']), "BB상단"), (adj(cp + atr*3), "ATR x3"), (adj(cp + atr*5), "ATR x5")]
    
    status = {"type": "💤 관망", "color": "#6c757d", "msg": "대기", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0][0]: status = {"type": "💰 익절", "color": "#28a745", "msg": "수익권", "alert": True}
        elif y < -3 and score >= 45: status = {"type": "❄️ 물타기", "color": "#00d2ff", "msg": f"추매({logic_mode})", "alert": True}
        elif y > 2 and ai_prob > 60: status = {"type": "🔥 불타기", "color": "#ff4b4b", "msg": "추세추종", "alert": True}

    return {"buy": final_buys, "sell": sell_pts, "avg": est_avg_price, "score": int(score), "status": status, "ai": ai_prob, "logic": logic_mode, "top_feat": top_feature, "ob": curr['OB']}

# ==========================================
# 🖥️ 4. 메인 UI (Integrated)
# ==========================================
with st.sidebar:
    # 실시간 시계
    now = get_now_kst()
    st.markdown(f'<div class="clock-box">⏰ {now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    st.title("📡 V67.2 Ultimate")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    
    scanner_alert = st.checkbox("📢 스캔 결과 자동 전송", value=True)
    auto_report = st.checkbox("16시 마감 리포트", value=True)
    min_m = st.number_input("최소 시총(억)", value=3000) * 100000000
    
    if auto_report and now.hour == 16 and now.minute == 0:
        pf_rep = get_portfolio_gsheets()
        if not pf_rep.empty:
            msg = "🔔 <b>[16시 마감 리포트]</b>\n"
            for _, r in pf_rep.iterrows():
                d = get_data_safe(r['Code'], days=5)
                if d is not None:
                    p = d['Close'].iloc[-1]; pct = (p-r['Buy_Price'])/r['Buy_Price']*100
                    msg += f"{r['Name']}: {pct:+.2f}%\n"
            send_telegram_msg(tg_token, tg_id, msg)

tabs = st.tabs(["📊 대시보드", "🔍 스캐너", "🧬 진화 검증", "💼 AI 리포트", "➕ 관리"])

with tabs[0]: # 대시보드
    pf = get_portfolio_gsheets()
    if not pf.empty:
        t_buy, t_eval, dash_list = 0, 0, []
        for _, row in pf.iterrows():
            df = get_all_indicators(get_data_safe(row['Code'], days=200))
            if df is not None:
                res = get_darwin_strategy(df, row['Buy_Price'])
                cp = df['Close'].iloc[-1]; t_buy += (row['Buy_Price']*row['Qty']); t_eval += (cp*row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)

with tabs[1]: # 스캐너 (Telegram Alert)
    if st.button("🧬 Darwin Evolution 스캔"):
        krx = get_safe_stock_listing(); targets = krx[krx['Marcap'] >= min_m].sort_values('Marcap', ascending=False).head(50)
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(get_all_indicators, get_data_safe(r['Code'], days=250)): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    s = get_darwin_strategy(res)
                    cp = res['Close'].iloc[-1]
                    found.append({"name": futs[f], "score": s['score'], "strat": s, "cp": cp})
                prog.progress((i+1)/len(targets))
        
        top_picks = sorted(found, key=lambda x: x['score'], reverse=True)[:15]
        
        # 텔레그램 자동 전송
        if scanner_alert and top_picks and tg_token and tg_id:
            msg = f"🚀 <b>[AI 스캔 Top 5]</b> ({now.strftime('%H:%M')})\n\n"
            for item in top_picks[:5]:
                s = item['strat']
                msg += f"<b>{item['name']}</b> ({s['logic']})\n💰 {item['cp']:,}원 / 🎯 {s['buy'][0][0]:,}원\n🏆 {s['score']}점 (AI:{s['ai']}%)\n\n"
            send_telegram_msg(tg_token, tg_id, msg)
            st.toast("📨 텔레그램 전송 완료!")

        for d in top_picks:
            s = d['strat']
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div><h3 style="margin:0;">{d['name']}</h3><span class="current-price">{d['cp']:,}원</span></div>
                        <div style="text-align:right;"><span class="mode-badge">{s['logic']}</span> <span class="ai-badge">AI: {s['ai']}%</span><br><span style="font-size:0.8em; color:#666;">Score: {d['score']}</span></div>
                    </div>
                    <div style="margin: 15px 0; display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div class="buy-box">
                            <b>🔵 3분할 매수</b><br>
                            1차: <b>{s['buy'][0][0]:,}원</b> <span class="logic-tag">{s['buy'][0][1]}</span><br>
                            2차: <b>{s['buy'][1][0]:,}원</b> <span class="logic-tag">{s['buy'][1][1]}</span><br>
                            3차: <b>{s['buy'][2][0]:,}원</b> <span class="logic-tag">{s['buy'][2][1]}</span>
                            <div class="avg-text">예상 평단: {s['avg']:,}원</div>
                        </div>
                        <div class="sell-box">
                            <b>🔴 3분할 매도</b><br>
                            1차: {s['sell'][0][0]:,}원 <span class="logic-tag">{s['sell'][0][1]}</span><br>
                            2차: {s['sell'][1][0]:,}원 <span class="logic-tag">{s['sell'][1][1]}</span><br>
                            3차: {s['sell'][2][0]:,}원 <span class="logic-tag">{s['sell'][2][1]}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

with tabs[2]: # 🧬 진화 검증 (Time Machine)
    st.subheader("🧬 Darwin 진화 성적표 (Time Machine)")
    st.info("💡 과거 데이터를 타임머신으로 분석하여 AI의 실력을 검증합니다.")
    
    if st.button("🚀 과거 데이터 검증 시작"):
        pf = get_portfolio_gsheets()
        sample_codes = pf['Code'].tolist() if not pf.empty else []
        fb = get_safe_stock_listing().head(5)['Code'].tolist()
        targets = list(set(sample_codes + fb))[:10]
        
        results = []
        prog = st.progress(0)
        
        for idx, code in enumerate(targets):
            full_df = get_data_safe(code, days=365)
            if full_df is not None and len(full_df) > 150:
                for i in range(24, 0, -1): # 24주 전부터 검사
                    past_date_idx = - (i * 5)
                    if abs(past_date_idx) < len(full_df) - 60:
                        past_df = full_df.iloc[:past_date_idx]
                        future_df = full_df.iloc[past_date_idx:]
                        
                        if len(future_df) >= 5:
                            res = get_darwin_strategy(past_df)
                            if res['score'] >= 50:
                                entry = past_df['Close'].iloc[-1]
                                exit_p = future_df['Close'].iloc[4]
                                results.append({"Date": past_df.index[-1], "Win": 1 if exit_p > entry else 0, "Count": 1})
            prog.progress((idx+1)/len(targets))
            
        if results:
            df_res = pd.DataFrame(results).sort_values('Date')
            df_res['Win_Rate'] = (df_res['Win'].cumsum() / df_res['Count'].cumsum() * 100)
            
            c1, c2 = st.columns(2)
            c1.metric("총 시그널", f"{len(df_res)}회")
            c2.metric("누적 승률", f"{df_res['Win_Rate'].iloc[-1]:.1f}%")
            
            fig = px.line(df_res, x='Date', y='Win_Rate', title="AI 승률 변화 추이", markers=True)
            fig.add_hline(y=50, line_dash="dot", line_color="gray")
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("데이터 부족")

with tabs[3]: # AI 리포트 (수동 전송)
    if not pf.empty:
        sel = st.selectbox("종목 선택", pf['Name'].unique())
        row = pf[pf['Name'] == sel].iloc[0]
        df_ai = get_all_indicators(get_data_safe(row['Code'], days=365))
        if df_ai is not None:
            res = get_darwin_strategy(df_ai, row['Buy_Price'])
            cp = df_ai['Close'].iloc[-1]
            
            if st.button("📡 텔레그램으로 전략 전송"):
                msg = f"💼 <b>[{sel}] 대응 전략</b>\n💰 현재가: {cp:,}원\n\n🔵 1차매수: {res['buy'][0][0]:,}원\n🔴 1차매도: {res['sell'][0][0]:,}원\n💡 예상평단: {res['avg']:,}원"
                send_telegram_msg(tg_token, tg_id, msg)
                st.success("전송 완료")
            
            # 전략 패널 (무조건 표시)
            buy_html = f"""<div class="buy-box"><b>🔵 3분할 매수</b><br>1차: <b>{res['buy'][0][0]:,}원</b> ({res['buy'][0][1]})<br>2차: <b>{res['buy'][1][0]:,}원</b> ({res['buy'][1][1]})<br>3차: <b>{res['buy'][2][0]:,}원</b> ({res['buy'][2][1]})<div class="avg-text">예상 평단: {res['avg']:,}원</div></div>"""
            sell_html = f"""<div class="sell-box"><b>🔴 3분할 매도</b><br>1차: <b>{res['sell'][0][0]:,}원</b> ({res['sell'][0][1]})<br>2차: <b>{res['sell'][1][0]:,}원</b> ({res['sell'][1][1]})<br>3차: <b>{res['sell'][2][0]:,}원</b> ({res['sell'][2][1]})</div>"""
            
            st.markdown(f"""<div class="metric-card" style="border-left:10px solid {res['status']['color']};">
                <div style="display:flex; justify-content:space-between;">
                    <div><h2>{sel} <span class="mode-badge">{res['logic']}</span></h2><p style="font-size:1.1em;">{res['status']['msg']} (AI승률: {res['ai']}%)</p></div>
                    <div style="text-align:right;"><h2 style="color:#333;">{cp:,}원</h2></div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:20px;">{buy_html} {sell_html}</div>
                </div>""", unsafe_allow_html=True)
            
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], close=df_ai['Close'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:])])
            fig.add_hline(y=res['ob'], line_color="purple", line_width=2, line_dash="dash", annotation_text="Order Block")
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

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
