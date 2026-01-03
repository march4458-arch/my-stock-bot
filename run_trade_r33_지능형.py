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

def check_market_open():
    now = get_now_kst()
    if now.weekday() >= 5: return False
    start_time = datetime.time(9, 0)
    end_time = datetime.time(15, 30)
    return start_time <= now.time() <= end_time

st.set_page_config(page_title="AI Master V70.2 Sort Fix", page_icon="📏", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #ff6d00; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    
    .buy-box { background-color: #fff3e0; padding: 15px; border-radius: 10px; border: 1px solid #ffe0b2; color: #e65100; margin-bottom: 10px; }
    .sell-box { background-color: #ffebee; padding: 15px; border-radius: 10px; border: 1px solid #ef9a9a; color: #b71c1c; margin-bottom: 10px; }
    .avg-text { font-weight: bold; color: #4a148c; text-align: center; background-color: #f3e5f5; padding: 5px; border-radius: 5px; margin-top: 5px; border: 1px solid #e1bee7; }
    
    .price-tag { font-weight: bold; font-size: 1.1em; }
    .current-price { font-size: 1.5em; font-weight: bold; color: #333; }
    .logic-tag { font-size: 0.75em; color: #444; background-color: #eceff1; padding: 2px 6px; border-radius: 4px; margin-left: 5px; border: 1px solid #cfd8dc; }
    .mode-badge { background-color: #263238; color: #00e676; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .ai-badge { background-color: #ff6d00; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .pro-tag { background-color: #e3f2fd; color: #0d47a1; font-size: 0.75em; padding: 2px 5px; border-radius: 4px; border: 1px solid #90caf9; font-weight:bold; }
    .hit-tag { background-color: #e8f5e9; color: #2e7d32; font-size: 0.8em; padding: 3px 6px; border-radius: 4px; margin-right: 5px; border: 1px solid #c8e6c9; display: inline-block; margin-bottom: 2px; }
    
    .clock-box { font-size: 1.2em; font-weight: bold; color: #333; text-align: center; margin-bottom: 5px; padding: 10px; background: #e0f7fa; border-radius: 8px; border: 1px solid #b2ebf2; }
    .source-box { background-color: #37474f; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .list-box { background-color: #546e7a; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    
    .status-open { color: #2e7d32; font-weight: bold; text-align: center; margin-bottom: 15px; }
    .status-closed { color: #c62828; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- [Data Loader] ---
@st.cache_data(ttl=60)
def get_data_safe(code, days=2000):
    start_date = (get_now_kst() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty:
            df.attrs['source'] = "🇰🇷 KRX (FDR)"
            return df
    except: pass
    try:
        df = yf.download(f"{code}.KS", start=start_date, progress=False)
        if not df.empty:
            df.attrs['source'] = "🇺🇸 Yahoo (KOSPI)"
            return df
    except: pass
    try:
        df = yf.download(f"{code}.KQ", start=start_date, progress=False)
        if not df.empty:
            df.attrs['source'] = "🇺🇸 Yahoo (KOSDAQ)"
            return df
    except: pass
    return None

@st.cache_data(ttl=86400)
def get_safe_stock_listing():
    try:
        df = fdr.StockListing('KRX')
        if not df.empty: return df, "⚡ KRX Live (전체)"
    except: pass
    fb = [['005930','삼성전자'],['000660','SK하이닉스'],['373220','LG에너지솔루션'],['207940','삼성바이오로직스'],['005380','현대차'],['000270','기아'],['005490','POSCO홀딩스'],['035420','NAVER'],['006400','삼성SDI'],['051910','LG화학'],['105560','KB금융'],['086520','에코프로'],['247540','에코프로비엠'],['042660','한화오션'],['010130','고려아연'],['034020','두산에너빌리티'],['035720','카카오'],['003670','포스코퓨처엠'],['028260','삼성물산'],['055550','신한지주']]
    return pd.DataFrame(fb, columns=['Code','Name']).assign(Marcap=10**15), "⚠️ Backup List (20개)"

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
    close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
    
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['Disp_20'] = (close / df['MA20']) * 100
    
    tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    tp = (high + low + close) / 3
    df['MVWAP'] = (tp * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    
    change = close.diff(10).abs(); volatility = close.diff().abs().rolling(10).sum()
    df['ER'] = change / (volatility + 1e-9)
    
    ma_bb = close.rolling(20).mean(); std_bb = close.rolling(20).std()
    df['BB_Up'] = ma_bb + (std_bb * 2); df['BB_Lo'] = ma_bb - (std_bb * 2)
    df['BB_Width'] = (df['BB_Up'] - df['BB_Lo']) / ma_bb
    df['Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(120).min() * 1.1

    df['Is_Impulse'] = (close > df['Open'] * 1.03) & (vol > vol.rolling(20).mean())
    ob_price = 0
    for i in range(len(df)-2, len(df)-60, -1):
        if df['Is_Impulse'].iloc[i] and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
            ob_price = (df['Open'].iloc[i-1] + df['Low'].iloc[i-1]) / 2; break
    df['OB'] = ob_price if ob_price > 0 else df['MA20'].iloc[-1]

    hi_1y = df.tail(252)['High'].max(); lo_1y = df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    
    df['SNOW_L'] = calc_stoch(df, 20, 12, 12)
    delta = close.diff(); g = delta.where(delta>0,0).rolling(14).mean(); l = -delta.where(delta<0,0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+(g/(l+1e-9))))
    
    exp1 = close.ewm(span=12, adjust=False).mean(); exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2; df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    mad = tp.rolling(14).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (tp - tp.rolling(14).mean()) / (0.015 * mad + 1e-9)
    
    raw_mf = tp * vol; pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(14).sum(); neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-9))))
    
    tr_atr = df['ATR']; dm_pos = high.diff().clip(lower=0); dm_neg = -low.diff().clip(upper=0)
    di_pos = 100 * (dm_pos.ewm(alpha=1/14).mean() / tr_atr); di_neg = 100 * (dm_neg.ewm(alpha=1/14).mean() / tr_atr)
    df['ADX'] = (100 * abs(di_pos - di_neg) / (di_pos + di_neg + 1e-9)).rolling(14).mean()

    hist = df.tail(20); counts, edges = np.histogram(hist['Close'], bins=10, weights=hist['Volume'])
    df['POC'] = edges[np.argmax(counts)]
    df['Vol_Z'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    return df

# ==========================================
# 🧠 3. 전략 엔진 (Sort Fix)
# ==========================================
def get_darwin_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]; cp = curr['Close']; atr = curr['ATR']; prev = df.iloc[-2]
    
    df['BB_Pos'] = (cp - curr['BB_Lo']) / (curr['BB_Up'] - curr['BB_Lo'] + 1e-9)
    features = ['RSI','SNOW_L','CCI','MFI','ADX','Vol_Z', 'BB_Pos', 'ER']
    data_ml = df.copy()[features].dropna()
    
    ai_prob = 50; logic_mode = "⚖️ Balanced"; top_feature = "None"
    if len(data_ml) > 60:
        try:
            model = RandomForestClassifier(n_estimators=60, random_state=42, max_depth=5)
            train_df = data_ml.iloc[:-1]; train_df['Target'] = (df['Close'].shift(-1).iloc[:-1] > df['Close'].iloc[:-1]).astype(int)
            model.fit(train_df.tail(200), train_df['Target'].tail(200))
            top_idx = np.argmax(model.feature_importances_); top_feature = features[top_idx]
            ai_prob = int(model.predict_proba(data_ml.iloc[-1:])[0][1] * 100)
        except: pass

    is_bull_setup = (curr['MA20'] > curr['MA60']) and (curr['MACD_Hist'] > -0.5)
    score = 0; hit_reasons = [] 
    
    if is_bull_setup:
        logic_mode = "🐆 Trend Hunter"
        if curr['Disp_20'] <= 105 and curr['Disp_20'] >= 98: score += 30; hit_reasons.append("20일선눌림")
        if curr['RSI'] < 65 and curr['RSI'] > 45: score += 15; hit_reasons.append("건전한조정")
        if cp >= curr['MVWAP']: score += 15; hit_reasons.append("기관수급")
        if curr['Vol_Z'] < 0: score += 10; hit_reasons.append("거래량감소")
    else:
        logic_mode = "🛡️ Sniper"
        if curr['RSI'] < 35: score += 20; hit_reasons.append("RSI과매도")
        if curr['CCI'] < -100: score += 20; hit_reasons.append("CCI침체")
        if cp <= curr['OB'] * 1.05: score += 20; hit_reasons.append("OB지지")
        if curr['MFI'] < 20: score += 10; hit_reasons.append("MFI바닥")
        if curr['Squeeze']: score += 15; hit_reasons.append("에너지응축")

    if curr['ER'] > 0.6: score += 10; hit_reasons.append("추세효율")
    
    if ai_prob >= 60: score += (ai_prob - 50) * 1.5
    elif ai_prob <= 40: score -= 20

    def adj(p):
        if np.isnan(p) or p <= 0: return 0
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500
        return int(round(p/t)*t)
    
    if is_bull_setup:
        final_buys = [(adj(curr['MA20']), "20일선"), (adj(curr['MVWAP']), "MVWAP"), (adj(cp*0.98), "눌림-2%")]
        final_buys = [x for x in final_buys if x[0] < cp * 1.01]
        if not final_buys: final_buys = [(adj(cp*0.99), "현재가"), (adj(curr['MA20']), "20일선"), (adj(curr['MA20']*0.97), "손절선")]
        elif len(final_buys) < 3: 
             while len(final_buys) < 3: final_buys.append((adj(final_buys[-1][0]*0.97), "비중확대"))
    else:
        candidates = [(adj(curr['MVWAP']), "MVWAP"), (adj(curr['OB']), "OB"), (adj(curr['Fibo_618']), "Fibo"), (adj(curr['BB_Lo']), "BB")]
        valid = [x for x in candidates if x[0] <= cp]
        final_buys = valid[:3] if len(valid)>=3 else (valid + [(adj(cp*0.95), "Low")]*3)[:3]

    # [FIX] 강제 정렬 (매수: 내림차순, 매도: 오름차순)
    final_buys.sort(key=lambda x: x[0], reverse=True) # 1차(높음) -> 3차(낮음)
    
    est_avg = int(sum([p[0] for p in final_buys]) / 3)
    sell_pts = [(adj(curr['BB_Up']), "BB 상단"), (adj(cp + atr*3), "ATR x3"), (adj(cp + atr*5), "ATR x5")]
    sell_pts.sort(key=lambda x: x[0]) # 1차(낮음) -> 3차(높음)
    
    status = {"type": "💤 관망", "color": "#78909c", "msg": "대기", "alert": False}
    if buy_price > 0:
        pct = (cp - buy_price) / buy_price * 100
        if cp >= sell_pts[0][0]: status = {"type": "💰 익절", "color": "#2e7d32", "msg": "수익권", "alert": True}
        elif pct < -3 and score >= 60: status = {"type": "❄️ 물타기", "color": "#0288d1", "msg": "추매", "alert": True}
        elif pct > 2 and is_bull_setup: status = {"type": "🔥 불타기", "color": "#ff6d00", "msg": "추세추종", "alert": True}
    
    return {"buy": final_buys, "sell": sell_pts, "avg": est_avg, "score": int(score), "status": status, "ai": ai_prob, "logic": logic_mode, "top_feat": top_feature, "reasons": hit_reasons, "mvwap": curr['MVWAP']}

# ==========================================
# 🖥️ 4. 메인 UI
# ==========================================
with st.sidebar:
    now = get_now_kst()
    is_market_open = check_market_open()
    
    st.markdown(f'<div class="clock-box">⏰ {now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    if is_market_open: st.markdown('<div class="status-open">🟢 KOSPI/KOSDAQ 장중</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="status-closed">🔴 정규장 마감 (휴장)</div>', unsafe_allow_html=True)
    
    source_container = st.empty()
    source_container.markdown('<div class="source-box">📡 Ready</div>', unsafe_allow_html=True)
    
    krx_list, list_src = get_safe_stock_listing()
    st.markdown(f'<div class="list-box">📋 {list_src}</div>', unsafe_allow_html=True)

    st.title("📏 V70.2 Sort Fix")
    
    with st.expander("⚙️ 설정 및 자동화", expanded=True):
        tg_token = st.text_input("Bot Token", type="password")
        tg_id = st.text_input("Chat ID")
        st.markdown("---")
        auto_report = st.checkbox("✅ 자동 리포트", value=True)
        report_time = st.time_input("발송 시간", datetime.time(16, 0))
        scanner_alert = st.checkbox("📢 스캔 자동 알림", value=True)
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 자동 갱신 (PC)", value=False)
        only_market_time = st.checkbox("⏰ 정규장에만 실행", value=True)
        refresh_min = st.slider("주기(분)", 1, 60, 5)
    
    min_m = st.number_input("최소 시총(억)", value=3000) * 100000000
    
    if auto_report and now.hour == report_time.hour and now.minute == report_time.minute:
        pf_rep = get_portfolio_gsheets()
        if not pf_rep.empty:
            msg = f"📏 <b>[{report_time.strftime('%H:%M')} 정기 리포트]</b>\n"
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
        last_source = "Loading..."
        for _, row in pf.iterrows():
            d = get_data_safe(row['Code'], days=200)
            if d is not None:
                last_source = d.attrs.get('source', 'Unknown')
                df = get_all_indicators(d)
                res = get_darwin_strategy(df, row['Buy_Price'])
                cp = df['Close'].iloc[-1]; t_buy += (row['Buy_Price']*row['Qty']); t_eval += (cp*row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
        
        source_container.markdown(f'<div class="source-box">📡 {last_source}</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_white"), use_container_width=True)
    
    if auto_refresh:
        if only_market_time and not is_market_open: st.warning("🌙 정규장 운영 시간이 아니므로 자동 갱신을 일시 정지합니다.")
        else: time.sleep(refresh_min * 60); st.rerun()

with tabs[1]: # 스캐너
    if st.button("📏 정렬된 스캔") or (auto_refresh and (not only_market_time or is_market_open)):
        if auto_refresh: st.info(f"🔄 자동 스캔 중... (주기: {refresh_min}분)")
        
        targets = krx_list[krx_list['Marcap'] >= min_m].sort_values('Marcap', ascending=False).head(50)
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
        
        source_container.markdown(f'<div class="source-box">📡 🇰🇷 KRX (Scanner)</div>', unsafe_allow_html=True)
        
        top_picks = sorted(found, key=lambda x: x['score'], reverse=True)[:15]
        if scanner_alert and top_picks and tg_token and tg_id:
            msg = f"🚀 <b>[AI 헌터 스캔 Top 5]</b>\n\n"
            for item in top_picks[:5]:
                s = item['strat']
                msg += f"<b>{item['name']}</b> ({s['logic']})\n💰 {item['cp']:,}원 / 🎯 {s['buy'][0][0]:,}원\n🏆 {s['score']}점 (MVWAP:{int(s['mvwap']):,})\n\n"
            send_telegram_msg(tg_token, tg_id, msg)
            st.toast("📨 텔레그램 전송 완료!")

        for d in top_picks:
            s = d['strat']
            reasons_html = "".join([f"<span class='hit-tag'>✅ {r}</span>" for r in s['reasons']])
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div><h3 style="margin:0;">{d['name']}</h3><span class="current-price">{d['cp']:,}원</span><span class="pro-tag" style="margin-left:5px;">MVWAP: {int(s['mvwap']):,}</span></div>
                        <div style="text-align:right;"><span class="ai-badge">AI: {s['ai']}%</span><span style="font-size:1.1em; font-weight:bold; color:#ff6d00; margin-left:5px;">Score: {s['score']}</span><br><span class="mode-badge" style="font-size:0.8em; margin-top:5px; display:inline-block;">{s['logic']}</span></div>
                    </div>
                    <div style="margin:5px 0;">{reasons_html}</div>
                    <div style="margin: 10px 0; display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div class="buy-box"><b>🔵 Smart Entry</b><br>1차: <b>{s['buy'][0][0]:,}원</b> <span class="logic-tag">{s['buy'][0][1]}</span><br>2차: <b>{s['buy'][1][0]:,}원</b> <span class="logic-tag">{s['buy'][1][1]}</span><br>3차: <b>{s['buy'][2][0]:,}원</b> <span class="logic-tag">{s['buy'][2][1]}</span><div class="avg-text">예상 평단: {s['avg']:,}원</div></div>
                        <div class="sell-box"><b>🔴 Smart Exit</b><br>1차: {s['sell'][0][0]:,}원 <span class="logic-tag">{s['sell'][0][1]}</span><br>2차: {s['sell'][1][0]:,}원 <span class="logic-tag">{s['sell'][1][1]}</span><br>3차: {s['sell'][2][0]:,}원 <span class="logic-tag">{s['sell'][2][1]}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    
    if auto_refresh:
        if only_market_time and not is_market_open: pass
        else: time.sleep(refresh_min * 60); st.rerun()

with tabs[2]: # 5년 검증
    st.subheader("🧬 5년 진화 성적표 (Dual + Sort)")
    if st.button("🚀 5년 데이터 검증 시작"):
        # (검증 코드 동일)
        pf = get_portfolio_gsheets()
        sample_codes = pf['Code'].tolist() if not pf.empty else []
        top5_codes = krx_list.head(5)['Code'].tolist()
        targets = list(set(sample_codes + top5_codes))[:10]
        results = []
        prog = st.progress(0)
        for idx, code in enumerate(targets):
            full_df_raw = get_data_safe(code, days=2000)
            if full_df_raw is not None and len(full_df_raw) > 300:
                full_df = get_all_indicators(full_df_raw)
                if full_df is not None:
                    for i in range(240, 0, -1):
                        past_idx = - (i * 5)
                        if abs(past_idx) < len(full_df) - 60 and abs(past_idx) < len(full_df):
                            past_df = full_df.iloc[:past_idx]; future_df = full_df.iloc[past_idx:]
                            if len(future_df) >= 5:
                                res = get_darwin_strategy(past_df)
                                if res['score'] >= 60:
                                    entry = past_df['Close'].iloc[-1]; exit_p = future_df['Close'].iloc[4]
                                    results.append({"Date": past_df.index[-1], "Win": 1 if exit_p > entry else 0, "Count": 1})
            prog.progress((idx+1)/len(targets))
        if results:
            df_res = pd.DataFrame(results).sort_values('Date')
            df_res['Win_Rate'] = (df_res['Win'].cumsum() / df_res['Count'].cumsum() * 100)
            c1, c2 = st.columns(2)
            c1.metric("총 검증 횟수", f"{len(df_res)}회"); c2.metric("누적 승률", f"{df_res['Win_Rate'].iloc[-1]:.1f}%")
            fig = px.line(df_res, x='Date', y='Win_Rate', title="5년 승률 변화 (Sorted)", markers=False)
            fig.add_hline(y=50, line_dash="dot", line_color="gray"); st.plotly_chart(fig, use_container_width=True)
        else: st.error("데이터 부족")

with tabs[3]: # AI 리포트
    if not pf.empty:
        sel = st.selectbox("종목 선택", pf['Name'].unique())
        row = pf[pf['Name'] == sel].iloc[0]
        raw_df = get_data_safe(row['Code'], days=365)
        if raw_df is not None:
            source_container.markdown(f'<div class="source-box">📡 {raw_df.attrs.get("source","Unknown")}</div>', unsafe_allow_html=True)
            df_ai = get_all_indicators(raw_df)
            res = get_darwin_strategy(df_ai, row['Buy_Price'])
            cp = df_ai['Close'].iloc[-1]
            if st.button("📡 전략 전송"):
                msg = f"📏 <b>[{sel}] 전략</b>\n💰 {cp:,}원\n\n🔵 1차: {res['buy'][0][0]:,}원\n🔴 1차: {res['sell'][0][0]:,}원\n💡 평단: {res['avg']:,}원"
                send_telegram_msg(tg_token, tg_id, msg); st.success("전송 완료")
            
            reasons_html = "".join([f"<span class='hit-tag'>✅ {r}</span>" for r in res['reasons']])
            buy_html = f"""<div class="buy-box"><b>🔵 Smart Entry</b><br>1차: <b>{res['buy'][0][0]:,}원</b> ({res['buy'][0][1]})<br>2차: <b>{res['buy'][1][0]:,}원</b> ({res['buy'][1][1]})<br>3차: <b>{res['buy'][2][0]:,}원</b> ({res['buy'][2][1]})<div class="avg-text">예상 평단: {res['avg']:,}원</div></div>"""
            sell_html = f"""<div class="sell-box"><b>🔴 Smart Exit</b><br>1차: <b>{res['sell'][0][0]:,}원</b> ({res['sell'][0][1]})<br>2차: <b>{res['sell'][1][0]:,}원</b> ({res['sell'][1][1]})<br>3차: <b>{res['sell'][2][0]:,}원</b> ({res['sell'][2][1]})</div>"""
            
            st.markdown(f"""<div class="metric-card" style="border-left:10px solid {res['status']['color']};"><div style="display:flex; justify-content:space-between;"><div><h2>{sel} <span class="mode-badge">{res['logic']}</span></h2><p style="font-size:1.1em;">{res['status']['msg']} (AI승률: {res['ai']}%)</p></div><div style="text-align:right;"><h2 style="color:#333;">{cp:,}원</h2><span class="pro-tag">MVWAP: {int(res['mvwap']):,}</span></div></div><div style="margin:5px 0;">{reasons_html}</div><div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:20px;">{buy_html} {sell_html}</div></div>""", unsafe_allow_html=True)
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], close=df_ai['Close'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:])])
            fig.add_hline(y=res['mvwap'], line_color="orange", line_width=2, annotation_text="MVWAP(기관)")
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]: # 관리
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            m = krx_list[krx_list['Name']==n]
            if not m.empty:
                new = pd.DataFrame([[m.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
    st.dataframe(df_p, use_container_width=True)
