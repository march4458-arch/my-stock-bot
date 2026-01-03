import streamlit as st
import pandas as pd
import pyupbit
import datetime, time, requests, os
from datetime import timezone, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# ⚙️ 1. 시스템 설정 (업비트 전용)
# ==========================================
def get_now_kst():
    return datetime.datetime.now(timezone(timedelta(hours=9)))

# 코인은 24시간 장이 열리므로 항상 True 반환
def check_market_open():
    return True

st.set_page_config(page_title="AI Master V71.2 Crypto Shield", page_icon="🪙", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; border-left: 5px solid #5e35b1; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 20px; border-radius: 15px; border: 1px solid #e0e0e0; margin-bottom: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    
    .buy-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; border: 1px solid #c8e6c9; color: #1b5e20; margin-bottom: 10px; }
    .sell-box { background-color: #ffebee; padding: 15px; border-radius: 10px; border: 1px solid #ffcdd2; color: #b71c1c; margin-bottom: 10px; }
    
    .price-tag { font-weight: bold; font-size: 1.1em; }
    .current-price { font-size: 1.5em; font-weight: bold; color: #333; }
    .logic-tag { font-size: 0.75em; color: #444; background-color: #eceff1; padding: 2px 6px; border-radius: 4px; margin-left: 5px; border: 1px solid #cfd8dc; }
    .mode-badge { background-color: #263238; color: #00e676; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .ai-badge { background-color: #5e35b1; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.85em; }
    .pro-tag { background-color: #ede7f6; color: #4527a0; font-size: 0.75em; padding: 2px 5px; border-radius: 4px; border: 1px solid #d1c4e9; font-weight:bold; }
    .exit-alert { color: #d32f2f; font-weight: bold; font-size: 0.9em; background: #ffQqee; padding: 5px; border-radius: 5px; margin-top: 5px; display: block; }
    
    .clock-box { font-size: 1.2em; font-weight: bold; color: #333; text-align: center; margin-bottom: 5px; padding: 10px; background: #e0f7fa; border-radius: 8px; border: 1px solid #b2ebf2; }
    .source-box { background-color: #4527a0; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .list-box { background-color: #546e7a; color: #fff; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.9em; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    
    .status-open { color: #5e35b1; font-weight: bold; text-align: center; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- [Data Loader: Upbit Shield System] ---
@st.cache_data(ttl=300) # 코인은 변동성이 크므로 캐시 시간을 5분으로 단축
def get_data_safe(ticker, days=200):
    # 로컬 저장소 폴더 생성 (데이터 백업용)
    save_dir = "coin_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    today_str = datetime.datetime.now().strftime('%Y%m%d_%H') # 시간 단위 저장
    clean_ticker = ticker.replace("-", "_")
    file_path = f"{save_dir}/{clean_ticker}_{today_str}.csv"

    # Upbit API 호출 (Data Shield: API 과부하 방지 딜레이)
    time.sleep(0.05) 
    
    try:
        # pyupbit는 컬럼명을 소문자로 반환하므로 대문자로 변환 필요
        df = pyupbit.get_ohlcv(ticker, interval="day", count=days)
        if df is not None and not df.empty:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Value']
            df.to_csv(file_path) # 백업
            df.attrs['source'] = "⚡ Upbit API"
            return df
    except:
        pass
    
    # API 실패 시 로컬 파일 로드 시도
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if not df.empty:
                df.attrs['source'] = "💾 Local Backup"
                return df
        except: pass

    return None

@st.cache_data(ttl=3600)
def get_coin_listing():
    try:
        # KRW 마켓의 모든 티커 가져오기
        tickers = pyupbit.get_tickers(fiat="KRW")
        return tickers, f"⚡ Upbit KRW ({len(tickers)})"
    except:
        return ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL'], "⚠️ Backup List"

def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            # 코인용 컬럼 매핑 (티커, 코인명)
            rename_map = {'코드':'Code','티커':'Code','Ticker':'Code','종목명':'Name','Name':'Name','평단가':'Buy_Price','Buy_Price':'Buy_Price','수량':'Qty','Qty':'Qty'}
            df = df.rename(columns=rename_map)
            if 'Code' in df.columns:
                df = df.dropna(subset=['Code'])
                # 코인 티커 포맷팅 (KRW- 생략시 자동 추가 등)
                df['Code'] = df['Code'].astype(str).apply(lambda x: x if x.startswith('KRW-') else f"KRW-{x}")
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
# 📊 2. Smart 지표 엔진 (공용 로직)
# ==========================================
def calc_stoch(df, n, m, t):
    l, h = df['Low'].rolling(n).min(), df['High'].rolling(n).max()
    return ((df['Close'] - l) / (h - l + 1e-9) * 100).rolling(m).mean().rolling(t).mean()

def get_all_indicators(df):
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']
    open_p = df['Open']
    
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    df['Disp_5'] = (close / df['MA5']) * 100 
    
    # Candle Patterns
    df['Upper_Shadow'] = high - df[['Open', 'Close']].max(axis=1)
    df['Body'] = (close - open_p).abs()
    df['Is_Shooting_Star'] = (df['Upper_Shadow'] > df['Body'] * 2) & (high > df['MA5'])
    
    prev_open = open_p.shift(1); prev_close = close.shift(1)
    df['Is_Bearish_Engulfing'] = (close < prev_open) & (open_p > prev_close) & (close < open_p) & (prev_close > prev_open)

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
# 🧠 3. Smart Exit 전략 (Upbit Tick 적용)
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

    is_bull_setup = (curr['MA20'] > curr['MA60']) and (curr['RSI'] > 55)
    
    score = 0; hit_reasons = [] 
    if is_bull_setup:
        logic_mode = "🐆 Trend Hunter"
        if curr['Disp_5'] <= 102 and curr['Disp_5'] >= 98: score += 30; hit_reasons.append("5일선지지")
        if curr['Vol_Z'] > 1.0: score += 15; hit_reasons.append("수급폭발")
        if cp >= curr['MVWAP']: score += 15; hit_reasons.append("고래수급")
        if curr['RSI'] > 60: score += 15; hit_reasons.append("RSI강세")
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

    # Sell Logic
    sell_score = 0; sell_reasons = []
    if curr['Is_Shooting_Star']: sell_score += 30; sell_reasons.append("🕯️ 유성형(고점)")
    if curr['Is_Bearish_Engulfing']: sell_score += 30; sell_reasons.append("🕯️ 하락장악형")
    if curr['RSI'] > 80: sell_score += 20; sell_reasons.append("⚠️ RSI초과열") # 코인은 80 기준
    if curr['MFI'] > 85: sell_score += 20; sell_reasons.append("⚠️ MFI초과열")
    if cp > curr['BB_Up'] * 1.05: sell_score += 20; sell_reasons.append("⚠️ BB이탈")
    
    if is_bull_setup:
        if cp < curr['MA5']: sell_score += 10; sell_reasons.append("📉 5일선붕괴")
        if cp < curr['MA20']: sell_score += 50; sell_reasons.append("⛔ 20일선붕괴")
    else:
        if cp < prev['Low']: sell_score += 20; sell_reasons.append("📉 전저점이탈")

    # [업비트 호가 단위 적용 함수]
    def adj(price):
        if np.isnan(price) or price <= 0: return 0
        if price >= 2000000: return int(round(price/1000)*1000)
        elif price >= 1000000: return int(round(price/500)*500)
        elif price >= 500000: return int(round(price/100)*100)
        elif price >= 100000: return int(round(price/50)*50)
        elif price >= 10000: return int(round(price/10)*10)
        elif price >= 1000: return int(round(price/5)*5)
        elif price >= 100: return int(round(price/1)*1)
        elif price >= 10: return round(price, 1)
        else: return round(price, 4)
    
    targets = []
    targets.append((adj(curr['BB_Up']), "BB상단(저항)"))
    if is_bull_setup:
        targets.append((adj(cp * 1.05), "추세+5%"))
        targets.append((adj(cp * 1.10), "추세+10%"))
    else:
        targets.append((adj(curr['MA60']), "60일선(저항)"))
        targets.append((adj(cp + atr*3), "반등목표"))
    
    sell_pts = sorted(list(set(targets)), key=lambda x: x[0])
    if len(sell_pts) < 3: sell_pts += [(adj(sell_pts[-1][0]*1.03), "Top")] * (3-len(sell_pts))
    sell_pts = sell_pts[:3]

    if is_bull_setup:
        final_buys = [(adj(curr['MA5']), "5일선"), (adj(curr['MA10']), "10일선"), (adj(curr['MA20']), "20일선")]
        final_buys = [x for x in final_buys if x[0] < cp * 1.005]
        if not final_buys: final_buys = [(adj(cp), "시장가"), (adj(curr['MA5']), "5일선"), (adj(curr['MA10']), "10일선")]
        elif len(final_buys) < 3: 
             while len(final_buys) < 3: final_buys.append((adj(final_buys[-1][0]*0.98), "불타기"))
    else:
        candidates = [(adj(curr['MVWAP']), "MVWAP"), (adj(curr['OB']), "OB"), (adj(curr['Fibo_618']), "Fibo"), (adj(curr['BB_Lo']), "BB")]
        valid = [x for x in candidates if x[0] <= cp]
        final_buys = valid[:3] if len(valid)>=3 else (valid + [(adj(cp*0.95), "Low")]*3)[:3]

    final_buys.sort(key=lambda x: x[0], reverse=True)
    est_avg = sum([p[0] for p in final_buys]) / 3 # 코인은 소수점 가능하므로 int 제거
    est_avg = adj(est_avg)
    
    status = {"type": "💤 관망", "color": "#78909c", "msg": "대기", "alert": False}
    if sell_score >= 50:
        status = {"type": "🔴 매도 경고", "color": "#d32f2f", "msg": f"{sell_reasons[0] if sell_reasons else '위험'} 발생", "alert": True}
    elif buy_price > 0:
        pct = (cp - buy_price) / buy_price * 100
        if sell_score >= 30: status = {"type": "⚠️ 주의", "color": "#f57f17", "msg": "분할매도 고려", "alert": True}
        elif pct > 0: status = {"type": "💰 수익중", "color": "#2e7d32", "msg": f"수익 +{pct:.1f}%", "alert": False}
        elif pct < -3: status = {"type": "❄️ 손실중", "color": "#1976d2", "msg": "버티기", "alert": False}
    
    return {"buy": final_buys, "sell": sell_pts, "avg": est_avg, "score": int(score), "status": status, "ai": ai_prob, "logic": logic_mode, "top_feat": top_feature, "reasons": hit_reasons, "sell_reasons": sell_reasons, "mvwap": curr['MVWAP']}

# ==========================================
# 🖥️ 4. 메인 UI
# ==========================================
with st.sidebar:
    now = get_now_kst()
    
    st.markdown(f'<div class="clock-box">⏰ {now.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-open">🟣 Upbit 24/7 Live</div>', unsafe_allow_html=True)
    
    source_container = st.empty()
    source_container.markdown('<div class="source-box">📡 Ready</div>', unsafe_allow_html=True)
    
    tickers_list, list_src = get_coin_listing()
    st.markdown(f'<div class="list-box">📋 {list_src}</div>', unsafe_allow_html=True)

    st.title("🪙 Crypto Master Shield")
    
    with st.expander("⚙️ 설정 및 자동화", expanded=True):
        tg_token = st.text_input("Bot Token", type="password")
        tg_id = st.text_input("Chat ID")
        st.markdown("---")
        auto_report = st.checkbox("✅ 자동 리포트", value=True)
        report_time = st.time_input("발송 시간", datetime.time(9, 0)) # 코인 리셋시간 09:00
        scanner_alert = st.checkbox("📢 스캔 자동 알림", value=True)
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 자동 갱신 (PC)", value=False)
        refresh_min = st.slider("주기(분)", 1, 60, 5)
    
    # 코인 특성상 시총 대신 거래대금이나 티커수로 제한 (여기선 단순화)
    # min_m = st.number_input("최소 시총(억)", value=3000) * 100000000 
    
    if auto_report and now.hour == report_time.hour and now.minute == report_time.minute:
        pf_rep = get_portfolio_gsheets()
        if not pf_rep.empty:
            msg = f"🛡️ <b>[{report_time.strftime('%H:%M')} 코인 정기 리포트]</b>\n"
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
                dash_list.append({"코인": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": res['status']['type']})
        
        source_container.markdown(f'<div class="source-box">📡 {last_source}</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='코인', y='수익', color='상태', template="plotly_white"), use_container_width=True)
    
    if auto_refresh:
        time.sleep(refresh_min * 60); st.rerun()

with tabs[1]: # 스캐너
    if st.button("🛡️ 쉴드 스캔") or auto_refresh:
        if auto_refresh: st.info(f"🔄 자동 스캔 중... (주기: {refresh_min}분)")
        
        # 업비트 스캔 대상: KRW 마켓 전체 (API 제한 고려하여 상위 40개만 샘플링하거나 전체 순회)
        # 전체를 다 하기엔 시간이 걸리므로 여기선 30개 무작위 or 주요 코인만
        targets_code = tickers_list[:40] # API 속도 고려 40개 제한
        
        found, prog = [], st.progress(0)
        with ThreadPoolExecutor(max_workers=5) as ex: # 업비트 API는 초당 10회 제한, 스레드 줄임
            futs = {ex.submit(get_all_indicators, get_data_safe(code, days=200)): code for code in targets_code}
            for i, f in enumerate(as_completed(futs)):
                code = futs[f]
                res = f.result()
                if res is not None:
                    s = get_darwin_strategy(res)
                    cp = res['Close'].iloc[-1]
                    found.append({"name": code, "score": s['score'], "strat": s, "cp": cp})
                prog.progress((i+1)/len(targets_code))
        
        source_container.markdown(f'<div class="source-box">📡 Upbit API (Scanner)</div>', unsafe_allow_html=True)
        
        top_picks = sorted(found, key=lambda x: x['score'], reverse=True)[:15]
        if scanner_alert and top_picks and tg_token and tg_id:
            msg = f"🛡️ <b>[Crypto Smart 스캔]</b>\n\n"
            for item in top_picks[:5]:
                s = item['strat']
                msg += f"<b>{item['name']}</b> ({s['logic']})\n💰 {item['cp']:,}원 / 🎯 {s['buy'][0][0]:,}원\n🏆 {s['score']}점\n\n"
            send_telegram_msg(tg_token, tg_id, msg)
            st.toast("📨 텔레그램 전송 완료!")

        for d in top_picks:
            s = d['strat']
            reasons_html = "".join([f"<span class='hit-tag'>✅ {r}</span>" for r in s['reasons']])
            sell_reasons_html = "".join([f"<span class='exit-alert'>🚨 {r}</span>" for r in s['sell_reasons']])
            
            st.markdown(f"""
                <div class="scanner-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div><h3 style="margin:0;">{d['name']}</h3><span class="current-price">{d['cp']:,}원</span><span class="pro-tag" style="margin-left:5px;">MVWAP: {s['mvwap']:,}</span></div>
                        <div style="text-align:right;"><span class="ai-badge">AI: {s['ai']}%</span><span style="font-size:1.1em; font-weight:bold; color:#5e35b1; margin-left:5px;">Score: {s['score']}</span><br><span class="mode-badge" style="font-size:0.8em; margin-top:5px; display:inline-block;">{s['logic']}</span></div>
                    </div>
                    <div style="margin:5px 0;">{reasons_html}</div><div>{sell_reasons_html}</div>
                    <div style="margin: 10px 0; display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div class="buy-box"><b>🔵 Smart Entry</b><br>1차: <b>{s['buy'][0][0]:,}원</b> <span class="logic-tag">{s['buy'][0][1]}</span><br>2차: <b>{s['buy'][1][0]:,}원</b> <span class="logic-tag">{s['buy'][1][1]}</span><br>3차: <b>{s['buy'][2][0]:,}원</b> <span class="logic-tag">{s['buy'][2][1]}</span><div class="avg-text">예상 평단: {s['avg']:,}원</div></div>
                        <div class="sell-box"><b>🔴 Smart Exit</b><br>1차: {s['sell'][0][0]:,}원 <span class="logic-tag">{s['sell'][0][1]}</span><br>2차: {s['sell'][1][0]:,}원 <span class="logic-tag">{s['sell'][1][1]}</span><br>3차: {s['sell'][2][0]:,}원 <span class="logic-tag">{s['sell'][2][1]}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    
    if auto_refresh:
        time.sleep(refresh_min * 60); st.rerun()

with tabs[2]: # 5년 검증
    st.subheader("🧬 과거 데이터 검증 (Crypto Shield)")
    status_text = st.empty()
    if st.button("🚀 데이터 검증 시작"):
        pf = get_portfolio_gsheets()
        sample_codes = pf['Code'].tolist() if not pf.empty else []
        top5_codes = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
        targets = list(set(sample_codes + top5_codes))[:10]
        
        if not targets: st.error("종목 없음")
        else:
            results = []
            prog = st.progress(0)
            for idx, code in enumerate(targets):
                status_text.write(f"⏳ **[{idx+1}/{len(targets)}] {code}** 데이터 분석 중...")
                time.sleep(0.1) 
                full_df_raw = get_data_safe(code, days=1000) # 코인은 1000일 정도가 적당
                if full_df_raw is not None and len(full_df_raw) > 300:
                    full_df = get_all_indicators(full_df_raw)
                    if full_df is not None:
                        for i in range(100, 0, -1):
                            past_idx = - (i * 5)
                            if abs(past_idx) < len(full_df) - 60:
                                past_df = full_df.iloc[:past_idx]; future_df = full_df.iloc[past_idx:]
                                if len(future_df) >= 5:
                                    res = get_darwin_strategy(past_df)
                                    if res['score'] >= 60:
                                        entry = past_df['Close'].iloc[-1]; exit_p = future_df['Close'].iloc[4]
                                        results.append({"Date": past_df.index[-1], "Win": 1 if exit_p > entry else 0, "Count": 1})
                prog.progress((idx+1)/len(targets))
            
            status_text.success("✅ 검증 완료!")
            if results:
                df_res = pd.DataFrame(results).sort_values('Date')
                df_res['Win_Rate'] = (df_res['Win'].cumsum() / df_res['Count'].cumsum() * 100)
                c1, c2 = st.columns(2)
                c1.metric("총 검증 횟수", f"{len(df_res)}회"); c2.metric("누적 승률", f"{df_res['Win_Rate'].iloc[-1]:.1f}%")
                fig = px.line(df_res, x='Date', y='Win_Rate', title="누적 승률 변화", markers=False)
                fig.add_hline(y=50, line_dash="dot", line_color="gray"); st.plotly_chart(fig, use_container_width=True)
            else: st.error("기록 없음")

with tabs[3]: # AI 리포트
    if not pf.empty:
        sel = st.selectbox("코인 선택", pf['Name'].unique())
        row = pf[pf['Name'] == sel].iloc[0]
        raw_df = get_data_safe(row['Code'], days=365)
        if raw_df is not None:
            source_container.markdown(f'<div class="source-box">📡 {raw_df.attrs.get("source","Unknown")}</div>', unsafe_allow_html=True)
            df_ai = get_all_indicators(raw_df)
            res = get_darwin_strategy(df_ai, row['Buy_Price'])
            cp = df_ai['Close'].iloc[-1]
            if st.button("📡 전략 전송"):
                msg = f"🛡️ <b>[{sel}] 전략</b>\n💰 {cp:,}원\n\n🔵 1차: {res['buy'][0][0]:,}원\n🔴 1차: {res['sell'][0][0]:,}원\n💡 평단: {res['avg']:,}원"
                send_telegram_msg(tg_token, tg_id, msg); st.success("전송 완료")
            
            reasons_html = "".join([f"<span class='hit-tag'>✅ {r}</span>" for r in res['reasons']])
            sell_reasons_html = "".join([f"<span class='exit-alert'>🚨 {r}</span>" for r in res['sell_reasons']])
            
            buy_html = f"""<div class="buy-box"><b>🔵 Smart Entry</b><br>1차: <b>{res['buy'][0][0]:,}원</b> ({res['buy'][0][1]})<br>2차: <b>{res['buy'][1][0]:,}원</b> ({res['buy'][1][1]})<br>3차: <b>{res['buy'][2][0]:,}원</b> ({res['buy'][2][1]})<div class="avg-text">예상 평단: {res['avg']:,}원</div></div>"""
            sell_html = f"""<div class="sell-box"><b>🔴 Smart Exit</b><br>1차: <b>{res['sell'][0][0]:,}원</b> ({res['sell'][0][1]})<br>2차: <b>{res['sell'][1][0]:,}원</b> ({res['sell'][1][1]})<br>3차: <b>{res['sell'][2][0]:,}원</b> ({res['sell'][2][1]})</div>"""
            
            st.markdown(f"""<div class="metric-card" style="border-left:10px solid {res['status']['color']};"><div style="display:flex; justify-content:space-between;"><div><h2>{sel} <span class="mode-badge">{res['logic']}</span></h2><p style="font-size:1.1em; color:{res['status']['color']}; font-weight:bold;">{res['status']['msg']} (AI승률: {res['ai']}%)</p></div><div style="text-align:right;"><h2 style="color:#333;">{cp:,}원</h2><span class="pro-tag">MVWAP: {res['mvwap']:,}</span></div></div><div style="margin:5px 0;">{reasons_html}</div><div>{sell_reasons_html}</div><div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:20px;">{buy_html} {sell_html}</div></div>""", unsafe_allow_html=True)
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-100:], open=df_ai['Open'][-100:], close=df_ai['Close'][-100:], high=df_ai['High'][-100:], low=df_ai['Low'][-100:])])
            fig.add_hline(y=res['mvwap'], line_color="orange", line_width=2, annotation_text="MVWAP(고래)")
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tabs[4]: # 관리
    df_p = get_portfolio_gsheets()
    with st.form("add"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("코인 티커 (예: BTC)"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            # 입력 편의성: KRW- 생략하고 입력해도 처리
            code = n.upper()
            if not code.startswith("KRW-"): code = "KRW-" + code
            
            if code in tickers_list:
                new = pd.DataFrame([[code, code, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new], ignore_index=True)); st.rerun()
            else:
                st.error("존재하지 않는 코인입니다.")
    st.dataframe(df_p, use_container_width=True)
