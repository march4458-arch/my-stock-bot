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

st.set_page_config(page_title="주식 비서 V64.9.4 High-End", page_icon="🛡️", layout="wide")

# 프리미엄 UI 디자인 CSS (V64.6~V64.9 스타일 완전 통합)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333333; }
    div[data-testid="stMetricValue"] { color: #007bff !important; font-weight: bold; }
    .guide-box { padding: 25px; border-radius: 12px; margin-bottom: 25px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .scanner-card { padding: 22px; border-radius: 15px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); background-color: white; }
    .buy-box { background-color: #f0f7ff; padding: 12px; border-radius: 10px; border: 1px solid #b3d7ff; color: #0056b3; }
    .sell-box { background-color: #fff5f5; padding: 12px; border-radius: 10px; border: 1px solid #ffcccc; color: #c82333; }
    .metric-card { background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- [유틸리티: 알림 및 데이터 연동] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

def get_portfolio_gsheets():
    try:
        if not os.path.exists(".streamlit/secrets.toml") and not hasattr(st, "secrets"):
            st.sidebar.error("🚨 .streamlit/secrets.toml 파일 누락!")
            return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
            
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is None or df.empty: return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

        df = df.dropna(how='all')
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        rename_map = {'Code': 'Code', '코드': 'Code', 'Name': 'Name', '종목명': 'Name', 'Buy_price': 'Buy_Price', '평단가': 'Buy_Price', 'Qty': 'Qty', '수량': 'Qty'}
        df = df.rename(columns=rename_map)
        for col in ['Buy_Price', 'Qty']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
        return df[['Code', 'Name', 'Buy_Price', 'Qty']]
    except:
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 하이브리드 분석 엔진 (2중 백업망 및 정밀 지표)
# ==========================================
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (get_now_kst() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = fdr.DataReader(code_str, start_date)
        if df is not None and not df.empty: return df
    except:
        try:
            ticker = f"{code_str}.KS" if int(code_str) < 900000 else f"{code_str}.KQ"
            df = yf.download(ticker, start=start_date, progress=False, timeout=7)
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
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9)).fillna(0)))
    
    std = close.rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    
    # 스토캐스틱 K/D 복구
    low_min, high_max = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['Stoch_K'] = ((close - low_min) / (high_max - low_min + 1e-9)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    avg_vol = df['Volume'].rolling(20).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(20).std() + 1e-9)
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    df['Fibo_618'] = hi_1y - ((hi_1y - lo_1y) * 0.618)
    
    ob_zones = [df['Low'].iloc[i-1] for i in range(len(df)-40, len(df)-1) 
                if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5]
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    counts, edges = np.histogram(df.tail(20)['Close'], bins=10, weights=df.tail(20)['Volume'])
    df['POC_Price'] = edges[np.argmax(counts)]
    
    df['Regime'] = "🚀 상승" if (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / (df['MA120'].iloc[-20] + 1e-9) > 0.004 else "📉 하락"
    return df

def get_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob, poc, f618, bbl, bbu = curr['Close'], curr['ATR'], curr['OB_Price'], curr['POC_Price'], curr['Fibo_618'], curr['BB_Lower'], curr['BB_Upper']
    hi_120 = df.tail(120)['High'].max()
    
    def adj(p): return int(round(p/10)*10) if p<100000 else int(round(p/100)*100)
    
    # [유기적 타점 재배치]
    candidates = [{"name": "매물대(POC)", "price": poc, "score": 0}, {"name": "피보나치(618)", "price": f618, "score": 0},
                  {"name": "세력선(OB)", "price": ob, "score": 0}, {"name": "밴드하단(BB)", "price": bbl, "score": 0}]
    for cand in candidates:
        if curr['RSI'] < 35: cand['score'] += 20
        dist = abs(cp - cand['price']) / (cp + 1e-9)
        if dist < 0.03: cand['score'] += 30

    sorted_cand = sorted(candidates, key=lambda x: x['score'], reverse=True)
    buy = [adj(sorted_cand[0]['price']), adj(sorted_cand[1]['price']), adj(sorted_cand[2]['price'])]
    buy_names = [sorted_cand[0]['name'], sorted_cand[1]['name'], sorted_cand[2]['name']]
    
    sell = [adj(cp + atr * 2.0), adj(max(cp + atr * 3.5, hi_120)), adj(max(cp + atr * 5.0, hi_120 + atr * 2.0))]
    
    stop_loss = adj(min(buy) * 0.93)
    pyramiding = {"type": "💤 관망", "msg": f"{buy_names[0]} 타점 대기", "color": "#6c757d", "alert": False}
    if buy_price > 0:
        y = (cp - buy_price) / (buy_price + 1e-9) * 100
        if cp >= sell[0]: pyramiding = {"type": "💰 익절", "msg": f"수익률 {y:.1f}% 달성!", "color": "#28a745", "alert": True}
        elif cp <= stop_loss: pyramiding = {"type": "⚠️ 손절", "msg": "리스크 관리 가동", "color": "#dc3545", "alert": True}
        elif y < -5: pyramiding = {"type": "💧 물타기", "msg": f"{buy_names[0]} 추매 구간", "color": "#d63384", "alert": True}

    return {"buy": buy, "buy_names": buy_names, "sell": sell, "stop": stop_loss, "regime": curr['Regime'], 
            "pyramiding": pyramiding, "poc": poc, "ob": ob, "bbl": bbl, "bbu": bbu, "rsi": curr['RSI'], "stoch": curr['Stoch_K']}

# ==========================================
# 🖥️ 3. 메인 인터페이스 (Full UI 복구)
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid V64.9.4")
    now_kst = get_now_kst()
    st.info(f"**KST: {now_kst.strftime('%H:%M')}**")
    tg_token = st.text_input("Telegram Bot Token", type="password")
    tg_id = st.text_input("Telegram Chat ID")
    st.markdown("---")
    min_marcap_input = st.number_input("최소 시총 (억 원)", value=5000)
    min_marcap = min_marcap_input * 100000000
    alert_portfolio = st.checkbox("보유종목 실시간 감시", value=True)
    alert_scanner = st.checkbox("스캐너 고득점 알림 발송", value=True)
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    interval = st.slider("주기(분)", 1, 60, 10)

shared_portfolio = get_portfolio_gsheets()
tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 전략 스캐너", "📈 트리플 복리 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    if not shared_portfolio.empty:
        t_buy, t_eval, dash_list, port_alert_msg, has_alert = 0.0, 0.0, [], "🚨 <b>포트폴리오 긴급 신호</b>\n\n", False
        for _, row in shared_portfolio.iterrows():
            df = fetch_stock_smart(row['Code'], days=200)
            idx_df = get_hybrid_indicators(df)
            if idx_df is not None:
                st_res = get_strategy(idx_df, row['Buy_Price'])
                cp = float(idx_df['Close'].iloc[-1])
                t_buy += (row['Buy_Price'] * row['Qty']); t_eval += (cp * row['Qty'])
                dash_list.append({"종목": row['Name'], "수익": (cp-row['Buy_Price'])*row['Qty'], "상태": st_res['pyramiding']['type']})
                if alert_portfolio and st_res['pyramiding']['alert']:
                    has_alert = True
                    port_alert_msg += f"<b>[{st_res['pyramiding']['type']}]</b> {row['Name']}\n{st_res['pyramiding']['msg']}\n\n"
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수", f"{int(t_buy):,}원")
        c2.metric("총 평가", f"{int(t_eval):,}원", f"{(t_eval-t_buy)/t_buy*100:+.2f}%" if t_buy>0 else "0%")
        c3.metric("손익", f"{int(t_eval-t_buy):,}원")
        if dash_list: st.plotly_chart(px.bar(pd.DataFrame(dash_list), x='종목', y='수익', color='상태', template="plotly_dark"), use_container_width=True)
        if has_alert: send_telegram_msg(tg_token, tg_id, port_alert_msg)
    else: st.info("➕ 관리 탭에서 종목을 먼저 등록하세요.")

# --- [💼 탭 1: AI 리포트 (차트 및 보조지표 설명 복구)] ---
with tabs[1]:
    if not shared_portfolio.empty:
        selected_name = st.selectbox("진단할 종목 선택", shared_portfolio['Name'].tolist())
        row = shared_portfolio[shared_portfolio['Name'] == selected_name].iloc[0]
        with st.spinner(f"{selected_name} 정밀 진단 중..."):
            df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
            if df_ai is not None:
                st_res = get_strategy(df_ai, row['Buy_Price'])
                py = st_res['pyramiding']
                st.markdown(f'<div class="guide-box" style="border-left:10px solid {py["color"]};"><h2>{py["type"]}</h2><p>{py["msg"]}</p></div>', unsafe_allow_html=True)
                
                col_b, col_s = st.columns(2)
                with col_b: st.markdown(f'<div class="buy-box"><b>🔵 유기적 3분할 매수</b><br>1차({st_res["buy_names"][0]}): {st_res["buy"][0]:,}원<br>2차: {st_res["buy"][1]:,}원<br>3차: {st_res["buy"][2]:,}원</div>', unsafe_allow_html=True)
                with col_s: st.markdown(f'<div class="sell-box"><b>🔴 트리플 3분할 매도</b><br>1차: {st_res["sell"][0]:,}원<br>2차: {st_res["sell"][1]:,}원<br>3차: {st_res["sell"][2]:,}원</div>', unsafe_allow_html=True)
                
                # 정밀 차트 시각화 (BB, POC, OB 포함)
                
                fig = go.Figure()
                d_tail = df_ai.tail(120)
                fig.add_trace(go.Candlestick(x=d_tail.index, open=d_tail['Open'], high=d_tail['High'], low=d_tail['Low'], close=d_tail['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=d_tail.index, y=d_tail['BB_Upper'], line=dict(color='rgba(173,216,230,0.5)'), name="BB Upper"))
                fig.add_trace(go.Scatter(x=d_tail.index, y=d_tail['BB_Lower'], line=dict(color='rgba(173,216,230,0.5)'), name="BB Lower"))
                fig.add_hline(y=st_res['poc'], line_width=2, line_color="green", annotation_text="POC")
                fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB")
                fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 보조지표 요약 섹션 복구
                c1, c2, c3 = st.columns(3)
                c1.metric("RSI (강도)", f"{st_res['rsi']:.1f}")
                c2.metric("Stoch (위치)", f"{st_res['stoch']:.1f}")
                c3.metric("트렌드", st_res['regime'])
    else: st.warning("포트폴리오가 비어 있습니다.")

# --- [🔍 탭 2: 전략 스캐너 (15스레드 고성능)] ---
with tabs[2]:
    if st.button(f"🚀 초고속 유기적 전수조사 (Top 100)"):
        krx = fdr.StockListing('KRX')
        targets = krx[krx['Marcap'] >= min_marcap].sort_values('Marcap', ascending=False).head(100)
        found, has_scan, scan_msg = [], False, "🔍 <b>V64.9.4 발굴 결과</b>\n\n"
        prog_bar = st.progress(0)
        
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'], days=300)): r['Name'] for _, r in targets.iterrows()}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res is not None:
                    curr = res.iloc[-1]; st_res = get_strategy(res)
                    # 수급 및 Confluence 점수 합산 로직
                    sc = curr['Vol_Zscore'] * 15 + (25 if curr['RSI'] < 35 else 0) + (25 if abs(curr['Close']-curr['POC_Price'])/(curr['POC_Price']+1) < 0.02 else 0)
                    found.append({"name": futs[f], "score": sc, "regime": st_res['regime'], "strat": st_res})
                prog_bar.progress((i + 1) / len(targets))

        for idx, d in enumerate(sorted(found, key=lambda x: x['score'], reverse=True)[:10]):
            acc_c = "#007bff" if d['regime'] == "🚀 상승" else "#dc3545"
            st.markdown(f"""<div class="scanner-card" style="border-left: 8px solid {acc_c};">
                <h3 style="margin:0; color:{acc_c};">{d['name']} <small>Score: {d['score']:.1f}</small></h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top:10px;">
                    <div class="buy-box"><b>🔵 유기적 매수 (30:30:40)</b><br>1차({d['strat']['buy_names'][0]}): {d['strat']['buy'][0]:,}원<br>2차: {d['strat']['buy'][1]:,}원<br>3차: {d['strat']['buy'][2]:,}원</div>
                    <div class="sell-box"><b>🔴 트리플 매도 (30:30:40)</b><br>1차: {d['strat']['sell'][0]:,}원<br>2차: {d['strat']['sell'][1]:,}원<br>3차: {d['strat']['sell'][2]:,}원</div>
                </div>
            </div>""", unsafe_allow_html=True)
            if alert_scanner and idx < 3:
                has_scan = True; scan_msg += f"🔥 <b>{d['name']}</b> ({d['score']:.1f}점)\n매수: {d['strat']['buy'][0]:,}원\n\n"
        if has_scan: send_telegram_msg(tg_token, tg_id, scan_msg)

# --- [📈 탭 3: 트리플 복리 백테스트 (통계 분석 강화)] ---
with tabs[3]:
    
    st.header("📈 트리플 분할 매매 복리 시뮬레이션")
    bt_name = st.text_input("검증 종목명", "에코프로비엠")
    init_seed = st.number_input("초기 자본 (원)", value=10000000)
    
    if st.button("📊 트리플 백테스트 실행"):
        krx = fdr.StockListing('KRX'); match = krx[krx['Name'] == bt_name]
        if not match.empty:
            with st.spinner('실전형 3단계 시뮬레이션 중...'):
                df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
                if df_bt is not None:
                    cash, stocks, in_pos, pos_size = init_seed, 0, False, 0
                    buy_levels, sell_levels, equity_curve = [], [], []
                    for i in range(120, len(df_bt)):
                        curr = df_bt.iloc[i]; strat = get_strategy(df_bt.iloc[:i]); cp = curr['Close']
                        if not in_pos:
                            if curr['Low'] <= strat['buy'][0]:
                                in_pos, buy_levels, sell_levels, pos_size = True, strat['buy'], strat['sell'], 0.3
                                buy_amt = cash * 0.3; stocks = buy_amt / buy_levels[0]; cash -= buy_amt
                        else:
                            # 3분할 추가 매수 및 매도 로직
                            if pos_size == 0.3 and curr['Low'] <= buy_levels[1]:
                                add_amt = (cash + (stocks*buy_levels[0])) / 0.7 * 0.3
                                stocks += (add_amt / buy_levels[1]); cash -= add_amt; pos_size = 0.6
                            elif pos_size == 0.6 and curr['Low'] <= buy_levels[2]:
                                add_amt = cash; stocks += (add_amt / buy_levels[2]); cash -= add_amt; pos_size = 1.0
                            if stocks > 0:
                                if curr['High'] >= sell_levels[0] and pos_size >= 0.3:
                                    s_qty = stocks * 0.3; cash += (s_qty * sell_levels[0]); stocks -= s_qty
                                if curr['High'] >= sell_levels[1] and stocks > 0:
                                    s_qty = stocks * 0.4; cash += (s_qty * sell_levels[1]); stocks -= s_qty
                                if curr['High'] >= sell_levels[2] or curr['Low'] <= strat['stop']:
                                    cash += (stocks * cp); stocks = 0; in_pos = False; pos_size = 0
                        equity_curve.append({'date': df_bt.index[i], 'total': cash + (stocks * cp)})
                    
                    edf = pd.DataFrame(equity_curve)
                    edf['peak'] = edf['total'].cummax(); edf['drawdown'] = (edf['total'] - edf['peak']) / (edf['peak'] + 1e-9) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("최종 자산", f"{int(edf['total'].iloc[-1]):,}원")
                    m2.metric("누적 수익률", f"{(edf['total'].iloc[-1]-init_seed)/init_seed*100:+.2f}%")
                    m3.metric("최대 낙폭(MDD)", f"{edf['drawdown'].min():.2f}%")
                    
                    
                    st.plotly_chart(px.line(edf, x='date', y='total', title="복리 자산 성장 곡선"), use_container_width=True)

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = shared_portfolio
    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3); n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("등록"):
            match = fdr.StockListing('KRX')[fdr.StockListing('KRX')['Name'] == n]
            if not match.empty:
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code', 'Name', 'Buy_Price', 'Qty'])
                st.connection("gsheets", type=GSheetsConnection).update(data=pd.concat([df_p, new_row]))
                st.rerun()
    st.dataframe(df_p, use_container_width=True)

if auto_refresh: time.sleep(interval * 60); st.rerun()
