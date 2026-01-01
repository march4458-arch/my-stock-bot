import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf
import datetime, os, time, requests, random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_gsheets import GSheetsConnection  # 구글 시트 연결 추가

# ==========================================
# ⚙️ 1. 시스템 설정 및 데이터 관리
# ==========================================
st.set_page_config(page_title="주식 비서 V62.1 Full Spec Pro", page_icon="⚡", layout="wide")

# 구글 시트 연결 함수 (데이터 로드 및 저장)
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # 캐시를 사용하지 않고 실시간으로 읽기 (ttl=0)
        df = conn.read(ttl=0)
        return df.dropna(how='all') if df is not None else pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
        st.error(f"구글 시트 연결 오류: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

def save_portfolio_gsheets(df):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.update(data=df)
        st.success("구글 시트에 동기화되었습니다!")
    except Exception as e:
        st.error(f"구글 시트 저장 실패: {e}")

def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            st.error(f"텔레그램 전송 실패: {e}")

@st.cache_data(ttl=3600)
def get_krx_list(): 
    return fdr.StockListing('KRX')

@st.cache_data(ttl=600)
def get_fear_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/static/data"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        return r.json()['now']['value'], r.json()['now']['value_text']
    except: return 50, "Neutral"


# ==========================================
# 🧠 2. 고도화된 분석 엔진 (전략 및 점수화)
# ==========================================
def fetch_stock_smart(code, days=1100):
    code_str = str(code).zfill(6)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
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
    if df is None or len(df) < 120: return None
    df = df.copy()
    close = df['Close']
    df['MA20'] = close.rolling(20).mean()
    df['MA120'] = close.rolling(120).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    
    ob_zones = []
    avg_vol = df['Volume'].rolling(20).mean()
    for i in range(len(df)-40, len(df)-1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] * 1.025 and df['Volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            ob_zones.append(df['Low'].iloc[i-1])
    df['OB_Price'] = np.mean(ob_zones) if ob_zones else df['MA20'].iloc[-1]
    
    hi_1y, lo_1y = df.tail(252)['High'].max(), df.tail(252)['Low'].min()
    range_1y = hi_1y - lo_1y
    df['Fibo_382'] = hi_1y - (range_1y * 0.382)
    df['Fibo_500'] = hi_1y - (range_1y * 0.500)
    df['Fibo_618'] = hi_1y - (range_1y * 0.618)
    
    slope = (df['MA120'].iloc[-1] - df['MA120'].iloc[-20]) / df['MA120'].iloc[-20] * 100
    df['Regime'] = "🚀 상승" if slope > 0.4 else "📉 하락" if slope < -0.4 else "↔️ 횡보"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp, atr, ob = curr['Close'], curr['ATR'], curr['OB_Price']
    f382, f500, f618 = curr['Fibo_382'], curr['Fibo_500'], curr['Fibo_618']
    
    def adj(p):
        t = 1 if p<2000 else 5 if p<5000 else 10 if p<20000 else 50 if p<50000 else 100 if p<200000 else 500 if p<500000 else 1000
        return int(round(p/t)*t)

    regime = df['Regime'].iloc[-1]
    if regime == "🚀 상승":
        buy = [adj(cp - atr*1.1), adj(ob), adj(f500)]
        sell = [adj(cp + atr*2.5), adj(cp + atr*4.5), adj(df.tail(252)['High'].max() * 1.1)]
    elif regime == "📉 하락":
        buy = [adj(f618), adj(df.tail(252)['Low'].min()), adj(df.tail(252)['Low'].min() - atr)]
        sell = [adj(f500), adj(ob), adj(df['MA120'].iloc[-1])]
    else:
        buy = [adj(f500), adj(ob), adj(f618)]
        sell = [adj(f382), adj(df.tail(252)['High'].max()), adj(df.tail(252)['High'].max() + atr)]

    # --- 물타기 / 불타기 분석 ---
    pyramiding = {"type": "💤 관망", "msg": "현재 신규 진입이나 비중 조절 구간이 아닙니다.", "color": "#777"}
    if buy_price > 0:
        yield_pct = (cp - buy_price) / buy_price * 100
        if yield_pct < -5:
            target = min(buy)
            pyramiding = {"type": "💧 물타기(추가매수)", "msg": f"평단 대비 {yield_pct:.1f}% 손실. {target:,}원 지점에서 비중 확대 권장", "color": "#FF4B4B"}
        elif yield_pct > 7 and regime == "🚀 상승":
            target = adj(cp + atr * 0.5)
            pyramiding = {"type": "🔥 불타기(수익강화)", "msg": f"수익률 {yield_pct:.1f}% 돌파. {target:,}원 상향 돌파 시 추가 매수 가능", "color": "#4FACFE"}

    return {
        "buy": buy, "sell": sell, "stop": adj(min(buy) * 0.93),
        "regime": regime, "ob": ob, "rsi": curr['RSI'], "pyramiding": pyramiding
    }

# ==========================================
# 🖥️ 3. UI 레이아웃 및 탭 구성
# ==========================================
with st.sidebar:
    st.title("🛡️ Hybrid Turbo V62.1")
    fg_val, fg_txt = get_fear_greed_index()
    st.metric("CNN Fear & Greed", f"{fg_val}pts", fg_txt)
    st.divider()
    st.subheader("🔔 텔레그램 설정")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    st.divider()
    auto_refresh = st.checkbox("자동 갱신 활성화")
    refresh_interval = st.slider("갱신 주기 (분)", 1, 60, 5)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드] ---
with tabs[0]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        total_buy, total_eval, dash_list = 0, 0, []
        with st.spinner('실시간 자산 동기화 중...'):
            for _, row in portfolio.iterrows():
                df = fetch_stock_smart(row['Code'], days=10)
                if df is not None and not df.empty:
                    cp = float(df.iloc[-1]['Close'])
                    b_total = row['Buy_Price'] * row['Qty']; e_total = cp * row['Qty']
                    total_buy += b_total; total_eval += e_total
                    dash_list.append({"종목": str(row['Name']), "수익": float(e_total - b_total), "평가액": float(e_total)})
        
        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            c1, c2, c3 = st.columns(3)
            c1.metric("총 매수액", f"{int(total_buy):,}원")
            c2.metric("총 평가액", f"{int(total_eval):,}원", f"{((total_eval-total_buy)/total_buy*100 if total_buy>0 else 0):+.2f}%")
            c3.metric("평가손익", f"{int(total_eval-total_buy):,}원")
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', title="종목별 손익", color_continuous_scale='RdBu_r'), use_container_width=True)
            col2.plotly_chart(px.pie(df_dash, values='평가액', names='종목', hole=0.3, title="자산 비중"), use_container_width=True)
    else: st.info("관리 탭에서 보유 종목을 등록하세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = load_portfolio()
    if not portfolio.empty:
        selected = st.selectbox("진단할 종목 선택", portfolio['Name'].unique())
        s_info = portfolio[portfolio['Name'] == selected].iloc[0]
        df_detail = get_hybrid_indicators(fetch_stock_smart(s_info['Code']))
        if df_detail is not None:
            strat = calculate_organic_strategy(df_detail, buy_price=s_info['Buy_Price'])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("국면", strat['regime']); c2.metric("RSI", f"{strat['rsi']:.1f}"); c3.metric("세력방어(OB)", f"{int(strat['ob']):,}원"); c4.error(f"손절가: {strat['stop']:,}원")
            
            # 대응 가이드
            py = strat['pyramiding']
            st.markdown(f"""<div style="background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:8px solid {py['color']}; margin-bottom:20px;">
                <h3 style="margin:0; color:{py['color']};">{py['type']} 가이드</h3>
                <p style="margin:5px 0; font-size:1.1em;">{py['msg']}</p></div>""", unsafe_allow_html=True)

            col_b, col_s = st.columns(2)
            col_b.info(f"🔵 **3분할 매수 타점**\n\n1차: {strat['buy'][0]:,}원\n\n2차: {strat['buy'][1]:,}원\n\n3차: {strat['buy'][2]:,}원")
            col_s.success(f"🔴 **3분할 매도 목표**\n\n1차: {strat['sell'][0]:,}원\n\n2차: {strat['sell'][1]:,}원\n\n3차: {strat['sell'][2]:,}원")
            
            fig = go.Figure()
            df_p = df_detail.tail(200)
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['MA120'], line=dict(color='royalblue', width=2), name="MA120"))
            fig.add_hline(y=strat['ob'], line_color="yellow", annotation_text="OB")
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)


# --- [🔍 스캐너] ---
with tabs[2]:
    if st.button("🚀 수급/신뢰도순 전수 조사"):
        stocks = get_krx_list()
        # 시총 5,000억 이상 상위 50개 종목 필터링
        targets = stocks[stocks['Marcap'] >= 500000000000].sort_values(by='Marcap', ascending=False).head(50)
        found = []
        
        with st.spinner("외인/기관 수급 데이터 정밀 분석 중..."):
            with ThreadPoolExecutor(max_workers=5) as exec:
                # 분석 엔진 실행 (수급 데이터 포함)
                futures = {exec.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                
                for f in as_completed(futures):
                    name = futures[f]
                    df_scan = f.result()
                    
                    if df_scan is not None and df_scan.iloc[-1]['RSI'] < 55: # 과열되지 않은 종목 위주
                        # 전략 산출
                        s = calculate_organic_strategy(df_scan)
                        cp = df_scan.iloc[-1]['Close']
                        
                        # [고도화] 신뢰 점수 계산 (RSI + 지지선 + 수급 + 기대수익)
                        # 수급 점수 추정: 거래량 폭발 + 양봉 여부
                        vol_avg = df_scan['Volume'].rolling(10).mean().iloc[-1]
                        supply_boost = 25 if (df_scan['Volume'].iloc[-1] > vol_avg * 1.3 and df_scan['Close'].iloc[-1] > df_scan['Open'].iloc[-1]) else 0
                        
                        rsi_score = max(0, (60 - df_scan.iloc[-1]['RSI']) * 0.41)
                        ob_dist = abs(cp - s['ob']) / s['ob']
                        ob_score = max(0, 25 * (1 - ob_dist * 10))
                        upside_score = min(25, ((s['sell'][0] - cp) / cp) * 100)
                        
                        # 최종 통합 점수 (100점 만점)
                        total_score = rsi_score + ob_score + supply_boost + upside_score
                        
                        found.append({
                            "name": name, 
                            "cp": cp, 
                            "strat": s, 
                            "score": total_score
                        })
        
        # 점수 높은 순으로 정렬
        found = sorted(found, key=lambda x: x['score'], reverse=True)
        
        # 결과 출력 (V62.1 고유 UI 유지)
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            # 75점 이상인 경우 테두리 강조 컬러 변경
            border_color = "#4FACFE" if d['score'] >= 75 else "#444"
            
            st.markdown(f"""
            <div style="background:#1E1E1E; padding:20px; border-radius:15px; border-left:10px solid {border_color}; margin-bottom:15px;">
                <h3 style="margin-bottom:5px;">{icon} {d['name']} <small style="color:#aaa;">(신뢰점수: {d['score']:.1f}점)</small></h3>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; font-family:monospace; font-size:15px;">
                    <div style="background:#1B2635; padding:10px; border-radius:8px;">
                        <b style="color:#4FACFE;">🔵 매수타점</b><br>
                        1차: {d['strat']['buy'][0]:>8,}원<br>
                        2차: {d['strat']['buy'][1]:>8,}원
                    </div>
                    <div style="background:#2D1B1B; padding:10px; border-radius:8px;">
                        <b style="color:#FF4B4B;">🔴 매도목표</b><br>
                        1차: {d['strat']['sell'][0]:>8,}원<br>
                        2차: {d['strat']['sell'][1]:>8,}원
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    st.header("📈 로직 실용성 백테스트")
    t_name = st.text_input("백테스트 종목명", "삼성전자")
    c1, c2 = st.columns(2)
    tp_pct, sl_pct = c1.slider("익절 목표 %", 3.0, 20.0, 7.0), c2.slider("손절 제한 %", 3.0, 20.0, 8.0)
    if st.button("📊 시뮬레이션 가동"):
        match = get_krx_list()[get_krx_list()['Name'] == t_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code']))
            if df_bt is not None:
                trades, in_pos = [], False
                for i in range(150, len(df_bt)-1):
                    strat = calculate_organic_strategy(df_bt.iloc[:i])
                    day_low, day_high = df_bt['Low'].iloc[i], df_bt['High'].iloc[i]
                    if not in_pos:
                        if day_low <= strat['buy'][0]:
                            entry_p = strat['buy'][0]
                            exit_tp, exit_sl = entry_p * (1+tp_pct/100), entry_p * (1-sl_pct/100)
                            entry_date, in_pos = df_bt.index[i], True
                    else:
                        if day_high >= exit_tp:
                            trades.append({"exit": df_bt.index[i], "ret": tp_pct, "res": "익절"})
                            in_pos = False
                        elif day_low <= exit_sl:
                            trades.append({"exit": df_bt.index[i], "ret": -sl_pct, "res": "손절"})
                            in_pos = False
                if trades:
                    tdf = pd.DataFrame(trades)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("승률", f"{(tdf['res']=='익절').sum()/len(tdf)*100:.1f}%")
                    r2.metric("누적 수익", f"{tdf['ret'].sum():.2f}%")
                    st.plotly_chart(px.line(tdf, x='exit', y=tdf['ret'].cumsum(), title="수익 곡선", template="plotly_dark"))
                else: st.warning("체결 내역 없음")

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 종목 추가")
        n = st.text_input("종목명"); p = st.number_input("평단가", 0); q = st.number_input("수량", 0)
        if st.button("저장"):
            match = get_krx_list()[get_krx_list()['Name'] == n]
            if not match.empty:
                df_p = load_portfolio()
                new_row = pd.DataFrame([[match.iloc[0]['Code'], n, p, q]], columns=['Code','Name','Buy_Price','Qty'])
                pd.concat([df_p, new_row]).to_csv(PORTFOLIO_FILE, index=False); st.rerun()
    with c2:
        st.subheader("🗑️ 종목 삭제")
        df_p = load_portfolio()
        if not df_p.empty:
            del_n = st.selectbox("삭제 종목", df_p['Name'].tolist())
            if st.button("삭제 실행"):
                df_p[df_p['Name'] != del_n].to_csv(PORTFOLIO_FILE, index=False); st.rerun()

if auto_refresh:
    time.sleep(refresh_interval * 60); st.rerun()
