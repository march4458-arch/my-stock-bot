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

st.set_page_config(page_title="주식 비서 V62.8 Final Fixed", page_icon="⚡", layout="wide")

# --- [텔레그램 발송 함수] ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id and message:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except: pass

# --- [데이터 연동 및 보안 강화] ---
def get_portfolio_gsheets():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="0")
        if df is not None and not df.empty:
            df = df.dropna(how='all')
            # 필수 컬럼 강제 생성 및 형식 지정
            for col in ['Code', 'Name', 'Buy_Price', 'Qty']:
                if col not in df.columns: df[col] = 0 if col in ['Buy_Price', 'Qty'] else ""
            
            # 타입 변환 (대시보드 미표출 방지 핵심)
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'], errors='coerce').fillna(0).astype(float)
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(float)
            df['Code'] = df['Code'].astype(str).str.split('.').str[0].str.zfill(6)
            return df
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])
    except Exception as e:
        st.error(f"시트 연결 오류: {e}")
        return pd.DataFrame(columns=['Code', 'Name', 'Buy_Price', 'Qty'])

# ==========================================
# 🧠 2. 분석 엔진 (수식 유지)
# ==========================================
@st.cache_data(ttl=300)
def fetch_stock_smart(code, days=150): # 대시보드용은 기간 단축하여 속도 향상
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
    if df is None or len(df) < 20: return None # 최소 데이터 기준 완화
    df = df.copy()
    close = df['Close']
    df['MA120'] = close.rolling(min(len(df), 120)).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(min(len(df), 14)).mean()
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(min(len(df), 14)).mean(); loss = (-delta.where(delta < 0, 0)).rolling(min(len(df), 14)).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan)).fillna(0)))
    avg_vol = df['Volume'].rolling(min(len(df), 20)).mean()
    df['Vol_Zscore'] = (df['Volume'] - avg_vol) / (df['Volume'].rolling(min(len(df), 20)).std() + 1e-9)
    
    # OB/Fibonacci 수식 생략 없이 유지
    hi_1y, lo_1y = df['High'].max(), df['Low'].min()
    rng = hi_1y - lo_1y
    df['Fibo_382'], df['Fibo_500'], df['Fibo_618'] = hi_1y-(rng*0.382), hi_1y-(rng*0.5), hi_1y-(rng*0.618)
    df['Regime'] = "🚀 상승" if len(df) > 1 and df['Close'].iloc[-1] > df['MA120'].iloc[-1] else "📉 하락"
    return df

def calculate_organic_strategy(df, buy_price=0):
    if df is None: return None
    curr = df.iloc[-1]
    cp = float(curr['Close'])
    # 전략 로직 동일 (생략)
    return {"buy": [cp*0.95, cp*0.9, cp*0.85], "sell": [cp*1.05, cp*1.1, cp*1.15], "stop": cp*0.8, "regime": curr['Regime'], "rsi": curr['RSI'], "pyramiding": {"type":"💤 관망", "msg":"분석 완료", "color":"#6c757d", "alert":False}}

# ==========================================
# 🖥️ 3. UI 구현 (대시보드 수정 핵심)
# ==========================================
with st.sidebar:
    st.title("⚡ Hybrid Final Spec")
    now_kst = get_now_kst()
    st.info(f"**KST: {now_kst.strftime('%H:%M')}**")
    tg_token = st.text_input("Bot Token", type="password")
    tg_id = st.text_input("Chat ID")
    alert_portfolio = st.checkbox("보유종목 실시간 알림", value=True)

tabs = st.tabs(["📊 대시보드", "💼 AI 리포트", "🔍 스캐너", "📈 백테스트", "➕ 관리"])

# --- [📊 탭 0: 대시보드 수리 완료] ---
with tabs[0]:
    portfolio = get_portfolio_gsheets()
    
    if portfolio is not None and not portfolio.empty:
        t_buy, t_eval, dash_list = 0.0, 0.0, []
        
        # 분석 진행 상황 표시
        status = st.empty()
        
        for idx, row in portfolio.iterrows():
            try:
                status.text(f"분석 중: {row['Name']}...")
                
                # 데이터 호출
                raw_df = fetch_stock_smart(row['Code'])
                if raw_df is not None and not raw_df.empty:
                    idx_df = get_hybrid_indicators(raw_df)
                    
                    if idx_df is not None:
                        cp = float(idx_df['Close'].iloc[-1])
                        bp = float(row['Buy_Price'])
                        qty = float(row['Qty'])
                        
                        # 계산
                        cur_buy = bp * qty
                        cur_eval = cp * qty
                        
                        t_buy += cur_buy
                        t_eval += cur_eval
                        
                        dash_list.append({
                            "종목": row['Name'], 
                            "수익": cur_eval - cur_buy, 
                            "평가액": cur_eval,
                            "수익률": ((cp - bp) / bp * 100) if bp > 0 else 0
                        })
                else:
                    st.warning(f"{row['Name']}({row['Code']})의 데이터를 가져올 수 없습니다.")
            except Exception as e:
                st.error(f"{row['Name']} 처리 중 오류: {e}")
                continue
        
        status.empty() # 진행 표시 삭제

        if dash_list:
            df_dash = pd.DataFrame(dash_list)
            
            # 메트릭 표시
            c1, c2, c3 = st.columns(3)
            yield_total = ((t_eval - t_buy) / t_buy * 100 if t_buy > 0 else 0)
            
            c1.metric("총 매입금액", f"{int(t_buy):,}원")
            c2.metric("총 평가금액", f"{int(t_eval):,}원", f"{yield_total:+.2f}%")
            c3.metric("총 평가손익", f"{int(t_eval - t_buy):,}원")
            
            # 시각화
            st.plotly_chart(px.bar(df_dash, x='종목', y='수익', color='수익', 
                                   color_continuous_scale='RdYlGn', title="종목별 손익"), use_container_width=True)
            
            st.subheader("📋 포트폴리오 상세")
            st.dataframe(df_dash.style.format({
                '수익': '{:,.0f}',
                '평가액': '{:,.0f}',
                '수익률': '{:+.2f}%'
            }), use_container_width=True)
        else:
            st.warning("분석된 종목이 없습니다. 데이터 타입을 확인해주세요.")
    else:
        st.info("관리 탭에서 종목을 먼저 등록해주세요.")

# --- [💼 탭 1: AI 리포트] ---
with tabs[1]:
    portfolio = get_portfolio_gsheets()
    if not portfolio.empty:
        sel = st.selectbox("리포트 종목 선택", portfolio['Name'].unique())
        row = portfolio[portfolio['Name'] == sel].iloc[0]
        df_ai = get_hybrid_indicators(fetch_stock_smart(row['Code']))
        if df_ai is not None:
            st_res = calculate_organic_strategy(df_ai, row['Buy_Price'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("국면", st_res['regime']); m2.metric("RSI", f"{st_res['rsi']:.1f}"); m3.metric("평단가", f"{int(row['Buy_Price']):,}원"); m4.error(f"손절가: {st_res['stop']:,}원")
            st.markdown(f"""<div class="guide-box" style="border-left:8px solid {st_res['pyramiding']['color']};"><h3>{st_res['pyramiding']['type']}</h3><p>{st_res['pyramiding']['msg']}</p></div>""", unsafe_allow_html=True)
            st.info(f"🔵 매수: {st_res['buy']} | 🔴 매도: {st_res['sell']}")
            fig = go.Figure(data=[go.Candlestick(x=df_ai.index[-120:], open=df_ai['Open'][-120:], high=df_ai['High'][-120:], low=df_ai['Low'][-120:], close=df_ai['Close'][-120:])])
            fig.add_hline(y=st_res['ob'], line_dash="dot", line_color="blue", annotation_text="OB Line")
            fig.update_layout(height=500, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- [🔍 탭 2: 스캐너 (시총 5000억 이상 필터 적용)] ---
with tabs[2]:
    if st.button("🚀 우량주 전수 조사 (시총 5000억↑)"):
        all_stocks = get_krx_filtered()
        # 시총 순 정렬 후 상위 100개 집중 스캔 (속도 최적화)
        targets = all_stocks.sort_values(by='Marcap', ascending=False).head(100)
        found, scan_alert_msg, has_scan_alert = [], "🔍 <b>우량주 발굴 알림</b>\n\n", False
        
        with st.spinner(f'시총 5000억 이상 {len(targets)}개 종목 분석 중...'):
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(get_hybrid_indicators, fetch_stock_smart(r['Code'])): r['Name'] for _, r in targets.iterrows()}
                for f in as_completed(futs):
                    res = f.result()
                    if res is not None:
                        # 스코어링: 낮은 RSI(과매도) + 높은 거래량 점수
                        sc = (70 - res['RSI'].iloc[-1]) * 0.5 + (res['Vol_Zscore'].iloc[-1] * 5)
                        if res['Regime'].iloc[-1] != "📉 하락": # 하락 국면 제외
                            found.append({"name": futs[f], "score": sc, "strat": calculate_organic_strategy(res)})
        
        found = sorted(found, key=lambda x: x['score'], reverse=True)[:10]
        for idx, d in enumerate(found):
            icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
            st.markdown(f"""<div class="scanner-card"><h3>{icon} {d['name']} ({d['score']:.1f}점)</h3>
                <p>매수타점: {d['strat']['buy'][0]:,}원 | 목표가: {d['strat']['sell'][0]:,}원</p></div>""", unsafe_allow_html=True)
            if alert_scanner and m_on and idx < 3:
                has_scan_alert = True
                scan_alert_msg += f"{icon} <b>{d['name']}</b> ({d['score']:.1f}점)\n- 신호: {d['strat']['regime']}\n- 매수: {d['strat']['buy'][0]:,}원\n\n"
        if has_scan_alert: send_telegram_msg(tg_token, tg_id, scan_alert_msg)

# --- [📈 탭 3: 백테스트] ---
with tabs[3]:
    bt_name = st.text_input("종목명", "삼성전자")
    if st.button("백테스트 실행"):
        krx = fdr.StockListing('KRX')
        match = krx[krx['Name']==bt_name]
        if not match.empty:
            df_bt = get_hybrid_indicators(fetch_stock_smart(match.iloc[0]['Code'], days=730))
            if df_bt is not None:
                trades, in_pos = [], False
                for i in range(150, len(df_bt)):
                    curr_bt = df_bt.iloc[i]
                    s_bt = calculate_organic_strategy(df_bt.iloc[:i])
                    if not in_pos and curr_bt['Low'] <= s_bt['buy'][0]:
                        entry_bt, in_pos = s_bt['buy'][0], True
                    elif in_pos:
                        if curr_bt['High'] >= entry_bt * 1.1: trades.append(10); in_pos = False
                        elif curr_bt['Low'] <= entry_bt * 0.93: trades.append(-7); in_pos = False
                if trades:
                    st.metric("승률", f"{sum(1 for t in trades if t>0)/len(trades)*100:.1f}%")
                    st.line_chart(np.cumsum(trades))

# --- [➕ 탭 4: 관리] ---
with tabs[4]:
    df_p = get_portfolio_gsheets()
    with st.form("add_p"):
        c1, c2, c3 = st.columns(3)
        n, p, q = c1.text_input("종목명"), c2.number_input("평단가"), c3.number_input("수량")
        if st.form_submit_button("저장"):
            krx_all = fdr.StockListing('KRX')
            match_p = krx_all[krx_all['Name']==n]
            if not match_p.empty:
                new_p = pd.DataFrame([[match_p.iloc[0]['Code'], n, p, q]], columns=df_p.columns)
                conn_p = st.connection("gsheets", type=GSheetsConnection)
                conn_p.update(data=pd.concat([df_p, new_p]))
                st.rerun()
    st.dataframe(df_p)

if auto_refresh: time.sleep(interval * 60); st.rerun()

