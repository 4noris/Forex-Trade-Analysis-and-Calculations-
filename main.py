import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import numpy as np
from io import BytesIO
import re

st.set_page_config(
    page_title="Rotex Trade Suite",
    page_icon="📈",
    layout="wide"
)

# ---------- Session State ----------
if "ip_history" not in st.session_state:
    st.session_state.ip_history = []
if "extracted_account_id" not in st.session_state:
    st.session_state.extracted_account_id = "N/A"

# ---------- Minimal CSS Theme ----------
THEME_CSS = r"""
<style>
html, body { overflow-x: hidden !important; }

/* Simple light background */
body {
  background-color: #f5f5f7 !important;
}

/* Hide default header + sidebar for cleaner layout */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.main .block-container {
  max-width: 100% !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  padding-top: 1.5rem !important;
  background: #f5f5f7 !important;
}

/* Simple, minimal buttons */
.stButton>button,
button[data-testid*="baseButton-"],
div[data-testid*="stDownloadButton"] > button {
  background: #ffffff !important;
  color: #111827 !important;
  border-radius: 6px !important;
  border: 1px solid #d1d5db !important;
  font-weight: 500 !important;
  padding: 0.4rem 1.1rem !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
  transition: background-color 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
}
.stButton>button:hover,
button[data-testid*="baseButton-"]:hover,
div[data-testid*="stDownloadButton"] > button:hover {
  background: #f3f4f6 !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.06) !important;
  transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border-radius: 6px !important;
  border: 1px solid #d1d5db !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  border-bottom: 1px solid #e5e7eb !important;
}
button[role="tab"] {
  border-radius: 6px 6px 0 0 !important;
}
button[role="tab"][aria-selected="true"] {
  background: #ffffff !important;
  border-bottom: 2px solid #111827 !important;
}

/* IP cards */
.ip-card {
  background: #ffffff;
  border-radius: 8px;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  margin-bottom: 1rem;
  text-align: left;
}

/* Logo alignment */
.logo-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.6rem;
}
.logo-row img {
  max-height: 70px;
  width: auto;
  object-fit: contain;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------- Forex Helpers ----------

@st.cache_data
def detect_pip_size(pair: str) -> float:
    pair = pair.upper()
    metals = ['XAUUSD', 'XAUAUD', 'XPDUSD', 'XAGAUD', 'XAGEUR', 'XAGUSD', 'XAUEUR', 'XPTUSD', 'XALUSD']
    indices = ['AUS200', 'CHINA50', 'ESP35', 'EU50', 'FRA40', 'GER40', 'HK50', 'JPN225', 'UK100', 'US100', 'US30',
               'US500']
    cryptos = ['ADAUSD', 'ATOUSD', 'BCHUSD', 'BNBUSD', 'BTCUSD', 'DOGUSD', 'DOTUSD', 'ETHUSD', 'LTCUSD', 'SHBUSD',
               'SOLUSD', 'TRXUSD', 'XRPUSD']

    if pair.endswith('JPY'):
        return 0.01
    if pair in metals:
        return 0.01
    if pair in indices:
        return 1
    if pair in cryptos:
        return 0.1
    return 0.0001


def contract_size(lot_type: str, custom_lot_size):
    if lot_type == 'standard':
        return 100000
    if lot_type == 'mini':
        return 10000
    if lot_type == 'micro':
        return 1000
    if lot_type == 'custom':
        try:
            return float(custom_lot_size)
        except (ValueError, TypeError):
            return 0
    return 0

# ---------- Trade Analyzer Core ----------

def parse_datetime_flex(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format='%Y.%m.%d %H:%M:%S.%f', errors='coerce')
    parsed = parsed.fillna(pd.to_datetime(series, format='%Y.%m.%d %H:%M:%S', errors='coerce'))
    return parsed


def analyze_trades(uploaded_file, scalping_limit: int = 3, file_name: str | None = None) -> dict:
    """
    Unified trade analyzer:
    - Extract Account ID from header or filename
    - Detect scalping / reversal / burst
    - Compute equity curve and symbol stats
    """
    extracted_account_id = None
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)

        # --- Try to extract Account from top rows ---
        df_head = df.head(10).fillna('')
        for _, row in df_head.iterrows():
            for col_idx, cell in row.items():
                if isinstance(cell, str) and re.search(r'\bAccount:\b', cell, re.IGNORECASE):
                    for val_col in range(col_idx + 1, min(col_idx + 6, len(row))):
                        account_info = str(row.get(val_col, '')).strip()
                        if account_info:
                            m = re.search(r'(\d+)', account_info)
                            if m:
                                extracted_account_id = m.group(1)
                                break
                    break
            if extracted_account_id:
                break

        # Fallback to filename
        if not extracted_account_id and file_name:
            m = re.search(r'ReportHistory[-_ ]?(\d+)', file_name, re.IGNORECASE)
            if m:
                extracted_account_id = m.group(1)

        # --- Locate Positions section ---
        start_idx = df.index[df[0].astype(str).str.contains(r'\bPositions\b', case=False, na=False)].tolist()
        end_idx = df.index[df[0].astype(str).str.contains(r'\bOrders\b', case=False, na=False)].tolist()

        if not start_idx:
            return {
                "error": "Could not find 'Positions' section in this report.",
                "extracted_account_id": extracted_account_id
            }

        start = start_idx[0] + 1
        end = end_idx[0] if end_idx else len(df)
        positions_raw = df.iloc[start:end].dropna(how='all')

        if positions_raw.empty:
            return {
                "error": "No position rows found under 'Positions' section.",
                "extracted_account_id": extracted_account_id
            }

        header_row = positions_raw.iloc[0]
        positions_df = positions_raw[1:].reset_index(drop=True)
        positions_df.columns = header_row

        positions_df = positions_df.rename(columns={"Time": "Open Time", "Price": "Open Price"})
        if 'Close Time' not in positions_df.columns and len(positions_df.columns) > 8:
            positions_df.columns.values[8] = 'Close Time'
        if 'Close Price' not in positions_df.columns and len(positions_df.columns) > 9:
            positions_df.columns.values[9] = 'Close Price'

        positions_df['Open Time'] = parse_datetime_flex(positions_df.get('Open Time'))
        positions_df['Close Time'] = parse_datetime_flex(positions_df.get('Close Time'))

        positions_df = positions_df.dropna(subset=['Open Time', 'Close Time'])
        if positions_df.empty:
            return {
                "error": "Could not parse any valid timestamps from the positions.",
                "extracted_account_id": extracted_account_id
            }

        positions_df['Profit'] = pd.to_numeric(positions_df.get('Profit'), errors='coerce').fillna(0)
        positions_df['Volume'] = pd.to_numeric(positions_df.get('Volume', 0), errors='coerce').fillna(0)
        positions_df['Hold_Time'] = positions_df['Close Time'] - positions_df['Open Time']

        # --- Scalping ---
        scalping_df = positions_df[positions_df['Hold_Time'] <= pd.Timedelta(minutes=scalping_limit)].copy()

        # Sort by open time for pattern detection and equity
        positions_df = positions_df.sort_values('Open Time').reset_index(drop=True)

        # --- Reversal (opposite Type within 20s, same Symbol) ---
        positions_df['Reversal'] = False
        for i in range(1, len(positions_df)):
            prev_close = positions_df.loc[i - 1, 'Close Time']
            curr_open = positions_df.loc[i, 'Open Time']
            prev_type = str(positions_df.loc[i - 1].get('Type', '')).strip().lower()
            curr_type = str(positions_df.loc[i].get('Type', '')).strip().lower()
            prev_symbol = str(positions_df.loc[i - 1].get('Symbol', '')).strip().upper()
            curr_symbol = str(positions_df.loc[i].get('Symbol', '')).strip().upper()

            if pd.notnull(prev_close) and pd.notnull(curr_open) and prev_symbol == curr_symbol:
                dt = (curr_open - prev_close).total_seconds()
                if dt <= 20 and (
                    (prev_type == 'buy' and curr_type == 'sell') or
                    (prev_type == 'sell' and curr_type == 'buy')
                ):
                    positions_df.loc[i, 'Reversal'] = True

        reversal_df = positions_df[positions_df['Reversal']].copy()

        # --- Burst (2+ trades within 2 seconds chain) ---
        positions_df['Burst'] = False
        burst_indices = set()
        i = 0
        while i < len(positions_df) - 1:
            current = []
            j = i
            while j < len(positions_df) - 1:
                t1 = positions_df.loc[j, 'Open Time']
                t2 = positions_df.loc[j + 1, 'Open Time']
                if pd.notnull(t1) and pd.notnull(t2):
                    dt = (t2 - t1).total_seconds()
                    if dt <= 2:
                        if not current:
                            current.append(j)
                        current.append(j + 1)
                        j += 1
                    else:
                        break
                else:
                    j += 1
            if len(current) >= 2:
                burst_indices.update(current)
                i = current[-1] + 1
            else:
                i += 1

        if burst_indices:
            positions_df.loc[list(burst_indices), 'Burst'] = True
        burst_df = positions_df[positions_df['Burst']].copy()

        # --- Stats ---
        total_positions = len(positions_df)
        total_profit = positions_df['Profit'].sum()
        total_volume = positions_df['Volume'].sum()

        scalping_count = len(scalping_df)
        scalping_profit = scalping_df['Profit'].sum()

        reversal_count = len(reversal_df)
        reversal_profit = reversal_df['Profit'].sum()

        burst_count = len(burst_df)
        burst_profit = burst_df['Profit'].sum()

        scalping_percentage = (scalping_count / total_positions * 100) if total_positions else 0
        reversal_percentage = (reversal_count / total_positions * 100) if total_positions else 0
        burst_percentage = (burst_count / total_positions * 100) if total_positions else 0

        scalping_profit_pct = (scalping_profit / total_profit * 100) if total_profit else 0
        reversal_profit_pct = (reversal_profit / total_profit * 100) if total_profit else 0
        burst_profit_pct = (burst_profit / total_profit * 100) if total_profit else 0

        avg_hold_time = positions_df['Hold_Time'].mean()
        avg_scalp_hold_time = scalping_df['Hold_Time'].mean() if scalping_count else pd.Timedelta(0)

        profit_by_symbol = positions_df.groupby('Symbol')['Profit'].sum() if 'Symbol' in positions_df.columns else pd.Series(dtype=float)
        trades_count = positions_df['Symbol'].value_counts() if 'Symbol' in positions_df.columns else pd.Series(dtype=int)

        equity_df = positions_df.sort_values('Close Time').copy()
        equity_df['Cumulative_Profit'] = equity_df['Profit'].cumsum()

        return {
            "error": None,
            "extracted_account_id": extracted_account_id,
            "total_positions": total_positions,
            "total_profit": total_profit,
            "total_volume": total_volume,
            "scalping_count": scalping_count,
            "scalping_profit": scalping_profit,
            "scalping_percentage": scalping_percentage,
            "scalping_profit_percentage": scalping_profit_pct,
            "reversal_count": reversal_count,
            "reversal_profit": reversal_profit,
            "reversal_percentage": reversal_percentage,
            "reversal_profit_percentage": reversal_profit_pct,
            "burst_count": burst_count,
            "burst_profit": burst_profit,
            "burst_percentage": burst_percentage,
            "burst_profit_percentage": burst_profit_pct,
            "avg_hold_time": avg_hold_time,
            "avg_scalping_hold_time": avg_scalp_hold_time,
            "scalping_df": scalping_df,
            "reversal_df": reversal_df,
            "burst_df": burst_df,
            "all_positions_df": positions_df,
            "profit_by_symbol": profit_by_symbol,
            "trades_count": trades_count,
            "equity_df": equity_df,
        }

    except Exception as e:
        return {
            "error": f"Error while reading file: {e}",
            "extracted_account_id": extracted_account_id
        }

# ---------- IP & Security Helpers ----------

def get_ip_details(ip_address: str) -> dict:
    if not ip_address or str(ip_address).lower() == "n/a":
        return {"error": "No IP provided."}
    try:
        resp = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def add_ip_to_history(ip: str, details: dict) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.ip_history.insert(0, {"timestamp": timestamp, "ip": ip, "details": details})
    st.session_state.ip_history = st.session_state.ip_history[:12]


def generate_security_report(analysis: dict, account_id: str, trade_ip: dict,
                             account_country: str, vps_used: str) -> str:
    total_positions = analysis['total_positions']
    total_profit = analysis['total_profit']
    scalping_count = analysis['scalping_count']
    scalping_pct = analysis['scalping_percentage']
    scalping_profit = analysis['scalping_profit']

    reversal_count = analysis['reversal_count']
    burst_count = analysis['burst_count']

    is_toxic = False
    patterns = []

    if scalping_pct >= 30:
        is_toxic = True
        patterns.append("Excessive Scalping (≥ 30% of total trades)")
    if reversal_count > total_positions * 0.03:
        is_toxic = True
        patterns.append("Frequent Reversal / Hedge-like trading")
    if burst_count > total_positions * 0.03:
        is_toxic = True
        patterns.append("Rapid Burst / HFT-like activity")

    toxic_status = "Toxic trading patterns detected." if is_toxic else "No major toxic trading pattern detected."

    city = trade_ip.get('city', 'N/A')
    country = trade_ip.get('country', 'N/A')
    trade_location = f"{city}, {country}"

    report = f"""Account {account_id}
Total trades: {total_positions} with overall profit of ${total_profit:.2f}.
Scalping trades: {scalping_count} ({scalping_pct:.1f}%) with profit of ${scalping_profit:.2f}.
Trading location appears to be {trade_location}, while registered country is {account_country}.

{toxic_status}
VPS used: {vps_used}.
"""
    if is_toxic and patterns:
        report += "Detected patterns: " + ", ".join(patterns)

    return report

# ---------- Layout: Logo + Title ----------
logo_col1, logo_col2, logo_col3 = st.columns([2, 3, 2])
with logo_col2:
    st.markdown('<div class="logo-row">', unsafe_allow_html=True)
    try:
        st.image("Rotex.png")
        st.image("Eagleeye.png")
    except Exception:
        st.write("")
    st.markdown('</div>', unsafe_allow_html=True)

st.title("Rotex EagleEye Trade Analyzer\n Forex Tools • IP Security")
st.markdown("---")

# ---------- Tabs ----------
tab_trade, tab_forex, tab_ip = st.tabs([
    "📊 Trade Analyzer",
    "🧮 Forex Calculator",
    "🌐 IP & Security",
])

# ---------- TAB 1: Trade Analyzer ----------
with tab_trade:
    st.subheader("Trade Analysis")

    upload_col, scalper_col = st.columns([3, 1])
    with upload_col:
        trade_file = st.file_uploader(
            "Upload MT4 / MT5 Trade History (.xlsx)",
            type=["xlsx"],
            key="trade_file_analyzer"
        )
    with scalper_col:
        scalping_limit = st.slider("Scalping (minutes)", 1, 5, 3)

    if trade_file:
        with st.spinner("Analyzing trade report..."):
            result = analyze_trades(trade_file, scalping_limit=scalping_limit, file_name=trade_file.name)

        if result.get("error"):
            st.error(result["error"])
        else:
            acc_id = result.get("extracted_account_id") or st.session_state.extracted_account_id or "Unknown"
            st.session_state.extracted_account_id = acc_id

            st.markdown(f"**Detected Account ID:** `{acc_id}`")

            # --- High-level Metrics ---
            st.subheader("Overall Statistics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Trades", result['total_positions'])
            with m2:
                st.metric("Total Profit", f"${result['total_profit']:.2f}")
            with m3:
                st.metric("Avg Hold Time", str(result['avg_hold_time']).split('.')[0])
            with m4:
                ppt = result['total_profit'] / result['total_positions'] if result['total_positions'] else 0
                st.metric("Profit / Trade", f"${ppt:.2f}")

            st.metric("Total Volume Traded", f"{result['total_volume']:.2f}")

            # --- Scalping metrics ---
            st.subheader(f"Scalping (≤ {scalping_limit} min)")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Scalping Trades", result['scalping_count'],
                          delta=f"{result['scalping_percentage']:.1f}% of total")
            with s2:
                st.metric("Scalping Profit", f"${result['scalping_profit']:.2f}",
                          delta=f"{result['scalping_profit_percentage']:.1f}% of total profit")
            with s3:
                if len(result['scalping_df']) > 0:
                    win_rate = (result['scalping_df']['Profit'] > 0).mean() * 100
                    st.metric("Scalping Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Scalping Win Rate", "N/A")
            with s4:
                avg_s = result['avg_scalping_hold_time']
                st.metric("Avg Scalp Time", str(avg_s).split('.')[0] if result['scalping_count'] else "N/A")

            # Downloads for scalping
            if result['scalping_count'] > 0:
                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_data = result['scalping_df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Scalping Trades (CSV)",
                        data=csv_data,
                        file_name=f"scalping_trades_{acc_id}.csv",
                        mime="text/csv",
                    )
                with dl2:
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        result['scalping_df'].to_excel(writer, index=False, sheet_name="Scalping")
                    buf.seek(0)
                    st.download_button(
                        "Scalping Trades (Excel)",
                        data=buf,
                        file_name=f"scalping_trades_{acc_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            # --- Reversal metrics ---
            st.subheader("Reversal Trades (Opposite side within 20s, same symbol)")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Reversal Trades", result['reversal_count'],
                          delta=f"{result['reversal_percentage']:.1f}% of total")
            with r2:
                st.metric("Reversal Profit", f"${result['reversal_profit']:.2f}",
                          delta=f"{result['reversal_profit_percentage']:.1f}% of total profit")
            with r3:
                if len(result['reversal_df']) > 0:
                    wr = (result['reversal_df']['Profit'] > 0).mean() * 100
                    st.metric("Reversal Win Rate", f"{wr:.1f}%")
                else:
                    st.metric("Reversal Win Rate", "N/A")
            with r4:
                avg_rev_profit = result['reversal_df']['Profit'].mean() if len(result['reversal_df']) > 0 else 0
                st.metric("Avg Reversal Profit", f"${avg_rev_profit:.2f}" if len(result['reversal_df']) else "N/A")

            if result['reversal_count'] > 0:
                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_data = result['reversal_df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Reversal Trades (CSV)",
                        data=csv_data,
                        file_name=f"reversal_trades_{acc_id}.csv",
                        mime="text/csv",
                    )
                with dl2:
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        result['reversal_df'].to_excel(writer, index=False, sheet_name="Reversal")
                    buf.seek(0)
                    st.download_button(
                        "Reversal Trades (Excel)",
                        data=buf,
                        file_name=f"reversal_trades_{acc_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            # --- Burst metrics ---
            st.subheader("Burst Trades (≥ 2 trades within 2 seconds)")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Burst Trades", result['burst_count'],
                          delta=f"{result['burst_percentage']:.1f}% of total")
            with b2:
                st.metric("Burst Profit", f"${result['burst_profit']:.2f}",
                          delta=f"{result['burst_profit_percentage']:.1f}% of total profit")
            with b3:
                if len(result['burst_df']) > 0:
                    wr = (result['burst_df']['Profit'] > 0).mean() * 100
                    st.metric("Burst Win Rate", f"{wr:.1f}%")
                else:
                    st.metric("Burst Win Rate", "N/A")
            with b4:
                avg_burst_profit = result['burst_df']['Profit'].mean() if len(result['burst_df']) > 0 else 0
                st.metric("Avg Burst Profit", f"${avg_burst_profit:.2f}" if len(result['burst_df']) else "N/A")

            if result['burst_count'] > 0:
                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_data = result['burst_df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Burst Trades (CSV)",
                        data=csv_data,
                        file_name=f"burst_trades_{acc_id}.csv",
                        mime="text/csv",
                    )
                with dl2:
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        result['burst_df'].to_excel(writer, index=False, sheet_name="Burst")
                    buf.seek(0)
                    st.download_button(
                        "Burst Trades (Excel)",
                        data=buf,
                        file_name=f"burst_trades_{acc_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            # --- Visuals ---
            st.subheader("Visual Analysis")
            v1, v2, v3 = st.columns(3)

            with v1:
                st.markdown("**Trade Type Distribution**")
                scalp = result['scalping_count']
                rev = result['reversal_count']
                burst = result['burst_count']
                others = result['total_positions'] - (scalp + rev + burst)
                pie_df = pd.DataFrame({
                    "Category": ["Scalping", "Reversal", "Burst", "Other"],
                    "Count": [scalp, rev, burst, max(0, others)],
                })
                fig = px.pie(pie_df, names="Category", values="Count")
                fig.update_traces(textinfo="percent+label", textposition="inside")
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.markdown("**Profit / Loss by Symbol**")
                pbs = result['profit_by_symbol']
                if not pbs.empty:
                    colors = ["#10b981" if v >= 0 else "#ef4444" for v in pbs.values]
                    fig2 = px.bar(
                        x=pbs.values,
                        y=pbs.index,
                        orientation="h",
                        labels={"x": "Profit / Loss", "y": "Symbol"},
                    )
                    fig2.update_traces(marker_color=colors)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No symbol column available in this report.")

            with v3:
                st.markdown("**Number of Trades per Symbol**")
                tc = result['trades_count']
                if not tc.empty:
                    pbs = result['profit_by_symbol']
                    colors = ["#10b981" if pbs.get(sym, 0) >= 0 else "#ef4444" for sym in tc.index]
                    fig3 = px.bar(
                        x=tc.index,
                        y=tc.values,
                        labels={"x": "Symbol", "y": "Trades"},
                    )
                    fig3.update_traces(marker_color=colors)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No symbol column available in this report.")

            st.markdown("### Equity Curve (Cumulative Profit)")
            eq = result['equity_df']
            fig_eq = px.line(eq, x="Close Time", y="Cumulative_Profit", markers=True)
            fig_eq.update_layout(
                xaxis_title="Time",
                yaxis_title="Cumulative Profit",
                height=450,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # --- Copiable Client Summary ---
            notes = []
            if result['scalping_percentage'] >= 30:
                notes.append("Scalping Acc.")
            if result['reversal_count'] > 0 and result['reversal_percentage'] > 0:
                notes.append("Performed Hedging")
            if result['burst_count'] > 5:
                notes.append("Performed Burst Trades")

            summary = f"""Trade_Analysis_Report - {acc_id}

Overall
- Total Trades: {result['total_positions']}
- Total Profit: ${result['total_profit']:.2f}

Scalping Trades
- Scalping Trades: {result['scalping_count']}
- Scalping Profit: ${result['scalping_profit']:.2f}
- Scalping % of trades: {result['scalping_percentage']:.1f}%
- Scalping profit % of total: {result['scalping_profit_percentage']:.1f}%

Reversal Trades
- Reversal Trades: {result['reversal_count']}
- Reversal Profit: ${result['reversal_profit']:.2f}
- Reversal Trades % of total trades: {result['reversal_percentage']:.1f}%
- Reversal Profit % of total profit: {result['reversal_profit_percentage']:.1f}%

Burst Trades
- Burst Trades: {result['burst_count']}
- Burst Profit: ${result['burst_profit']:.2f}
- Burst Trades % of total trades: {result['burst_percentage']:.1f}%
- Burst Profit % of total: {result['burst_profit_percentage']:.1f}%

{("Notes: " + ", ".join(notes)) if notes else ""}"""

            st.markdown("#### Copy-ready Summary")
            st.code(summary, language="text")
            st.caption("You can copy the summary above and send it directly to the client.")

    else:
        st.info("Upload a trade history Excel file to begin analysis.")

# ---------- TAB 2: Forex Calculator ----------
with tab_forex:
    st.header("Forex Calculator")

    center = st.columns([1, 3, 1])[1]
    with center:
        calc_type = st.selectbox(
            "Calculation Type",
            ["Pip Difference", "Margin Calculator", "Pip Value & Spread Cost", "Swap Calculator"]
        )

        st.divider()

        if calc_type == "Pip Difference":
            st.subheader("Pip Difference")
            c1, c2 = st.columns(2)
            with c1:
                pair = st.text_input("Pair (e.g., EURUSD)", value="EURUSD").upper()
            with c2:
                open_price = st.number_input("Opening Price", format="%.6f")
            close_price = st.number_input("Closing Price", format="%.6f")

            placeholder = st.empty()
            if open_price > 0 and close_price > 0 and len(pair) == 6:
                pip = detect_pip_size(pair)
                pips = abs((close_price - open_price) / pip)
                placeholder.success(f"Pip Difference: {pips:.2f} pips (pip size {pip})")
            else:
                placeholder.warning("Enter valid pair (6 letters) and positive prices.")

        elif calc_type in ["Margin Calculator", "Pip Value & Spread Cost"]:
            if calc_type == "Margin Calculator":
                st.subheader("Margin Calculator")
            else:
                st.subheader("Pip Value & Spread Cost")

            c1, c2 = st.columns(2)
            with c1:
                pair = st.text_input("Pair (BASEQUOTE)", value="AUDCAD").upper()
            with c2:
                lot_type = st.selectbox("Lot Type", ["standard", "mini", "micro", "custom"])

            custom_lot = None
            if lot_type == "custom":
                custom_lot = st.number_input("Custom Contract Size", min_value=1.0, step=1.0)

            c1, c2, c3 = st.columns(3)
            with c1:
                lots = st.number_input("Lots", min_value=0.01, value=1.0, step=0.01)
            with c2:
                price = st.number_input("Current Market Price", format="%.6f")
            with c3:
                leverage = st.number_input("Leverage (e.g., 100 for 1:100)", min_value=1, value=100, step=1)

            cross_rate = st.number_input(
                "Cross Rate (USD/Quote)",
                min_value=0.0,
                format="%.6f",
                help="If QUOTE is not USD, enter USD/QUOTE rate."
            )

            if calc_type == "Margin Calculator":
                equity = st.number_input("Account Equity (USD)", min_value=0.0, value=1000.0)

            base = pair[:3]
            quote = pair[3:]
            contract = contract_size(lot_type, custom_lot)

            margin_usd = 0.0
            margin_quote = 0.0
            formula_text = ""

            if len(pair) == 6 and lots > 0 and price > 0 and leverage > 0 and contract > 0:
                if quote == "USD":
                    margin_usd = (lots * contract * price) / leverage
                    margin_quote = margin_usd / price
                    formula_text = "((lots * contract * price) / leverage)"
                elif base == "USD":
                    margin_quote = (lots * contract) / leverage
                    margin_usd = margin_quote * price
                    formula_text = "((lots * contract) / leverage) * price"
                else:
                    margin_quote = (lots * contract * price) / leverage
                    if cross_rate > 0:
                        margin_usd = margin_quote / cross_rate
                        formula_text = "((lots * contract * price) / leverage) / crossRate"
                    else:
                        formula_text = "Missing cross rate for non-USD pair."

            if calc_type == "Margin Calculator":
                st.subheader("Margin Result")
                if margin_usd > 0:
                    st.info(f"Formula (USD): {formula_text}")
                    st.metric(f"Margin ({quote})", f"{margin_quote:.4f} {quote}")
                    st.metric("Blocked Margin (USD)", f"${margin_usd:.2f}")
                    if equity > 0 and margin_usd > 0:
                        ml = (equity / margin_usd) * 100
                        st.metric("Margin Level %", f"{ml:.2f}%")
                else:
                    st.warning("Provide valid pair, price, leverage, lots and cross rate (if needed).")

            if calc_type == "Pip Value & Spread Cost":
                st.subheader("Pip Value & Spread Cost")
                spread = st.number_input("Spread (in pips)", min_value=0.0, value=1.0, step=0.1)

                if price > 0 and lots > 0 and contract > 0:
                    pip = detect_pip_size(pair)

                    if quote == "USD":
                        pip_value_unit = contract * pip
                    elif base == "USD":
                        pip_value_unit = (contract * pip) / price
                    else:
                        if cross_rate > 0:
                            pip_value_unit = (contract * pip) / cross_rate
                        else:
                            pip_value_unit = 0.0

                    pip_value_usd = pip_value_unit * lots
                    spread_cost = spread * pip_value_usd

                    if pip_value_usd > 0:
                        st.success(
                            f"Pip Value for {lots} lot(s): ${pip_value_usd:.2f} | "
                            f"Spread Cost per trade: ${spread_cost:.2f}"
                        )
                    else:
                        st.warning("Cannot compute pip value — check cross rate or inputs.")
                else:
                    st.warning("Enter valid price, lots, and contract parameters.")

        elif calc_type == "Swap Calculator":
            st.subheader("Swap Calculator")

            c1, c2 = st.columns(2)
            with c1:
                pair = st.text_input("Pair (e.g., EURUSD)", value="EURUSD").upper()
                trade_type = st.radio("Trade Type", ["Buy", "Sell"])
            with c2:
                lots = st.number_input("Lots", min_value=0.01, value=1.0, step=0.01)
                days = st.number_input("Days Held", min_value=1, value=1, step=1)

            c3, c4, c5 = st.columns(3)
            with c3:
                swap_rate = st.number_input(
                    f"Swap Rate ({trade_type})",
                    value=-7.5,
                    help="Swap rate per lot from broker (may be positive or negative)."
                )
            with c4:
                price = st.number_input("Current Market Price", value=1.10000, format="%.6f")
            with c5:
                cross_rate = st.number_input(
                    "Cross Rate (USD/Quote)",
                    min_value=0.0,
                    value=1.0,
                    format="%.6f",
                    help="If QUOTE is not USD, enter USD/QUOTE rate."
                )

            lot_sel = st.selectbox("Lot Type", ["standard", "mini", "micro"])
            contract = contract_size(lot_sel, None)

            if lots > 0 and contract > 0 and price > 0:
                base = pair[:3]
                quote = pair[3:]

                total_swap_quote = swap_rate * lots * days

                if quote == "USD":
                    final_swap_usd = total_swap_quote
                elif cross_rate > 0:
                    final_swap_usd = total_swap_quote / cross_rate
                else:
                    st.warning("Enter cross rate to convert swap into USD.")
                    final_swap_usd = 0

                if final_swap_usd != 0:
                    st.success(f"Estimated Swap over {days} day(s): ${final_swap_usd:.2f}")
                    st.caption(
                        f"Computed as Swap Rate per Lot ({swap_rate}) × Lots ({lots}) × Days ({days}), "
                        "converted to USD."
                    )
            else:
                st.warning("Enter valid pair, lots, lot type, and price.")

# ---------- TAB 3: IP & Security ----------
with tab_ip:
    st.header("Trade Security & IP Intelligence")

    # --- Section 1: Trade + IP Security for one account ---
    st.subheader("Trade & Security Analyzer (Single Account)")

    col_file, col_country = st.columns(2)
    with col_file:
        sec_file = st.file_uploader(
            "Trade History Excel (.xlsx)",
            type=["xlsx"],
            key="trade_file_security"
        )
    with col_country:
        account_country = st.text_input("Registered Account Country", value="United Arab Emirates")

    st.markdown(f"**Last Extracted Account ID:** `{st.session_state.extracted_account_id}`")

    col_ip, col_vps = st.columns(2)
    with col_ip:
        trade_ip = st.text_input(
            "Last Trading IP",
            value="103.1.200.1",
            help="Use a sample IP to test lookup."
        )
    with col_vps:
        vps_used = st.selectbox("VPS Used?", ["No", "Yes"])

    if st.button("Run Trade + IP Security Check"):
        if sec_file is None:
            st.error("Please upload a trade history file first.")
        else:
            with st.spinner("Analyzing trade patterns..."):
                analysis = analyze_trades(sec_file, scalping_limit=3, file_name=sec_file.name)

            acc = analysis.get("extracted_account_id") or st.session_state.extracted_account_id or "Unknown"
            st.session_state.extracted_account_id = acc

            if analysis.get("error"):
                st.error(analysis["error"])
            else:
                st.success(f"Trade analysis completed for Account {acc}.")

            with st.spinner(f"Looking up IP {trade_ip}..."):
                ip_info = get_ip_details(trade_ip)

            if "error" in ip_info:
                st.warning(f"IP lookup issue for {trade_ip}: {ip_info['error']}")
                ip_info = {"city": "N/A", "country": "N/A"}

            add_ip_to_history(trade_ip, ip_info)

            if not analysis.get("error") and analysis.get("total_positions", 0) > 0:
                st.subheader("Security Summary")
                report = generate_security_report(
                    analysis=analysis,
                    account_id=acc,
                    trade_ip=ip_info,
                    account_country=account_country,
                    vps_used=vps_used
                )
                st.code(report, language="text")

                # quick risk snapshot
                total_pos = analysis["total_positions"]
                total_profit = analysis["total_profit"]
                wins = analysis["all_positions_df"][analysis["all_positions_df"]["Profit"] > 0].shape[0]
                win_rate = (wins / total_pos * 100) if total_pos else 0

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Total Trades", total_pos)
                with r2:
                    st.metric("Total Profit", f"${total_profit:.2f}")
                with r3:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.info("No valid trades, so detailed security report could not be produced.")

    st.markdown("---")

    # --- Section 2: Bulk IP Lookup with cards ---
    st.subheader("Bulk IP Lookup")

    ip_text = st.text_area(
        "IP addresses (comma or newline separated)",
        placeholder="8.8.8.8, 1.1.1.1\n203.0.113.1",
        height=100
    )
    lookup_btn = st.button("Lookup IPs")

    if lookup_btn and ip_text.strip():
        ip_list = [x.strip() for x in ip_text.replace("\n", ",").split(",") if x.strip()]
        st.info(f"Looking up {len(ip_list)} IP(s)...")
        for ip in ip_list:
            with st.spinner(f"Looking up {ip}..."):
                info = get_ip_details(ip)
                add_ip_to_history(ip, info)

    if st.session_state.ip_history:
        st.subheader("Recent IP Lookups")
        per_row = 3
        entries = st.session_state.ip_history

        for i in range(0, len(entries), per_row):
            row = st.columns(per_row)
            for j, entry in enumerate(entries[i:i+per_row]):
                with row[j]:
                    details = entry["details"]
                    if "error" in details:
                        st.error(f"{entry['ip']}: {details['error']}")
                        continue

                    city = details.get("city", "N/A")
                    region = details.get("region", "N/A")
                    country = details.get("country", "N/A")
                    org = details.get("org", "N/A")
                    loc = details.get("loc")
                    tz = details.get("timezone", "N/A")

                    st.markdown(
                        f"""
                        <div class="ip-card">
                          <strong>{entry['ip']}</strong><br/>
                          <span>{city}, {region}, {country}</span><br/><br/>
                          <span><b>ISP:</b> {org}</span><br/>
                          <span><b>Timezone:</b> {tz}</span><br/>
                          <span><b>Coords:</b> {loc if loc else "N/A"}</span><br/>
                          <span style="font-size:0.75rem; color:#6b7280;">⏱ {entry['timestamp']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if loc:
                        try:
                            lat, lon = map(float, loc.split(","))
                            st.map(
                                pd.DataFrame({"lat": [lat], "lon": [lon]}),
                                use_container_width=True,
                                height=160,
                            )
                        except Exception:
                            pass

        c = st.columns([1, 4, 1])[1]
        with c:
            if st.button("Clear IP History", use_container_width=True):
                st.session_state.ip_history = []
                st.experimental_rerun()
    else:
        st.info("No IP lookups yet — run a lookup above to see results.")

st.markdown("Built with ❤ using Streamlit • Rotex • EagleEye")
