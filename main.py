import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import numpy as np
import io
import re

# Page config (unified)
st.set_page_config(page_title="Rotex Forex & Trade Analyzer", page_icon="📈", layout="wide")

# Initialize session state for IP history
if 'ip_history' not in st.session_state:
    st.session_state.ip_history = []
if 'extracted_account_id' not in st.session_state:
    st.session_state.extracted_account_id = 'N/A'

# --- FULL ISO 3166-1 ALPHA-2 COUNTRY CODE MAPPING ---
COUNTRY_CODE_MAP = {
    'AD': 'Andorra', 'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AG': 'Antigua and Barbuda',
    'AI': 'Anguilla', 'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AQ': 'Antarctica',
    'AR': 'Argentina', 'AS': 'American Samoa', 'AT': 'Austria', 'AU': 'Australia', 'AW': 'Aruba',
    'AX': 'Åland Islands', 'AZ': 'Azerbaijan', 'BA': 'Bosnia and Herzegovina', 'BB': 'Barbados',
    'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso', 'BG': 'Bulgaria', 'BH': 'Bahrain',
    'BI': 'Burundi', 'BJ': 'Benin', 'BL': 'Saint Barthélemy', 'BM': 'Bermuda', 'BN': 'Brunei',
    'BO': 'Bolivia', 'BQ': 'Caribbean Netherlands', 'BR': 'Brazil', 'BS': 'The Bahamas', 'BT': 'Bhutan',
    'BV': 'Bouvet Island', 'BW': 'Botswana', 'BY': 'Belarus', 'BZ': 'Belize', 'CA': 'Canada',
    'CC': 'Cocos (Keeling) Islands', 'CD': 'Democratic Republic of the Congo', 'CF': 'Central African Republic',
    'CG': 'Republic of the Congo', 'CH': 'Switzerland', 'CI': 'Ivory Coast', 'CK': 'Cook Islands',
    'CL': 'Chile', 'CM': 'Cameroon', 'CN': 'China', 'CO': 'Colombia', 'CR': 'Costa Rica',
    'CU': 'Cuba', 'CV': 'Cape Verde', 'CW': 'Curaçao', 'CX': 'Christmas Island', 'CY': 'Cyprus',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DJ': 'Djibouti', 'DK': 'Denmark', 'DM': 'Dominica',
    'DO': 'Dominican Republic', 'DZ': 'Algeria', 'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egypt',
    'EH': 'Western Sahara', 'ER': 'Eritrea', 'ES': 'Spain', 'ET': 'Ethiopia', 'FI': 'Finland',
    'FJ': 'Fiji', 'FK': 'Falkland Islands', 'FM': 'Micronesia', 'FO': 'Faroe Islands', 'FR': 'France',
    'GA': 'Gabon', 'GB': 'United Kingdom', 'GD': 'Grenada', 'GE': 'Georgia', 'GF': 'French Guiana',
    'GG': 'Guernsey', 'GH': 'Ghana', 'GI': 'Gibraltar', 'GL': 'Greenland', 'GM': 'The Gambia',
    'GN': 'Guinea', 'GP': 'Guadeloupe', 'GQ': 'Equatorial Guinea', 'GR': 'Greece',
    'GS': 'South Georgia and the South Sandwich Islands',
    'GT': 'Guatemala', 'GU': 'Guam', 'GW': 'Guinea-Bissau', 'GY': 'Guyana', 'HK': 'Hong Kong',
    'HM': 'Heard Island and McDonald Islands', 'HN': 'Honduras', 'HR': 'Croatia', 'HT': 'Haiti',
    'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel', 'IM': 'Isle of Man',
    'IN': 'India', 'IO': 'British Indian Ocean Territory', 'IQ': 'Iraq', 'IR': 'Iran', 'IS': 'Iceland',
    'IT': 'Italy', 'JE': 'Jersey', 'JM': 'Jamaica', 'JO': 'Jordan', 'JP': 'Japan',
    'KE': 'Kenya', 'KG': 'Kyrgyzstan', 'KH': 'Cambodia', 'KI': 'Kiribati', 'KM': 'Comoros',
    'KN': 'Saint Kitts and Nevis', 'KP': 'North Korea', 'KR': 'South Korea', 'KW': 'Kuwait', 'KY': 'Cayman Islands',
    'KZ': 'Kazakhstan', 'LA': 'Laos', 'LB': 'Lebanon', 'LC': 'Saint Lucia', 'LI': 'Liechtenstein',
    'LK': 'Sri Lanka', 'LR': 'Liberia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'LY': 'Libya', 'MA': 'Morocco', 'MC': 'Monaco', 'MD': 'Moldova',
    'ME': 'Montenegro', 'MF': 'Saint-Martin', 'MG': 'Madagascar', 'MH': 'Marshall Islands', 'MK': 'North Macedonia',
    'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia', 'MO': 'Macau', 'MP': 'Northern Mariana Islands',
    'MQ': 'Martinique', 'MR': 'Mauritania', 'MS': 'Montserrat', 'MT': 'Malta', 'MU': 'Mauritius',
    'MV': 'Maldives', 'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NC': 'New Caledonia', 'NE': 'Niger', 'NF': 'Norfolk Island', 'NG': 'Nigeria',
    'NI': 'Nicaragua', 'NL': 'Kingdom of the Netherlands', 'NO': 'Norway', 'NP': 'Nepal', 'NR': 'Nauru',
    'NU': 'Niue', 'NZ': 'New Zealand', 'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru',
    'PF': 'French Polynesia', 'PG': 'Papua New Guinea', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland',
    'PM': 'Saint Pierre and Miquelon', 'PN': 'Pitcairn Islands', 'PR': 'Puerto Rico', 'PS': 'State of Palestine',
    'PT': 'Portugal',
    'PW': 'Palau', 'PY': 'Paraguay', 'QA': 'Qatar', 'RE': 'Réunion', 'RO': 'Romania',
    'RS': 'Serbia', 'RU': 'Russia', 'RW': 'Rwanda', 'SA': 'Saudi Arabia', 'SB': 'Solomon Islands',
    'SC': 'Seychelles', 'SD': 'Sudan', 'SE': 'Sweden', 'SG': 'Singapore',
    'SH': 'Saint Helena, Ascension and Tristan da Cunha',
    'SI': 'Slovenia', 'SJ': 'Svalbard and Jan Mayen', 'SK': 'Slovakia', 'SL': 'Sierra Leone', 'SM': 'San Marino',
    'SN': 'Senegal', 'SO': 'Somalia', 'SR': 'Suriname', 'SS': 'South Sudan', 'ST': 'São Tomé and Príncipe',
    'SV': 'El Salvador', 'SX': 'Sint Maarten', 'SY': 'Syria', 'SZ': 'Eswatini', 'TC': 'Turks and Caicos Islands',
    'TD': 'Chad', 'TF': 'French Southern and Antarctic Lands', 'TG': 'Togo', 'TH': 'Thailand', 'TJ': 'Tajikistan',
    'TK': 'Tokelau', 'TL': 'Timor-Leste', 'TM': 'Turkmenistan', 'TN': 'Tunisia', 'TO': 'Tonga',
    'TR': 'Turkey', 'TT': 'Trinidad and Tobago', 'TV': 'Tuvalu', 'TW': 'Taiwan', 'TZ': 'Tanzania',
    'UA': 'Ukraine', 'UG': 'Uganda', 'UM': 'United States Minor Outlying Islands', 'US': 'United States',
    'UY': 'Uruguay',
    'UZ': 'Uzbekistan', 'VA': 'Vatican City', 'VC': 'Saint Vincent and the Grenadines', 'VE': 'Venezuela',
    'VG': 'British Virgin Islands',
    'VI': 'United States Virgin Islands', 'VN': 'Vietnam', 'VU': 'Vanuatu', 'WF': 'Wallis and Futuna', 'WS': 'Samoa',
    'YE': 'Yemen', 'YT': 'Mayotte', 'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'
}


def get_country_name(code):
    """Converts a 2-letter country code to its full name."""
    if not code:
        return "N/A"
    code = code.upper().strip()
    return COUNTRY_CODE_MAP.get(code, code)


# --- Other Helper Functions (detect_pip_size, contract_size, etc.) ---

@st.cache_data
def detect_pip_size(pair):
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


def contract_size(lot_type, custom_lot_size):
    if lot_type == 'standard':
        return 100000
    elif lot_type == 'mini':
        return 10000
    elif lot_type == 'micro':
        return 1000
    elif lot_type == 'custom':
        try:
            return float(custom_lot_size)
        except (ValueError, TypeError):
            return 0
    return 0


# --- Trade Analysis Function (File Name Fallback) ---
def analyze_trades(uploaded_file, file_name=None):
    """Analyze trading positions from Excel file and extract Account ID, with file name fallback."""

    extracted_account_id = None

    try:
        df = pd.read_excel(uploaded_file, sheet_name=0, header=None)

        # --- 1. Robust Extract Account ID from File Content ---
        df_head = df.head(10).fillna('')

        for i in df_head.index:
            row = df_head.loc[i]
            for col_index, cell_value in row.items():
                if isinstance(cell_value, str) and re.search(r'\bAccount:\b', cell_value, re.IGNORECASE):
                    for val_col in range(col_index + 1, min(col_index + 6, len(row))):
                        account_info = str(row.get(val_col, '')).strip()

                        if account_info:
                            match = re.search(r'(\d+)', account_info)
                            if match:
                                extracted_account_id = match.group(1)
                                break

                    if extracted_account_id:
                        break

            if extracted_account_id:
                break

                # --- 2. Fallback: Extract Account ID from File Name ---
        file_name_account_id = None
        if file_name:
            match = re.search(r'ReportHistory[-_](\d+)', file_name, re.IGNORECASE)
            if match:
                file_name_account_id = match.group(1)

        # If content extraction failed, use file name extraction
        if not extracted_account_id and file_name_account_id:
            extracted_account_id = file_name_account_id

        # --- 3. Start Trade Position Analysis ---
        start_idx = df.index[df[0].astype(str).str.contains(r'\bPositions\b', case=False, na=False)].tolist()
        end_idx = df.index[df[0].astype(str).str.contains(r'\bOrders\b', case=False, na=False)].tolist()

        if not start_idx:
            # If the positions section can't be found, return a specific error
            return {"error": "Could not find 'Positions' section in the file. Ensure the structure is correct.",
                    "extracted_account_id": extracted_account_id}

        start = start_idx[0] + 1
        end = end_idx[0] if end_idx else len(df)

        positions_raw = df.iloc[start:end]
        positions_raw = positions_raw.dropna(how='all')
        if len(positions_raw) == 0:
            return {"error": "No data found in Positions section.", "extracted_account_id": extracted_account_id}

        header_row = positions_raw.iloc[0]
        positions_df = positions_raw[1:].reset_index(drop=True)
        positions_df.columns = header_row

        positions_df = positions_df.rename(columns={
            'Time': 'Open Time',
            'Price': 'Open Price'
        })

        if 'Close Time' not in positions_df.columns and len(positions_df.columns) > 8:
            positions_df.columns.values[8] = 'Close Time'
        if 'Close Price' not in positions_df.columns and len(positions_df.columns) > 9:
            positions_df.columns.values[9] = 'Close Price'

        def parse_datetime_flex(series):
            parsed = pd.to_datetime(series, format='%Y.%m.%d %H:%M:%S.%f', errors='coerce')
            parsed = parsed.fillna(pd.to_datetime(series, format='%Y.%m.%d %H:%M:%S', errors='coerce'))
            return parsed

        positions_df['Open Time'] = parse_datetime_flex(positions_df.get('Open Time'))
        positions_df['Close Time'] = parse_datetime_flex(positions_df.get('Close Time'))

        positions_df = positions_df.dropna(subset=['Open Time', 'Close Time'])
        if len(positions_df) == 0:
            return {"error": "No valid timestamps found for analysis.", "extracted_account_id": extracted_account_id}

        positions_df['Profit'] = pd.to_numeric(positions_df['Profit'], errors='coerce').fillna(0)
        positions_df['Hold_Time'] = positions_df['Close Time'] - positions_df['Open Time']

        scalping_df = positions_df[positions_df['Hold_Time'] <= pd.Timedelta(minutes=3)].copy()

        positions_df = positions_df.sort_values(by='Open Time').reset_index(drop=True)

        positions_df['Reversal'] = False
        for i in range(1, len(positions_df)):
            prev_close = positions_df.loc[i - 1, 'Close Time']
            curr_open = positions_df.loc[i, 'Open Time']
            prev_type = str(positions_df.loc[i - 1, 'Type']).strip().lower()
            curr_type = str(positions_df.loc[i, 'Type']).strip().lower()
            prev_symbol = str(positions_df.loc[i - 1, 'Symbol']).strip().upper()
            curr_symbol = str(positions_df.loc[i, 'Symbol']).strip().upper()

            if pd.notnull(prev_close) and pd.notnull(curr_open) and prev_symbol == curr_symbol:
                time_diff = abs((curr_open - prev_close).total_seconds())
                if time_diff <= 20 and (
                        (prev_type == 'buy' and curr_type == 'sell') or
                        (prev_type == 'sell' and curr_type == 'buy')
                ):
                    positions_df.loc[i, 'Reversal'] = True

        reversal_df = positions_df[positions_df['Reversal']].copy()

        positions_df['Burst'] = False
        burst_indices = set()

        i = 0
        while i < len(positions_df) - 1:
            current_burst = []
            j = i
            while j < len(positions_df) - 1:
                curr_open = positions_df.loc[j, 'Open Time']
                next_open = positions_df.loc[j + 1, 'Open Time']

                if pd.notnull(curr_open) and pd.notnull(next_open):
                    time_diff = (next_open - curr_open).total_seconds()
                    if time_diff <= 2:
                        if not current_burst:
                            current_burst.append(j)
                        current_burst.append(j + 1)
                        j += 1
                    else:
                        break
                else:
                    j += 1

            if len(current_burst) >= 2:
                burst_indices.update(current_burst)
                i = current_burst[-1] + 1
            else:
                i += 1

        positions_df.loc[list(burst_indices), 'Burst'] = True
        burst_df = positions_df[positions_df['Burst']].copy()

        total_positions = len(positions_df)
        total_profit = positions_df['Profit'].sum()
        scalping_count = len(scalping_df)
        scalping_profit = scalping_df['Profit'].sum()
        reversal_count = len(reversal_df)
        reversal_profit = reversal_df['Profit'].sum()
        burst_count = len(burst_df)
        burst_profit = burst_df['Profit'].sum()

        scalping_percentage = (scalping_count / total_positions * 100) if total_positions > 0 else 0
        reversal_percentage = (reversal_count / total_positions * 100) if total_positions > 0 else 0
        burst_percentage = (burst_count / total_positions * 100) if total_positions > 0 else 0

        avg_hold_time = positions_df['Hold_Time'].mean()
        avg_scalping_hold_time = scalping_df['Hold_Time'].mean() if scalping_count > 0 else pd.Timedelta(0)

        return {
            "total_positions": total_positions,
            "total_profit": total_profit,
            "scalping_count": scalping_count,
            "scalping_profit": scalping_profit,
            "scalping_percentage": scalping_percentage,
            "reversal_count": reversal_count,
            "reversal_profit": reversal_profit,
            "reversal_percentage": reversal_percentage,
            "burst_count": burst_count,
            "burst_profit": burst_profit,
            "burst_percentage": burst_percentage,
            "avg_hold_time": avg_hold_time,
            "avg_scalping_hold_time": avg_scalping_hold_time,
            "scalping_df": scalping_df,
            "reversal_df": reversal_df,
            "burst_df": burst_df,
            "all_positions_df": positions_df,
            "extracted_account_id": extracted_account_id
        }

    except Exception as e:
        # If an error occurs during processing, ensure a default structure is returned
        return {"error": f"Error processing file: {str(e)}", "extracted_account_id": extracted_account_id}


# --- IP Lookup Helpers & Report Generation (Unchanged) ---
def get_ip_details(ip_address):
    if not ip_address or ip_address.lower() == 'n/a':
        return {"error": "No IP provided."}
    try:
        response = requests.get(f'https://ipinfo.io/{ip_address}/json', timeout=5)
        response.raise_for_status()

        data = response.json()

        if 'country' in data:
            data['full_country'] = get_country_name(data['country'])

        return data
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def generate_report(analysis_result, account_id, trade_ip_details, account_country, vps_used):
    total_positions = analysis_result['total_positions']
    total_profit = analysis_result['total_profit']
    scalping_count = analysis_result['scalping_count']
    scalping_profit = analysis_result['scalping_profit']
    scalping_percentage = analysis_result['scalping_percentage']

    is_toxic = False
    toxic_patterns = []

    if scalping_percentage >= 30:
        is_toxic = True
        toxic_patterns.append("Excessive Scalping (>= 30% of total trades)")

    if analysis_result['reversal_count'] > (total_positions * 0.03):
        is_toxic = True
        toxic_patterns.append("Frequent Reversal Patterns (Hedge-and-Dump)")

    if analysis_result['burst_count'] > (total_positions * 0.03):
        is_toxic = True
        toxic_patterns.append("Rapid Burst Trading (HFT-like activity)")

    if scalping_percentage >= 30:
        toxic_status = "Toxic trading patterns were found in this account."
    else:
        toxic_status = "No toxic trading pattern found in this account."

    trade_location_country_name = trade_ip_details.get('full_country', trade_ip_details.get('country', 'N/A'))
    trade_location_text = f"{trade_ip_details.get('city', 'N/A')}, {trade_location_country_name}"

    report = f"""Account {account_id}
Total trades: {total_positions} with an overall profit of ${total_profit:.2f}.
Scalping trades: {scalping_count} ({scalping_percentage:.1f}%) with a profit of ${scalping_profit:.2f}.
Trading activity was done in {trade_location_text}, but the account country is registered as {account_country}.

{toxic_status}
VPS used: {vps_used}.
"""
    if is_toxic and scalping_percentage >= 30 and len(toxic_patterns) > 1:
        report += f"Detected patterns: {', '.join(toxic_patterns)}"

    return report


# --- Main App with Tabs (Modified only in the analysis result handling) ---
st.title("📈 Rotex Forex & Trade Analyzer")

tab1, tab2 = st.tabs(["💱 Forex Calculator", "📊 Trade & Security Analyzer"])

# ===========================================================
# 🔹 TAB 1: FOREX CALCULATOR (Unchanged)
# ===========================================================
with tab1:
    st.header("ROTEX FOREX CALCULATOR")

    col_spacer_left, col_center, col_spacer_right = st.columns([1, 3, 1])

    with col_center:
        calculation_type = st.selectbox(
            "Select Calculation Type",
            ['Pip Difference', 'Margin Calculator', 'Pip Value & Spread Cost', 'Swap Calculator'],
            key="calc_type_select"
        )

        st.divider()

        # --- Pip Difference Section ---
        if calculation_type == 'Pip Difference':
            st.subheader("Pip Difference")
            col1, col2 = st.columns(2)
            with col1:
                pip_pair = st.text_input("Pair (e.g. EURUSD)", value="EURUSD", key="pip_pair").upper()
            with col2:
                pip_open = st.number_input("Opening Price", format="%.6f", key="pip_open")
            pip_close = st.number_input("Closing Price", format="%.6f", key="pip_close")

            pip_result = st.empty()
            if pip_open > 0 and pip_close > 0 and len(pip_pair) == 6:
                pip = detect_pip_size(pip_pair)
                pips = abs((pip_close - pip_open) / pip)
                pip_result.success(f"Pip Difference: {pips:.2f} pips (pip size {pip})")
            else:
                pip_result.warning("⚠️ Enter valid inputs (pair must be 6 chars and prices > 0)")
            st.caption("Note: JPY pairs use pip = 0.01; others use 0.0001. Metals/Indices/Cryptos adjusted.")

        # --- Margin Calculator & Pip Value Section ---
        elif calculation_type == 'Margin Calculator' or calculation_type == 'Pip Value & Spread Cost':

            if calculation_type == 'Margin Calculator':
                st.subheader("Forex Margin Calculator")
            else:
                st.subheader("Pip Value & Spread Cost Calculator")

            col1, col2 = st.columns(2)
            with col1:
                pair = st.text_input("Pair (BASEQUOTE, e.g. AUDCAD)", value="AUDCAD", key="margin_pair").upper()
            with col2:
                lot_type = st.selectbox("Lot Type", ['standard', 'mini', 'micro', 'custom'], key="margin_lot_type")

            custom_lot = None
            if lot_type == 'custom':
                custom_lot = st.number_input("Custom Lot Size", min_value=1.0, step=1.0, key="margin_custom")

            col1, col2, col3 = st.columns(3)
            with col1:
                lots = st.number_input("Lots", value=1.0, min_value=0.01, step=0.01, key="margin_lots")
            with col2:
                price = st.number_input("Current Market Price", format="%.6f", key="margin_price")
            with col3:
                leverage = st.number_input("Leverage (e.g., 100 for 1:100)", value=100, min_value=1, step=1,
                                           key="margin_leverage")

            cross_rate = st.number_input("Cross Rate (USD/Quote)", min_value=0.0, format="%.6f",
                                         key="margin_cross",
                                         help="If QUOTE is not USD, enter USD/QUOTE rate (e.g., USDGBP rate for EURGBP)")

            if calculation_type == 'Margin Calculator':
                equity = st.number_input("Account Equity (USD)", value=1000.0, min_value=0.01, step=0.01,
                                         key="margin_equity")

            # Calculation Logic (Unified)
            base = pair[:3]
            quote = pair[3:]
            contract = contract_size(lot_type, custom_lot)
            calculated_margin_usd = 0.0
            margin_quote = 0.0

            if len(pair) == 6 and lots > 0 and price > 0 and leverage > 0 and contract > 0:

                if quote == 'USD':
                    margin_usd = (lots * contract * price) / leverage
                    margin_quote = margin_usd / price
                    formula_text = '((lots * contract * price) / leverage)'
                    calculated_margin_usd = margin_usd

                elif base == 'USD':
                    margin_quote = (lots * contract) / leverage
                    margin_usd = margin_quote * price
                    formula_text = '((lots * contract) / leverage) * price'
                    calculated_margin_usd = margin_usd

                else:  # Non-USD pair
                    margin_quote = (lots * contract * price) / leverage
                    if cross_rate > 0:
                        margin_usd = margin_quote / cross_rate
                        formula_text = '((lots * contract * price) / leverage) / crossRate'
                        calculated_margin_usd = margin_usd
                    else:
                        formula_text = "Error: Cross Rate Missing"

            if calculation_type == 'Margin Calculator':
                st.subheader("Margin Results")
                if calculated_margin_usd > 0:
                    st.info(f"Formula (USD): {formula_text}")
                    st.metric(f"Margin ({quote})", f"{margin_quote:.6f} {quote}")
                    st.metric("Blocked Margin (USD)", f"${calculated_margin_usd:.6f}")

                    if equity > 0 and calculated_margin_usd > 0:
                        margin_level = (equity / calculated_margin_usd) * 100
                        st.metric("Margin Level %", f"{margin_level:.2f}%")
                    else:
                        st.warning("⚠️ Enter valid equity/margin for Margin Level.")
                else:
                    st.warning("⚠️ Enter valid inputs (pair, lots, price, leverage, and cross rate if needed).")

            # --- Pip Value & Spread Cost Results ---
            if calculation_type == 'Pip Value & Spread Cost':
                st.subheader("Pip Value & Spread Cost Results")
                spread = st.number_input("Spread (in pips)", value=1.0, min_value=0.0, step=0.1)

                if price > 0 and lots > 0 and contract > 0:
                    pip = detect_pip_size(pair)

                    if quote == 'USD':
                        pip_value_unit = contract * pip
                    elif base == 'USD':
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
                            f"Pip Value (per {lots} lots): **${pip_value_usd:.2f}** | Spread Cost (per trade): **${spread_cost:.2f}**")
                    else:
                        st.info("Pip Value: **Cannot calculate** (Check Cross Rate or inputs)")
                else:
                    st.info("Enter valid price, lots, and lot info to calculate.")


        # --- Swap Calculator Section ---
        elif calculation_type == 'Swap Calculator':
            st.subheader("Forex Swap Calculator")

            col1, col2 = st.columns(2)
            with col1:
                swap_pair = st.text_input("Pair (e.g. EURUSD)", value="EURUSD", key="swap_pair").upper()
                swap_type = st.radio("Trade Type", options=["Buy", "Sell"], key="swap_trade_type")
            with col2:
                swap_lots = st.number_input("Lots", value=1.0, min_value=0.01, step=0.01, key="swap_lots")
                swap_days = st.number_input("Number of Nights/Days Held", value=1, min_value=1, step=1, key="swap_days")

            col3, col4, col5 = st.columns(3)
            with col3:
                swap_rate = st.number_input(f"Swap Rate ({swap_type})", value=-7.5,
                                            help="Enter the swap rate given by your broker (can be positive or negative).",
                                            key="swap_rate")
            with col4:
                swap_price = st.number_input("Current Market Price", format="%.6f", value=1.10000, key="swap_price")
            with col5:
                swap_cross_rate = st.number_input("Cross Rate (USD/Quote)", min_value=0.0, format="%.6f", value=1.0,
                                                  key="swap_cross", help="If QUOTE is not USD, enter USD/QUOTE rate.")

            swap_contract = contract_size(
                st.selectbox("Lot Type", ['standard', 'mini', 'micro'], key="swap_lot_type_sel"), None)

            if swap_lots > 0 and swap_contract > 0 and swap_price > 0:

                base = swap_pair[:3]
                quote = swap_pair[3:]

                total_swap_value_quote = swap_rate * swap_lots * swap_days

                if quote == 'USD':
                    final_swap_usd = total_swap_value_quote
                    swap_currency = 'USD'
                elif swap_cross_rate > 0:
                    final_swap_usd = total_swap_value_quote / swap_cross_rate
                    swap_currency = quote
                else:
                    st.warning("⚠️ Enter a valid USD/Quote Cross Rate to calculate swap in USD.")
                    st.stop()

                st.success(f"Estimated Total Swap ({swap_days} days): **${final_swap_usd:.2f}**")
                st.info(
                    f"Swap calculated based on: **Swap Rate per Lot ({swap_rate})** * **Lots ({swap_lots})** * **Days ({swap_days})** converted from {swap_currency} to USD.")

            else:
                st.warning("⚠️ Enter valid inputs (Pair, Lots, Lot Type, Price).")

# ===========================================================
# 🔹 TAB 2: TRADE & SECURITY ANALYZER (FIXED)
# ===========================================================
with tab2:
    st.header("📊 TRADE & SECURITY ANALYZER")

    st.subheader("1. File Upload & Account Details")

    col_a, col_c = st.columns(2)

    with col_a:
        uploaded_file_obj = st.file_uploader("📂 Upload Trade History Excel (.xlsx)", type=["xlsx"])

    with col_c:
        account_country = st.text_input("Registered Account Country", value="United Arab Emirates")

    st.markdown(f"**Extracted Account ID:** `**{st.session_state.extracted_account_id}**`")

    st.divider()

    st.subheader("2. Trading Location & Security Check")
    col_d, col_e = st.columns(2)

    with col_d:
        trade_ip = st.text_input("Last Trading IP Address", value="103.1.200.1",
                                 help="Use a test IP like 103.1.200.1 (India) or 203.0.113.1 (Test IP) to see results.")

    with col_e:
        vps_used = st.selectbox("Was VPS used?", options=['No', 'Yes'])

    # Button to start analysis
    if st.button("🚀 Run Comprehensive Analysis"):

        # 1. Trade Analysis
        if uploaded_file_obj is None:
            st.error("Please upload the Trade History Excel file.")
            st.stop()

        file_name = uploaded_file_obj.name

        with st.spinner(f"Analyzing trading history and extracting Account ID (File: {file_name})..."):
            analysis_result = analyze_trades(uploaded_file_obj, file_name=file_name)

        # --- Handle Account ID Extraction ---
        extracted_account_id = analysis_result.get("extracted_account_id")

        if not extracted_account_id:
            st.error(
                "❌ Could not automatically extract Account ID from the file content or file name. Please check the file format.")
            account_id_to_use = "Unknown Account ID"
        else:
            st.session_state.extracted_account_id = extracted_account_id
            account_id_to_use = extracted_account_id

            # **FIXED LOGIC**: Safely check for error key and evaluate success source
            file_name_match = re.search(r'ReportHistory[-_](\d+)', file_name, re.IGNORECASE)
            error_message = analysis_result.get('error')

            # The file name ID was used if:
            # 1. We found a match in the file name, AND
            # 2. The error message indicates that the content-based extraction failed (i.e., couldn't find 'Account:')
            if file_name_match and (error_message and 'Account:' in error_message):
                st.success(f"✅ Extracted Account ID: **{account_id_to_use}** (Found via **File Name Fallback**)")
            else:
                st.success(f"✅ Extracted Account ID: **{account_id_to_use}** (Found via File Content)")

        # --- Handle General Analysis Error ---
        # Safely use .get('error') here as well
        if analysis_result.get("error") and analysis_result["error"] != "No valid timestamps found for analysis.":
            st.error(f"Trade Analysis Failed: {analysis_result['error']}")
            # Continue execution here only if we successfully extracted the ID and want to proceed with IP check
            if not extracted_account_id:
                st.stop()

        if analysis_result['total_positions'] == 0:
            st.warning("No valid closed positions were found for analysis.")

        # Only show success if no critical error prevented the whole analysis
        if not analysis_result.get("error"):
            st.success("Trade Analysis Complete!")

        # 2. IP Lookup
        with st.spinner(f"Looking up IP: {trade_ip}..."):
            trade_ip_details = get_ip_details(trade_ip)

        if "error" in trade_ip_details:
            st.warning(f"IP Lookup Warning for {trade_ip}: {trade_ip_details['error']}. Using N/A for report.")
            trade_ip_details = {"full_country": "N/A", "city": "N/A", "country": "N/A"}
            ip_display_country = "N/A"
        else:
            ip_display_country = trade_ip_details.get('full_country', trade_ip_details.get('country', 'N/A'))
            st.success(f"IP Lookup Complete: Located in {trade_ip_details.get('city', 'N/A')}, {ip_display_country}")

        # Log to history for display later
        st.session_state.ip_history.append({
            "ip": trade_ip,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": trade_ip_details
        })

        st.divider()
        st.subheader("3. Copiable Report and Visuals")

        # --- Generate and Display Copiable Report ---
        if extracted_account_id and analysis_result['total_positions'] > 0:
            report_text = generate_report(
                analysis_result=analysis_result,
                account_id=account_id_to_use,
                trade_ip_details=trade_ip_details,
                account_country=account_country,
                vps_used=vps_used
            )

            st.markdown("**Copiable Report:**")
            st.code(report_text, language='text')

        # --- Display Visuals and Details ---

        if analysis_result['total_positions'] > 0:
            st.subheader("Trade Summary")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total Positions", analysis_result['total_positions'])
            col_s2.metric("Total Profit", f"${analysis_result['total_profit']:.2f}")

            win_count = analysis_result['all_positions_df'][analysis_result['all_positions_df']['Profit'] > 0].shape[0]
            total_trades = analysis_result['total_positions']
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            col_s3.metric("Win Rate", f"{win_rate:.1f}%")
            col_s4.metric("Avg Hold Time", str(analysis_result['avg_hold_time']).split('.')[0])

            st.divider()

            st.subheader("High-Risk Trade Patterns")

            scalping_style = "background-color: #f63366; color: white; font-weight: bold;" if analysis_result[
                                                                                                  'scalping_percentage'] >= 30 else ""

            summary_data = {
                'Trade Type': ['Scalping (< 3 min)', 'Reversal (< 20 sec)', 'Burst (< 2 sec)'],
                'Count': [analysis_result['scalping_count'], analysis_result['reversal_count'],
                          analysis_result['burst_count']],
                'Count %': [f"{analysis_result['scalping_percentage']:.1f}%",
                            f"{analysis_result['reversal_percentage']:.1f}%",
                            f"{analysis_result['burst_percentage']:.1f}%"],
                'Profit': [f"${analysis_result['scalping_profit']:.2f}", f"${analysis_result['reversal_profit']:.2f}",
                           f"${analysis_result['burst_profit']:.2f}"],
            }
            summary_df = pd.DataFrame(summary_data)


            def highlight_scalping(s):
                is_scalping_row = s['Trade Type'] == 'Scalping (< 3 min)'
                is_toxic = analysis_result['scalping_percentage'] >= 30
                return [scalping_style] * len(s) if is_scalping_row and is_toxic else [''] * len(s)


            st.dataframe(
                summary_df.style.apply(highlight_scalping, axis=1),
                hide_index=True
            )

            st.caption(f"Average Scalping Hold Time: {str(analysis_result['avg_scalping_hold_time']).split('.')[0]}")

            df_plot = analysis_result['all_positions_df'].copy()
            df_plot['Cumulative Profit'] = df_plot['Profit'].cumsum()
            fig_cum = px.line(df_plot, x='Close Time', y='Cumulative Profit',
                              title="Cumulative Profit Curve (Equity Growth)",
                              line_shape='spline')
            st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.info("No trading summary or visuals generated because no valid trades were found in the file.")

    st.divider()
    st.subheader("Recent IP Lookups")
    if st.session_state.ip_history:
        for entry in st.session_state.ip_history:
            display_country = entry['details'].get('full_country', entry['details'].get('country', 'N/A'))
            st.write(
                f"**{entry['timestamp']}** — {entry['ip']} → {entry['details'].get('city', 'N/A')}, **{display_country}**")
    else:
        st.info("No IP lookups yet.")