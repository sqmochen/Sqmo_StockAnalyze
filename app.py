# ── 套件匯入 ──────────────────────────────────────────────────
import anthropic
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()

def get_stock_data(symbol, api_token, start_date, end_date):
    """
    從 FinMind API 獲取台股歷史股價數據（TaiwanStockPrice dataset）

    FinMind API 規格：
        端點: https://api.finmindtrade.com/api/v4/data
        認證: Authorization: Bearer {token}  (Header)
        dataset: TaiwanStockPrice
        回傳欄位: date, stock_id, Trading_Volume, Trading_money,
                  open, max, min, close, spread, Trading_turnover

    欄位對應（統一為小寫標準名稱）：
        max  → high
        min  → low
        Trading_Volume → volume

    Args:
        symbol: 台股股票代碼（如 2330、0050）
        api_token: FinMind API Token
        start_date: 起始日期（date 物件）
        end_date: 結束日期（date 物件）

    Returns:
        DataFrame: 含標準欄位（date, open, high, low, close, volume）的數據，
                   若失敗則回傳 None
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        headers = {"Authorization": f"Bearer {api_token}"}
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": symbol,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()

        # FinMind v4 回傳格式：{"msg": "success", "status": 200, "data": [...]}
        if result.get("status") != 200 or not result.get("data"):
            msg = result.get("msg", "未知錯誤")
            st.error(f"無法獲取股票 {symbol} 的數據：{msg}。請確認股票代碼和 Token 是否正確。")
            return None

        df = pd.DataFrame(result["data"])

        # 將 FinMind 欄位名稱統一對應為標準欄位名稱
        df = df.rename(columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
        })

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 確保必要欄位存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"API 回傳數據缺少欄位：{missing}，請確認資料集是否正確。")
            return None

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"數據處理錯誤：{str(e)}")
        return None

def filter_by_date_range(df, start_date, end_date):
    """
    根據日期範圍過濾數據
    
    Args:
        df: 股票數據DataFrame
        start_date: 起始日期
        end_date: 結束日期
    
    Returns:
        DataFrame: 過濾後的數據
    """
    if df is None:
        return None
    
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask].copy()
    
    return filtered_df.reset_index(drop=True)

def get_moving_averages(df):
    """
    計算移動平均線（MA5, MA10, MA20, MA60）
    
    Args:
        df: 股票數據DataFrame
    
    Returns:
        DataFrame: 包含移動平均線的數據
    """
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    
    # 計算移動平均線
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()
    
    return df

def calculate_rsi(df, period=14):
    """
    計算RSI相對強弱指標
    
    RSI計算公式：
        RSI = 100 - (100 / (1 + RS))
        RS = 指定期間內平均漲幅 / 指定期間內平均跌幅
    
    Args:
        df: 包含收盤價的股票數據DataFrame
        period: RSI計算天數，預設為14天
    
    Returns:
        DataFrame: 新增RSI欄位的數據
    """
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    
    try:
        # 計算每日收盤價變化
        delta = df['close'].diff()
        
        # 分離漲幅和跌幅
        gain = delta.clip(lower=0)   # 只保留正值（漲幅）
        loss = -delta.clip(upper=0)  # 只保留正值（跌幅取絕對值）
        
        # 使用指數移動平均計算平均漲跌幅（更符合傳統RSI計算方式）
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        # 計算RS值，避免除以零
        rs = avg_gain / avg_loss.replace(0, np.nan)
        
        # 計算RSI值
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 填補前period筆無法計算的NaN值（設為50作為中性值）
        df['RSI'] = df['RSI'].fillna(50)
        
    except Exception as e:
        st.warning(f"RSI計算發生錯誤：{str(e)}，RSI欄位將填入預設值50")
        df['RSI'] = 50
    
    return df

def create_candlestick_chart(df, symbol):
    """
    創建K線圖、移動平均線、成交量和RSI圖表
    
    圖表結構（由上至下）：
        Row 1: K線圖 + 移動平均線（佔70%高度）
        Row 2: 成交量柱狀圖（佔15%高度）
        Row 3: RSI走勢圖（佔15%高度）
    
    Args:
        df: 包含股票數據、移動平均線和RSI的DataFrame
        symbol: 股票代碼
    
    Returns:
        plotly.graph_objects.Figure: 互動式圖表
    """
    # 建立三列子圖，共用X軸
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('價格與移動平均線', '成交量', f'RSI (14)'),
        row_width=[0.15, 0.15, 0.70]  # 由下到上的高度比例
    )
    
    # ── Row 1：K線圖 ──────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ),
        row=1, col=1
    )
    
    # 移動平均線
    ma_colors = {
        'MA5': '#ff6b6b',
        'MA10': '#4ecdc4', 
        'MA20': '#45b7d1',
        'MA60': '#96ceb4'
    }
    
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[ma],
                mode='lines',
                name=ma,
                line=dict(color=ma_colors[ma], width=2)
            ),
            row=1, col=1
        )
    
    # ── Row 2：成交量柱狀圖 ────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color='#a55eea',
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # ── Row 3：RSI走勢圖 ──────────────────────────────────────
    # RSI主線（藍色）
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['RSI'],
            mode='lines',
            name='RSI(14)',
            line=dict(color='#1e90ff', width=2),
            hovertemplate='日期: %{x}<br>RSI: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 超買警戒線（RSI = 70，紅色虛線）
    fig.add_hline(
        y=70,
        line_dash='dash',
        line_color='red',
        line_width=1.5,
        annotation_text='超買(70)',
        annotation_position='bottom right',
        annotation_font_color='red',
        row=3, col=1
    )
    
    # 超賣警戒線（RSI = 30，綠色虛線）
    fig.add_hline(
        y=30,
        line_dash='dash',
        line_color='green',
        line_width=1.5,
        annotation_text='超賣(30)',
        annotation_position='top right',
        annotation_font_color='green',
        row=3, col=1
    )
    
    # 超買區域背景（70~100，半透明紅色）
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor='rgba(255, 0, 0, 0.08)',
        line_width=0,
        row=3, col=1
    )
    
    # 超賣區域背景（0~30，半透明綠色）
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor='rgba(0, 200, 0, 0.08)',
        line_width=0,
        row=3, col=1
    )
    
    # ── 整體佈局設定 ──────────────────────────────────────────
    fig.update_layout(
        title=f'{symbol} 股價技術分析圖表',
        xaxis_title='日期',
        yaxis_title='價格 (TWD)',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # 關閉K線圖的rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    
    # Y軸標籤
    fig.update_yaxes(title_text="價格 (TWD)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig

def get_rsi_status(rsi_value):
    """
    根據RSI數值判斷目前狀態
    
    Args:
        rsi_value: RSI數值
    
    Returns:
        tuple: (狀態文字, 說明文字)
    """
    if rsi_value >= 70:
        return "🔴 超買", f"RSI={rsi_value:.1f}，技術指標顯示處於超買區間（≥70）"
    elif rsi_value <= 30:
        return "🟢 超賣", f"RSI={rsi_value:.1f}，技術指標顯示處於超賣區間（≤30）"
    elif rsi_value >= 60:
        return "🟡 偏強", f"RSI={rsi_value:.1f}，技術指標顯示動能偏強（60~70）"
    elif rsi_value <= 40:
        return "🟡 偏弱", f"RSI={rsi_value:.1f}，技術指標顯示動能偏弱（30~40）"
    else:
        return "⚪ 中性", f"RSI={rsi_value:.1f}，技術指標顯示動能中性（40~60）"

def generate_ai_insights(symbol, stock_data, anthropic_api_key, start_date, end_date):
    """
    使用 Anthropic Claude API 進行技術分析（含RSI指標分析）

    Args:
        symbol: 股票代碼
        stock_data: 含RSI的股票數據DataFrame
        anthropic_api_key: Anthropic Claude API 金鑰
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        str: AI分析結果
    """
    try:
        # 建立 Anthropic 客戶端
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # 準備數據摘要資訊
        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100
        
        # 取得最新RSI數值供提示語使用
        latest_rsi = stock_data['RSI'].iloc[-1]
        rsi_status, _ = get_rsi_status(latest_rsi)
        
        # 轉換數據為JSON格式
        data_json = stock_data.to_json(orient='records', date_format='iso')
        
        # System Prompt
        system_message = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識，包含RSI動量指標解讀

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""
        
        # User Prompt
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% (從 NT${start_price:.2f} 變化到 NT${end_price:.2f})
- 最新RSI(14)數值：{latest_rsi:.2f}（狀態：{rsi_status}）

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量、移動平均線和RSI指標：
{data_json}

### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. RSI分析（請務必包含此章節）
- 當前RSI(14)數值（{latest_rsi:.2f}）的歷史意義解讀
- RSI在分析期間的高低點變化紀錄
- 超買（RSI≥70）或超賣（RSI≤30）區間的歷史出現次數與持續時間
- RSI與價格走勢的背離現象（若存在）
- RSI動量強弱的客觀描述

#### 4. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 5. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 6. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）
- 關鍵價位觀察點
- 技術面風險因子

### 綜合評估要求
#### 輸出格式要求
- 條理清晰，分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測，強調分析的局限性
- 在適當位置使用表格或重點標記

分析目標：{symbol}"""
        
        # 呼叫 Anthropic Claude API
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # ✅ 正確的模型名稱
            max_tokens=2000,
            system=system_message,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # 解析回應內容
        return response.content[0].text
        
    except anthropic.AuthenticationError as e:
        st.error(f"Anthropic API 金鑰驗證失敗：{str(e)}，請確認金鑰是否正確。")
        return "AI分析暫時無法使用，請檢查 Anthropic API 金鑰後再試。"
    except anthropic.RateLimitError as e:
        st.error(f"Anthropic API 請求頻率超限：{str(e)}，請稍後再試。")
        return "AI分析暫時無法使用，請稍後再試。"
    except anthropic.APIError as e:
        st.error(f"Anthropic API 錯誤：{str(e)}")
        return "AI分析暫時無法使用，請稍後再試。"
    except Exception as e:
        st.error(f"AI分析失敗：{str(e)}")
        return "AI分析暫時無法使用，請檢查API金鑰或稍後再試。"

# ── 側邊欄設置 ────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 分析設定")
st.sidebar.divider()

# 輸入控制項
symbol = st.sidebar.text_input(
    "股票代碼",
    value="2330",
    help="輸入台股股票代碼，例如：2330（台積電）、2317（鴻海）、0050（元大台灣50）"
)

finmind_token = st.sidebar.text_input(
    "FinMind API Token",
    type="password",
    help="請輸入您的 FinMind API Token。前往 https://finmindtrade.com 註冊並於個人頁面取得 Token。"
)

# Anthropic API Key 輸入欄位
anthropic_api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="請輸入您的 Anthropic API 金鑰。前往 https://console.anthropic.com 註冊並取得 API Key。"
)

# 日期選擇
default_start_date = datetime.now() - timedelta(days=90)
default_end_date = datetime.now()

start_date = st.sidebar.date_input(
    "起始日期",
    value=default_start_date,
    help="選擇分析的起始日期"
)

end_date = st.sidebar.date_input(
    "結束日期", 
    value=default_end_date,
    help="選擇分析的結束日期"
)

# RSI參數設定（支援自訂）
st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ RSI 參數設定")
rsi_period = st.sidebar.number_input(
    "RSI 計算天數",
    min_value=2,
    max_value=100,
    value=14,
    step=1,
    help="RSI指標的計算週期，標準為14天。數值越小越靈敏，數值越大越平滑。"
)

# 分析按鈕
st.sidebar.markdown("---")
analyze_button = st.sidebar.button("🚀 開始分析", type="primary")

# 免責聲明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明
本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")

# ── 主要分析邏輯 ──────────────────────────────────────────────
if analyze_button:
    # 輸入驗證
    if not symbol.strip():
        st.error("請輸入股票代碼")
    elif not finmind_token.strip():
        st.error("請輸入 FinMind API Token")
    elif not anthropic_api_key.strip():
        st.error("請輸入 Anthropic API Key")
    elif start_date >= end_date:
        st.error("起始日期不能晚於或等於結束日期")
    else:
        # 開始分析流程
        with st.spinner("正在獲取股票數據..."):
            stock_data = get_stock_data(symbol.strip(), finmind_token, start_date, end_date)
            
            if stock_data is not None and len(stock_data) > 0:
                st.success(f"成功獲取 {len(stock_data)} 筆交易數據")
                
                # 過濾日期範圍
                filtered_data = filter_by_date_range(stock_data, start_date, end_date)
                
                if filtered_data is not None and len(filtered_data) > 0:
                    
                    with st.spinner("正在計算技術指標（移動平均線 + RSI）..."):
                        # 計算移動平均線
                        data_with_ma = get_moving_averages(filtered_data)
                        # 計算RSI（使用側邊欄設定的天數）
                        data_with_indicators = calculate_rsi(data_with_ma, period=rsi_period)
                    
                    if data_with_indicators is not None:
                        
                        # ── K線圖與技術指標（含RSI子圖）────────────────────
                        st.markdown("### 📊 股價K線圖與技術指標")
                        chart = create_candlestick_chart(data_with_indicators, symbol.strip())
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # ── 基本統計資訊 ──────────────────────────────────
                        st.markdown("### 📈 基本統計資訊")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        start_price = data_with_indicators['close'].iloc[0]
                        end_price = data_with_indicators['close'].iloc[-1]
                        price_change = end_price - start_price
                        price_change_pct = (price_change / start_price) * 100
                        latest_rsi = data_with_indicators['RSI'].iloc[-1]
                        rsi_status_label, rsi_status_desc = get_rsi_status(latest_rsi)
                        
                        with col1:
                            st.metric(
                                "起始價格",
                                f"NT${start_price:.2f}",
                                help="分析期間第一個交易日的收盤價"
                            )
                        
                        with col2:
                            st.metric(
                                "結束價格",
                                f"NT${end_price:.2f}",
                                help="分析期間最後一個交易日的收盤價"
                            )
                        
                        with col3:
                            st.metric(
                                "價格變化",
                                f"NT${price_change:.2f}",
                                f"{price_change_pct:.2f}%",
                                help="期間內的價格變化金額和百分比"
                            )
                        
                        with col4:
                            st.metric(
                                f"RSI({rsi_period})",
                                f"{latest_rsi:.1f}",
                                rsi_status_label,
                                help=rsi_status_desc
                            )
                        
                        # ── RSI狀態警告提示 ────────────────────────────────
                        if latest_rsi >= 70:
                            st.warning(
                                f"⚠️ **超買警告**：當前 RSI({rsi_period}) = {latest_rsi:.1f}，"
                                f"技術指標顯示價格處於歷史超買區間（≥70）。歷史表現不代表未來結果。"
                            )
                        elif latest_rsi <= 30:
                            st.info(
                                f"📉 **超賣提示**：當前 RSI({rsi_period}) = {latest_rsi:.1f}，"
                                f"技術指標顯示價格處於歷史超賣區間（≤30）。歷史表現不代表未來結果。"
                            )
                        
                        # ── AI技術分析 ────────────────────────────────────
                        st.markdown("### 🤖 AI技術分析")
                        with st.spinner("Claude AI 正在分析中（含RSI動量解讀）..."):
                            ai_analysis = generate_ai_insights(
                                symbol.strip(), 
                                data_with_indicators, 
                                anthropic_api_key,
                                start_date, 
                                end_date
                            )
                        
                        if ai_analysis:
                            st.markdown(ai_analysis)
                        
                        # ── 歷史數據表格（含RSI欄位）──────────────────────
                        st.markdown("### 📋 歷史數據表格")
                        display_data = data_with_indicators.tail(10).copy()
                        display_data = display_data.sort_values('date', ascending=False)
                        
                        # 格式化欄位（含RSI）
                        display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60', 'RSI']
                        display_data_formatted = display_data[display_columns].copy()
                        display_data_formatted.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60', f'RSI({rsi_period})']
                        
                        st.dataframe(
                            display_data_formatted,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.success("✅ 分析完成！")
                        
                else:
                    st.warning("所選日期範圍內沒有交易數據，請調整日期範圍。")
            else:
                st.error("無法獲取股票數據，請檢查股票代碼和API金鑰。")

# ── 初始頁面說明 ──────────────────────────────────────────────
if not analyze_button:
    st.markdown("""
    ## 歡迎使用 AI 股票趨勢分析系統 👋
    
    ### 🚀 功能特色
    - **專業K線圖表**: 互動式價格圖表，包含移動平均線技術指標
    - **RSI相對強弱指標**: 14日RSI動量指標，直觀顯示超買超賣區間
    - **AI智能分析**: 使用 Anthropic Claude 模型進行深度技術面分析（含RSI解讀）
    - **歷史數據**: 詳細的股票歷史價格和成交量數據
    - **教育導向**: 客觀的技術分析，僅供學習研究使用
    
    ### 📝 使用方法
    1. 在左側輸入台股股票代碼（如：2330、2317、0050）
    2. 輸入您的 FinMind API Token 和 Anthropic API 金鑰
    3. 選擇分析的日期範圍
    4. （選填）調整RSI計算天數，預設為14天
    5. 點擊「開始分析」按鈕
    
    ### 💡 技術指標說明
    - **MA5**: 5日移動平均線，短期趨勢指標
    - **MA10**: 10日移動平均線，短中期趨勢指標  
    - **MA20**: 20日移動平均線，中期趨勢指標
    - **MA60**: 60日移動平均線，長期趨勢指標
    - **RSI(14)**: 相對強弱指標，數值介於0~100；≥70為超買、≤30為超賣、40~60為中性
    
    ### 🔑 API 金鑰獲取
    - **FinMind Token**: 前往 [FinMind 官網](https://finmindtrade.com) 註冊，登入後至個人頁面取得 Token
    - **Anthropic API**: 前往 [Anthropic Console](https://console.anthropic.com) 註冊並取得 API Key
    
    ---
    **開始您的技術分析之旅吧！** 📈
    """)