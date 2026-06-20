import streamlit as st
import pandas as pd
import akshare as ak
from openai import OpenAI
from datetime import datetime
import os

# ================= 把你的 DeepSeek Key 填在这里 =================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
# =================================================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ---------- 数据获取函数 ----------
@st.cache_data(ttl=600)
def get_market_flow():
    try:
        df = ak.stock_market_fund_flow()
        latest = df.iloc[-1]
        return {"日期": latest["日期"], "主力净流入(亿)": round(latest["主力净流入"]/1e8,2), "超大单净流入(亿)": round(latest["超大单净流入"]/1e8,2)}
    except: return None

@st.cache_data(ttl=600)
def get_sector_rank():
    try:
        df = ak.stock_board_industry_name_em()
        df = df.sort_values("涨跌幅", ascending=False)
        return df.head(5)[["板块名称","涨跌幅"]], df.tail(5)[["板块名称","涨跌幅"]]
    except: return None, None

def get_today_stock_summary(symbol):
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=today, end_date=today)
        if df.empty:
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot["代码"]==symbol]
            if not row.empty:
                r = row.iloc[0]
                return f"现价{r['最新价']}，涨跌幅{r['涨跌幅']}%，换手率{r['换手率']}%，量比{r['量比']}"
            return "暂无今日数据"
        r = df.iloc[-1]
        return f"开{r['开盘']}收{r['收盘']}，高{r['最高']}低{r['最低']}，涨幅{r['涨跌幅']}%，振幅{r['振幅']}%，成交量{r['成交量']}手"
    except: return "获取失败"

# ---------- 日记存储 ----------
DIARY_FILE = "trade_diary.csv"
def save_diary(entry):
    df = pd.DataFrame([entry])
    if not os.path.exists(DIARY_FILE): df.to_csv(DIARY_FILE, index=False)
    else: df.to_csv(DIARY_FILE, mode='a', header=False, index=False)
def load_all_diaries():
    if os.path.exists(DIARY_FILE): return pd.read_csv(DIARY_FILE)
    return pd.DataFrame()

# ---------- AI 深度复盘（升级版） ----------
def get_deep_advice(stock_code, name, direction, emotion, stock_summary, market, sector, entry_price, trade_reason):
    price_info = f"，交易价格是{entry_price}元" if entry_price else "（未提供具体价格）"
    reason_info = f"，理由是：{trade_reason}" if trade_reason else "（未提供理由）"
    
    prompt = f"""你是一位顶尖的股票交易心理教练和盘面复盘专家。你的任务不是预测未来，而是基于当天已发生的盘面事实，对用户的操作进行客观、锐利的复盘。

【今日盘面事实】
个股({name} {stock_code})今日走势摘要：{stock_summary}
大盘资金：{market}
板块表现：{sector}

【用户操作】
动作：{direction}
价格：{price_info}
操作理由：{reason_info}
操作时情绪：{emotion}

请你按以下结构，输出一份不超过300字的深度复盘建议：
1. **对错评估**：结合今日的真实走势（开盘、收盘、最高、最低、振幅），判断用户的买/卖点属于“追高/杀跌”、“精准把握”还是“过早/过晚”。直接指出问题，语气温和但直击要害。
2. **理想买卖点复盘**：根据今日盘面，给出一个“事后诸葛亮”式的、相对正确的理想买点或卖点作为参考（例如“理想买点在早盘回踩均价线时的10.20元附近”）。这能帮助用户建立盘感。
3. **心理与纪律分析**：分析用户的操作理由和情绪，指出其中可能存在的认知偏差（如锚定效应、损失厌恶、羊群效应等），并给出下次遇到类似情况时的1条具体行动建议。

结尾加上：本建议仅用于复盘学习，不构成投资指导。"""
    
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return resp.choices[0].message.content

def get_longterm_summary(diaries_text, market_text=""):
    prompt = f"""交易日记汇总：{diaries_text}，当前市场：{market_text}。
请分析用户交易模式，指出三个主要情绪/纪律问题并给出改进行动。"""
    resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], max_tokens=600)
    return resp.choices[0].message.content

# ---------- 界面 ----------
st.set_page_config(page_title="股票日记AI", layout="wide")
st.title("📈 股票交易日记 · AI深度复盘助手")

# 侧边栏：市场数据
st.sidebar.header("今日市场")
market_data = get_market_flow()
if market_data:
    st.sidebar.write(market_data["日期"])
    st.sidebar.metric("主力净流入(亿)", market_data["主力净流入(亿)"])
    st.sidebar.metric("超大单净流入(亿)", market_data["超大单净流入(亿)"])
top5, bottom5 = get_sector_rank()
if top5 is not None:
    st.sidebar.subheader("🔥涨幅前5")
    st.sidebar.dataframe(top5, hide_index=True)
    st.sidebar.subheader("❄️跌幅前5")
    st.sidebar.dataframe(bottom5, hide_index=True)

# 主区域：交易日记表单
st.header("✍️ 记录今天操作与情绪")
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        code = st.text_input("股票代码", "000001")
        name = st.text_input("股票名称（选填）", "平安银行")
    with c2:
        dir = st.selectbox("操作方向", ["买入", "卖出", "持有观望"])
        emo = st.selectbox("操作时主要情绪", ["平静自信","焦虑犹豫","懊悔","兴奋贪婪","麻木"])

    st.markdown("---")
    st.subheader("📝 交易细节（用于深度复盘，不填也没事）")
    col_price, col_reason = st.columns(2)
    with col_price:
        entry_price = st.text_input("你的买入/卖出价（元）", placeholder="例如：12.50")
    with col_reason:
        trade_reason = st.text_area("你买入/卖出的核心理由", placeholder="例如：看着跌不动了就买了...", height=100)
        
    sub = st.form_submit_button("📝 提交并获取深度复盘建议")

if sub:
    if not code: st.error("请输入股票代码")
    else:
        with st.spinner("获取个股数据..."):
            summary = get_today_stock_summary(code)
        mkt = f"主力净流入{market_data['主力净流入(亿)']}亿" if market_data else "数据缺失"
        sec = f"领涨:{top5.iloc[0]['板块名称']} 领跌:{bottom5.iloc[0]['板块名称']}" if top5 is not None else ""
        with st.spinner("AI深度分析中..."):
            advice = get_deep_advice(code, name, dir, emo, summary, mkt, sec, entry_price, trade_reason)
        st.subheader("🧠 深度复盘建议")
        st.info(advice)
        save_diary({"日期":datetime.now().strftime("%Y-%m-%d"), "时间":datetime.now().strftime("%H:%M"),
                    "股票代码":code, "操作":dir, "情绪":emo, "价格":entry_price, "理由":trade_reason,
                    "个股走势":summary, "AI建议":advice})
        st.success("已保存")

st.header("📚 历史日记")
diaries = load_all_diaries()
if not diaries.empty:
    st.dataframe(diaries[["日期","股票代码","操作","情绪"]], height=200)
    if st.button("生成长期交易总结"):
        txt = " ".join([f"{r['日期']}{r['操作']}{r['股票代码']}情绪{r['情绪']}" for _,r in diaries.iterrows()])
        with st.spinner("深度复盘..."):
            long = get_longterm_summary(txt, mkt)
        st.subheader("📋 你的交易特征与改进")
        st.write(long)
