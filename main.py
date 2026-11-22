import os
import json
import time
import random
import datetime
import threading
import sys
import asyncio  # 新增：导入asyncio模块
from pathlib import Path
from typing import List, Dict, Any
from filelock import FileLock

# 新增：动态查询股票代码的依赖
import requests
from bs4 import BeautifulSoup
import re

# 移除proxybroker依赖（解决loop参数问题）
import aiohttp

# 数据可视化与交互依赖
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory  # 终端输入历史
from prompt_toolkit.styles import Style

# LangChain 核心依赖（适配旧版 LangChain 0.0.x）
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory  # 新增ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# 终端美化依赖
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

# 移除yfinance，使用东方财富接口
import pytz
# 新增：重试机制
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ============================ 全局配置 ============================
# 1. 密钥配置
dashscope_key = os.getenv("DASHSCOPE_API_KEY", "你的APPKEY")
embedding_key = os.getenv("DASHSCOPE_EMBEDDING_KEY", dashscope_key)

# 2. 路径配置
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
MEMORY_DIR = BASE_DIR / "memory"
USER_PROFILE_PATH = MEMORY_DIR / "user_profile.json"
LOCK_PATH = MEMORY_DIR / "user_profile.lock"
FAISS_DB_DIR = BASE_DIR / "faiss_cache"
ALLOWED_EXTENSIONS = ['.pdf', '.md', '.txt', '.csv']
CN_TZ = pytz.timezone('Asia/Shanghai')

# 3. 防爬虫/限流配置（适配东方财富）
REQUEST_INTERVAL = 8   # 东方财富限流宽松
USER_AGENT_POOL = [    # 真实浏览器UA池
    # 360急速浏览器（真实特征）
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.124 Safari/537.36 QIHU 360SE",
    # Edge/必应浏览器（真实特征）
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0 BingPreview/1.0",
    # Google Chrome（纯净版）
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox（火狐，差异化特征）
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Mac版Chrome（补充跨平台特征）
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Linux版Firefox（差异化）
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"
]
RETRY_MAX_ATTEMPTS = 1  # 仅1次重试
RETRY_INITIAL_WAIT = 2  # 初始重试等待2秒

# 4. 代理IP配置（新增，移除proxybroker依赖）
PROXY_CHECK_TIMEOUT = 10  # 代理验证超时时间
PROXY_POOL_SIZE = 5       # 代理池大小
PROXY_REFRESH_INTERVAL = 3600  # 代理池刷新间隔（1小时）
PROXY_TYPES = ['HTTP', 'HTTPS']  # 支持的代理类型
# 商业代理配置（可选，优先使用免费代理）
COMMERCIAL_PROXIES = []  # 格式：["http://user:pass@ip:port", ...]

# 5. 全局状态（增强版）
FAISS_CACHE: Dict[str, Dict[str, Any]] = {}
console = Console()
# 请求计时锁（控制频率）
last_request_time = 0
request_lock = threading.Lock()
# 限流状态缓存
rate_limit_status = {
    "is_limited": False,
    "limited_time": 0,
    "retry_after": 300,  # 东方财富限流后等待5分钟
    "block_exit": False  # 新增：是否触发爬虫校验需要退出
}
# 短期缓存（延长缓存时间）
price_cache = {}
code_cache = {}
CACHE_EXPIRE = 1800  # 价格缓存30分钟
CODE_CACHE_EXPIRE = 86400

# 6. 代理IP池（新增）
proxy_pool = []  # 可用代理列表
proxy_lock = threading.Lock()
last_proxy_refresh = 0  # 最后一次刷新代理池时间

# 7. 股票代码映射（适配东方财富，新增协鑫集成）
STOCK_CODE_TYPE = {
    # A股
    "600744": {"name": "华银电力", "market": "沪市", "eastmoney_code": "600744"},
    "600519": {"name": "贵州茅台", "market": "沪市", "eastmoney_code": "600519"},
    "300750": {"name": "宁德时代", "market": "深市", "eastmoney_code": "300750"},
    "002594": {"name": "比亚迪", "market": "深市", "eastmoney_code": "002594"},
    "600759": {"name": "洲际油气", "market": "沪市", "eastmoney_code": "600759"},
    "601857": {"name": "中国石油", "market": "沪市", "eastmoney_code": "601857"},
    "600028": {"name": "中国石化", "market": "沪市", "eastmoney_code": "600028"},
    "601398": {"name": "工商银行", "market": "沪市", "eastmoney_code": "601398"},
    "600036": {"name": "招商银行", "market": "沪市", "eastmoney_code": "600036"},
    "002490": {"name": "山东墨龙", "market": "深市", "eastmoney_code": "002490"},
    "000333": {"name": "美的集团", "market": "深市", "eastmoney_code": "000333"},
    "000651": {"name": "格力电器", "market": "深市", "eastmoney_code": "000651"},
    "000858": {"name": "五粮液", "market": "深市", "eastmoney_code": "000858"},
    "600722": {"name": "金牛化工", "market": "沪市", "eastmoney_code": "600722"},
    "601899": {"name": "紫金矿业", "market": "沪市", "eastmoney_code": "601899"},
    "002506": {"name": "协鑫集成", "market": "深市", "eastmoney_code": "002506"},  # 新增协鑫集成
    # 美股（保留映射）
    "AAPL": {"name": "苹果", "market": "美股", "eastmoney_code": "AAPL"},
    "MSFT": {"name": "微软", "market": "美股", "eastmoney_code": "MSFT"},
    # 港股
    "0700": {"name": "腾讯控股", "market": "港股", "eastmoney_code": "0700"},
    "9988": {"name": "阿里巴巴-SW", "market": "港股", "eastmoney_code": "9988"}
}

# 股票名称映射（新增协鑫集成）
STOCK_NAME_MAP = {
    "华银电力": "600744",
    "贵州茅台": "600519",
    "茅台": "600519",
    "宁德时代": "300750",
    "比亚迪": "002594",
    "洲际油气": "600759",
    "中国石油": "601857",
    "中国石化": "600028",
    "工商银行": "601398",
    "招商银行": "600036",
    "山东墨龙": "002490",
    "美的集团": "000333",
    "格力电器": "000651",
    "五粮液": "000858",
    "金牛化工": "600722",
    "紫金矿业": "601899",
    "协鑫集成": "002506",  # 新增协鑫集成
    "苹果": "AAPL",
    "微软": "MSFT",
    "谷歌": "GOOGL",
    "特斯拉": "TSLA",
    "腾讯控股": "0700",
    "腾讯": "0700",
    "阿里巴巴-SW": "9988"
}

# 8. 退出标记（新增）
QUERY_COMPLETED = False

# 9. 交易时间配置（A股）
CN_TZ = pytz.timezone('Asia/Shanghai')
TRADING_HOURS = {
    "morning": (9, 30, 11, 30),   # 9:30-11:30
    "afternoon": (13, 0, 15, 0)   # 13:00-15:00
}
# 2026年节假日（可根据实际调整）
HOLIDAYS = [
    "2026-01-01", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13", "2026-02-14",
    "2026-04-04", "2026-04-05", "2026-05-01", "2026-06-01", "2026-09-27", "2026-10-01",
    "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"
]

# ============================ 交易日/交易时间判断函数 ============================
def is_trading_day(check_date: datetime.date = None) -> bool:
    """判断是否为A股交易日（周一到周五，排除节假日）"""
    if check_date is None:
        check_date = datetime.datetime.now(CN_TZ).date()
    
    # 周末不是交易日
    if check_date.weekday() >= 5:
        return False
    
    # 节假日不是交易日
    if check_date.strftime("%Y-%m-%d") in HOLIDAYS:
        return False
    
    return True

def get_latest_trading_day() -> datetime.date:
    """获取最近一个交易日"""
    current_date = datetime.datetime.now(CN_TZ).date()
    delta = 0
    while True:
        check_date = current_date - datetime.timedelta(days=delta)
        if is_trading_day(check_date):
            return check_date
        delta += 1

def is_in_trading_hours() -> bool:
    """判断当前是否处于A股交易时间"""
    now = datetime.datetime.now(CN_TZ)
    current_hour = now.hour
    current_minute = now.minute
    
    # 先判断是否为交易日
    if not is_trading_day():
        return False
    
    # 上午交易时间：9:30-11:30
    morning_start = datetime.time(TRADING_HOURS["morning"][0], TRADING_HOURS["morning"][1])
    morning_end = datetime.time(TRADING_HOURS["morning"][2], TRADING_HOURS["morning"][3])
    # 下午交易时间：13:00-15:00
    afternoon_start = datetime.time(TRADING_HOURS["afternoon"][0], TRADING_HOURS["afternoon"][1])
    afternoon_end = datetime.time(TRADING_HOURS["afternoon"][2], TRADING_HOURS["afternoon"][3])
    
    current_time = datetime.time(current_hour, current_minute)
    
    # 判断是否在交易时间段内
    if (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end):
        return True
    return False

# ============================ 代理IP核心函数（重构，移除proxybroker） ============================
async def fetch_proxies():
    """异步获取高可用免费代理IP（兼容新版Python，移除proxybroker依赖）"""
    valid_proxies = []
    # 改用公开免费代理接口（替代proxybroker）
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=PROXY_CHECK_TIMEOUT)) as session:
            # 免费代理接口1：proxy-list.org
            try:
                resp = await session.get("https://proxy-list.org/english/index.php?p=1", timeout=PROXY_CHECK_TIMEOUT)
                text = await resp.text()
                # 解析代理IP和端口
                proxy_pattern = re.compile(r"\d+\.\d+\.\d+\.\d+:\d+")
                proxies = proxy_pattern.findall(text)
                # 验证代理有效性
                for proxy in proxies[:PROXY_POOL_SIZE*3]:
                    try:
                        async with session.get(
                            "http://www.baidu.com", 
                            proxy=f"http://{proxy}", 
                            timeout=5
                        ):
                            valid_proxies.append(f"http://{proxy}")
                            if len(valid_proxies) >= PROXY_POOL_SIZE:
                                break
                    except:
                        continue
            except:
                # 备用接口：免费代理池
                try:
                    resp = await session.get("https://free-proxy-list.net/", timeout=PROXY_CHECK_TIMEOUT)
                    text = await resp.text()
                    proxy_pattern = re.compile(r"\d+\.\d+\.\d+\.\d+:\d+")
                    proxies = proxy_pattern.findall(text)
                    for proxy in proxies[:PROXY_POOL_SIZE*3]:
                        try:
                            async with session.get(
                                "http://www.baidu.com", 
                                proxy=f"http://{proxy}", 
                                timeout=5
                            ):
                                valid_proxies.append(f"http://{proxy}")
                                if len(valid_proxies) >= PROXY_POOL_SIZE:
                                    break
                        except:
                            continue
                except:
                    pass
    except Exception as e:
        console.print(f"[dim]❌ 获取代理失败：{str(e)}[/dim]")
    return valid_proxies[:PROXY_POOL_SIZE]

def refresh_proxy_pool():
    """刷新代理池（线程安全）"""
    global proxy_pool, last_proxy_refresh
    
    with proxy_lock:
        # 检查是否需要刷新
        if time.time() - last_proxy_refresh < PROXY_REFRESH_INTERVAL and proxy_pool:
            return
        
        console.print("[dim]🔄 正在刷新代理IP池...[/dim]")
        
        # 优先使用商业代理
        if COMMERCIAL_PROXIES:
            proxy_pool = COMMERCIAL_PROXIES.copy()
            console.print(f"[dim]✅ 使用商业代理池，共{len(proxy_pool)}个代理[/dim]")
        else:
            # 获取免费代理（重构后）
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_proxies = loop.run_until_complete(fetch_proxies())
                proxy_pool = new_proxies
                loop.close()  # 确保循环关闭
                
                if proxy_pool:
                    console.print(f"[dim]✅ 刷新代理池成功，获取到{len(proxy_pool)}个可用代理[/dim]")
                else:
                    console.print("[dim]⚠️  未获取到免费代理，将使用直连模式[/dim]")
            except Exception as e:
                console.print(f"[dim]❌ 获取代理失败：{str(e)}，将使用直连模式[/dim]")
                proxy_pool = []
        
        last_proxy_refresh = time.time()

def get_random_proxy():
    """获取随机代理IP（自动刷新）"""
    # 检查并刷新代理池
    if not proxy_pool or time.time() - last_proxy_refresh > PROXY_REFRESH_INTERVAL:
        refresh_proxy_pool()
    
    with proxy_lock:
        if proxy_pool:
            return random.choice(proxy_pool)
    return None

def get_request_proxies():
    """获取requests可用的代理配置"""
    proxy = get_random_proxy()
    if not proxy:
        return None
    
    # 适配requests的代理格式
    proxies = {
        "http": proxy,
        "https": proxy
    }
    return proxies

# ============================ 核心工具函数 ============================
def check_rate_limit():
    """检查是否处于限流状态"""
    global rate_limit_status
    with request_lock:
        if rate_limit_status["is_limited"]:
            if time.time() - rate_limit_status["limited_time"] > rate_limit_status["retry_after"]:
                rate_limit_status["is_limited"] = False
                console.print("[bold green]✅ 限流状态已解除[/bold green]")
            else:
                remaining = int(rate_limit_status["retry_after"] - (time.time() - rate_limit_status["limited_time"]))
                console.print(f"[bold yellow]⚠️  仍处于限流状态，剩余等待时间：{remaining//60}分{remaining%60}秒[/bold yellow]")
        return rate_limit_status["is_limited"]

def set_rate_limit(retry_after=300, block_exit=False):
    """设置限流状态（新增block_exit控制是否自动退出）"""
    global rate_limit_status
    with request_lock:
        rate_limit_status["is_limited"] = True
        rate_limit_status["limited_time"] = time.time()
        rate_limit_status["retry_after"] = retry_after
        rate_limit_status["block_exit"] = block_exit  # 标记是否需要退出
        
        # 如果触发爬虫校验，直接提示并标记退出
        if block_exit:
            console.print("[bold red]🚨 检测到爬虫校验触发，程序将自动退出！[/bold red]")

def control_request_frequency():
    """请求频率控制"""
    # 检查是否需要强制退出
    if rate_limit_status.get("block_exit", False):
        console.print("[bold red]🛑 爬虫校验触发，程序退出中...[/bold red]")
        sys.exit(1)
    
    if check_rate_limit():
        raise Exception(f"当前处于限流状态，请等待{rate_limit_status['retry_after']//60}分钟后重试")
    
    global last_request_time
    with request_lock:
        current_time = time.time()
        elapsed = current_time - last_request_time
        random_sleep = random.uniform(1, 3)
        if elapsed < REQUEST_INTERVAL:
            sleep_time = REQUEST_INTERVAL - elapsed + random_sleep
            console.print(f"[dim]🕒 限流防护：等待 {sleep_time:.1f} 秒后再请求[/dim]")
            time.sleep(sleep_time)
        last_request_time = time.time()

def get_random_user_agent():
    """获取随机User-Agent"""
    return random.choice(USER_AGENT_POOL)

def check_cache(stock_code):
    """检查价格缓存"""
    if stock_code in price_cache:
        cache_time, data = price_cache[stock_code]
        if time.time() - cache_time < CACHE_EXPIRE:
            # 更新查询时间
            data["update_time"] = datetime.datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
            data["query_time"] = datetime.datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
            return data
        else:
            # 缓存过期，删除
            del price_cache[stock_code]
    return None


def update_cache(stock_code, data):
    """更新价格缓存"""
    price_cache[stock_code] = (time.time(), data)

def detect_anti_crawler(response_text):
    """检测是否触发爬虫校验"""
    anti_crawler_keywords = [
        "验证码", "验证", "人机验证", "滑块验证", "安全验证",
        "请先验证", "访问验证", "Anti-Spider", "antispider",
        "爬虫", "禁止访问", "403 Forbidden", "503 Service"
    ]
    
    for keyword in anti_crawler_keywords:
        if keyword in response_text:
            return True
    return False

def check_code_cache(stock_name):
    """检查股票代码缓存"""
    if stock_name in code_cache:
        cache_time, code = code_cache[stock_name]
        if time.time() - cache_time < CODE_CACHE_EXPIRE:
            return code
        else:
            del code_cache[stock_name]
    return None

def update_code_cache(stock_name, code):
    """更新股票代码缓存"""
    code_cache[stock_name] = (time.time(), code)

def get_stock_code_dynamically(stock_name):
    """动态查询股票代码（从东方财富搜索）"""
    stock_name = stock_name.strip()
    
    # 先查缓存
    cached_code = check_code_cache(stock_name)
    if cached_code:
        return cached_code
    
    # 限流状态下跳过
    if check_rate_limit():
        console.print("[dim]🔴 处于限流状态，跳过动态代码查询[/dim]")
        return None
    
    # 构建请求头
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "http://quote.eastmoney.com/"
    }
    
    try:
        # 东方财富搜索URL
        eastmoney_url = f"http://quote.eastmoney.com/search.aspx?keywords={stock_name}"
        control_request_frequency()
        
        # 获取代理
        proxies = get_request_proxies()
        if proxies:
            console.print(f"[dim]🌐 使用代理IP：{list(proxies.values())[0]}[/dim]")
        
        # 发送请求
        response = requests.get(
            eastmoney_url, 
            headers=headers, 
            proxies=proxies,
            timeout=10
        )
        
        # 检测反爬虫
        if detect_anti_crawler(response.text):
            set_rate_limit(block_exit=True)
            return None
        
        response.raise_for_status()
        
        # 解析结果
        soup = BeautifulSoup(response.text, 'html.parser')
        result_table = soup.select_one('.quotebody table')
        if result_table:
            rows = result_table.select('tr')
            for row in rows[1:]:  # 跳过表头
                cols = row.select('td')
                if len(cols) >= 2:
                    name_text = cols[0].get_text().strip()
                    code_text = cols[1].get_text().strip()
                    
                    # 匹配股票名称
                    if stock_name in name_text or name_text in stock_name:
                        update_code_cache(stock_name, code_text)
                        update_code_cache(name_text, code_text)
                        return code_text
        
        return None
        
    except Exception as e:
        console.print(f"[dim]动态查询代码失败：{str(e)}[/dim]")
        if "验证" in str(e) or "403" in str(e):
            set_rate_limit(block_exit=True)
        return None
    
# ============================ 核心：东方财富股价查询 ============================
def get_stock_price_from_eastmoney(stock_code):
    """
    从东方财富获取股价：
    - 交易时间：返回实时股价（f62字段）
    - 非交易时间：返回最近交易日收盘价（f43字段）
    """
    # 构建请求字段和URL
    fields = "f43,f44,f45,f46,f47,f48,f49,f50,f57,f58,f60,f62"
    
    # 美股处理
    if stock_code in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
        url = f"https://quote.eastmoney.com/usstock/{stock_code.lower()}.html"
    # 港股处理（4位数字）
    elif stock_code.startswith('0') and len(stock_code) == 4:
        url = f"https://quote.eastmoney.com/hkstock/{stock_code}.html"
    # 沪市A股（6/9开头）
    elif stock_code.startswith(('6', '9')):
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=1.{stock_code}&fields={fields}"
    # 深市A股（0/3开头）
    elif stock_code.startswith(('0', '3')):
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=0.{stock_code}&fields={fields}"
    else:
        return None, "不支持的股票代码类型"
    
    # 构建请求头
    headers = {
        "User-Agent": get_random_user_agent(),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*"
    }
    
    try:
        # 控制请求频率
        control_request_frequency()
        
        # 获取代理
        proxies = get_request_proxies()
        if proxies:
            console.print(f"[dim]🌐 使用代理IP：{list(proxies.values())[0]}[/dim]")
        
        # 发送请求
        response = requests.get(
            url, 
            headers=headers, 
            proxies=proxies,
            timeout=10,
            verify=False  # 忽略SSL验证
        )
        
        # 记录响应信息
        response_text = response.text
        response_status = response.status_code
        
        # 检测反爬虫
        if detect_anti_crawler(response_text):
            set_rate_limit(block_exit=True)
            return None, "触发爬虫校验，程序将退出"
        
        response.raise_for_status()
        
        # 基础时间信息
        current_real_time = datetime.datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        in_trading = is_in_trading_hours()
        latest_trading_day = get_latest_trading_day().strftime("%Y-%m-%d")
        
        # 美股处理（网页解析）
        if stock_code in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
            soup = BeautifulSoup(response_text, 'html.parser')
            price_elem = soup.select_one('.price')
            if not price_elem:
                return None, "未找到价格数据"
            price = float(price_elem.get_text().strip().replace(',', ''))
            
            close_elem = soup.select_one('.close')
            close_price = price if not close_elem else float(close_elem.get_text().strip().replace(',', ''))
            
            name = STOCK_CODE_TYPE.get(stock_code, {}).get("name", stock_code)
            
            result = {
                "price": price,
                "close": close_price,
                "name": name,
                "currency": "USD",
                "update_time": current_real_time,
                "data_date": latest_trading_day,
                "is_real_time": in_trading,
                "open": price,
                "pre_close": close_price,
                "high": price,
                "low": price,
                "volume": 0,
                "amount": 0,
                "response_status": response_status,
                "response_info": f"Status: {response_status}, URL: {url}"
            }
            return result, None
        
        # A股/港股处理（JSON解析）
        else:
            data = response.json()
            if data.get("data"):
                # 交易时间取实时价，非交易时间取收盘价
                if in_trading:
                    current_price = round(float(data["data"]["f62"]) / 100, 2)
                else:
                    current_price = round(float(data["data"]["f43"]) / 100, 2)
                
                # 解析其他字段
                close_price = round(float(data["data"]["f43"]) / 100, 2)
                pre_close = round(float(data["data"]["f60"]) / 100, 2)
                open_price = round(float(data["data"]["f46"]) / 100, 2)
                high = round(float(data["data"].get("f44", 0)) / 100, 2)
                low = round(float(data["data"].get("f45", 0)) / 100, 2)
                volume = round(float(data["data"].get("f47", 0)) / 100, 2)
                amount = round(float(data["data"].get("f48", 0)) / 10000, 2)
                name = data["data"].get("f58", stock_code)
                code = data["data"].get("f57", stock_code)
                
                # 计算涨跌
                change = current_price - pre_close
                change_pct = (change / pre_close) * 100 if pre_close != 0 else 0
                
                result = {
                    "price": current_price,
                    "close": close_price,
                    "pre_close": pre_close,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "amount": amount,
                    "change": change,
                    "change_pct": change_pct,
                    "name": name,
                    "code": code,
                    "currency": "CNY",
                    "update_time": current_real_time,
                    "data_date": latest_trading_day,
                    "is_real_time": in_trading,
                    "market": "沪市A股" if code.startswith('6') else "深市A股",
                    "response_status": response_status,
                    "response_info": f"Status: {response_status}, URL: {url}"
                }
                return result, None
            else:
                return None, "未返回有效数据"
    except Exception as e:
        console.print(f"[dim]获取股价失败：{str(e)}[/dim]")
        return None, str(e)

# ============================ LangChain回调类 ============================
class HackerMatrixCallback(BaseCallbackHandler):
    """自定义回调类，显示查询进度"""
    def __init__(self):
        self._ignore_chain = False
        self._raise_error = False
        self.progress = None
        self.task_id = None
        self.retry_count = 0

    def on_chain_start(self, serialized, inputs,** kwargs):
        """链开始时触发"""
        if rate_limit_status.get("block_exit", False):
            sys.exit(1)
        if not self._ignore_chain:
            if self.progress:
                self.progress.stop()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold green]🤖 正在查询东方财富实时数据...[/bold green]"),
                console=console
            )
            self.task_id = self.progress.add_task("", total=None)
            self.progress.start()

    def on_chain_end(self, outputs, **kwargs):
        """链结束时触发"""
        if not self._ignore_chain and self.progress:
            self.progress.stop()
            self.progress = None
            console.print("[bold green]✅ 查询完成[/bold green]")

    def on_chain_error(self, error,** kwargs):
        """链出错时触发"""
        if not self._ignore_chain and self.progress:
            self.progress.stop()
            self.progress = None
        console.print(f"[bold red]❌ 查询出错：{str(error)}[/bold red]")

# ============================ LangChain工具函数 ============================
@tool
@retry(stop=stop_after_attempt(RETRY_MAX_ATTEMPTS), 
       wait=wait_exponential(multiplier=1, min=RETRY_INITIAL_WAIT, max=10),
       retry=retry_if_exception_type(Exception))
def get_universal_stock_price(code_or_name: str) -> str:
    """
    获取任意股票的价格（支持A股/美股/港股）：
    - 交易时间：返回实时股价
    - 非交易时间：返回最近交易日收盘价
    :param code_or_name: 股票代码或名称（如：600519、贵州茅台、AAPL、招商轮船）
    :return: 格式化的股价信息
    """
    global QUERY_COMPLETED
    
    # 1. 识别股票代码
    if code_or_name in STOCK_CODE_TYPE:
        stock_code = code_or_name
    elif code_or_name in STOCK_NAME_MAP:
        stock_code = STOCK_NAME_MAP[code_or_name]
    else:
        stock_code = get_stock_code_dynamically(code_or_name)
        if not stock_code:
            return f"❌ 未找到股票「{code_or_name}」的相关信息"
    
    # 2. 查询股价（强制实时查询，跳过缓存）
    result_data, error = get_stock_price_from_eastmoney(stock_code)
    if error:
        return f"❌ 获取「{code_or_name}」股价失败：{error}"
    
    # 3. 格式化输出
    name = result_data.get("name", code_or_name)
    code = result_data.get("code", stock_code)
    price = result_data.get("price", 0)
    currency = result_data.get("currency", "CNY")
    update_time = result_data.get("update_time", "")
    data_date = result_data.get("data_date", "")
    is_real_time = result_data.get("is_real_time", False)
    market = result_data.get("market", "未知市场")
    
    # 区分实时/非实时数据提示
    time_tip = "📈 交易时间，返回实时股价" if is_real_time else f"📊 非交易时间，返回最近交易日（{data_date}）收盘价"
    
    # 完整数据详情
    details = f"""
{time_tip}

📊 股票行情数据
 ───────────────────────────────────────────
 股票名称    {name}（{code}）
 交易市场    {market}
 当前价格    {currency}¥{price:.2f}
 涨跌幅      {result_data.get('change_pct', 0):+.2f}%
 涨跌额      {result_data.get('change', 0):+.2f}
 今开        {currency}¥{result_data.get('open', 0):.2f}
 昨收        {currency}¥{result_data.get('pre_close', 0):.2f}
 最高        {currency}¥{result_data.get('high', 0):.2f}
 最低        {currency}¥{result_data.get('low', 0):.2f}
 成交量      {result_data.get('volume', 0):.1f} 手
 成交额      {result_data.get('amount', 0):,.0f} 万元
 数据日期    {data_date}
 查询时间    {update_time}
 数据类型    {"实时数据" if is_real_time else "收盘数据"}
 ───────────────────────────────────────────
 接口信息    {result_data.get('response_info', '未知')}
 状态码      {result_data.get('response_status', '未知')}
    """
    
    QUERY_COMPLETED = True
    
    return f"✅ 查询到「{name}（{code}）」的最新股价：{currency}¥{price:.2f}\n{details}"

# ============================ 主程序逻辑 ============================
def create_agent():
    """创建LangChain Agent（强制实时查询）"""
    # 初始化LLM（temperature=0，禁用随机性）
    llm = ChatOpenAI(
        model="qwen3.5-plus",
        temperature=0.0,  # 禁用随机性，确保输出稳定
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=dashscope_key,
        request_timeout=60
    )
    
    tools = [get_universal_stock_price]
    
    # 系统提示词（强制返回真实数据，禁止幻觉）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是严格的股票数据查询助手，必须遵守以下规则：
        1. 仅调用get_universal_stock_price工具获取数据，禁止使用任何缓存/记忆/历史数据；
        2. 必须原样返回工具返回的所有内容，不增删、不修改、不编造任何信息（包括时间、价格）；
        3. 数据时间以工具返回的「查询时间」和「数据日期」为准，禁止使用任何其他时间；
        4. 一次查询仅调用一次工具，直接返回结果，不添加任何额外解释；
        5. 禁止使用任何幻觉数据，所有信息必须来自工具返回值。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | OpenAIFunctionsAgentOutputParser()
    )
    
    # 创建Agent执行器, 限制Agent仅执行1次迭代，强制实时查询
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[HackerMatrixCallback()],
        max_iterations=1,  # 仅1次迭代，避免重复调用
        early_stopping_method="force",
        handle_parsing_errors=True,
        verbose=False
    )
    
    return agent_executor

def main():
    """主程序入口"""
    global QUERY_COMPLETED
    QUERY_COMPLETED = False
    
    # 初始化代理IP池
    console.print("[dim]🔄 初始化代理IP池...[/dim]")
    refresh_proxy_pool()
    
    # 初始化Agent
    agent_executor = create_agent()
    
    # 清空对话历史和缓存
    chat_history = ChatMessageHistory()
    price_cache.clear()
    code_cache.clear()
    
    # 显示当前时间（上海时区）
    current_query_time = datetime.datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"✅ 缓存已清空，当前查询时间（上海时区）：{current_query_time}")
    
    # 启动提示
    console.print("🤖 全球股票分析 Agent 已启动（东方财富实时接口版）！")
    console.print("📌 核心特性：区分交易时间/非交易时间，返回最新可用真实数据")
    console.print("📌 交易时间（9:30-11:30/13:00-15:00）：返回实时股价")
    console.print("📌 非交易时间：返回最近交易日收盘价")
    console.print("📌 支持：A股/美股/港股（包含招商轮船等股票）")
    console.print("📌 输入 'quit'/'exit'/'退出' 结束对话")
    console.print("-" * 80)
    
    # 交互循环
    session = PromptSession(history=InMemoryHistory())
    while True:
        try:
            user_input = session.prompt("你: ")
            
            # 退出指令
            if user_input in ["quit", "exit", "退出"]:
                console.print("[bold blue]👋 程序已退出[/bold blue]")
                sys.exit(0)
            
            # 执行查询
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history.messages,
                "agent_scratchpad": []
            })
            
            # 输出结果
            console.print("-" * 40 + " SYS.RESPONSE " + "-" * 40)
            console.print(response["output"])
            console.print("-" * 87 + " EOF " + "-" * 87)
            
            # 查询完成后自动退出
            if QUERY_COMPLETED:
                console.print("[bold green]✅ 查询完成，程序自动退出[/bold green]")
                sys.exit(0)
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]⏹ 操作取消，程序退出[/bold yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[bold red]❌ 程序出错：{str(e)}[/bold red]")
            sys.exit(1)

if __name__ == "__main__":
    main()
