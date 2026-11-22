import os
import json
import time
import random
import datetime
import threading
import sys
from pathlib import Path
from typing import List, Dict, Any
from filelock import FileLock

# 新增：动态查询股票代码的依赖
import requests
from bs4 import BeautifulSoup
import re

# 新增：代理IP相关依赖
import asyncio
from proxybroker import Broker
import aiohttp

# 数据可视化与交互依赖
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

# LangChain 核心依赖（适配旧版 LangChain 0.0.x）
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
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

# 3. 防爬虫/限流配置（适配东方财富）
REQUEST_INTERVAL = 5  # 东方财富限流宽松
USER_AGENT_POOL = [    # 真实浏览器UA池
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
]
RETRY_MAX_ATTEMPTS = 1  # 仅1次重试
RETRY_INITIAL_WAIT = 2  # 初始重试等待2秒

# 4. 代理IP配置（新增）
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

# 7. 股票代码映射（适配东方财富）
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
    "601899": {"name": "紫金矿业", "market": "沪市", "eastmoney_code": "601899"},  # 新增紫金矿业
    # 美股（保留映射）
    "AAPL": {"name": "苹果", "market": "美股", "eastmoney_code": "AAPL"},
    "MSFT": {"name": "微软", "market": "美股", "eastmoney_code": "MSFT"},
    # 港股
    "0700": {"name": "腾讯控股", "market": "港股", "eastmoney_code": "0700"},
    "9988": {"name": "阿里巴巴-SW", "market": "港股", "eastmoney_code": "9988"}
}

# 股票名称映射
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
    "紫金矿业": "601899",  # 新增紫金矿业名称映射
    "苹果": "AAPL",
    "微软": "MSFT",
    "谷歌": "GOOGL",
    "特斯拉": "TSLA",
    "腾讯控股": "0700",
    "腾讯": "0700",
    "阿里巴巴-SW": "9988"
}

# ============================ 代理IP核心函数（新增） ============================
async def fetch_proxies():
    """异步获取高可用免费代理IP"""
    proxies = []
    broker = Broker(proxies)
    
    # 启动代理抓取
    tasks = asyncio.gather(
        broker.find(
            types=PROXY_TYPES,
            limit=PROXY_POOL_SIZE * 3,  # 多获取一些用于筛选
            timeout=PROXY_CHECK_TIMEOUT,
            strict=True  # 严格验证
        ),
        asyncio.sleep(PROXY_CHECK_TIMEOUT)  # 超时控制
    )
    
    try:
        await tasks
    except asyncio.TimeoutError:
        pass
    
    # 格式化代理地址
    valid_proxies = []
    for proxy in proxies:
        if proxy and proxy.host and proxy.port:
            proxy_str = f"{proxy.scheme.lower()}://{proxy.host}:{proxy.port}"
            valid_proxies.append(proxy_str)
    
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
            # 获取免费代理
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_proxies = loop.run_until_complete(fetch_proxies())
                proxy_pool = new_proxies
                loop.close()
                
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
    """检查价格缓存是否有效"""
    if stock_code in price_cache:
        cache_time, data = price_cache[stock_code]
        if time.time() - cache_time < CACHE_EXPIRE:
            return data
        else:
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

# ============================ 股票代码查询函数 ============================
def check_code_cache(stock_name):
    """检查股票代码缓存是否有效"""
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
    """动态查询股票代码（东方财富接口 + 代理IP）"""
    stock_name = stock_name.strip()
    
    # 先查缓存
    cached_code = check_code_cache(stock_name)
    if cached_code:
        return cached_code
    
    if check_rate_limit():
        console.print("[dim]🔴 处于限流状态，跳过动态代码查询[/dim]")
        return None
    
    # 防爬虫准备
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "http://quote.eastmoney.com/"
    }
    
    try:
        # 东方财富搜索接口
        eastmoney_url = f"http://quote.eastmoney.com/search.aspx?keywords={stock_name}"
        control_request_frequency()
        
        # 获取代理IP
        proxies = get_request_proxies()
        if proxies:
            console.print(f"[dim]🌐 使用代理IP：{list(proxies.values())[0]}[/dim]")
        
        # 发送请求（带代理）
        response = requests.get(
            eastmoney_url, 
            headers=headers, 
            proxies=proxies,
            timeout=10
        )
        
        # 检测爬虫校验
        if detect_anti_crawler(response.text):
            set_rate_limit(block_exit=True)  # 触发自动退出
            return None
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        result_table = soup.select_one('.quotebody table')
        if result_table:
            rows = result_table.select('tr')
            for row in rows[1:]:
                cols = row.select('td')
                if len(cols) >= 2:
                    name_text = cols[0].get_text().strip()
                    code_text = cols[1].get_text().strip()
                    
                    if stock_name in name_text or name_text in stock_name:
                        update_code_cache(stock_name, code_text)
                        update_code_cache(name_text, code_text)
                        return code_text
        
        return None
        
    except Exception as e:
        console.print(f"[dim]动态查询代码失败：{str(e)}[/dim]")
        # 如果是爬虫校验相关错误，标记退出
        if "验证" in str(e) or "403" in str(e):
            set_rate_limit(block_exit=True)
        return None

def get_stock_price_from_eastmoney(stock_code):
    """从东方财富获取股票价格（带代理IP）- 增强版：返回完整当日数据"""
    # 构建请求URL（支持更多字段，获取完整当日数据）
    if stock_code in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
        # 美股特殊处理（使用简化接口）
        url = f"https://quote.eastmoney.com/usstock/{stock_code.lower()}.html"
    elif stock_code.startswith('0') and len(stock_code) == 4:
        # 港股
        url = f"https://quote.eastmoney.com/hkstock/{stock_code}.html"
    elif stock_code.startswith(('6', '9')):
        # 沪市 - 增加更多字段：收盘价、成交量、成交额、换手率等
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=1.{stock_code}&fields=f43,f57,f58,f168,f107,f60,f46,f51,f177,f169,f170,f152,f151,f171,f172,f173,f174,f175,f176"
    elif stock_code.startswith(('0', '3')):
        # 深市 - 增加更多字段
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=0.{stock_code}&fields=f43,f57,f58,f168,f107,f60,f46,f51,f177,f169,f170,f152,f151,f171,f172,f173,f174,f175,f176"
    else:
        return None, "不支持的股票代码类型"
    
    headers = {
        "User-Agent": get_random_user_agent(),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*"
    }
    
    try:
        control_request_frequency()
        
        # 获取代理IP
        proxies = get_request_proxies()
        if proxies:
            console.print(f"[dim]🌐 使用代理IP：{list(proxies.values())[0]}[/dim]")
        
        # 发送请求（带代理）
        response = requests.get(
            url, 
            headers=headers, 
            proxies=proxies,
            timeout=10
        )
        
        # 检测爬虫校验
        if detect_anti_crawler(response.text):
            set_rate_limit(block_exit=True)
            return None, "触发爬虫校验，程序将退出"
        
        response.raise_for_status()
        
        if stock_code in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
            # 解析美股页面
            soup = BeautifulSoup(response.text, 'html.parser')
            price_elem = soup.select_one('.price')
            if not price_elem:
                return None, "未找到价格数据"
            price = float(price_elem.get_text().strip().replace(',', ''))
            
            # 尝试获取收盘价（美股）
            close_elem = soup.select_one('.close')
            close_price = price if not close_elem else float(close_elem.get_text().strip().replace(',', ''))
            
            name = STOCK_CODE_TYPE.get(stock_code, {}).get("name", stock_code)
            return {
                "price": price,          # 当前价
                "close": close_price,    # 收盘价
                "name": name,
                "currency": "USD",
                "update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, None
        else:
            # 解析A股/港股JSON数据（增强版）
            data = response.json()
            if data.get("data"):
                # 核心价格字段
                current_price = float(data["data"]["f43"])  # 当前价
                close_price = float(data["data"]["f60"])    # 收盘价（当日）
                pre_close = float(data["data"]["f46"])      # 昨收价
                open_price = float(data["data"]["f51"])     # 今开价
                high = float(data["data"].get("f169", 0))   # 最高价
                low = float(data["data"].get("f170", 0))    # 最低价
                name = data["data"]["f57"]                  # 股票名称
                update_time = data["data"].get("f168", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                # 新增当日交易数据
                volume = int(data["data"].get("f177", 0))   # 成交量（手）
                turnover = float(data["data"].get("f152", 0))/10000  # 成交额（万元）
                turnover_rate = float(data["data"].get("f151", 0))   # 换手率（%）
                amplitude = float(data["data"].get("f173", 0))      # 振幅（%）
                pe = float(data["data"].get("f174", 0))             # 市盈率（TTM）
                pb = float(data["data"].get("f175", 0))              # 市净率
                
                return {
                    "price": current_price,   # 当前价
                    "close": close_price,     # 当日收盘价
                    "pre_close": pre_close,   # 昨日收盘价
                    "open": open_price,       # 今日开盘价
                    "high": high,             # 今日最高价
                    "low": low,               # 今日最低价
                    "name": name,             # 股票名称
                    "currency": "CNY" if stock_code.isdigit() else "HKD",
                    "volume": volume,         # 成交量（手）
                    "turnover": turnover,     # 成交额（万元）
                    "turnover_rate": turnover_rate,  # 换手率（%）
                    "amplitude": amplitude,   # 振幅（%）
                    "pe": pe,                 # 市盈率
                    "pb": pb,                 # 市净率
                    "update_time": update_time
                }, None
    
    except Exception as e:
        error_msg = str(e)
        # 检测爬虫校验并标记退出
        if detect_anti_crawler(error_msg) or "403" in error_msg or "验证" in error_msg:
            set_rate_limit(block_exit=True)
        return None, f"获取价格失败：{error_msg}"
    
    return None, "未获取到价格数据"

# ============================ 回调类 ============================
class HackerMatrixCallback(BaseCallbackHandler):
    """黑客风格回调类"""
    def __init__(self):
        self._ignore_chain = False
        self._raise_error = False
        self.progress = None
        self.task_id = None
        self.retry_count = 0

    @property
    def ignore_chain(self):
        return self._ignore_chain

    @ignore_chain.setter
    def ignore_chain(self, value):
        self._ignore_chain = value

    @property
    def raise_error(self):
        return self._raise_error

    @raise_error.setter
    def raise_error(self, value):
        self._raise_error = value

    def on_chain_start(self, serialized, inputs,** kwargs):
        # 检查是否需要退出
        if rate_limit_status.get("block_exit", False):
            sys.exit(1)
            
        if not self._ignore_chain:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold green]🤖 正在查询东方财富数据...[/bold green]"),
                console=console
            )
            self.task_id = self.progress.add_task("", total=None)
            self.progress.start()

    def on_chain_end(self, outputs, **kwargs):
        if not self._ignore_chain and self.progress:
            self.progress.stop()
            console.print("[bold green]✅ 查询完成[/bold green]")

    def on_chain_error(self, error,** kwargs):
        if not self._ignore_chain and self.progress:
            self.progress.stop()
        
        # 检测爬虫校验错误，触发退出
        error_msg = str(error)
        if detect_anti_crawler(error_msg) or rate_limit_status.get("block_exit", False):
            console.print("[bold red]🚨 爬虫校验触发，程序退出！[/bold red]")
            sys.exit(1)
            
        self.retry_count += 1
        if self.retry_count >= RETRY_MAX_ATTEMPTS:
            console.print(f"[bold red]❌ 重试次数已达上限：{str(error)}[/bold red]")
            set_rate_limit()
        else:
            console.print(f"[bold yellow]⚠️  第{self.retry_count}次重试失败：{str(error)}[/bold yellow]")
        
        if self._raise_error:
            raise error

    def on_llm_error(self, error, **kwargs):
        console.print(f"[bold red]❌ LLM 调用出错：{str(error)}[/bold red]")
        if self._raise_error:
            raise error

# ============================ 核心工具函数 ============================
@tool
@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=RETRY_INITIAL_WAIT, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=False
)
def get_universal_stock_price(code_or_name: str) -> str:
    """
    查询全球股票最新价格（东方财富接口 + 浮动IP）- 增强版：
    1. 支持任意股票查询（无需本地映射）
    2. 返回完整的当日股价信息（收盘价、成交量、换手率等）
    支持：股票名称、股票代码
    """
    code_or_name = code_or_name.strip()
    
    # 检查是否需要退出
    if rate_limit_status.get("block_exit", False):
        return "🚨 触发爬虫校验，程序即将退出！"
    
    # 先检查限流状态
    if check_rate_limit():
        remaining = int(rate_limit_status["retry_after"] - (time.time() - rate_limit_status["limited_time"]))
        return (f"❌ 系统当前处于查询速率限制状态\n"
                f"⏳ 建议等待 {remaining//60}分{remaining%60}秒后重试\n"
                f"📞 限流解除后可正常查询 {code_or_name} 的股价")
    
    # 解析股票代码 - 优化逻辑：
    # 1. 优先查本地映射
    # 2. 本地没有则动态查询
    # 3. 如果是纯数字，直接作为代码使用
    stock_code = STOCK_NAME_MAP.get(code_or_name)
    
    if not stock_code:
        # 本地映射没有，尝试动态查询
        console.print(f"[dim]🔍 本地未找到{code_or_name}，正在动态查询代码...[/dim]")
        stock_code = get_stock_code_dynamically(code_or_name)
    
    # 如果动态查询也失败，但输入是数字，直接作为代码使用
    if not stock_code:
        if code_or_name.isdigit() or code_or_name in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
            stock_code = code_or_name
        else:
            return f"❌ 未查询到 {code_or_name} 的股票代码，请确认名称是否正确"

    # 检查缓存（优先返回缓存）
    cached_data = check_cache(stock_code)
    if cached_data:
        return f"✅ 缓存数据（30分钟内）：\n{cached_data}"

    # 从东方财富获取价格（增强版：返回完整数据）
    price_data, error = get_stock_price_from_eastmoney(stock_code)
    if not price_data:
        # 检查是否触发退出
        if rate_limit_status.get("block_exit", False):
            sys.exit(1)
        return f"❌ {error}"

    # 构建返回结果（增强版：包含完整当日数据）
    currency_symbol = {"CNY": "¥", "USD": "$", "HKD": "HK$"}.get(price_data["currency"], price_data["currency"])
    
    # 核心价格信息
    result = (
        f"{price_data['name']}（{stock_code}）\n"
        f"📊 当日收盘价：{currency_symbol}{price_data['close']:.2f} ({price_data['currency']})\n"
        f"📈 当前价：{currency_symbol}{price_data['price']:.2f}\n"
        f"📅 数据更新时间：{price_data['update_time']}（北京时间）\n"
        f"🔍 数据源：东方财富（防爬虫模式 + 浮动IP）"
    )
    
    # 涨跌信息
    if "pre_close" in price_data:
        change = price_data['close'] - price_data['pre_close']  # 用收盘价计算涨跌
        change_pct = (change / price_data['pre_close']) * 100
        change_color = "🟢" if change >= 0 else "🔴"
        result += f"\n{change_color} 涨跌：{change:.2f} ({change_pct:.2f}%)"
        result += f"\n📊 今开：{price_data['open']:.2f} | 最高：{price_data['high']:.2f} | 最低：{price_data['low']:.2f}"
    
    # 新增当日交易数据
    if "volume" in price_data and price_data["volume"] > 0:
        result += f"\n📈 成交量：{price_data['volume']:,} 手 | 成交额：{price_data['turnover']:,.2f} 万元"
    if "turnover_rate" in price_data and price_data["turnover_rate"] > 0:
        result += f"\n📊 换手率：{price_data['turnover_rate']:.2f}% | 振幅：{price_data['amplitude']:.2f}%"
    if "pe" in price_data and price_data["pe"] > 0:
        result += f"\n📈 市盈率(TTM)：{price_data['pe']:.2f} | 市净率：{price_data['pb']:.2f}"

    # 更新缓存
    update_cache(stock_code, result)
    
    # 检查是否触发退出
    if rate_limit_status.get("block_exit", False):
        sys.exit(1)
        
    return f"✅ {result}"

@tool
def search_company_ticker(company_name: str) -> str:
    """查询股票代码（支持精准/模糊匹配+动态查询 + 浮动IP）"""
    # 检查退出标记
    if rate_limit_status.get("block_exit", False):
        sys.exit(1)
        
    company_name = company_name.strip()
    
    if company_name in STOCK_NAME_MAP:
        code = STOCK_NAME_MAP[company_name]
        return f"✅ {company_name} 股票代码：{code}"
    
    matches = []
    for name, code in STOCK_NAME_MAP.items():
        if company_name in name:
            matches.append(f"{name} → {code}")
    
    if matches:
        return f"✅ 模糊匹配到以下股票：\n" + "\n".join(matches)
    
    if check_rate_limit():
        return f"❌ 当前处于限流状态，无法动态查询股票代码，请稍后重试"
    
    console.print(f"[dim]🔍 正在动态查询{company_name}的股票代码...[/dim]")
    dynamic_code = get_stock_code_dynamically(company_name)
    
    # 检查是否触发退出
    if rate_limit_status.get("block_exit", False):
        sys.exit(1)
    
    if dynamic_code:
        return f"✅ 动态查询到 {company_name} 股票代码：{dynamic_code}"
    
    return f"❌ 未查询到 {company_name} 的股票代码\n提示：可手动输入代码（如600744、AAPL、0700）"

@tool
def read_local_file(file_name: str) -> str:
    """读取知识库中的文件内容"""
    target_path = KB_DIR / file_name
    if not target_path.exists():
        return f"❌ 文件不存在：{file_name}"
    
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"✅ 读取 {file_name} 成功：\n{content}"
    except Exception as e:
        return f"❌ 读取文件失败：{type(e).__name__} - {str(e)}"

@tool
def write_local_file(file_name: str, content: str) -> str:
    """写入内容到知识库文件"""
    target_path = KB_DIR / file_name
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ 文件已保存：{target_path.resolve()}"
    except Exception as e:
        return f"❌ 写入文件失败：{type(e).__name__} - {str(e)}"

@tool
def analyze_local_document(file_name: str, query: str) -> str:
    """分析本地文档并检索相关信息"""
    target_path = KB_DIR / file_name
    if not target_path.is_file() or target_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return "❌ 安全拦截：仅允许访问知识库内的指定格式文件！"
    if not target_path.exists():
        return f"❌ 找不到文件: {file_name}。请先调用 list_kb_files 查看可用文件。"
    
    current_mtime = os.path.getmtime(target_path)
    target_path_str = str(target_path)
    doc_cache_dir = FAISS_DB_DIR / f"{file_name}_vstore"
    meta_file = doc_cache_dir / "meta.json"
    
    try:
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=embedding_key,
            model="text-embedding-v3",
        )
        
        if target_path_str in FAISS_CACHE and FAISS_CACHE[target_path_str]["mtime"] == current_mtime:
            console.print(f"[bold yellow]⚡ L1 缓存命中：复用 {file_name} 向量索引[/bold yellow]")
            vectorstore = FAISS_CACHE[target_path_str]["vectorstore"]
        else:
            loaded_from_disk = False
            if doc_cache_dir.exists() and meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    if isinstance(meta, dict) and meta.get("mtime") == current_mtime:
                        console.print(f"[bold cyan]💾 L2 缓存命中：加载 {file_name} 持久化索引[/bold cyan]")
                        vectorstore = FAISS.load_local(
                            str(doc_cache_dir), 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                        FAISS_CACHE[target_path_str] = {"mtime": current_mtime, "vectorstore": vectorstore}
                        loaded_from_disk = True
                except (json.JSONDecodeError, TypeError, Exception) as e:
                    console.print(f"[bold red]读取硬盘缓存失败：{str(e)}，准备重建[/bold red]")
            
            if not loaded_from_disk:
                console.print(f"[bold blue]🔄 正在构建 {file_name} 向量索引...[/bold blue]")
                ext = target_path.suffix.lower()
                loader = PyPDFLoader(target_path_str) if ext == '.pdf' else TextLoader(target_path_str, encoding='utf-8')
                
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = [s for s in text_splitter.split_documents(docs) if s.page_content.strip()]
                
                if not splits:
                    return f"❌ 文件 {file_name} 无有效文本内容"
                
                vectorstore = FAISS.from_documents(splits, embeddings)
                FAISS_CACHE[target_path_str] = {"mtime": current_mtime, "vectorstore": vectorstore}
                
                doc_cache_dir.mkdir(parents=True, exist_ok=True)
                vectorstore.save_local(str(doc_cache_dir))
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump({"mtime": current_mtime, "file_name": file_name}, f)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        return f"✅ 从 {file_name} 检索到的核心信息：\n{context}"
    
    except json.JSONDecodeError:
        return f"❌ 文档元数据损坏：JSONDecodeError"
    except FileNotFoundError:
        return f"❌ 文档不存在：{file_name}"
    except Exception as e:
        return f"❌ 检索文档出错：{type(e).__name__} - {str(e)}"

@tool
def list_kb_files() -> str:
    """列出知识库中所有可用的文件"""
    try:
        files = [f.name for f in KB_DIR.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        return "当前知识库文件列表：\n" + "\n".join(files) if files else "知识库为空，无可用文件"
    except FileNotFoundError:
        return "❌ 知识库目录不存在"
    except PermissionError:
        return "❌ 无权限访问知识库目录"
    except Exception as e:
        return f"❌ 读取目录出错：{type(e).__name__} - {str(e)}"

@tool
def update_user_memory(key: str, value: str) -> str:
    """更新用户长期记忆"""
    try:
        if not USER_PROFILE_PATH.exists():
            USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        
        with FileLock(LOCK_PATH, timeout=5):
            with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data[key] = value
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return f"✅ 记忆已更新：[{key}] -> '{value}'"
    except json.JSONDecodeError:
        return "❌ 记忆文件损坏：JSONDecodeError"
    except TimeoutError:
        return "❌ 文件锁超时：其他进程正在写入记忆"
    except Exception as e:
        return f"❌ 记忆写入失败：{type(e).__name__} - {str(e)}"

@tool
def append_transaction_log(action: str, target: str, details: str) -> str:
    """追加交易日志记录"""
    try:
        log_path = MEMORY_DIR / "transaction_logs.jsonl"
        log_entry = json.dumps({
            "timestamp": time.time(),
            "action": action,
            "target": target,
            "details": details
        }, ensure_ascii=False)
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        return "✅ 交易日志已追加记录"
    except Exception as e:
        return f"❌ 记录日志失败：{type(e).__name__} - {str(e)}"

# ============================ 工具列表 ============================
tools = [
    get_universal_stock_price,
    search_company_ticker,
    read_local_file,
    write_local_file,
    list_kb_files,
    analyze_local_document,
    update_user_memory,
    append_transaction_log
]

# ============================ 记忆引擎 ============================
def get_session_history(session_id: str):
    """获取会话历史记录"""
    memory_file = str(MEMORY_DIR / f"{session_id}.json")
    history = FileChatMessageHistory(memory_file)
    
    if len(history.messages) > 10:
        kept_messages = history.messages[-10:]
        history.clear()
        for msg in kept_messages:
            history.add_message(msg)
    
    return history

def get_user_profile():
    """获取用户长期记忆配置"""
    if not USER_PROFILE_PATH.exists():
        return "暂无长期记忆"
    try:
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "\n".join([f"- 【{k}】: {v}" for k, v in data.items()]) if data else "暂无长期记忆"
    except:
        return "暂无长期记忆"

# ============================ LLM 配置 ============================
llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=dashscope_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    request_timeout=60,
    max_retries=1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是极客风格的量化分析师助手，核心规则：
1. 实时时间：{current_time}，需以此为基准理解用户的时间描述
2. 用户记忆：{user_profile}，分析时需结合使用
3. 工具调用规则：
   - 优先使用缓存数据，避免重复请求
   - 检测到限流状态时，立即返回限流提示，不重复调用工具
   - 查股价 → get_universal_stock_price（东方财富接口 + 浮动IP）
   - 查代码 → search_company_ticker（优先本地映射 + 浮动IP）
4. 限流处理：
   - 触发限流后，明确告知用户等待时间（5分钟）
   - 触发爬虫校验时，程序会自动退出，需重启使用
5. 记忆存储：
   - 状态/偏好 → update_user_memory（覆盖式）
   - 交易动作 → append_transaction_log（追加式）
   - 深度报告 → write_local_file（持久化）
   - 简单问答 → 不调用记忆工具"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        "user_profile": lambda x: x["user_profile"],
        "current_time": lambda x: x["current_time"],
    }
    | prompt
    | llm
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False,
    max_iterations=2,  # 减少迭代次数避免重复请求
    handle_parsing_errors="返回友好的错误提示，告知用户当前系统限流或触发爬虫校验，请稍后重试或重启程序"
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ============================ 主程序入口 ============================
if __name__ == "__main__":
    # 安装缺失依赖
    try:
        import requests
        import bs4
        import proxybroker
    except ImportError:
        console.print("[bold yellow]⚠️  检测到缺失依赖，正在安装...[/bold yellow]")
        os.system("pip install requests beautifulsoup4 proxybroker aiohttp")
        import requests
        from bs4 import BeautifulSoup
        import proxybroker
    
    # 初始化代理池（新增）
    console.print("[dim]🔄 初始化代理IP池...[/dim]")
    refresh_proxy_pool()
    
    # 创建必要目录
    for dir_path in [KB_DIR, MEMORY_DIR, FAISS_DB_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 启动信息
    console.print("\n🤖 全球股票分析 Agent 已启动（东方财富接口 + 浮动IP版）！")
    console.print("📌 防护策略：5秒请求间隔 + 爬虫校验检测 + 30分钟缓存 + 浮动IP轮换")
    console.print("📌 新增功能：支持任意股票查询（无需本地映射）+ 返回完整当日股价数据")
    console.print("📌 支持：A股/美股/港股")
    console.print("📌 输入 'quit'/'exit'/'退出' 结束对话")
    console.print("-" * 80)
    
    # 初始化交互
    session = PromptSession(history=InMemoryHistory())
    style = Style([('prompt', 'ansicyan bold')])

    while True:
        try:
            # 前置检查：如果标记退出，直接结束程序
            if rate_limit_status.get("block_exit", False):
                console.print("[bold red]🛑 检测到爬虫校验触发，程序安全退出！[/bold red]")
                sys.exit(0)
            
            user_input = session.prompt('\n你: ', style=style)
            
            # 退出指令
            if user_input.lower() in ['quit', 'exit', '退出']:
                console.print("👋 再见！祝你投资顺利。")
                break
            
            # 空输入处理
            if not user_input.strip():
                continue
            
            # 限流检查
            if check_rate_limit():
                remaining = int(rate_limit_status["retry_after"] - (time.time() - rate_limit_status["limited_time"]))
                console.print(f"\n[bold yellow]⚠️  系统当前处于限流状态[/bold yellow]")
                console.print(f"⏳ 请等待 {remaining//60}分{remaining%60}秒后再查询")
                continue
            
            # 调用Agent
            response = agent_with_chat_history.invoke(
                {
                    "input": user_input,
                    "user_profile": get_user_profile(),
                    "current_time": datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                },
                config={
                    "configurable": {"session_id": "terminal_session_01"},
                    "callbacks": [HackerMatrixCallback()]
                }
            )
            
            # 输出结果
            console.print()
            console.print(Rule("[bold cyan]SYS.RESPONSE[/bold cyan]", style="cyan"))
            console.print(Markdown(response['output']))
            console.print(Rule("[dim cyan]EOF[/dim cyan]", style="cyan"))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[操作取消，程序退出]")
            sys.exit(0)
        except EOFError:
            console.print("\n👋 再见！")
            sys.exit(0)
        except SystemExit:
            # 捕获主动退出信号
            console.print("\n[bold red]🛑 程序已安全退出[/bold red]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[bold red]❌ 程序异常： {type(e).__name__} - {str(e)}[/bold red]")
            # 如果是爬虫校验相关异常，直接退出
            if detect_anti_crawler(str(e)) or "block_exit" in str(e):
                sys.exit(1)
            continue
