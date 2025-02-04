import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from langchain.schema import SystemMessage
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import yfinance as yf

# Load environment variables
load_dotenv()

def get_openrouter_client():
    """Configure and return the OpenRouter chat model."""
    return ChatOpenAI(
        model_name="openai/gpt-4o-2024-11-20",
        temperature=0.5,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

def get_stock_data(symbol: str):
    """Fetch stock market data for a given symbol."""
    try:
        stock = yf.Ticker(symbol)
        price_info = stock.history(period='1d')
        current_price = price_info['Close'].iloc[-1] if not price_info.empty else None
        info = stock.info
        return {
            "price": current_price,
            "volume": info.get("regularMarketVolume"),
            "pe_ratio": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

def get_news(query: str):
    """Fetch recent news articles about a company using SerpAPI."""
    try:
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        params = {
            "engine": "google_news",
            "q": query,
            "gl": "us",
            "hl": "en",
            "api_key": serpapi_key,
            "num": 3
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        news_results = results.get("news_results", [])
        
        if not news_results:
            print("⚠️ No news found for query:", query)
            return {"error": "No news found"}
        
        formatted_news = []
        for article in news_results:
            formatted_news.append({
                "title": article.get("title", "No Title"),
                "date": article.get("date", "Unknown Date"),
                "link": article.get("link", "No Link"),
                "source": article.get("source", "Unknown Source"),
                "summary": article.get("snippet", "No Summary Available")
            })
        
        print("📰 News Articles Found:", formatted_news)
        return formatted_news
    
    except Exception as e:
        print(f"❌ Error fetching news: {str(e)}")
        return {"error": str(e)}

# Define tools
stock_tool = Tool(
    name="Stock Market Data",
    func=lambda x: get_stock_data(x),
    description="Fetch financial data for a given stock symbol."
)
news_tool = Tool(
    name="News Fetcher",
    func=lambda x: get_news(x),
    description="Fetch recent news articles for a given query."
)

# Define agents
financial_analyst_agent = initialize_agent(
    tools=[stock_tool],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一名金融分析师。
        你的任务是使用股票数据工具分析市场数据，并将结果移交给规划协调员。
        """
    ),
)

news_analyst_agent = initialize_agent(
    tools=[news_tool],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一名新闻分析师。
        你的任务是使用新闻工具获取相关新闻，并将结果移交给规划协调员。
        """
    ),
)

writer_agent = initialize_agent(
    tools=[],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一名财经报告撰写员。
        你的任务是基于市场数据和新闻分析，撰写最终的财经报告，并移交给规划协调员。
        """
    ),
)

planner_agent = initialize_agent(
    tools=[],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一名研究规划协调员。
        你的任务是协调金融分析师、新闻分析师和撰写员的工作。
        先请求金融分析师获取市场数据，然后请求新闻分析师收集相关新闻，
        最后将所有信息移交给撰写员生成最终的财经报告。
        """
    ),
)

def main():
    """Execute the market research task."""
    task = "为特斯拉(TSLA)股票进行市场研究，并用中文回答"
    stock_data = financial_analyst_agent.invoke({"input": task, "chat_history": []})
    news_data = news_analyst_agent.invoke({"input": task, "chat_history": []})
    report = writer_agent.invoke({"input": f"Stock Data: {stock_data}, News Data: {news_data}", "chat_history": []})
    
    print("📊 Market Research Report:\n", report)

if __name__ == "__main__":
    main()
