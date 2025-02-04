import os
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_openrouter_client():
    """Configure and return the OpenRouter chat model."""
    return ChatOpenAI(
        model_name="openai/gpt-4o-2024-11-20",
        temperature=0.5,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

# Define memory for agents
programmer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
reviewer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the Programmer Agent
programmer_agent = initialize_agent(
    tools=[],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=programmer_memory,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一个专业的Python开发工程师。
        请基于需求编写清晰、可维护且符合PEP8规范的Python代码。
        代码应包含:
        - 清晰的注释和文档字符串
        - 适当的错误处理
        - 代码性能优化
        - 单元测试
        """
    ),
)

# Define the Code Reviewer Agent
reviewer_agent = initialize_agent(
    tools=[],
    llm=get_openrouter_client(),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=reviewer_memory,
    verbose=True,
    system_message=SystemMessage(
        content="""
        你是一位资深的代码审查专家。请对代码进行全面的评审,包括:
        - 代码规范性和可读性
        - 设计模式的使用
        - 性能和效率
        - 安全性考虑
        - 测试覆盖率
        - 潜在问题
        当代码符合要求时,回复'同意通过'。
        """
    ),
)

def execute_task(task):
    """Executes the task between Programmer and Reviewer without an explicit loop."""
    print("Starting code review process...\n")
    
    response = programmer_agent.run(task)
    print("🛠️ Programmer's response:\n", response)
    
    review_response = reviewer_agent.run(response)
    print("🔍 Reviewer's feedback:\n", review_response)
    
    if "同意通过" in review_response:
        print("✅ Code review approved! Task completed.")
    else:
        print("❌ Code needs improvement. Please revise the implementation.")

if __name__ == "__main__":
    task_description = """
    请实现一个文件处理类 FileProcessor,要求:
    1. 支持读取、写入和追加文本文件
    2. 包含基本的文件统计功能(行数、字符数、单词数)
    3. 支持文件加密/解密功能
    4. 实现异常处理
    5. 编写完整的单元测试
    """
    execute_task(task_description)
