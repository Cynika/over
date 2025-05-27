# main.py
import os
from database import init_db_from_csv
from llm_interface import LLMClient
from tools import ALL_TOOLS
from react_agent import ReActAgent

# CSV文件路径
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_call_records.csv')
TABLE_NAME = "call_records"


def main():
    # 1. 初始化数据库并从CSV加载数据
    print(f"Initializing database from {CSV_FILE_PATH}...")
    if not init_db_from_csv(CSV_FILE_PATH, TABLE_NAME):
        print("Database initialization failed. Exiting.")
        return

    # 2. 设置LLM API Key和API Base URL
    # 替换为您的SiliconFlow API Key
    os.environ["SILICONFLOW_API_KEY"] = "YOUR_SILICONFLOW_API_KEY"
    # 或者其他兼容OpenAI API的服务商URL
    os.environ["LLM_API_BASE"] = "https://api.siliconflow.cn/v1/"
    os.environ["LLM_MODEL_NAME"] = "mistralai/mixtral-8x7b-instruct-v0.1"  # 检查模型名称

    sf_api_key = os.getenv("SILICONFLOW_API_KEY")
    sf_api_base = os.getenv("LLM_API_BASE", "https://api.siliconflow.cn/v1/")  # 默认SiliconFlow
    sf_model_name = os.getenv("LLM_MODEL_NAME", "mistralai/mixtral-8x7b-instruct-v0.1")  # 默认Mixtral

    if not sf_api_key:
        print("Error: SILICONFLOW_API_KEY environment variable not set.")
        print("Please set it or replace os.getenv() with your actual key in llm_interface.py.")
        return

    # 3. 初始化LLM客户端
    try:
        llm_client = LLMClient(api_base=sf_api_base, api_key=sf_api_key, model=sf_model_name)
    except ValueError as e:
        print(f"Error initializing LLM client: {e}")
        return

    # 4. 初始化ReAct代理
    agent = ReActAgent(llm_client=llm_client, tools=ALL_TOOLS)

    # 5. 定义任务并运行代理
    tasks = [
        "分析通话记录中投诉电话的主要问题和趋势，找出其中潜在的风险点。",
        "找出通话时长超过300秒的销售咨询电话，分析这些电话中哪些是潜在的高价值客户？",
        "根据通话内容和情感，找出客户最不满意的产品或服务特征。",
        "过去一周内（指CSV数据中的时间），哪种通话类型最多？它的平均通话时长是多少？"
    ]

    for i, task in enumerate(tasks):
        print(f"\n======== Running Task {i + 1}/{len(tasks)} ========")
        final_answer = agent.run(task)
        print(f"\nTask {i + 1} Final Answer:\n{final_answer}")
        print("=" * 60)
        # 清空历史以处理下一个任务
        # agent.chat_history = [] # ReActAgent._initialize_system_message handles this now
        # agent._initialize_system_message()


if __name__ == "__main__":
    main()
