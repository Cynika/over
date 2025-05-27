# react_agent.py
import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml
import os

from tools import ALL_TOOLS, Tool  # 我们需要保留ALL_TOOLS来实际获取工具实例
from llm_interface import LLMClient

# 定义一个全局变量或在类内部加载YAML
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'prompts.yaml')


class ReActAgent:
    def __init__(self, llm_client: LLMClient, tools: Dict[str, Tool]):
        self.llm_client = llm_client
        self.tools = tools
        self.chat_history: List[Dict[str, Any]] = []
        self._prompts = self._load_prompts()  # 加载提示词

    def _load_prompts(self) -> Dict[str, str]:
        """从YAML文件加载提示词模板"""
        try:
            with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found at {PROMPTS_FILE_PATH}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing prompts YAML file: {e}")

    def _initialize_system_message(self, initial_db_info: Optional[str] = None):
        """
        初始化系统消息，定义代理的行为、可用的工具及其格式。
        可以传入初始的数据库信息，如表名和列名。
        """
        tool_descriptions = "\n".join([
            f"  - {tool.name}: {tool.description}" for tool in self.tools.values()
        ])

        # 使用加载的模板
        system_message = self._prompts['system_message_template'].format(
            tool_descriptions=tool_descriptions,
            initial_db_info=initial_db_info if initial_db_info else ""
        )

        self.chat_history = [{"role": "system", "content": system_message.strip()}]  # .strip()去除末尾多余换行

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """
        解析LLM的文本响应，识别Thought, Action, Final Answer。
        """
        content_lower = response_content.strip().lower()

        if content_lower.startswith("thought:"):
            return {"type": "thought", "content": response_content[len("thought:"):].strip()}
        # For function calling, we rely on LLMClient to return tool_calls, not parse raw text
        # This part is mostly for legacy/fallback if LLM doesn't use tool_calls mechanism correctly
        elif content_lower.startswith("action:"):
            match = re.match(r"Action:\s*(\w+)\((.*)\)", response_content, re.IGNORECASE)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                args = {}
                # This part is to handle cases where LLM *generates* tool-like text, not actual function calls.
                # If using proper Function Calling, this block might be less critical.
                try:
                    # Attempt to parse as JSON first
                    args = json.loads("{" + args_str + "}")
                except json.JSONDecodeError:
                    # Fallback to key=value parsing
                    for arg_pair in args_str.split(','):
                        if '=' in arg_pair:
                            key, value = arg_pair.split('=', 1)
                            try:
                                args[key.strip()] = json.loads(value.strip())
                            except json.JSONDecodeError:
                                args[key.strip()] = value.strip().strip("'\"")
                return {"type": "action", "tool_name": tool_name, "tool_args": args}
            else:
                return {"type": "error", "content": f"Invalid Action format: {response_content}"}
        elif content_lower.startswith("final answer:"):
            return {"type": "final_answer", "content": response_content[len("final answer:"):].strip()}
        else:
            return {"type": "text", "content": response_content.strip()}

    def run(self, task: str, max_steps: int = 10) -> str:
        """
        运行ReAct代理以完成给定任务。
        在开始前，代理会尝试获取表结构信息。
        """
        print("\n--- Initializing Agent and Discovering DB Schema ---")

        initial_user_query_for_schema = self._prompts['initial_user_query_for_schema']

        schema_tool = self.tools.get("describe_table")
        initial_db_info = ""
        if schema_tool:
            print(
                f"Action (Agent Init): describe_table(table_name='call_records') (using prompt: {initial_user_query_for_schema})")
            try:  # Add try-except for tool execution during init
                schema_df = schema_tool.run(table_name="call_records")
                if isinstance(schema_df, pd.DataFrame) and not schema_df.empty:
                    initial_db_info = f"Table 'call_records' schema:\n{schema_df.to_markdown(index=False)}"
                    print(f"Observation (Agent Init): \n{initial_db_info}")
                else:
                    initial_db_info = f"Could not retrieve schema for 'call_records': {schema_df}"
                    print(f"Observation (Agent Init): \n{initial_db_info}")
            except Exception as e:
                initial_db_info = f"Error executing describe_table during init: {e}"
                print(f"Observation (Agent Init): \n{initial_db_info}")
        else:
            print("Warning: describe_table tool not found. LLM might struggle with schema.")
            initial_db_info = "Table 'call_records' is the primary table. Its schema is unknown; LLM should use `describe_table` tool to find it."

        self._initialize_system_message(initial_db_info=initial_db_info)
        self.chat_history.append({"role": "user", "content": task})
        print(f"\n--- Starting ReAct Agent for task: '{task}' ---")

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")

            llm_response = self.llm_client.call_llm(
                messages=self.chat_history,
                tools=[tool.to_openai_function_format() for tool in self.tools.values()]
            )

            if llm_response["type"] == "tool_calls":
                tool_calls = llm_response["tool_calls"]
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args_str = tool_call["arguments"]  # This is already a JSON string from LLM

                    # Ensure the tool exists before attempting to parse args and run
                    if tool_name not in self.tools:
                        observation = f"Error: Tool '{tool_name}' not found."
                        print(f"Observation: {observation}")
                        self.chat_history.append({"role": "tool", "tool_call_id": "mock_id", "content": observation})
                        continue

                    try:
                        tool_args = json.loads(tool_args_str)  # Directly parse the JSON string
                    except json.JSONDecodeError as e:
                        observation = f"Error parsing tool arguments for {tool_name}: {e}. Raw arguments: {tool_args_str}"
                        print(f"Observation: {observation}")
                        self.chat_history.append({"role": "tool", "tool_call_id": "mock_id", "content": observation})
                        continue

                    print(f"Action: {tool_name}({tool_args})")
                    self.chat_history.append({
                        "role": "assistant",
                        "tool_calls": [{"id": "mock_id", "function": {"name": tool_name, "arguments": tool_args_str}}]
                    })

                    try:
                        observation_result = self.tools[tool_name].run(
                            **tool_args)  # `run` now handles Pydantic validation
                        if isinstance(observation_result, pd.DataFrame):
                            observation_str = observation_result.to_markdown(index=False)
                            if len(observation_str) > 2000:
                                observation_str = observation_result.head(5).to_markdown(
                                    index=False) + "\n... (results truncated due to length, showing first 5 rows)"
                        else:
                            observation_str = str(observation_result)

                    except Exception as e:  # Catch exceptions from tool.run()
                        observation_str = f"Tool execution failed: {e}"

                    print(f"Observation: {observation_str}")
                    self.chat_history.append({"role": "tool", "tool_call_id": "mock_id", "content": observation_str})

            elif llm_response["type"] == "text":
                parsed_response = self._parse_llm_response(llm_response["content"])

                if parsed_response["type"] == "thought":
                    print(f"Thought: {parsed_response['content']}")
                    self.chat_history.append({"role": "assistant", "content": llm_response["content"]})
                elif parsed_response["type"] == "final_answer":
                    print(f"Final Answer: {parsed_response['content']}")
                    self.chat_history.append({"role": "assistant", "content": llm_response["content"]})
                    return parsed_response["content"]
                else:
                    print(f"LLM said (unparsed): {llm_response['content']}")
                    self.chat_history.append({"role": "assistant", "content": llm_response["content"]})
                    print("Agent could not parse LLM response for ReAct format. Trying to continue or stopping.")
            elif llm_response["type"] == "error":
                print(f"LLM communication error: {llm_response['content']}")
                return f"Agent terminated due to LLM communication error: {llm_response['content']}"
            else:
                print(f"Unexpected LLM response type: {llm_response['type']}")
                return "Agent terminated due to unexpected LLM response type."

        print("\n--- Max steps reached. ---")
        return "Max steps reached without providing a final answer."
