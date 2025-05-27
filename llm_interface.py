# llm_interface.py
import os
import requests
import json
from typing import List, Dict, Any, Optional


class LLMClient:
    def __init__(self, api_base: str, api_key: Optional[str] = None,
                 model: str = "mistralai/mixtral-8x7b-instruct-v0.1"):
        """
        :param api_base: API请求的基础URL，例如 "https://api.siliconflow.cn/v1/"
        :param api_key: API Key
        :param model: 要使用的模型名称，例如 "mistralai/mixtral-8x7b-instruct-v0.1"
                      请根据SiliconFlow或其他服务提供的模型名称调整。
        """
        self.api_base = api_base
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")  # 假设环境变量为 SILICONFLOW_API_KEY
        self.model = model

        if not self.api_key:
            raise ValueError("API key not provided and SILICONFLOW_API_KEY environment variable not set.")

    def call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        调用LLM并处理其响应，包括潜在的工具调用。
        使用requests库。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"  # 告诉模型可以自动选择工具

        url = f"{self.api_base}/chat/completions"  # 兼容OpenAI API规范

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            choice = response_data["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                tool_calls_info = []
                for tc in message["tool_calls"]:
                    tool_calls_info.append({
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]  # This is a JSON string
                    })
                return {"type": "tool_calls", "tool_calls": tool_calls_info}
            elif message.get("content"):
                return {"type": "text", "content": message["content"]}
            else:
                return {"type": "empty", "content": ""}

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - {response.text}")
            return {"type": "error", "content": f"HTTP Error: {str(http_err)} - {response.text}"}
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            return {"type": "error", "content": f"Connection Error: {str(conn_err)}"}
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            return {"type": "error", "content": f"Timeout Error: {str(timeout_err)}"}
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")
            return {"type": "error", "content": f"Request Error: {str(req_err)}"}
        except json.JSONDecodeError as json_err:
            print(f"JSON decoding error: {json_err} - Response: {response.text}")
            return {"type": "error", "content": f"JSON Decode Error: {str(json_err)}"}
        except Exception as e:
            print(f"An unknown error occurred: {e}")
            return {"type": "error", "content": f"Unknown Error: {str(e)}"}
