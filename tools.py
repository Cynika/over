# tools.py
from typing import Callable, Dict, Any, Type
from pydantic import BaseModel, Field, ValidationError  # 导入 ValidationError 用于可能的类型检查错误
import yaml
import os

from database import execute_sql_query, get_table_schema

# 定义一个全局变量或在类内部加载YAML
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), 'prompts.yaml')

# 提前加载提示词，以便在定义工具时使用
_prompts = {}
try:
    with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f:
        _prompts = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: Prompts file not found at {PROMPTS_FILE_PATH}. Using default descriptions.")
except yaml.YAMLError as e:
    print(f"Warning: Error parsing prompts YAML file: {e}. Using default descriptions.")


class Tool:
    """基类，定义LLM可调用的工具接口"""

    def __init__(self, name: str, description: str, func: Callable, args_schema: Type[BaseModel] = None):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def run(self, **kwargs) -> Any:
        """执行工具函数，并尝试用args_schema验证参数"""
        if self.args_schema:
            try:
                # 使用 Pydantic v2 的模型验证
                validated_args = self.args_schema(**kwargs).model_dump()
                return self.func(**validated_args)
            except ValidationError as e:
                raise ValueError(f"Tool '{self.name}' argument validation failed: {e}")
        else:
            return self.func(**kwargs)

    def to_openai_function_format(self) -> Dict[str, Any]:
        """将工具转换为OpenAI Function Calling所需的格式 (使用 Pydantic v2 的 model_json_schema())"""
        function_def = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
        if self.args_schema:
            # Pydantic v2: 使用 model_json_schema()
            schema = self.args_schema.model_json_schema()
            function_def["parameters"]["properties"] = schema.get("properties", {})
            function_def["parameters"]["required"] = schema.get("required", [])
        return function_def


# 定义SQL查询工具的参数 (Pydantic v2)
class SQLQueryParams(BaseModel):
    # Field() 在 Pydantic v2 中仍然可以使用，但通常推荐直接类型注解
    # query: str = Field(description="The SQL query to execute...")
    query: str = Field(
        description="The SQL query to execute against the call records database. Ensure the query is valid SQLite syntax. The table name is 'call_records'.")


# 实例化SQL查询工具
sql_query_tool = Tool(
    name="sql_query",
    description=_prompts.get('sql_query_tool_description', "Executes a SQL query on the 'call_records' database."),
    # 从YAML获取描述
    func=execute_sql_query,
    args_schema=SQLQueryParams
)


# 定义获取表结构工具的参数 (Pydantic v2)
class DescribeTableParams(BaseModel):
    table_name: str = Field(description="The name of the table to describe. Only 'call_records' is available.")


# 实例化获取表结构工具
describe_table_tool = Tool(
    name="describe_table",
    description=_prompts.get('describe_table_tool_description', "Returns the schema of a specified table."),
    # 从YAML获取描述
    func=get_table_schema,
    args_schema=DescribeTableParams
)

ALL_TOOLS = {
    sql_query_tool.name: sql_query_tool,
    describe_table_tool.name: describe_table_tool,
}
