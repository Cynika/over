# database.py
import sqlite3
import pandas as pd
import os

DATABASE_NAME = "call_records.db"  # 使用文件数据库，方便持久化和查看


def init_db_from_csv(csv_path: str, table_name: str = "call_records"):
    """
    从CSV文件读取数据并初始化/填充SQLite数据库表。
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    try:
        df = pd.read_csv(csv_path)

        # 简单的数据清洗/类型转换，确保兼容SQLite
        if 'call_time' in df.columns:
            df['call_time'] = pd.to_datetime(df['call_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # 检查表是否存在，如果存在则删除，以确保每次运行都从CSV加载最新数据
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()

        # 将DataFrame写入SQLite，如果表不存在则创建
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Successfully loaded data from {csv_path} into table '{table_name}' in {DATABASE_NAME}.")
        print(f"Total rows in '{table_name}': {len(df)}")
        return True
    except Exception as e:
        print(f"Error loading CSV to DB: {e}")
        return False
    finally:
        conn.close()


def execute_sql_query(query: str) -> pd.DataFrame:
    """
    执行SQL查询并返回结果。
    在实际百万级数据中，此函数应限制结果行数，或返回聚合/抽样数据。
    """
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    finally:
        conn.close()


def get_table_schema(table_name: str) -> pd.DataFrame:
    """
    获取指定表的Schema信息。
    """
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        # SQLite的PRAGMA table_info可以获取表的结构
        schema_df = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        # 选取需要的列，并改名方便LLM理解
        if not schema_df.empty:
            schema_df = schema_df[['name', 'type', 'notnull', 'pk']]
            schema_df.columns = ['Column Name', 'Data Type', 'Not Null', 'Primary Key']
        return schema_df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    finally:
        conn.close()


if __name__ == "__main__":
    CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_call_records.csv')
    init_db_from_csv(CSV_FILE_PATH)

    print("\nExample Query Results:")
    df = execute_sql_query("SELECT call_type, COUNT(*) as count FROM call_records GROUP BY call_type;")
    print(df)

    print("\nTable Schema:")
    schema = get_table_schema("call_records")
    print(schema)
