import sqlite3
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


# 定义状态
class GraphState(TypedDict):
    messages: Annotated[List, add_messages]


# 初始化大模型
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key="sk-14fd29f4451449cbb013458844d60fa3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 定义节点函数
def call_llm(state: GraphState):
    """调用大模型生成回复"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def main():
    print("=== LangGraph Agent + SQLite 演示 ===\n")

    # 1. 创建SQLite连接（连接到具体的.db文件）
    conn = sqlite3.connect("D:\\develop\\database\\langgraph_checkpoint.db", check_same_thread=False)
    # 2. 创建SqliteSaver用于保存对话历史
    sqliteDB = SqliteSaver(conn)
    # 3. 构建LangGraph工作流
    builder = StateGraph(GraphState)
    # 添加节点
    builder.add_node("agent", call_llm)
    # 添加边
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    # 4. 编译图并添加checkpoint保存器
    graph = builder.compile(checkpointer=sqliteDB)
    # 5. 使用配置（thread_id用于区分不同会话）
    config = {"configurable": {"thread_id": "conversation_1"}}
    # 6. 第一次对话
    print("用户:我是小明")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="我是小明")]},
        config=config
    )
    print(f"助手: {result1['messages'][-1].content}\n")
    # 7. 继续对话（会自动加载历史记录）
    print("用户: 我是谁")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="我是谁")]},
        config=config
    )
    print(f"助手: {result2['messages'][-1].content}\n")
    # 8. 查看对话历史
    print("=== 对话历史 ===")
    for message in result2["messages"]:
        print(f"{message.type}: {message.content}")
    # 9. 关闭数据库连接
    conn.close()
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
    main()