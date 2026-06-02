from typing import TypedDict, Annotated, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model


#state定义状态
class GraphLLM_state(TypedDict):
    messages:Annotated[List,add_messages]
    # messages 是一个消息列表，Annotated + add_messages 表示支持自动追加消息
model= init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key="sk-14fd29f4451449cbb013458844d60fa3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#节点函数
def model_note(graph:GraphLLM_state):
    reply=model.invoke(graph["messages"])
    return {"messages":[reply]}
graph= StateGraph(GraphLLM_state)
graph.add_node("model",model_note)
graph.add_edge(START,"model")
graph.add_edge("model",END)
app= graph.compile()


#result= app.invoke({"messages":"请用一句话解释什么是 LangGraph。"})
result = app.invoke({"messages": [HumanMessage(content="请用一句话解释什么是 LangGraph。")]})
print( result["messages"][-1].content)

print(app.get_graph().print_ascii())
print("="*50)

#2. 打印图的Mermaid代码可视化结构并通过https://www.processon.com/mermaid编辑器查看
print(app.get_graph().draw_mermaid())



