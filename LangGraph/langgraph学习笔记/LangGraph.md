# LangGraph概述

LangGraph就是用图形化来实行langchain,不是线性执行.

**基于 LangChain 构建的、面向智能体多轮交互 / 状态持久化 / 分支并行执行的图结构工作流框架**

**LangGraph = LangChain + 图编排 + 状态机**

langchain会遇到智能体乱操作,出现黑箱操作

langgraph需要人机合作工作

**人机协作(HITL)：将人类决策融入工作流关键节点，构建可信AI系统**

**多智能体协作：分层规划与共创协作两种模式，模拟现实团队工作方式**

![image-20260522160745239](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522160745239.png)

![image-20260522160758008](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522160758008.png)

*<u>**LangGraph的灵魂：**</u>*
*<u>**State(状态)、Nodes（节点)、Edges(边)、Graph(图)。***</u>

![image-20260522160828398](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260522160828398.png)

## LangGraph技术架构

![image-20260523080326891](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523080326891.png)

**打印图的ascii可视化结构**

**打印图的Mermaid代码可视化结构并通过processon编辑器查看**

**生成 PNG并写入文件**

## Hello world

```python
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
import uuid

# 1．定义State(可选)
class HelloState(TypedDict):
    name: str
    greeting: str


# 2.定义节点Node
def greet(helloState: HelloState) -> dict:
    name = helloState["name"]
    return {"greeting": f"你好,{name}"}

def add_emoji(helloState:HelloState) -> dict:
    greeting = helloState["greeting"]
    return {"greeting": greeting + "  。。。😄"}


# 3.构建图graph
graph = StateGraph(HelloState)

graph.add_node("greeting",greet)
graph.add_node("add_emoji",add_emoji)

graph.add_edge(START, "greeting")
graph.add_edge("greeting","add_emoji")
graph.add_edge("add_emoji",END)


# 4.编译图
app = graph.compile()

# 5.运行
# invoke()方法只接收状态字典作为核心参数
result = app.invoke({"name":"z3"})
print(result)
print(result["greeting"])



#
# #6 打印图的边和节点信息
# #6.1 打印图的ascii可视化结构
print(app.get_graph().print_ascii())
print("="*50)
# #
# #6.2 打印图的Mermaid代码可视化结构并通过https://www.processon.com/mermaid编辑器查看
print(app.get_graph().draw_mermaid())
print("="*50)


#
# #6.3 生成 PNG并写入文件
# png_bytes = app.get_graph().draw_mermaid_png(max_retries=2,retry_delay=2.0)
# output_path = "langgraph" + str(uuid.uuid4())[:8] + ".png"
# with open(output_path, "wb") as f:
#     f.write(png_bytes)
# print(f"图片已生成：{output_path}")

"""
上面第3种方式，容易bug,时好时坏
ValueError: Failed to reach https://mermaid.ink  API while trying to render your graph after 1 retries. 
To resolve this issue:
1. Check your internet connection and try again
2. Try with higher retry settings: `draw_mermaid_png(..., max_retries=5, retry_delay=2.0)`
3. Use the Pyppeteer rendering method which will render your graph locally in a browser: 
`draw_mermaid_png(..., draw_method=MermaidDrawMethod.PYPPETEER)`
"""
```

由State(状态)、Nodes（节点)、Edges(边)构成Graph(图)

```python
# 1．定义State(可选)
class HelloState(TypedDict):
    name: str
    greeting: str
```

第一步定义stata

```python
# 2.定义节点Node
def greet(helloState: HelloState) -> dict:
    name = helloState["name"]
    return {"greeting": f"你好,{name}"}

def add_emoji(helloState:HelloState) -> dict:
    greeting = helloState["greeting"]
    return {"greeting": greeting + "  。。。😄"}
```

定义两个节点

```python
# 3.构建图graph
graph = StateGraph(HelloState)

graph.add_node("greeting",greet)
graph.add_node("add_emoji",add_emoji)

graph.add_edge(START, "greeting")
graph.add_edge("greeting","add_emoji")
graph.add_edge("add_emoji",END)
```

用边构建图 

```python
# 4.编译图
app = graph.compile()
```

将图编译

```python
# invoke()方法只接收状态字典作为核心参数
result = app.invoke({"name":"z3"})
print(result)
print(result["greeting"])
```

运行

```python
print(app.get_graph().print_ascii())
```

1.将图打印成可视化图如下图

![image-20260523092116237](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523092116237.png)

```python
print(app.get_graph().draw_mermaid())
```

2.打印出代码格式到mermaid格式中的代码生成图![image-20260523092226931](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523092226931.png)

![image-20260523092236394](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523092236394.png)

```python
# #6.3 生成 PNG并写入文件
# png_bytes = app.get_graph().draw_mermaid_png(max_retries=2,retry_delay=2.0)
# output_path = "langgraph" + str(uuid.uuid4())[:8] + ".png"
# with open(output_path, "wb") as f:
#     f.write(png_bytes)
# print(f"图片已生成：{output_path}")
```

3.要访问国外所以时好时坏

## 连接大模型

```python
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
```

上为完整代码

tips:在节点函数中连接大模型

要注意先连接节点再加边

在定义状态中的变量名和后续变量名保持一致

