# Graph API

Graph API分为

1.Graph API之Graph(图)

2.Graph API之State(状态)

3.Graph API之Node(节点)

4.Graph API之Edge(边)

5.Graph API之Send/Command/Runtime context

## Graph API之Graph(图)

是一种通过状态,节点,边相连创造出的

![image-20260523104809194](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523104809194.png)

流程:

1.初始化一个StateGraph实例

2.加节点

3.定义边，将所有的节点连接起来

4.设置特殊节点，入口和出口（可选）

5.编译图

6.执行工作流

## Graph API之State(状态)

状态是一直流动的从第一个节点到最后一个节点

**在LangGraph中，State是一个贯穿整个工作流执行过程中的共享数据的结构，代表当前快照，它存储了从工作流开始到结束的所有必要的信息（历史对话、检索到的文档、工具执行结果等），在各个节点中共享，且每个节点都可以修改。**

**状态包含两部分：**
**一是图的模式（schema），**
**二是“规约函数”（reducer functions）；后者指明如何把更新应用到状态上。**



### 1.图的模式（schema）

![image-20260523105314971](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523105314971.png)

(1)state_schema:

图的完整内部状态，包含了所有节点可能读写的字段，必须指定，不能为空

是图的"全局状态空间"

所有节点都可以访问和写入这个 schema 中的任何字段

```python
class OverallState(InputState, OutputState):
    pass
```

(2)input_schema

定义图接受什么输入，是 state_schema 的子集

可选参数，如果不指定，默认等于 state_schema

限制图的输入接口，只能传入这些字段

```python
class InputState(TypedDict):
    question: str
```

(3)output_schema

定义图返回什么输出，是 state_schema 的子集

可选参数，如果不指定，默认等于 state_schema

限制图的输出接口，只返回这些字段

```python
class OutputState(TypedDict):
    answer: str
```

------

State可以是TypedDict类型，
也可以是pydantic中的BaseModel类型

![image-20260523105613071](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523105613071.png)

'一句话选型
想要 轻量、无运行时开销、习惯字典写法 → 用 TypedDict
想要 自动校验、默认值、嵌套结构、字段描述 → 用 pydantic.BaseModel
两种写法在 LangGraph 里都能一键编译，只需按上例规则声明字段即可''

```python
def answer_node(state: InputState):
    """
    处理输入并生成答案的节点
    Args:
        state: 输入状态
    Returns:
        dict: 包含答案的字典
    """
    print(f"执行 answer_node 节点:")
    print(f"  输入: {state}")

    # 示例答案
    answer = "再见" if "bye" in state["question"].lower() else "你好"
    result = {"answer": answer, "question": state["question"]}

    print(f"  输出: {result}")
    return result
```

```python
def demo_input_output_schema():
    """演示输入输出模式"""
    print("=== 演示输入输出模式 ===")

    # 使用指定的输入和输出模式构建图
    builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    builder.add_edge(START, "answer_node")  # 定义起始边
    builder.add_node("answer_node", answer_node)  # 添加答案节点
    builder.add_edge("answer_node", END)  # 定义结束边
    graph = builder.compile()  # 编译图

    # 使用输入调用图并打印结果
    result = graph.invoke({"question": "你好"})
    print(f"图调用结果: {result}")
    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
    print()
```

类似上述代码

### 规约函数”（reducer functions）重要!!!

决定对state的数据的操作方法

**“字段级合并策略”，它让节点只需吐出“增量”，框架负责按规则把增量焊进全局 State**

------

**Reducer函数在LangGraph中的作用：**
**Ø 控制状态更新方式：决定新值如何与现有值合并。**
**Ø 处理并行更新：当多个节点同时更新同一字段时，确保数据一致性。**
**Ø 提供灵活性：支持不同的合并策略，如覆盖、追加、相加等。**
**Ø 增强表达力：允许开发者根据业务需求自定义合并逻辑。**
**通过合理使用Reducer函数，可以构建更强大和灵活的状态管理机制，特别是在处理复杂工作流和并行执行场景时。**

------

#### default：未指定Reducer时使用覆盖更新

在不指定时都使用覆盖更新

```python
class DefaultReducerState(TypedDict):
    foo: int
    bar: List[str]

def node_default_1(state: DefaultReducerState) -> dict:
    print(state["foo"])
    print(state["bar"])
    return {"foo": 22}

def node_default_2(state: DefaultReducerState) -> dict:
    print()
    print(state["foo"])
    print(state["bar"])
    return {"bar": ["bye1","bye2","bye3"]}


def main():
    print("1. 默认Reducer（覆盖更新）演示:\n")
    builder = StateGraph(DefaultReducerState)

    builder.add_node("node1", node_default_1)
    builder.add_node("node2", node_default_2)

    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)

    graph = builder.compile()

    result = graph.invoke(input={"foo": 1, "bar": ["hi"]})
    #print(f"初始状态: {{'foo': 1, 'bar': ['hi']}}")
    print(f"执行结果: {result}\n")
```

在代码

```python
class DefaultReducerState(TypedDict):
    foo: int
    bar: List[str]
```

中没有规定就是默认覆盖,所以在最后的运行结果中运行结果是

执行结果: {'foo': 22, 'bar': ['bye1', 'bye2', 'bye3']}

#### add_messages：用于消息列表追加

大多用于消息传递如之前连接大模型例子中规定的add_messages一样

```python
class AddMessagesState(TypedDict):
	messages: Annotated[List, add_messages]
```

如上述代码

```python
def chat_node_1(state: AddMessagesState) -> dict:
    return {"messages": [("assistant", "Hello from node 1")]}

def chat_node_2(state: AddMessagesState) -> dict:
    return {"messages": [("assistant", "Hello from node 2")]}
```

两个节点方法

```python
def run_demo():
    print("2. add_messages Reducer（消息列表专用）演示:")
    builder = StateGraph(AddMessagesState)
    builder.add_node("chat1", chat_node_1)
    builder.add_node("chat2", chat_node_2)

    builder.add_edge(START, "chat1")
    builder.add_edge(START, "chat2")  # 并行执行
    builder.add_edge("chat1", END)
    builder.add_edge("chat2", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [("user", "Hi there!")]})
    print(f"初始状态: {{'messages': [('user', 'Hi there!')]}}")
    print(f"执行结果: {result}\n")

    print("*" * 60)

    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
```

运行方法,注意这两个节点是并行的所以打印出的可视化结构也是并行的

![image-20260523141708614](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523141708614.png)

执行结果也是在后面追加

![image-20260523141827199](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523141827199.png)

#### operator.add

**将元素追加到现有元素中，**
**支持列表、字符串、数值类型的追加**

有列表追加,字符串连接,数值累加 这里只以列表追加为例

```python
class ListAddState(TypedDict):
	 data: Annotated[List[int], operator.add]
```

定义状态

```python
def producer_1(state: ListAddState) -> dict:
    return {"data": [1, 2]}


def producer_2(state: ListAddState) -> dict:
    return {"data": [3, 4]}
```

节点方法 

```python
def run_demo():
    builder = StateGraph(ListAddState)
    # 注册节点
    builder.add_node("producer1", producer_1)
    builder.add_node("producer2", producer_2)
    # 顺序执行边
    builder.add_edge(START, "producer1")
    builder.add_edge("producer1", "producer2")
    builder.add_edge("producer2", END)

    graph = builder.compile()
    result = graph.invoke({"data": [0]})
    print(f"初始状态: {{'data': [0]}}")
    print(f"执行结果: {result}\n")
```

初始状态: {'data': [0]}
执行结果: {'data': [0, 1, 2, 3, 4]}

#### operator.mul：用于数值相乘

注意会有bug,langgraph框架中会有一个初始值,加法的话就是0+咱们认为的初始值再继续相加不会有什么变化无伤大雅,乘法的话会直接相乘0,所以乘法我们推荐使用自定义规约函数

乘法使用类似add的使用不再赘述

#### 自定义Reducer：支持用户自定义合并逻辑

```python
def MyOperatorMul(current: float, update: float) -> float:
    """自定义乘法reducer，处理初始值为1.0"""
    # 如果是第一次调用，current会是默认值0.0
    if current == 0.0:
        # 对于乘法，恒等元应该是1.0或者 return 1.0 * update
        print(f"current:{current}")
        print(f"update:{update}")
        return 1.0 * update
    return current * update
class MultiplyState(TypedDict):
    factor: Annotated[float, MyOperatorMul]
```

函数MyOperatorMul是自定义函数

在后面factor: Annotated[float, MyOperatorMul]是使用自定义规约函数来使用的

#### 案例

```python
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史
    tags: Annotated[List[str], operator.add]  # 标签列表
    score: Annotated[float, operator.add]     # 累计分数

def process_user_message(state: ChatState) -> dict:
    user_message = state["messages"][-1]  # 获取最新消息
    # 修复：正确访问消息内容
    return {
        "messages": [("assistant", f"Echo: {user_message.content}")],
        "tags": ["processed"],
        "score": 1.0
    }

def add_sentiment_tag(state: ChatState) -> dict:
    return {
        "tags": ["positive"],
        "score": 0.5
    }

# 构建图
builder = StateGraph(ChatState)
builder.add_node("process", process_user_message)
builder.add_node("sentiment", add_sentiment_tag)

builder.add_edge(START, "process")
builder.add_edge(START, "sentiment")
builder.add_edge("process", END)
builder.add_edge("sentiment", END)

graph = builder.compile()

# 使用示例 -使用正确的消息格式
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "tags": ["greeting"],
    "score": 0.0
})

print(result)
```

以上是完整代码

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史
    tags: Annotated[List[str], operator.add]  # 标签列表
    score: Annotated[float, operator.add]     # 累计
```

消息是追加规约函数,标签是加拼接,分数是加载一块

```python
def process_user_message(state: ChatState) -> dict:
    user_message = state["messages"][-1]  # 获取最新消息
    # 修复：正确访问消息内容
    return {
        "messages": [("assistant", f"Echo: {user_message.content}")],
        "tags": ["processed"],
        "score": 1.0
    }
```

user_message是消息列表最后一条消息改成"assistant", f"Echo: {user_message.content}追加上去

tags": ["processed"]在列表添加上去

"score": 1.0分数相加

```python
def add_sentiment_tag(state: ChatState) -> dict:
    return {
        "tags": ["positive"],
        "score": 0.5
    }
```

在"tags": ["positive"]列表加上"positive",分数加上0.5

```python
builder = StateGraph(ChatState)
builder.add_node("process", process_user_message)
builder.add_node("sentiment", add_sentiment_tag)

builder.add_edge(START, "process")
builder.add_edge(START, "sentiment")
builder.add_edge("process", END)
builder.add_edge("sentiment", END)

graph = builder.compile()
```

这个图是并行的,最后合并

```python
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "tags": ["greeting"],
    "score": 0.0
})
```

运行结果

{'messages': [HumanMessage(content='Hello, how are you?', additional_kwargs={}, response_metadata={}, id='b996fa2d-9320-45ad-804c-082841ba2ec5'), AIMessage(content='Echo: Hello, how are you?', additional_kwargs={}, response_metadata={}, id='d631e1d8-dbca-49d8-8aca-3576eeb4b369', tool_calls=[], invalid_tool_calls=[])], 'tags': ['greeting', 'processed', 'positive'], 'score': 1.5}

分数是0+1+0.5=1.5,tags': ['greeting', 'processed', 'positive']

消息AIMessage(content='Echo: Hello, how are you?', 

## Graph API之Node(节点)

相当于一个中间处理站,在这里通过一些操作来改变state

**在LangGraph中，节点(Node)就是是Python函数（可以是同步的，也可以是异步的），它们接受以下参数：**
**Ø state：图的状态**
**Ø config：一个RunnableConfig对象，包含诸如thread_id之类的配置信息以及诸如tags之类的跟踪信息**
**Ø runtime：一个Runtime对象，包含运行时context以及其他信息，如store和stream_writer**
**定义好node函数后，使用add_node方法将这些节点添加到图中。如果在向图中添加节点时未指定名称，系统会为其分配一个与函数名相同的默认名称。**

------

**Node是LangGraph中的一个基本处理单元，代表工作流中的一个操作步骤，**
**可以是一个Agent、调用大模型、工具或一个函数（说白了就是绑定一个python函数，具体逻辑可以干任何事情）**

------

![image-20260523150634488](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523150634488.png)

### Node的设计原则

单一职责原则：每个节点应该只负责一项职责，避免功能过于复杂

无状态设计：节点本身不应该保存状态，所有数据都通过输入状态传递

幂等性：相同的输入应该产生相同的输出，确保可重试性

可测试性：节点逻辑应该易于单元测试

### 特殊的节点 __START__ （开始节点）和 __END__（结束节点）

__START__节点：开始节点，确定应该首先调用哪些节点

也可以通过graph.set_entry_point("node_start") 函数设置起始节点，等价于graph.add_edge(START, "node_start")

__END__节点：终止节点，表示后续没有其他节点可以继续执行了

也可以通过graph.set_finish_point("node_end") 函数设置结束节点，
等价于graph.add_edge("node_start"， END)

### 节点缓存Node Caching

就是为了可以快速加载已经查找过的内容

**LangGraph支持基于节点输入对任务/节点进行缓存。使用缓存的方法如下：**
**编译图（或指定入口点）时指定缓存。**
**为节点指定缓存策略。每个缓存策略支持：**
**Ø key_func用于根据节点的输入生成缓存键。**
**Ø ttl，即缓存的生存时间（以秒为单位）。如果未指定，缓存将永不过期。**

------

**缓存键与命中：**
**当一个节点开始执行时，系统会使用其配置的 key_func 根据当前节点的输入数据生成一个唯一的键。LangGraph 会检查缓存中是否存在这个键。**

**如果存在（缓存命中），则直接返回之前存储的结果，跳过该节点的实际执行。**
**如果不存在（缓存未命中），则正常执行节点函数，并将结果与缓存键关联后存入缓存。**

**缓存有效期：**
**ttl 参数能控制缓存的有效期。**
**例如，对于依赖实时数据的天气查询节点，可以设置较短的 ttl（如60秒）。而对于处理静态信息或变化不频繁数据的节点，则可以设置较长的 ttl甚至不设置（None），让缓存永久有效，直到手动清除**

------

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


# 定义状态类，也就是你的业务实体entity
class State(TypedDict):
    x: int
    result: int

# 创建图
builder = StateGraph(State)

# 定义节点：模拟耗时计算（sleep3秒）
def expensive_node(state: State) -> dict[str, int]:
    time.sleep(3)
    return {"result": state["x"] * 2}

#     builder.add_node("node1", node_default_1)

# 添加节点
builder.add_node(node="expensive_node",action=expensive_node,
    # 不用传key_fn，底层自动用默认逻辑
    cache_policy=CachePolicy(ttl=8)
)

# 设置入口和出口
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")

# 编译图，指定内存缓存
app = builder.compile(cache=InMemoryCache())

# 第一次执行：耗时3秒（无缓存）
print("第一次执行（无缓存，耗时3秒）：")
print(app.invoke({"x": 5}))
# 第二次执行：瞬间返回（利用缓存，8秒内有效）
print("\n1111111111111111111111111111")
print("第二次运行利用缓存并快速返回：")
print(app.invoke({"x": 5}))

# 可选：测试8秒后缓存过期（取消注释查看）
print("\n等待8秒，缓存过期...")
time.sleep(8)
print("8秒后第三次执行（重新计算，耗时3秒）：")
print(app.invoke({"x": 5}))
```

以上为示例代码

```python
def expensive_node(state: State) -> dict[str, int]:
    time.sleep(3)
    return {"result": state["x"] * 2}
```

模拟大模型思考时间

```python
builder.add_node(node="expensive_node",action=expensive_node,
    # 不用传key_fn，底层自动用默认逻辑
    cache_policy=CachePolicy(ttl=8)
)
```

缓存时间是8秒

```python
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")
```

设置开始节点和结束节点

```python
app = builder.compile(cache=InMemoryCache())
```

存储短期的记忆,缓存

```python
# 第一次执行：耗时3秒（无缓存）
print("第一次执行（无缓存，耗时3秒）：")
print(app.invoke({"x": 5}))
# 第二次执行：瞬间返回（利用缓存，8秒内有效）
print("\n1111111111111111111111111111")
print("第二次运行利用缓存并快速返回：")
print(app.invoke({"x": 5}))

# 可选：测试8秒后缓存过期（取消注释查看）
print("\n等待8秒，缓存过期...")
time.sleep(8)
print("8秒后第三次执行（重新计算，耗时3秒）：")
print(app.invoke({"x": 5}))
```

输出结果

### 错误处理和重试机制

相当于java程序中的异常处理器

**LangGraph还提供了错误处理和重试机制来指定重试次数、重试间隔、重试异常等，用于保证系统的可靠性**

**为节点添加重试策略，需要在add_node中设置retry_policy参数。retry_policy参数接受一个RetryPolicy命名元组对象。默认情况下，retry_on参数使用default_retry_on函数，该函数会在遇到任何异常时重试**

![image-20260523155503803](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260523155503803.png)

------

共有三种策略

****

```python
from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# 定义状态类型
class AtguiguState(TypedDict):
    result: str

# 全局计数器：记录API尝试次数
attempt_counter = 0


# 工具函数
def build_retry_graph(node_name: str, node_func, retry_policy: RetryPolicy):
    builder = StateGraph(AtguiguState)
    #为节点添加重试策略，需要在add_node中设置retry_policy参数。
    # retry_policy参数接受一个RetryPolicy命名元组对象。
    # 默认情况下，retry_on参数使用default_retry_on函数，该函数会在遇到任何异常时重试
    builder.add_node(node_name, node_func, retry_policy=retry_policy)
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()


# 模拟不稳定的API调用，使用全局变量跟踪尝试次数
def unstable_api_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟不稳定API：前2次失败，第3次成功（全局计数器记录尝试次数）"""
    global attempt_counter
    attempt_counter += 1
    # 纯文本打印尝试次数
    print(f"尝试调用API，这是第 {attempt_counter} 次尝试")

    # 模拟失败/成功逻辑：前2次抛异常，第3次返回结果
    if attempt_counter < 3:
        raise Exception(f"模拟API调用失败abcd (尝试 {attempt_counter})")
    return {"result": f"API调用成功，经过 {attempt_counter} 次尝试"}

# 自定义重试条件判断函数
def custom_retry_on(exception: Exception) -> bool:
    """自定义重试规则：只对包含「模拟API调用失败」的异常重试"""
    print("########################:  "+str(exception))
    err_msg = str(exception)
    if "模拟API调用失败" in err_msg:
        print(f"捕获到可重试异常: {err_msg}")
        return True
    print(f"捕获到不可重试异常: {err_msg}")
    return False

# 模拟抛出 ValueError 的节点
def value_error_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟抛出ValueError：默认重试策略对这类异常不重试"""
    print("调用会抛出 ValueError 的节点")
    raise ValueError("模拟 ValueError 异常")


# 测试方法1：默认重试策略
def test_default_retry():
    global attempt_counter
    print("1. 使用默认重试策略:")
    print("   默认策略会对除特定异常外的所有异常进行重试")
    print("   不会重试的异常包括: ValueError, TypeError, ArithmeticError, ImportError,")
    print("                     LookupError, NameError, SyntaxError, RuntimeError,")
    print("                     ReferenceError, StopIteration, StopAsyncIteration, OSError\n")

    print("测试默认重试策略:")
    attempt_counter = 0  # 重置计数器
    default_graph = build_retry_graph(
        node_name="unstable_api",
        node_func=unstable_api_call,
        retry_policy=RetryPolicy(max_attempts=5)  # 最多5次尝试，足够重试成功
    )
    try:
        result = default_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 测试方法2：自定义重试策略（输出完全匹配要求）
def test_custom_retry():
    global attempt_counter
    print("2. 使用自定义重试策略:")
    print("   自定义策略只对特定错误进行重试\n")
    print("测试自定义重试策略:")
    attempt_counter = 0  # 重置计数器
    custom_graph = build_retry_graph(
        node_name="custom_retry_api",
        node_func=unstable_api_call,
        retry_policy=RetryPolicy(max_attempts=5, retry_on=custom_retry_on)
    )
    try:
        result = custom_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 测试方法3：不可重试异常演示,测试 ValueError（默认策略不会重试）
def test_no_retry_exception():
    print("3. 测试不会重试的异常类型:")
    print("测试 ValueError（默认策略不会重试）:")
    no_retry_graph = build_retry_graph(
        node_name="value_error_node",
        node_func=value_error_call,
        retry_policy=RetryPolicy(max_attempts=3)
    )
    try:
        result = no_retry_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 主演示函数
def run_demo():
    print("=== LangGraph 节点重试策略完整演示===")
    print("-" * 80 + "\n")
    #test_default_retry()
    #test_custom_retry()
    test_no_retry_exception()
    print("-" * 80)
    print("=== 演示结束 ===")


# 程序入口
if __name__ == "__main__":
    run_demo()
```

以上为案例代码

```python
def build_retry_graph(node_name: str, node_func, retry_policy: RetryPolicy):
    builder = StateGraph(AtguiguState)
    #为节点添加重试策略，需要在add_node中设置retry_policy参数。
    # retry_policy参数接受一个RetryPolicy命名元组对象。
    # 默认情况下，retry_on参数使用default_retry_on函数，该函数会在遇到任何异常时重试
    builder.add_node(node_name, node_func, retry_policy=retry_policy)
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()
```

使用上述函数可快速创建图

```python
def unstable_api_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟不稳定API：前2次失败，第3次成功（全局计数器记录尝试次数）"""
    global attempt_counter
    attempt_counter += 1
    # 纯文本打印尝试次数
    print(f"尝试调用API，这是第 {attempt_counter} 次尝试")

    # 模拟失败/成功逻辑：前2次抛异常，第3次返回结果
    if attempt_counter < 3:
        raise Exception(f"模拟API调用失败abcd (尝试 {attempt_counter})")
    return {"result": f"API调用成功，经过 {attempt_counter} 次尝试"}
```

1.默认重试策略,不断重新调用,调用三次成功

```python
# 自定义重试条件判断函数
def custom_retry_on(exception: Exception) -> bool:
    """自定义重试规则：只对包含「模拟API调用失败」的异常重试"""
    print("########################:  "+str(exception))
    err_msg = str(exception)
    if "模拟API调用失败" in err_msg:
        print(f"捕获到可重试异常: {err_msg}")
        return True
    print(f"捕获到不可重试异常: {err_msg}")
    return False
```

2.自定义重试策略,

```python
# 模拟抛出 ValueError 的节点
def value_error_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟抛出ValueError：默认重试策略对这类异常不重试"""
    print("调用会抛出 ValueError 的节点")
    raise ValueError("模拟 ValueError 异常")
```

3.不可重试异常演示,大部分是操作代码问题

## Graph API之Edge(边)

相当于一个执行顺序

**Edge定义了节点之间的连接和执行顺序，以及不同节点之间是如何通讯的，**
**一个节点可以有多个出边（指向多个节点），多个节点也可以指向同一个节点（Map-Reduce）**

------

![image-20260525084718150](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260525084718150.png)

### 条件边和普通边

#### 普通边

```python
def main():
    """演示普通边"""
    print("=== 普通边演示 ===")

    # 创建图
    builder = StateGraph(AtguiguState)

    # 添加节点
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    # 添加普通边
    builder.add_edge(START, "node_a")  # 从开始到A
    builder.add_edge("node_a", "node_b")  # 从A到B
    builder.add_edge("node_b", "node_c")  # 从B到C
    builder.add_edge("node_c", END)  # 从C到结束

    # 编译图
    app = builder.compile()

    # 执行图
    result = app.invoke({"value": 1})
    print(f"执行结果: {result}\n")
```

```python
builder.add_edge(START, "node_a")  # 从开始到A
builder.add_edge("node_a", "node_b")  # 从A到B
builder.add_edge("node_b", "node_c")  # 从B到C
builder.add_edge("node_c", END)  # 从C到结束
```

#### 条件边

```python
builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)

builder.add_conditional_edges("check_x", is_even, {
    True: "handle_even",
    False: "handle_odd"
})

# 添加起始边，从START节点流向check_x节点
builder.add_edge(START, "check_x")

# 添加结束边，从处理节点流向END节点
builder.add_edge("handle_even", END)
builder.add_edge("handle_odd", END)

# 编译图结构
graph = builder.compile()

# 打印图的可视化结构
print(graph.get_graph().print_ascii())

# 测试用例：输入偶数4
logger.info("输入 x=4（偶数）")
graph.invoke(MyState(x=4))
```

ps:此段代码是判断奇偶数的代码

案例二:

```python
def route_by_sentiment(state: AtguiguState) -> str:
    # 路由逻辑...返回最终的条件
    flag = state["x"]
    if flag == 1:
        return "condition_1"
    elif flag == 2:
        return "condition_2"
    else:
        return "condition_3"

graph = StateGraph(AtguiguState)
graph.add_node("node1", addition1)
graph.add_node("node2", addition2)
graph.add_node("node3", addition3)
# 添加路由函数，参数：当前节点，路由函数，路由函数返回的条件与node的映射
graph.add_conditional_edges(
    START,
    route_by_sentiment,
    {
        "condition_1": "node1",
        "condition_2": "node2",
        "condition_3": "node3"
    }
)
```

addition1分别是加一加二加三,在graph.add_conditional_edges中第二个参数就是判断的返回

#### 开始点

```python
builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.set_finish_point("node_b")
```

使用set_entry_point来添加start,用set_finish_point来设置结束点

#### 条件开始点

```python
def route_input(state: SimpleState) -> str:
    """根据用户输入决定去哪个节点"""
    text = state["user_input"].lower()

    if "hello" in text or "hi" in text:
        return "greeting"  # 返回路由键
    elif "bye" in text or "exit" in text:
        return "farewell"  # 返回路由键
    else:
        return "question"  # 返回路由键
```

路由函数,决定开始节点

```python
stateGraph.add_conditional_edges(
    START,  # 起点
    route_input,  # 路由函数
    # 路由映射（可选）：路由函数的返回值 -> 节点名
    {
        "greeting": "greeting_node",  # route_input返回"greeting"时，去greeting_node
        "farewell": "farewell_node",  # route_input返回"farewell"时，去farewell_node
        "question": "question_node"  # route_input返回"question"时，去question_node
    }
)
```

添加开始节点,重点在于START,有这个就是添加条件开始点

## Graph API之Send/Command/Runtime context

**Send和Command是两种用于实现高级工作流控制的核心机制，用于支持动态地决定下一步执行哪个节点**

### Send

<u>**多路并进，汇总规约**</u>

#### Map-Reduce场景

**Map-Reduce(映射 - 归约):将一个「大规模的复杂计算任务」拆解成无数个「小任务」并行处理，最后再把小任务的结果汇总得到最终答案。**

**Map 直译是「映射」，核心行为是：「拆分 + 局部处理」**

**Reduce 直译是「归约」，核心行为是：「聚合 + 全局计算」**

通俗讲就是把每个班级成绩优秀,及格,不及格的学生分开,再在全年部统计出优秀及格不及格的学生

**核心思想：拆分任务 →并行执行 → 统一汇总结果**

以下为案例完整代码

```python
from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 定义状态
class AtguiguState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], lambda x, y: x + y]  # 使用列表合并的方式


# 第一个节点：生成需要处理的主题列表
def generate_subjects(state: AtguiguState) -> dict:
    """生成需要处理的主题列表"""
    print("执行节点(第一个节点：生成需要处理的主题列表): generate_subjects")
    subjects = ["猫", "狗", "程序员"]
    print(f"生成主题列表: {subjects}")
    return {"subjects": subjects}


# Map节点：为每个主题生成笑话
def make_joke(state: AtguiguState) -> dict:
    """为单个主题生成笑话"""
    subject = state.get("subject", "未知")
    print(f"执行节点: make_joke，处理主题: {subject}")

    # 根据主题生成相应笑话
    jokes_map = {
        "猫": "为什么猫不喜欢在线购物？因为它们更喜欢实体店！",
        "狗": "为什么狗不喜欢计算机？因为它们害怕被鼠标咬！",
        "程序员": "为什么程序员喜欢洗衣服？因为他们在寻找bugs！",
        "未知": "这是一个关于未知主题的神秘笑话。"
    }

    joke = jokes_map.get(subject, f"这是一个关于{subject}的即兴笑话。")
    print(f"生成笑话: {joke}")
    return {"jokes": [joke]}


# 条件边函数：根据主题列表生成Send对象列表
def map_subjects_to_jokes(state: AtguiguState) -> List[Send]:
    """将主题列表映射到joke生成任务"""
    print("执行条件边函数: map_subjects_to_jokes")
    subjects = state["subjects"]
    print(f"映射主题到joke任务: {subjects}")

    # 为每个主题创建一个Send对象，指向make_joke节点
    # 每个Send对象包含节点名称和传递给该节点的状态
    send_list = [Send("make_joke", {"subject": subject}) for subject in subjects]
    print(f"生成Send对象列表: {send_list}")
    return send_list


def main():
    """演示Map-Reduce模式"""
    print("=== Map-Reduce 模式演示 ===\n")

    # 创建图
    builder = StateGraph(AtguiguState)

    # 添加节点
    builder.add_node("generate_subjects", generate_subjects)
    builder.add_node("make_joke", make_joke)

    # 添加边
    builder.add_edge(START, "generate_subjects")

    # 添加条件边，使用Send对象实现map-reduce
    builder.add_conditional_edges(
        "generate_subjects",  # 源节点
        map_subjects_to_jokes  # 路由函数，返回Send对象列表
    )

    # 从make_joke到结束
    builder.add_edge("make_joke", END)

    # 编译图
    graph = builder.compile()
    print(graph.get_graph().print_ascii())

    # 执行图
    initial_state = {"subjects": [], "jokes": []}
    print("初始状态:", initial_state)
    print("\n开始执行图...")

    result = graph.invoke(initial_state)
    print(f"\n最终结果: {result}")

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
```

------

------

```python
def generate_subjects(state: AtguiguState) -> dict:
    """生成需要处理的主题列表"""
    print("执行节点(第一个节点：生成需要处理的主题列表): generate_subjects")
    subjects = ["猫", "狗", "程序员"]
    print(f"生成主题列表: {subjects}")
    return {"subjects": subjects}
```

生成需要处理的主题,返回{"名称":[主题]}

```python
def make_joke(state: AtguiguState) -> dict:
    """为单个主题生成笑话"""
    subject = state.get("subject", "未知")
    print(f"执行节点: make_joke，处理主题: {subject}")

    # 根据主题生成相应笑话
    jokes_map = {
        "猫": "为什么猫不喜欢在线购物？因为它们更喜欢实体店！",
        "狗": "为什么狗不喜欢计算机？因为它们害怕被鼠标咬！",
        "程序员": "为什么程序员喜欢洗衣服？因为他们在寻找bugs！",
        "未知": "这是一个关于未知主题的神秘笑话。"
    }

    joke = jokes_map.get(subject, f"这是一个关于{subject}的即兴笑话。")
    print(f"生成笑话: {joke}")
    return {"jokes": [joke]}
```

生成对应的笑话,返回{"jokes":[具体笑话]}

```python
def map_subjects_to_jokes(state: AtguiguState) -> List[Send]:
    """将主题列表映射到joke生成任务"""
    print("执行条件边函数: map_subjects_to_jokes")
    subjects = state["subjects"]
    print(f"映射主题到joke任务: {subjects}")

    # 为每个主题创建一个Send对象，指向make_joke节点
    # 每个Send对象包含节点名称和传递给该节点的状态
    send_list = [Send("make_joke", {"subject": subject}) for subject in subjects]
    print(f"生成Send对象列表: {send_list}")
    return send_list
```

拆分处理每个主题,之后合并成一个send对象

```python
builder.add_node("generate_subjects", generate_subjects)
builder.add_node("make_joke", make_joke)

# 添加边
builder.add_edge(START, "generate_subjects")

# 添加条件边，使用Send对象实现map-reduce
builder.add_conditional_edges(
    "generate_subjects",  # 源节点
    map_subjects_to_jokes  # 路由函数，返回Send对象列表
)

# 从make_joke到结束
builder.add_edge("make_joke", END)
```

从在map_subjects_to_jokes中处理完后生成结果

最后的图是这样的

![image-20260525100136207](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260525100136207.png)

### Command

Command就是路由到一个节点,并更新数据

**在同一个节点中既执行状态更新，又决定下一步前往哪个节点，LangGraph 提供了一种实现方式，即从节点函数返回一个 Command 对象**

------

![image-20260525105656138](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260525105656138.png)

**走条新路，状态更新**

------

```python
def decision_agent(state: AgentState) -> Command[AgentState]:
    """根据消息内容路由代理，任务完成则直接终止"""
    print("执行节点: decision_agent")
    # 优先终止流程（核心防循环逻辑）
    if state["task_completed"]:
        print("✅ 检测到任务已完成，直接终止流程")
        return Command(
            update={"messages": [("system", "所有任务处理完成，流程正常结束")]},
            goto=END
        )
    # 提取消息文本（兼容空消息）
    last_message = state["messages"][-1] if state["messages"] else ("", "")
    last_msg_content = last_message[1]
    print(f"最新消息文本: {last_msg_content}")

    # 动态路由
    if "数学" in last_msg_content:
        print("✅ 检测到数学任务，路由到数学代理")
        return Command(
            update={"messages": [("system", "路由到数学代理")], "current_agent": "math_agent"},
            goto="math_agent"
        )
    elif "翻译" in last_msg_content:
        print("✅ 检测到翻译任务，路由到翻译代理")
        return Command(
            update={"messages": [("system", "路由到翻译代理")], "current_agent": "translation_agent"},
            goto="translation_agent"
        )
    else:
        print("❌ 未识别任务类型，标记任务完成并终止")
        return Command(
            update={"messages": [("system", "任务完成")], "task_completed": True},
            goto=END
        )
```

核心在与return的Command对象update是更新的数据goto是要去的节点

与条件边不同的在于Command可以更新数据,带着数据去下一个节点

### Runtime context运行时上下文

类似微服务的YML文件，配置和代码分离，信息从配置文件读取

#### 使用方式

1.使用 @dataclass 定义了一个 ContextSchema类，定义的内容不属于图的状态，但在运行时需要被节点访问

2.节点函数定义:节点函数接收两个参数：state（图的状态）和 runtime（运行时上下文）

通过 runtime.context 访问上下文信息，如 runtime.context.model_name

3.图的创建和执行:

创建 StateGraph 时传入  context_schema=ContextSchema 参数

调用 graph.invoke() 时通过 context 参数传递上下文数据

以下为案例完整代码

```python
"""
RuntimeContextDemo.py

LangGraph Context Schema 演示

演示如何在 LangGraph 1.0 中使用 context_schema 向节点传递不属于图表状态的信息。
这在传递模型名称、数据库连接等依赖项时非常有用。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage
from dataclasses import dataclass


# 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    response: str


# 定义上下文结构
@dataclass
class ContextSchema:
    model_name: str
    db_connection: str
    api_key: str


# 节点函数：处理用户消息
def process_message(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """处理用户消息的节点，使用context中的信息"""
    print("执行节点: process_message")

    # 获取最新的用户消息
    last_message = state["messages"][-1].content if state["messages"] else ""
    print(f"用户消息: {last_message}")
    print("=========以下是从RuntimeContext中获得信息=========")
    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    db_connection = runtime.context.db_connection
    api_key = runtime.context.api_key

    print(f"使用的模型: {model_name}")
    print(f"数据库连接: {db_connection}")
    print(f"API密钥前缀: {api_key[:5]}***")  # 只显示前5位，隐藏其余部分

    # 模拟使用这些信息处理请求
    response = f"使用 {model_name} 处理了您的请求，已连接到 {db_connection}"

    return {
        "messages": [AIMessage(content=response)],
        "response": response
    }


# 节点函数：生成最终响应
def generate_response(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """生成最终响应的节点"""
    print("执行节点: generate_response")

    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    print(f"使用模型 {model_name} 生成最终响应")

    # 获取之前的结果
    previous_response = state["response"]

    # 生成更详细的响应
    final_response = f"{previous_response}\n\n这是使用 {model_name} 生成的完整响应。"

    return {
        "messages": [AIMessage(content=final_response)],
        "response": final_response
    }


def main():
    """演示 context_schema 的使用"""
    print("=== Context Schema 演示 ===\n")

    # 定义上下文
    context = ContextSchema(
        model_name="gpt-4-turbo",
        db_connection="postgresql://user:pass@localhost:5432/orders_db",
        api_key="sk-abcdefghijklmnopqrstuvwxyz123456"
    )

    # 创建图，指定state_schema和context_schema
    builder = StateGraph(AgentState, context_schema=ContextSchema)

    # 添加节点
    builder.add_node("process_message", process_message)
    builder.add_node("generate_response", generate_response)

    # 添加边
    builder.add_edge(START, "process_message")
    builder.add_edge("process_message", "generate_response")
    builder.add_edge("generate_response", END)

    # 编译图
    graph = builder.compile()

    # 定义初始状态
    initial_state = {
        "messages": [HumanMessage(content="请帮我查询最新的订单信息")],
        "response": ""
    }

    print("初始状态:", initial_state)
    print()
    print("上下文信息:\n", {
        "model_name": context.model_name,
        "db_connection": context.db_connection,
        "api_key": f"{context.api_key[:5]}***"
    })
    print("\n" + "-" * 50 + "\n")

    # 执行图，通过context参数传递上下文
    result = graph.invoke(initial_state, context=context)

    print("\n" + "=" * 50)
    print("最终状态:", result)
    print("\n最终响应:")
    print(result["response"])


if __name__ == "__main__":
    main()
```



```python
# 定义上下文结构
@dataclass
class ContextSchema:
    model_name: str
    db_connection: str
    api_key: str
```

以此定义上下文结构

```python
def process_message(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """处理用户消息的节点，使用context中的信息"""
    print("执行节点: process_message")

    # 获取最新的用户消息
    last_message = state["messages"][-1].content if state["messages"] else ""
    print(f"用户消息: {last_message}")
    print("=========以下是从RuntimeContext中获得信息=========")
    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    db_connection = runtime.context.db_connection
    api_key = runtime.context.api_key

    print(f"使用的模型: {model_name}")
    print(f"数据库连接: {db_connection}")
    print(f"API密钥前缀: {api_key[:5]}***")  # 只显示前5位，隐藏其余部分

    # 模拟使用这些信息处理请求
    response = f"使用 {model_name} 处理了您的请求，已连接到 {db_connection}"

    return {
        "messages": [AIMessage(content=response)],
        "response": response
    }
```

处理用户消息

```python
def generate_response(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """生成最终响应的节点"""
    print("执行节点: generate_response")

    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    print(f"使用模型 {model_name} 生成最终响应")

    # 获取之前的结果
    previous_response = state["response"]

    # 生成更详细的响应
    final_response = f"{previous_response}\n\n这是使用 {model_name} 生成的完整响应。"

    return {
        "messages": [AIMessage(content=final_response)],
        "response": final_response
    }
```

返回大模型的响应消息

主函数中的

```python
context = ContextSchema(
    model_name="gpt-4-turbo",
    db_connection="postgresql://user:pass@localhost:5432/orders_db",
    api_key="sk-abcdefghijklmnopqrstuvwxyz123456"
)
```

传递数据库和大模型信息

```python
builder = StateGraph(AgentState, context_schema=ContextSchema)
```

创建对象时context_schema=ContextSchema上下文消息传进去

@dataclass相当于省掉"init","eq","repr"的魔法方法,更专业有了dataclass不允许再写init魔法方法,要写的话使用![image-20260525113231683](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260525113231683.png)

![image-20260525113346402](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260525113346402.png)