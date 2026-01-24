import os
import json
from openai import OpenAI
from typing import List, Dict, Any, Callable

# ==========================================
# 1. 定义基础抽象 (First Principles)
# Google 风格：接口清晰，没有多余的魔法
# ==========================================

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def to_openai_schema(self) -> Dict:
        """将工具转换为 OpenAI Function Calling 格式"""
        # 为了演示从简，这里假设所有工具都接受单一字符串参数 'query'
        # 在生产环境中，这里应该根据 func 的签名自动生成 JSON Schema
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The input to the tool"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

class SimpleAgent:
    def __init__(self, system_prompt: str = ""):
        # 使用 DeepSeek 配置
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {}
        self.history: List[Dict] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def run(self, user_input: str):
        """
        核心 ReAct 循环 (Think -> Act -> Observe)
        Google 面试考点：如何手动管理 Context Window 和 Message History
        """
        print(f"User: {user_input}")
        self.history.append({"role": "user", "content": user_input})

        max_turns = 5  # 防止无限循环 (Safety)
        turn = 0

        while turn < max_turns:
            turn += 1
            print(f"\n--- Turn {turn} (Thinking...) ---")
            
            # 1. Think: 调用 LLM 决策
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",  # 使用 DeepSeek 模型
                    messages=self.history,
                    tools=[t.to_openai_schema() for t in self.tools.values()] if self.tools else None,
                    tool_choice="auto" if self.tools else None,
                    timeout=30
                )
            except Exception as e:
                print(f"Error during API call: {e}")
                print("⚠️ API Failed, switching to MOCK mode for demonstration...")
                
                # Mock logic for demonstration
                import collections
                MockMessage = collections.namedtuple('MockMessage', ['content', 'tool_calls'])
                MockToolCall = collections.namedtuple('MockToolCall', ['id', 'function'])
                MockFunction = collections.namedtuple('MockFunction', ['name', 'arguments'])

                if "北京" in user_input and turn == 1:
                    # Mock Turn 1: Call Weather
                    message = MockMessage(
                        content=None,
                        tool_calls=[MockToolCall(
                            id='call_1',
                            function=MockFunction(name='get_weather', arguments='{"query": "北京"}')
                        )]
                    )
                elif "calculator" in [t.name for t in self.tools.values()] and turn == 2:
                     # Mock Turn 2: Call Calculator (assuming we got weather)
                     # In a real agent, we'd check history, but here we hardcode for the specific demo case
                    message = MockMessage(
                        content=None,
                        tool_calls=[MockToolCall(
                            id='call_2',
                            function=MockFunction(name='calculator', arguments='{"query": "25 + 10"}')
                        )]
                    )
                else:
                    message = MockMessage(content="根据计算，结果是 35 度。", tool_calls=None)
                
                self.history.append(message)
                
                # Skip the "if not message.tool_calls" check below by handling it here or ensuring structure matches
                if not message.tool_calls:
                     print(f"Agent (Mock): {message.content}")
                     return message.content

                # Continue to tool execution logic...
                pass 
            
            # message is already set above if mock, or from response if success
            if 'response' in locals():
                 message = response.choices[0].message
                 self.history.append(message)

            if not message.tool_calls:
                print(f"Agent: {message.content}")
                return message.content

            # 2. Act: 解析并执行工具
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"-> Act: Calling Tool '{function_name}' with args {arguments}")

                if function_name in self.tools:
                    tool_instance = self.tools[function_name]
                    # 3. Observe: 获取执行结果
                    try:
                        # 简化起见，我们解包 arguments
                        # 实际生产代码需要更健壮的参数验证
                        tool_result = tool_instance.func(**arguments)
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                    
                    print(f"<- Observe: Result: {tool_result}")

                    # 将执行结果反馈给 LLM
                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result)
                    })
                else:
                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": "Error: Tool not found"
                    })

        print("Error: Max turns reached.")
        return "Max turns reached"

# ==========================================
# 2. 模拟工具 (Mock Tools)
# ==========================================

def get_weather(query: str):
    """查询天气的模拟工具"""
    if "Beijing" in query or "北京" in query:
        return "北京今天是晴天，气温 25度。"
    elif "Shanghai" in query or "上海" in query:
        return "上海今天是小雨，气温 22度。"
    else:
        return "未知城市的天气数据。"

def calculator(query: str):
    """简单的计算器"""
    try:
        # ⚠️ 危险：在生产环境中永远不要直接 eval
        # 这里仅作演示 First Principles
        return eval(query)
    except:
        return "计算错误"

# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == "__main__":
    # 初始化 Agent
    agent = SimpleAgent(system_prompt="你是一个有用的 AI 助手，可以使用工具解决问题。")

    # 注册工具
    agent.register_tool(Tool("get_weather", "查询城市天气", get_weather))
    agent.register_tool(Tool("calculator", "计算数学表达式", calculator))

    # 执行任务
    # print("Task 1: 查询天气")
    # agent.run("北京今天天气怎么样？")

    # print("\n" + "="*30 + "\n")

    print("Task 2: 复杂推理")
    agent.run("北京的气温加上 10 度是多少？")
