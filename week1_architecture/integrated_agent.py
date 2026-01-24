import os
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from openai import OpenAI
from datetime import datetime
import numpy as np

from memory_manager import MemoryManager, MemoryItem
from context_manager import ContextManager, Message

class Tool:
    def __init__(self, name: str, description: str, func: Callable, parameters: Dict = None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input to the tool"
                }
            },
            "required": ["query"]
        }

    def to_openai_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

class IntegratedAgent:
    def __init__(self,
                 system_prompt: str = "",
                 token_limit: int = 8000,
                 short_term_limit: int = 20,
                 long_term_limit: int = 1000,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        整合的记忆+上下文+工具 Agent

        Args:
            system_prompt: 系统提示词
            token_limit: 上下文窗口Token限制
            short_term_limit: 短期记忆最大条数
            long_term_limit: 长期记忆最大条数
            api_key: API密钥
            base_url: API base URL
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_mock = False
            print("✅ IntegratedAgent: Using real API")
        else:
            self.client = None
            self.use_mock = True
            print("⚠️ IntegratedAgent: Using MOCK mode")

        self.tools: Dict[str, Tool] = {}

        self.memory_manager = MemoryManager(
            short_term_limit=short_term_limit,
            long_term_limit=long_term_limit,
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.context_manager = ContextManager(
            token_limit=token_limit,
            model_name="deepseek-chat",
            system_prompt=system_prompt
        )

        self.system_prompt = system_prompt
        self.conversation_history: List[Dict] = []
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})

        self.stats = {
            'total_queries': 0,
            'memory_retrievals': 0,
            'tool_calls': 0,
            'context_trims': 0
        }

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> str:
        """检索相关记忆并返回格式化的上下文"""
        results = self.memory_manager.retrieve_memory(query, k=k)
        self.stats['memory_retrievals'] += 1

        if not results:
            return ""

        memory_context = "Relevant memories from previous conversations:\n"
        for memory, score in results:
            memory_context += f"- [{memory.memory_type}] {memory.content}\n"

        return memory_context

    def _build_context(self, user_input: str) -> List[Dict[str, str]]:
        """构建最优上下文"""
        memory_context = self._retrieve_relevant_memories(user_input, k=5)

        return self.context_manager.get_optimal_context(
            user_input=user_input,
            memory_context=memory_context,
            memory_importance=0.8
        )

    def run(self, user_input: str, max_turns: int = 5) -> str:
        """
        执行 ReAct 循环

        Args:
            user_input: 用户输入
            max_turns: 最大轮次

        Returns:
            Agent 的最终响应
        """
        self.stats['total_queries'] += 1
        print(f"\n👤 User: {user_input}")

        context = self._build_context(user_input)

        context.append({"role": "user", "content": user_input})
        self.conversation_history = context.copy()

        turn = 0
        final_response = ""

        while turn < max_turns:
            turn += 1
            print(f"\n--- Turn {turn} (Thinking...) ---")

            message = self._think(context)

            if not message.get("tool_calls"):
                final_response = message.get("content", "")
                print(f"🤖 Agent: {final_response}")
                break

            self.stats['tool_calls'] += len(message["tool_calls"])

            for tool_call in message["tool_calls"]:
                result = self._act(tool_call, context)
                context.append(result)
                self.conversation_history.append(result)

                if self._should_continue(context, max_turns):
                    continue
                else:
                    final_response = result.get("content", "")
                    print(f"🤖 Agent: {final_response}")
                    return final_response

        if not final_response:
            final_response = "I need more time to think about this..."
            print(f"🤖 Agent: {final_response}")

        self._update_memory(user_input, final_response)

        return final_response

    def _think(self, context: List[Dict]) -> Dict:
        """调用 LLM 进行思考决策"""
        if self.use_mock:
            return self._mock_think(context)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=context,
                tools=[t.to_openai_schema() for t in self.tools.values()] if self.tools else None,
                tool_choice="auto" if self.tools else None,
                timeout=30
            )
            return json.loads(response.model_dump_json())
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            return self._mock_think(context)

    def _mock_think(self, context: List[Dict]) -> Dict:
        """Mock 思考逻辑"""
        user_content = ""
        for msg in reversed(context):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if "天气" in user_content or "weather" in user_content.lower():
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"query": user_content})
                    }
                }]
            }
        elif "计算" in user_content or any(op in user_content for op in ["+", "-", "*", "/"]):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"query": user_content})
                    }
                }]
            }
        else:
            return {
                "role": "assistant",
                "content": f"我收到了你的消息：「{user_content}」。这是通过 Mock 模式生成的回复。",
                "tool_calls": None
            }

    def _act(self, tool_call: Dict, context: List[Dict]) -> Dict:
        """执行工具调用"""
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        print(f"-> Act: Calling Tool '{function_name}' with args {arguments}")

        if function_name in self.tools:
            try:
                result = self.tools[function_name].func(**arguments)
            except Exception as e:
                result = f"Error: {str(e)}"
        else:
            result = "Error: Tool not found"

        print(f"<- Observe: Result: {result}")

        return {
            "tool_call_id": tool_call["id"],
            "role": "tool",
            "name": function_name,
            "content": str(result)
        }

    def _should_continue(self, context: List[Dict], max_turns: int) -> bool:
        """判断是否需要继续思考"""
        last_msg = context[-1] if context else {}

        if last_msg.get("role") == "tool":
            content = last_msg.get("content", "")
            if not content.startswith("Error:") and len(content) < 100:
                return False

        return True

    def _update_memory(self, user_input: str, response: str):
        """更新记忆系统"""
        self.memory_manager.add_to_memory(
            content=f"User asked: {user_input}",
            memory_type="episodic",
            importance=0.6,
            metadata={"type": "conversation"}
        )

        self.memory_manager.add_to_memory(
            content=f"Agent responded: {response}",
            memory_type="episodic",
            importance=0.5,
            metadata={"type": "conversation"}
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "conversation": self.stats.copy(),
            "memory": self.memory_manager.get_stats(),
            "context": self.context_manager.get_stats()
        }

    def clear_history(self):
        """清空对话历史"""
        self.context_manager.clear_history(keep_system=True)
        self.conversation_history = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []


def get_weather(query: str) -> str:
    """查询天气的模拟工具"""
    if "北京" in query or "Beijing" in query:
        return "北京今天是晴天，气温 25 度。"
    elif "上海" in query or "Shanghai" in query:
        return "上海今天是小雨，气温 22 度。"
    elif "深圳" in query or "Shenzhen" in query:
        return "深圳今天是多云，气温 28 度。"
    else:
        return f"未知城市「{query}」的天气数据。"

def calculator(query: str) -> str:
    """简单的计算器"""
    try:
        import re
        safe_chars = re.sub(r'[^0-9+\-*/.()\s]', '', query)
        result = eval(safe_chars)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def search_code(query: str) -> str:
    """代码搜索模拟工具"""
    codebase = {
        "python": "Python 是一种高级编程语言，特点是语法简洁、易读性强。",
        "react": "React 是一个用于构建用户界面的 JavaScript 库。",
        "api": "API (Application Programming Interface) 是软件系统间交互的接口。",
        "database": "Database 是用于存储和管理数据的系统。"
    }

    for key, value in codebase.items():
        if key in query.lower():
            return value

    return f"没有找到与「{query}」相关的代码文档。"

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Integrated Agent Demo")
    print("   Memory + Context + Tools")
    print("=" * 60)

    agent = IntegratedAgent(
        system_prompt="你是一个智能助手，可以使用工具解决问题，并能记住之前的对话内容。",
        token_limit=2000,
        short_term_limit=10,
        long_term_limit=50
    )

    agent.register_tool(Tool(
        "get_weather",
        "查询城市天气",
        get_weather,
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "城市名称"}
            },
            "required": ["query"]
        }
    ))

    agent.register_tool(Tool(
        "calculator",
        "计算数学表达式",
        calculator,
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "数学表达式，如 25 + 10"}
            },
            "required": ["query"]
        }
    ))

    agent.register_tool(Tool(
        "search_code",
        "搜索代码相关文档",
        search_code,
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    ))

    print("\n📌 Task 1: 天气查询")
    agent.run("北京今天天气怎么样？")

    print("\n" + "-" * 60)

    print("\n📌 Task 2: 计算")
    agent.run("25 + 17 等于多少？")

    print("\n" + "-" * 60)

    print("\n📌 Task 3: 代码搜索")
    agent.run("请介绍一下 Python 的特点")

    print("\n" + "-" * 60)

    print("\n📌 Task 4: 记忆测试")
    agent.run("还记得我刚才问了什么吗？")

    print("\n" + "=" * 60)
    print("📊 Statistics:")
    print("-" * 60)
    stats = agent.get_stats()
    for section, data in stats.items():
        print(f"\n{section}:")
        for key, value in data.items():
            print(f"   {key}: {value}")

    print("\n✅ Demo complete!")
