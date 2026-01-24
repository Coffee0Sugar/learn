import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1_architecture'))

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from openai import OpenAI
from datetime import datetime

from memory_manager import MemoryManager, MemoryItem
from context_manager import ContextManager, Message

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDING_MODEL = True
except ImportError:
    HAS_EMBEDDING_MODEL = False
    print("⚠️ sentence-transformers not installed, using mock embeddings")

class Tool:
    def __init__(self, name: str, description: str, func: Callable, parameters: Dict = None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The input to the tool"}
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

class RAGAgent:
    def __init__(self,
                 system_prompt: str = "",
                 knowledge_base: List[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 token_limit: int = 8000,
                 short_term_limit: int = 20,
                 long_term_limit: int = 1000,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        整合 Memory + Context + Tools + RAG 的完整 Agent

        Args:
            system_prompt: 系统提示词
            knowledge_base: 知识库文档列表
            embedding_model: Embedding 模型名称
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
            print("✅ RAGAgent: Using real API")
        else:
            self.client = None
            self.use_mock = True
            print("⚠️ RAGAgent: Using MOCK mode")

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

        self.rag_embeddings = None
        self.rag_documents = []

        if knowledge_base:
            self.build_knowledge_base(knowledge_base, embedding_model)

        self.stats = {
            'total_queries': 0,
            'rag_retrievals': 0,
            'memory_retrievals': 0,
            'tool_calls': 0
        }

    def build_knowledge_base(self, docs: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """构建知识库向量索引"""
        self.rag_documents = docs
        print(f"📚 Building knowledge base with {len(docs)} documents...")

        if HAS_EMBEDDING_MODEL:
            print(f"   Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            self.rag_embeddings = model.encode(docs, normalize_embeddings=True)
            self.rag_embedding_dim = self.rag_embeddings.shape[1]
            print(f"   Embedding dimension: {self.rag_embedding_dim}")
        else:
            self.rag_embedding_dim = 384
            np.random.seed(42)
            self.rag_embeddings = np.random.rand(len(docs), self.rag_embedding_dim)
            self.rag_embeddings = self.rag_embeddings / np.linalg.norm(self.rag_embeddings, axis=1)[:, np.newaxis]
            print(f"   Using mock embeddings (dimension: {self.rag_embedding_dim})")

        print("   Knowledge base ready!")

    def rag_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        RAG 向量检索

        Returns:
            [(文档内容, 相似度分数)] 列表
        """
        if self.rag_embeddings is None or len(self.rag_documents) == 0:
            return []

        self.stats['rag_retrievals'] += 1

        if HAS_EMBEDDING_MODEL and not self.use_mock:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_vec = model.encode([query], normalize_embeddings=True)[0]
        else:
            np.random.seed(hash(query) % (2**32 - 1))
            query_vec = np.random.rand(self.rag_embedding_dim)
            query_vec = query_vec / np.linalg.norm(query_vec)

        scores = np.dot(self.rag_embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.rag_documents[i], float(scores[i])) for i in top_indices]

    def get_rag_context(self, query: str, top_k: int = 3) -> str:
        """获取 RAG 检索结果作为上下文"""
        results = self.rag_search(query, top_k)

        if not results:
            return ""

        context = "Relevant information from knowledge base:\n"
        for doc, score in results:
            context += f"- [{score:.3f}] {doc}\n"

        return context

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _build_context(self, user_input: str) -> List[Dict[str, str]]:
        """构建最优上下文"""
        memory_context = self._retrieve_memories(user_input)
        rag_context = self.get_rag_context(user_input)

        context_messages = []

        if self.system_prompt:
            context_messages.append({"role": "system", "content": self.system_prompt})

        if rag_context:
            context_messages.append({
                "role": "system",
                "content": rag_context,
                "metadata": {"type": "rag_context"}
            })

        if memory_context:
            context_messages.append({
                "role": "system",
                "content": memory_context,
                "metadata": {"type": "memory_context"}
            })

        context_messages.append({"role": "user", "content": user_input})

        return context_messages

    def _retrieve_memories(self, query: str, k: int = 3) -> str:
        """检索相关记忆"""
        results = self.memory_manager.retrieve_memory(query, k=k)
        self.stats['memory_retrievals'] += 1

        if not results:
            return ""

        context = "Relevant memories:\n"
        for memory, score in results:
            context += f"- [{memory.memory_type}] {memory.content}\n"

        return context

    def run(self, user_input: str, max_turns: int = 5) -> str:
        """执行 ReAct 循环"""
        self.stats['total_queries'] += 1
        print(f"\n👤 User: {user_input}")

        context = self._build_context(user_input)
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
                result = self._act(tool_call)
                context.append(result)
                self.conversation_history.append(result)

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
            response_dict = response.model_dump()
            return response_dict.get("choices", [{}])[0].get("message", {})
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

        for msg in context:
            if msg.get("metadata", {}).get("type") == "rag_context":
                rag_info = msg.get("content", "")
                return {
                    "role": "assistant",
                    "content": f"Based on the knowledge base: {rag_info}\n\nThis is a mock response for: {user_content}",
                    "tool_calls": None
                }

        return {
            "role": "assistant",
            "content": f"I received: 「{user_content}」. (Mock mode response)",
            "tool_calls": None
        }

    def _act(self, tool_call: Dict) -> Dict:
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

    def _update_memory(self, user_input: str, response: str):
        """更新记忆系统"""
        self.memory_manager.add_to_memory(
            content=f"User asked: {user_input}",
            memory_type="episodic",
            importance=0.6
        )
        self.memory_manager.add_to_memory(
            content=f"Agent responded: {response}",
            memory_type="episodic",
            importance=0.5
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "conversation": self.stats.copy(),
            "memory": self.memory_manager.get_stats(),
            "knowledge_base": {
                "doc_count": len(self.rag_documents),
                "embedding_dim": getattr(self, 'rag_embedding_dim', 0)
            }
        }


def get_weather(query: str) -> str:
    """天气查询工具"""
    if "北京" in query:
        return "北京今天晴天，25°C"
    elif "上海" in query:
        return "上海今天小雨，22°C"
    else:
        return f"未知城市天气"

def calculator(query: str) -> str:
    """计算器工具"""
    try:
        import re
        safe = re.sub(r'[^0-9+\-*/.()\s]', '', query)
        return str(eval(safe))
    except:
        return "计算错误"


if __name__ == "__main__":
    print("=" * 60)
    print("📚 RAG Agent Demo")
    print("   Memory + Context + Tools + RAG Retrieval")
    print("=" * 60)

    knowledge_base = [
        "Python is a high-level programming language known for simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with generation.",
        "Vector databases store embeddings for semantic search.",
        "Transformers use self-attention mechanisms.",
        "Google was founded in 1998 by Larry Page and Sergey Brin.",
        "OpenAI developed GPT models and ChatGPT.",
    ]

    agent = RAGAgent(
        system_prompt="你是一个知识渊博的助手，可以根据知识库回答问题。",
        knowledge_base=knowledge_base,
        token_limit=2000,
        short_term_limit=10
    )

    agent.register_tool(Tool("get_weather", "查询天气", get_weather))
    agent.register_tool(Tool("calculator", "计算", calculator))

    print("\n📌 Test 1: RAG 知识库查询")
    agent.run("Python 有什么特点？")

    print("\n" + "-" * 60)

    print("\n📌 Test 2: RAG 检索")
    agent.run("什么是深度学习？")

    print("\n" + "-" * 60)

    print("\n📌 Test 3: RAG + 工具调用")
    agent.run("北京天气怎么样？顺便帮我算一下 100 * 7 + 50")

    print("\n" + "-" * 60)

    print("\n📌 Test 4: 记忆测试")
    agent.run("还记得我刚才问了什么吗？")

    print("\n" + "=" * 60)
    print("📊 Statistics:")
    print("-" * 60)
    stats = agent.get_stats()
    for section, data in stats.items():
        print(f"\n{section}:")
        for k, v in data.items():
            print(f"   {k}: {v}")

    print("\n✅ Demo complete!")
