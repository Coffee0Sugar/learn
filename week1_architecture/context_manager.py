import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("⚠️ tiktoken not installed, using simple token estimation")

# ==========================================
# Week 1.2: Context Window Management
# Google 考点：智能Token管理和上下文优化
# ==========================================

@dataclass
class Message:
    """消息类：表示一条对话消息"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: datetime = None
    importance: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class ContextManager:
    """Context Window管理器：智能管理对话历史"""
    
    def __init__(self, 
                 token_limit: int = 4000,
                 model_name: str = "gpt-4o-mini",
                 system_prompt: str = ""):
        """
        Args:
            token_limit: 最大Token数
            model_name: 模型名称（用于选择正确的tokenizer）
            system_prompt: 系统提示词
        """
        self.token_limit = token_limit
        self.model_name = model_name
        self.messages: List[Message] = []

        # 初始化tokenizer
        if tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                print(f"⚠️ Model '{model_name}' not found, using cl100k_base encoding")
        else:
            self.encoding = None

        # 统计信息
        self.stats = {
            'total_messages': 0,
            'trimmed_messages': 0,
            'current_tokens': 0,
            'total_tokens_consumed': 0
        }

        # 添加系统提示词
        if system_prompt:
            self.add_message(
                role="system",
                content=system_prompt,
                importance=1.0  # 系统消息总是最重要
            )
    
    def calculate_tokens(self, text: str) -> int:
        """
        准确计算Token数
        
        Args:
            text: 输入文本
        
        Returns:
            Token数
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # 简单估算：1 token ≈ 4 字符（英文）或 1.5 字符（中文）
            # 这里使用更精确的估算
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)
    
    def calculate_message_tokens(self, message: Message) -> int:
        """
        计算单个消息的Token数
        
        Args:
            message: 消息对象
        
        Returns:
            Token数
        """
        # 基础Token数（消息格式开销）
        base_tokens = 4  # role + content分隔符等
        
        # 内容Token数
        content_tokens = self.calculate_tokens(message.content)
        
        # Tool调用有额外开销
        if message.role == 'tool':
            base_tokens += 10  # tool_call_id, name等
        
        return base_tokens + content_tokens
    
    def add_message(self, 
                   role: str, 
                   content: str,
                   importance: float = 1.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加消息
        
        Args:
            role: 消息角色
            content: 消息内容
            importance: 重要性分数 (0-1)
            metadata: 元数据
        """
        message = Message(
            role=role,
            content=content,
            importance=importance,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.stats['total_messages'] += 1
    
    def rank_messages(self, messages: List[Message]) -> List[Message]:
        """
        按重要性排序消息
        
        排序规则：
        1. 系统消息（总是保留）
        2. 高重要性分数
        3. 最近的消息
        4. 包含工具调用的消息
        
        Args:
            messages: 要排序的消息列表
        
        Returns:
            排序后的消息列表
        """
        def importance_score(msg: Message) -> float:
            # 基础重要性
            score = msg.importance
            
            # 系统消息最高优先级
            if msg.role == 'system':
                score += 10.0
            
            # 工具调用次优先级
            if msg.role == 'tool' or 'tool_calls' in (msg.metadata or {}):
                score += 2.0
            
            # 时间衰减：越新的消息权重越高
            age_hours = (datetime.now() - msg.timestamp).total_seconds() / 3600
            time_factor = max(0.1, np.exp(-age_hours / 12))  # 12小时半衰期
            score *= time_factor
            
            return score
        
        import numpy as np
        return sorted(messages, key=importance_score, reverse=True)
    
    def trim_history(self, 
                     messages: List[Message], 
                     token_limit: Optional[int] = None) -> Tuple[List[Message], int]:
        """
        智能裁剪历史，保留最重要的消息
        
        Args:
            messages: 要裁剪的消息列表
            token_limit: Token限制（默认使用实例的token_limit）
        
        Returns:
            (裁剪后的消息列表, 移除的消息数)
        """
        if token_limit is None:
            token_limit = self.token_limit
        
        # 计算当前Token数
        current_tokens = sum(self.calculate_message_tokens(m) for m in messages)
        
        if current_tokens <= token_limit:
            return messages, 0
        
        # 按重要性排序
        ranked_messages = self.rank_messages(messages)
        
        # 选择最重要的消息
        selected_messages = []
        total_tokens = 0
        
        for msg in ranked_messages:
            msg_tokens = self.calculate_message_tokens(msg)
            
            if total_tokens + msg_tokens <= token_limit:
                selected_messages.append(msg)
                total_tokens += msg_tokens
            else:
                # 尝试只保留部分内容（对长消息）
                if total_tokens < token_limit * 0.9:  # 只在还有空间时尝试
                    remaining_tokens = token_limit - total_tokens
                    # 简单截断（实际可以使用更智能的摘要）
                    if len(msg.content) > 0:
                        chars_per_token = 4 if self.encoding else 3
                        max_chars = int(remaining_tokens * chars_per_token)
                        truncated_msg = Message(
                            role=msg.role,
                            content=msg.content[:max_chars] + "... [truncated]",
                            importance=msg.importance,
                            timestamp=msg.timestamp,
                            metadata=msg.metadata
                        )
                        selected_messages.append(truncated_msg)
                        total_tokens += remaining_tokens
                break
        
        # 按原始顺序重新排序
        selected_messages.sort(key=lambda m: m.timestamp)
        
        removed_count = len(messages) - len(selected_messages)
        return selected_messages, removed_count
    
    def get_optimal_context(self, 
                           user_input: str,
                           memory_context: str = "",
                           memory_importance: float = 0.8) -> List[Dict[str, str]]:
        """
        获取最优的上下文，整合消息历史和记忆
        
        Args:
            user_input: 当前用户输入
            memory_context: 记忆上下文
            memory_importance: 记忆的重要性分数
        
        Returns:
            OpenAI格式的消息列表
        """
        # 添加用户输入
        self.add_message(role="user", content=user_input, importance=0.7)
        
        # 如果有记忆，插入到合适的位置
        if memory_context:
            # 找到最后一条系统消息之后的位置
            insert_index = 0
            for i, msg in enumerate(self.messages):
                if msg.role == 'system':
                    insert_index = i + 1
                elif msg.role in ['user', 'assistant']:
                    break
            
            # 插入记忆作为系统消息
            memory_msg = Message(
                role="system",
                content=f"Relevant context from memory:\n{memory_context}",
                importance=memory_importance,
                metadata={'type': 'memory'}
            )
            self.messages.insert(insert_index, memory_msg)
        
        # 裁剪到Token限制
        trimmed_messages, removed_count = self.trim_history(
            self.messages, 
            self.token_limit
        )
        
        if removed_count > 0:
            self.stats['trimmed_messages'] += removed_count
            print(f"⚠️ Trimmed {removed_count} messages to fit token limit")
        
        # 更新消息列表
        self.messages = trimmed_messages
        
        # 计算当前Token数
        current_tokens = sum(self.calculate_message_tokens(m) for m in self.messages)
        self.stats['current_tokens'] = current_tokens
        self.stats['total_tokens_consumed'] += current_tokens
        
        # 转换为OpenAI格式
        return self._to_openai_format(self.messages)
    
    def _to_openai_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """转换为OpenAI API格式"""
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            if msg.metadata:
                openai_msg.update(msg.metadata)
            openai_messages.append(openai_msg)
        return openai_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['message_count'] = len(self.messages)
        return stats
    
    def export_history(self, filepath: str) -> None:
        """导出对话历史"""
        export_data = []
        for msg in self.messages:
            export_data.append({
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'importance': msg.importance,
                'metadata': msg.metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(export_data)} messages to {filepath}")
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        清空对话历史
        
        Args:
            keep_system: 是否保留系统消息
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == 'system']
        else:
            self.messages = []
        
        print(f"✅ Cleared history. Remaining messages: {len(self.messages)}")

# ==========================================
# 实战演示
# ==========================================

if __name__ == "__main__":
    print("📊 Context Window Management System")
    print("=" * 60)
    
    # 初始化上下文管理器
    context_manager = ContextManager(
        token_limit=1000,  # 较小的限制用于演示
        model_name="gpt-4o-mini",
        system_prompt="You are a helpful AI assistant."
    )
    
    # 添加一些对话
    print("\n📝 Adding messages to simulate a long conversation...")
    for i in range(10):
        context_manager.add_message(
            role="user",
            content=f"This is message number {i} with some additional text to use more tokens.",
            importance=0.5
        )
        
        context_manager.add_message(
            role="assistant",
            content=f"Response to message {i}: Here's a detailed explanation about the topic you asked about.",
            importance=0.4
        )
    
    # 添加一些高重要性消息
    context_manager.add_message(
        role="user",
        content="IMPORTANT: The API key for production is PROD-KEY-12345",
        importance=1.0,
        metadata={'critical': True}
    )
    
    # 显示当前状态
    print("\n📈 Current statistics:")
    print("-" * 60)
    stats = context_manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 测试Token计算
    print("\n🔢 Token calculation examples:")
    print("-" * 60)
    test_texts = [
        "Hello, world!",
        "这是一个中文测试句子。",
        "This is a much longer text that should contain significantly more tokens than the previous examples. It demonstrates how the token counting function works with different types of content."
    ]
    
    for text in test_texts:
        tokens = context_manager.calculate_tokens(text)
        print(f"   Text: '{text[:50]}...'")
        print(f"   Tokens: {tokens}")
    
    # 测试智能裁剪
    print("\n✂️ Testing intelligent trimming...")
    print("-" * 60)
    optimal_context = context_manager.get_optimal_context(
        user_input="What did you tell me in message 3?",
        memory_context="User is interested in token optimization.",
        memory_importance=0.9
    )
    
    print(f"Optimal context contains {len(optimal_context)} messages")
    print(f"Total tokens: {context_manager.calculate_tokens(''.join([m['content'] for m in optimal_context]))}")
    
    # 显示排序后的消息
    print("\n📊 Messages ranked by importance:")
    print("-" * 60)
    ranked = context_manager.rank_messages(context_manager.messages)
    for i, msg in enumerate(ranked[:5], 1):
        print(f"{i}. [{msg.role}] {msg.content[:50]}...")
        print(f"   Importance: {msg.importance}")
    
    # 导出历史
    print("\n💾 Exporting conversation history...")
    context_manager.export_history("week1_architecture/conversation_history.json")
    
    print("\n✅ Demo complete!")
