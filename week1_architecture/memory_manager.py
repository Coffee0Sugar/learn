import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("⚠️ OpenAI not installed, running in MOCK mode")
    OpenAI = None

# ==========================================
# Week 1.1: Memory Management System
# Google 考点：企业级记忆系统设计
# ==========================================

@dataclass
class MemoryItem:
    """记忆项：表示一条记忆"""
    content: str
    memory_type: str  # 'episodic' or 'semantic'
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # 重要性分数 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

class MemoryManager:
    """记忆管理系统：管理短期和长期记忆"""
    
    def __init__(self, 
                 short_term_limit: int = 20,
                 long_term_limit: int = 1000,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Args:
            short_term_limit: 短期记忆最大条数
            long_term_limit: 长期记忆最大条数
            api_key: OpenAI API密钥（用于生成embedding）
            base_url: API base URL
        """
        # 初始化API客户端
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_mock = False
            print("✅ MemoryManager: Using real embeddings")
        else:
            self.client = None
            self.use_mock = True
            print("⚠️ MemoryManager: Using MOCK embeddings")
        
        # 记忆存储
        self.short_term_memory: List[MemoryItem] = []
        self.long_term_memory: List[MemoryItem] = []
        
        # 限制
        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit
        
        # 向量维度（使用模拟的384维）
        self.embedding_dim = 384
        
        # 统计信息
        self.stats = {
            'total_memories': 0,
            'episodic_count': 0,
            'semantic_count': 0,
            'retrieval_count': 0,
            'compression_count': 0
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        if self.use_mock or not self.client:
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest()[:16], 16)
            hash_int = hash_int % (2**32 - 1)
            np.random.seed(hash_int)
            return np.random.randn(self.embedding_dim)
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=30
            )
            embedding = np.array(response.data[0].embedding)
            return embedding / np.linalg.norm(embedding)  # 归一化
        except Exception as e:
            print(f"⚠️ Embedding API failed: {e}, using mock")
            return np.random.randn(self.embedding_dim)
    
    def add_to_memory(self, 
                     content: str, 
                     memory_type: str = 'episodic',
                     importance: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> MemoryItem:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 ('episodic' or 'semantic')
            importance: 重要性分数 (0-1)
            metadata: 元数据
        
        Returns:
            添加的MemoryItem
        """
        if memory_type not in ['episodic', 'semantic']:
            raise ValueError(f"Invalid memory_type: {memory_type}")
        
        # 生成embedding
        embedding = self._get_embedding(content)
        
        # 创建记忆项
        memory_item = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # 添加到相应的记忆存储
        if memory_type == 'episodic':
            self.short_term_memory.append(memory_item)
            # 限制短期记忆数量
            if len(self.short_term_memory) > self.short_term_limit:
                self.short_term_memory.pop(0)
        else:
            self.long_term_memory.append(memory_item)
            # 限制长期记忆数量
            if len(self.long_term_memory) > self.long_term_limit:
                # 移除最不重要且最旧的记忆
                self.long_term_memory.sort(
                    key=lambda x: (x.importance, x.timestamp)
                )
                self.long_term_memory.pop(0)
        
        # 更新统计
        self.stats['total_memories'] += 1
        if memory_type == 'episodic':
            self.stats['episodic_count'] += 1
        else:
            self.stats['semantic_count'] += 1
        
        return memory_item
    
    def retrieve_memory(self, 
                       query: str, 
                       k: int = 5,
                       memory_types: Optional[List[str]] = None,
                       time_window: Optional[Tuple[datetime, datetime]] = None) -> List[Tuple[MemoryItem, float]]:
        """
        检索相关记忆
        
        Args:
            query: 查询文本
            k: 返回的Top-K数量
            memory_types: 要搜索的记忆类型（None表示全部）
            time_window: 时间范围（只检索此范围内的记忆）
        
        Returns:
            [(MemoryItem, 相似度分数)] 列表，按相似度降序排列
        """
        # 准备搜索空间
        if memory_types is None:
            all_memories = self.short_term_memory + self.long_term_memory
        else:
            all_memories = []
            if 'episodic' in memory_types:
                all_memories.extend(self.short_term_memory)
            if 'semantic' in memory_types:
                all_memories.extend(self.long_term_memory)
        
        # 时间过滤
        if time_window:
            start_time, end_time = time_window
            all_memories = [m for m in all_memories 
                          if start_time <= m.timestamp <= end_time]
        
        if not all_memories:
            return []
        
        # 获取query的embedding
        query_embedding = self._get_embedding(query)
        
        # 计算相似度（点积，因为已经归一化）
        similarities = []
        for memory in all_memories:
            if memory.embedding is not None:
                sim = np.dot(query_embedding, memory.embedding)
                # 应用时间衰减因子
                age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
                time_decay = np.exp(-age_hours / 24)  # 24小时半衰期
                # 综合分数：相似度 * 时间衰减 * 重要性
                final_score = sim * time_decay * memory.importance
                similarities.append((memory, final_score))
        
        # 排序并返回Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        self.stats['retrieval_count'] += 1
        return similarities[:k]
    
    def summarize_memory(self, token_limit: int = 1000) -> str:
        """
        压缩记忆，生成摘要
        
        Args:
            token_limit: 目标Token数
        
        Returns:
            压缩后的摘要文本
        """
        # 按重要性排序记忆
        all_memories = self.short_term_memory + self.long_term_memory
        all_memories.sort(key=lambda x: x.importance, reverse=True)
        
        # 估算当前Token数（简单估算：1 token ≈ 4字符）
        total_chars = sum(len(m.content) for m in all_memories)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= token_limit:
            # 不需要压缩
            return "\n".join([f"[{m.memory_type}] {m.content}" 
                            for m in all_memories])
        
        # 需要压缩
        # 策略1：移除不重要的记忆
        compressed_memories = []
        current_tokens = 0
        
        for memory in all_memories:
            tokens = len(memory.content) // 4
            if current_tokens + tokens <= token_limit:
                compressed_memories.append(memory)
                current_tokens += tokens
            else:
                break
        
        # 如果使用真实LLM，可以用LLM进一步压缩
        if not self.use_mock and self.client:
            try:
                summary_text = "\n".join([f"[{m.memory_type}] {m.content}" 
                                        for m in compressed_memories])
                prompt = f"""
                Please summarize the following memories concisely.
                Keep the most important information and key details.
                
                Memories:
                {summary_text}
                
                Summary:
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=token_limit // 2,
                    timeout=30
                )
                
                self.stats['compression_count'] += 1
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"⚠️ LLM compression failed: {e}, using simple truncation")
        
        # 简单截断
        summary = "\n".join([f"[{m.memory_type}] {m.content}" 
                          for m in compressed_memories])
        
        self.stats['compression_count'] += 1
        return summary
    
    def cleanup_old_memories(self, days: int = 7) -> int:
        """
        清理旧的记忆
        
        Args:
            days: 保留最近N天的记忆
        
        Returns:
            清理的记忆数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 清理短期记忆
        old_short = [m for m in self.short_term_memory 
                    if m.timestamp < cutoff_date]
        self.short_term_memory = [m for m in self.short_term_memory 
                                 if m.timestamp >= cutoff_date]
        
        # 清理长期记忆（保留高重要性的）
        old_long = [m for m in self.long_term_memory 
                   if m.timestamp < cutoff_date and m.importance < 0.5]
        self.long_term_memory = [m for m in self.long_term_memory 
                                if not (m.timestamp < cutoff_date and m.importance < 0.5)]
        
        cleaned_count = len(old_short) + len(old_long)
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['short_term_size'] = len(self.short_term_memory)
        stats['long_term_size'] = len(self.long_term_memory)
        return stats
    
    def export_memories(self, filepath: str) -> None:
        """导出记忆到文件"""
        all_memories = self.short_term_memory + self.long_term_memory
        export_data = []
        
        for memory in all_memories:
            export_data.append({
                'content': memory.content,
                'memory_type': memory.memory_type,
                'timestamp': memory.timestamp.isoformat(),
                'importance': memory.importance,
                'metadata': memory.metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(export_data)} memories to {filepath}")
    
    def import_memories(self, filepath: str) -> int:
        """从文件导入记忆"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            self.add_to_memory(
                content=item['content'],
                memory_type=item['memory_type'],
                importance=item['importance'],
                metadata=item['metadata']
            )
            count += 1
        
        print(f"✅ Imported {count} memories from {filepath}")
        return count

# ==========================================
# 实战演示
# ==========================================

if __name__ == "__main__":
    print("🧠 Memory Management System")
    print("=" * 60)
    
    # 初始化记忆管理器
    memory_manager = MemoryManager(
        short_term_limit=10,
        long_term_limit=50
    )
    
    # 添加一些情节记忆（Episodic）
    print("\n📝 Adding episodic memories...")
    memory_manager.add_to_memory(
        "User asked about Python syntax for loops",
        memory_type='episodic',
        importance=0.6,
        metadata={'topic': 'python', 'user_id': 'user123'}
    )
    
    memory_manager.add_to_memory(
        "User requested help with debugging a race condition",
        memory_type='episodic',
        importance=0.8,
        metadata={'topic': 'debugging', 'severity': 'high'}
    )
    
    memory_manager.add_to_memory(
        "User expressed interest in learning about distributed systems",
        memory_type='episodic',
        importance=0.7,
        metadata={'topic': 'distributed-systems'}
    )
    
    # 添加一些语义记忆（Semantic）
    print("\n📚 Adding semantic memories...")
    memory_manager.add_to_memory(
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        memory_type='semantic',
        importance=0.9,
        metadata={'domain': 'programming', 'language': 'python'}
    )
    
    memory_manager.add_to_memory(
        "A race condition occurs when multiple threads access shared data concurrently, leading to unpredictable behavior.",
        memory_type='semantic',
        importance=0.9,
        metadata={'domain': 'concurrency', 'concept': 'race-condition'}
    )
    
    memory_manager.add_to_memory(
        "Distributed systems are collections of independent computers that appear to users as a single coherent system.",
        memory_type='semantic',
        importance=0.85,
        metadata={'domain': 'distributed-systems'}
    )
    
    # 检索相关记忆
    print("\n🔍 Retrieving relevant memories for query: 'How do I debug race conditions?'")
    print("-" * 60)
    results = memory_manager.retrieve_memory("race condition debugging", k=3)
    
    for i, (memory, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Type: {memory.memory_type}")
        print(f"   Content: {memory.content}")
        print(f"   Importance: {memory.importance}")
    
    # 压缩记忆
    print("\n📦 Summarizing memories...")
    print("-" * 60)
    summary = memory_manager.summarize_memory(token_limit=500)
    print(summary)
    
    # 显示统计信息
    print("\n📊 Statistics:")
    print("-" * 60)
    stats = memory_manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 导出记忆
    print("\n💾 Exporting memories...")
    memory_manager.export_memories("week1_architecture/memories_export.json")
    
    print("\n✅ Demo complete!")
