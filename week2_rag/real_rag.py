import numpy as np
from typing import List, Tuple
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Error: sentence-transformers not installed.")
    print("Please run: pip install -r week2_rag/requirements.txt")
    exit(1)

# ==========================================
# Week 2: Real RAG with Semantic Embeddings
# 使用真正的 Embedding 模型将文本转换为向量
# ==========================================

class RealRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        model_name: 
        - 'all-MiniLM-L6-v2': 小快灵 (384 dim)，适合 CPU
        - 'all-mpnet-base-v2': 精度更高 (768 dim)
        - 'text-embedding-3-small': OpenAI 的模型 (需调用 API)
        """
        print(f"Loading embedding model '{model_name}'... (this may take a moment)")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.vectors = None
        self.input_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {self.input_dim}")
    
    def add_documents(self, docs: List[str]):
        self.documents = docs
        print(f"Encoding {len(docs)} documents...")
        
        # 真正地将文本转换为向量！
        # encode() 返回 numpy array
        self.vectors = self.model.encode(docs, normalize_embeddings=True)
        print("Encoding complete.")

    def dense_search(self, query: str, top_k=3) -> List[Tuple[str, float]]:
        # 1. 将 Query 也转换为向量
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        
        # 2. 计算 Cosine Similarity (因为已经 normalize 了，所以 dot product 等价于 cosine)
        scores = np.dot(self.vectors, query_vec)
        
        # 3. 排序
        # np.argsort 返回从小到大的索引，[::-1] 反转为从大到小
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

# ==========================================
# 实战演示：语义匹配的威力
# ==========================================
if __name__ == "__main__":
    # 这是一个充满"陷阱"的知识库
    # 包含了字面不相关但语义相关的句子
    knowledge_base = [
        "The quick brown fox jumps over the lazy dog.",  # 这里的 dog 和 puppy 语义相关
        "A puppy is a juvenile dog.",
        "Python is a high-level programming language.", # 这里的 Python 和 coding 相关
        "I love eating apples and bananas.",            # 水果
        "Machine learning focuses on data and algorithms.",
        "Google's headquarters is in Mountain View.",
    ]
    
    rag = RealRAG()
    rag.add_documents(knowledge_base)
    
    # Test 1: 同义词测试 (Semantics)
    # Query 里没有 "dog", 只有 "canine" (犬科动物)，看能否搜到 dog/puppy
    query1 = "tell me about small canines"
    print(f"\nQuery: '{query1}'")
    print("-" * 30)
    for doc, score in rag.dense_search(query1):
        print(f"[{score:.4f}] {doc}")
        
    # Test 2: 概念关联测试
    # Query 里没有 "Python"，只有 "coding"
    query2 = "tools for writing code script"
    print(f"\nQuery: '{query2}'")
    print("-" * 30)
    for doc, score in rag.dense_search(query2):
        print(f"[{score:.4f}] {doc}")
