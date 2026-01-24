import numpy as np
from typing import List, Dict, Any, Tuple

# ==========================================
# Week 2: RAG First Principles
# Google 考点：不使用 VectorDB (Milvus/Chroma) 
# 手写一个简单的 Hybrid Search (Dense + Sparse)
# ==========================================

class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.vectors = None
        self.vocab = set()
    
    def add_documents(self, docs: List[str]):
        """
        加载文档并构建简单的向量索引
        """
        self.documents = docs
        print(f"Loading {len(docs)} documents...")
        
        # 1. Sparse Embedding (模拟 BM25/TF-IDF)
        # 简单起见，我们用简单的词袋模型 (Bag of Words)
        for doc in docs:
            self.vocab.update(doc.split())
        self.vocab_list = sorted(list(self.vocab))
        self.word_to_id = {w: i for i, w in enumerate(self.vocab_list)}
        
        # 2. Dense Embedding (模拟 BERT/OpenAI Embedding)
        # 这里用随机向量模拟，实际生产中需调用 embedding 模型
        # Google 考点：Why 768 dimensions? (BERT-base hidden size)
        self.input_dim = 768 
        self.vectors = np.random.rand(len(docs), self.input_dim)
        
        # 归一化以便计算 Cosine Similarity
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]
        print("Index build complete.")

    def dense_search(self, query: str, top_k=3) -> List[Tuple[str, float]]:
        """
        向量检索 (Dense Retrieval)
        擅长：语义匹配 (Semantics)
        """
        # 模拟 Query Embedding
        query_vec = np.random.rand(self.input_dim)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # 计算相似度 (Dot Product)
        scores = np.dot(self.vectors, query_vec)
        
        # 获取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def keyword_search(self, query: str, top_k=3) -> List[Tuple[str, float]]:
        """
        关键词检索 (Keyword/Sparse Retrieval)
        擅长：精确匹配 (Exact Match)
        """
        query_words = set(query.split())
        scores = []
        for doc in self.documents:
            doc_words = set(doc.split())
            # Jaccard Similarity 简单模拟 BM25
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            score = intersection / union if union > 0 else 0
            scores.append(score)
            
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def hybrid_search(self, query: str, alpha=0.5, top_k=3):
        """
        混合检索 (Hybrid Search)
        Google 生产环境标配：Weighted Sum of Sparse + Dense
        """
        # 1. 获取两路召回结果
        dense_results = self.dense_search(query, top_k=len(self.documents))
        keyword_results = self.keyword_search(query, top_k=len(self.documents))
        
        # 2. 融合分数 (Reciprocal Rank Fusion RRF 或 简单加权)
        # 这里演示简单的加权融合
        doc_scores = {}
        
        # 建立 doc -> score 映射
        for doc, score in dense_results:
            doc_scores[doc] = doc_scores.get(doc, 0) + alpha * score
            
        for doc, score in keyword_results:
            doc_scores[doc] = doc_scores.get(doc, 0) + (1 - alpha) * score
            
        # 3. 排序返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

# ==========================================
# 实战演示
# ==========================================
if __name__ == "__main__":
    knowledge_base = [
        "Google is a technology company founded in 1998.",
        "Python is a popular programming language for AI.",
        "DeepMind is an AI research laboratory acquired by Google.",
        "The transformer architecture was introduced in the paper 'Attention Is All You Need'.",
        "Agentic workflows allow LLMs to use tools and reasoning."
    ]
    
    rag = SimpleRAG()
    rag.add_documents(knowledge_base)
    
    query = "Google AI research"
    
    print(f"\nQuery: {query}")
    print("-" * 30)
    
    print("\n[Dense Search Results] (Focus on semantics)")
    for doc, score in rag.dense_search(query):
        print(f"{score:.4f}: {doc}")
        
    print("\n[Keyword Search Results] (Focus on exact words)")
    for doc, score in rag.keyword_search(query):
        print(f"{score:.4f}: {doc}")
        
    print("\n[Hybrid Search Results] (Best of both worlds)")
    for doc, score in rag.hybrid_search(query, alpha=0.7):
        print(f"{score:.4f}: {doc}")
