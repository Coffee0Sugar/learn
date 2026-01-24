import numpy as np
from real_rag import RealRAG

# ==========================================
# Week 2 Extra: RAG Evaluation System
# Google 面试核心：如何科学地评估检索质量？
# ==========================================

class RAGEvaluator(RealRAG):
    def evaluate(self, test_set, k=3):
        """
        运行评估并打印指标
        test_set: List of (query, correct_doc_content)
        """
        print(f"\n🚀 Starting Evaluation on {len(test_set)} queries (k={k})...")
        
        hits = 0        # 命中的次数 (用于计算 Recall)
        reciprocal_ranks = [] # 倒数排名 (用于计算 MRR)
        
        for query, ground_truth in test_set:
            # 1. 运行检索
            results = self.dense_search(query, top_k=k)
            retrieved_docs = [doc for doc, score in results]
            
            # 2. 检查是否命中
            rank = -1
            for i, doc in enumerate(retrieved_docs):
                if doc == ground_truth:
                    rank = i + 1 # 排名从 1 开始
                    break
            
            # 3. 记录指标
            if rank > 0:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                print(f"✅ Hit! Rank: {rank} | Query: '{query}'")
            else:
                reciprocal_ranks.append(0.0)
                print(f"❌ Miss | Query: '{query}'")
                
        # 4. 计算最终分数
        recall_at_k = hits / len(test_set)
        mrr = np.mean(reciprocal_ranks)
        
        print("\n" + "="*40)
        print(f"📊 Evaluation Results (k={k}):")
        print(f"   Recall@{k}: {recall_at_k:.2%} (越高越好)")
        print(f"   MRR:       {mrr:.4f} (越高越好, max=1.0)")
        print("="*40 + "\n")

if __name__ == "__main__":
    # 1. 准备知识库
    rag = RAGEvaluator()
    docs = [
        "A puppy is a juvenile dog.",                 # Doc 0
        "The quick brown fox jumps over the lazy dog.", # Doc 1
        "Python is a high-level programming language.", # Doc 2
        "Machine learning focuses on data and algorithms.", # Doc 3
        "Google's headquarters is in Mountain View.",   # Doc 4
        "Deep learning is part of specific machine learning methods.", # Doc 5
    ]
    rag.add_documents(docs)
    
    # 2. 准备 "Golden Dataset" (测试集)
    # (Query, Ground Truth)
    test_set = [
        # 容易题：直接语义匹配
        ("tell me about small dogs", "A puppy is a juvenile dog."),
        
        # 中等题：跨语言/同义词
        ("what language is good for coding", "Python is a high-level programming language."),
        
        # 难题：干扰项 (Deep Learning vs Machine Learning)
        # 这是一个常见的 "Distractor" 陷阱
        ("neural networks training data", "Machine learning focuses on data and algorithms."),
        
        # 陷阱题：知识库里没有 (应该 Miss)
        ("how to bake a cake", "N/A"), 
    ]
    
    # 3. 运行评估
    rag.evaluate(test_set, k=3)
