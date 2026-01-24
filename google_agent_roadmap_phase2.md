# 🚀 Google 级 Agent 工程师进阶指南 (Phase 2 Deep Dive)

既然你的目标是 Google，我们需要从"会用框架"进阶到"理解原理、能造轮子、懂评估与安全"。Google 的面试和工程标准非常看重**底层原理 (First Principles)**、**可扩展性 (Scalability)** 和 **系统设计 (System Design)**。

这份指南假设你已经跳过了基础的 API 调用阶段，直接进入核心深水区。

> **当前工作区准备**:
> 我已经为你 Clone 了两个可以"当字典查"的核心仓库到当前目录 `AgentGuide` 和 `dive-into-llms`。

---

## 📅 Week 1: Agent 架构底层原理 (Beyond LangChain)

Google 工程师通常不依赖臃肿的框架，而是喜欢更轻量的抽象。你需要理解 Agent 思考的"原子级"操作。

### 🎯 核心目标
不要只用 LangChain 的 `ZeroShotAgent`，要能手写一个 ReAct 循环。

### 📚 学习资源
1.  **理论**: 阅读 `AgentGuide/docs/01-theory/` 下的架构文章。
2.  **代码**: `dive-into-llms/documents/chapter9/README.md` (GUI Agent)
    *   即使你不是做 GUI，也要看它如何处理 Observation 和 Action 的循环。

### 💻 实战任务 (Hardcore)
*   **任务**: 编写一个不到 100 行 Python 代码的 Agent 类。
*   **要求**:
    *   实现 `think() -> act() -> observe()` 循环。
    *   实现一个简单的 Tool Registry（工具注册表）。
    *   **禁止使用 LangChain**，只用 OpenAI/Anthropic 原生 API。
*   **思考题**: 如果 Tool 的输出超过了 Context Window 怎么办？(Google 面试常见 System Design 题)

---

## 📅 Week 2: Enterprise RAG (RAG at Scale)

Google 是搜索公司，RAG (Retrieval-Augmented Generation) 是重中之重。简单的 cosine similarity 检索在 Google 规模下远远不够。

### 🎯 核心目标
掌握 "Advanced RAG" & "GraphRAG"。

### 📚 学习资源
1.  **面试准备**: `AgentGuide/docs/04-interview/02-rag-questions.md`
    *   重点看：Hybrid Search, Re-ranking, Context Compression.
2.  **论文阅读**: `AgentGuide/resources/rag/papers/`
    *   阅读 **GraphRAG** 相关资料。理解知识图谱如何帮助"多跳推理" (Multi-hop reasoning)。

### 💻 实战任务
*   **任务**: 为你的 Agent 加上长短期记忆。
*   **挑战**:
    *   实现一个 Hybrid Search (关键词 BM25 + 向量检索)。
    *   或者：尝试复现 GraphRAG 的核心思想（提取 Entity -> 建图 -> 检索子图）。

---

## 📅 Week 3: Evaluation & Safety (The Google Differentiator)

这是区分"玩票"和"大厂工程师"的分水岭。Google 极度重视 AI 安全和可测量性。

### 🎯 核心目标
学会如何"给 AI 打分"以及如何"防止 AI 作恶"。

### 📚 学习资源
1.  **评估**: `AgentGuide/docs/04-interview/13-model-evaluation.md`
    *   核心概念: **LLM-as-a-Judge** (使用 GPT-4 评估小模型), **AgentBench**.
    *   思考: 怎么评估 Agent 的"规划能力"？
2.  **安全**: `dive-into-llms/documents/chapter10/README.md`
    *   了解 **Red Teaming** (红队测试), Prompt Injection (提示词注入), Jailbreaking (越狱)。

### 💻 实战任务
*   **任务**: 攻击你自己的 Agent。
    *   尝试写一个 "Prompt Injection" 让你的 Agent 泄露系统提示词。
    *   设计一套简单的评估集（Golden Dataset），计算你 Agent 的准确率（Pass@1）。

---

## 📅 Week 4: Alignment & RLHF (The Brain Surgery)

理解模型是如何"学会听话"的。虽然你可能不自己训练基座模型，但理解 RLHF 对调优 Agent 至关重要。

### 🎯 核心目标
理解 **SFT** (Supervised Fine-Tuning) vs **RLHF** (PPO/DPO).

### 📚 学习资源
1.  **实战教程 (强烈推荐)**: `dive-into-llms/documents/chapter11/`
    *   **阅读**: `README.md` (理解 PPO 流程图)
    *   **运行**: `RLHF.ipynb` (这是一个完整的 PPO 实战，微调 GPT-2 产生积极评论)。
2.  **关键概念**:
    *   Reward Model (奖励模型) 是什么？
    *   KL Divergence (KL散度) 在这里的作用是什么？(防止模型崩坏)

---

## 🏆 Google 面试模拟 (System Design)

在学完以上内容后，尝试回答以下 System Design 问题：

1.  **Design a Coding Agent**:
    *   如何处理代码执行的安全性？(Sandbox)
    *   如果代码运行死循环了怎么办？
    *   如何支持多文件项目？(Context Management)
2.  **Design a Customer Support Agent at Google Scale**:
    *   QPS 很高时，如何缓存 RAG 的结果？(Semantic Cache)
    *   如何保证 Agent 不会对用户辱骂？(Guardrails)
