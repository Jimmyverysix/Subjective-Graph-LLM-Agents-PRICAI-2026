"""LLM 智能体核心：消息生成、信任评估、贝叶斯更新。

- `LLMClient`：统一封装 OpenRouter chat/completions（支持并行/缓存/无 key 的 mock）
- `LLMAgent`：在“受限视野”的证据上下文中生成评估消息，并对收到消息进行信任门控与信念更新
"""
from __future__ import annotations

import os
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path

import numpy as np
import httpx

from .data_loader import StudentRecord, ClassData, DatasetSplit
from .subjective_graph import SubjectiveGraph, RashomonSet
from .rag import RAGEngine, RetrievalResult


@dataclass
class EvaluationMessage:
    """智能体对目标的评估消息。
    
    m_{i→j}^{(t,r)} = {ŝ, û, rationale}
    """
    sender_id: int  # 发送者
    target_id: int  # 被评估者
    ability_score: float  # ŝ ∈ [0, 1]
    uncertainty: float  # û ∈ [0, 1]
    rationale: str  # 理由（≤40字）
    timestamp: Tuple[int, int] = (0, 0)  # (epoch, round)
    
    def to_dict(self) -> Dict:
        return {
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "ability_score": self.ability_score,
            "uncertainty": self.uncertainty,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class TrustWeight:
    """可信度权重。
    
    ω_j(i→k; t,r) ∈ [0, 1]
    """
    receiver_id: int  # 接收者
    sender_id: int  # 发送者
    target_id: int  # 关于谁的消息
    trust: float  # 信任度
    uncertainty: float  # 不确定性
    
    def to_dict(self) -> Dict:
        return {
            "receiver_id": self.receiver_id,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "trust": self.trust,
            "uncertainty": self.uncertainty,
        }


@dataclass
class BeliefState:
    """智能体对某个同学能力的信念分布。
    
    B_{jk}^{(t)}(θ) = N(μ, σ)
    """
    target_id: int
    mu: float = 0.5  # 均值
    sigma: float = 0.3  # 标准差
    
    def update_bayesian(
        self,
        observation: float,
        obs_precision: float,
    ) -> "BeliefState":
        """贝叶斯更新（精度加权融合）。
        
        μ' = (σ^{-1}·μ + τ·ŝ) / (σ^{-1} + τ)
        σ' = 1 / (σ^{-1} + τ)
        """
        prior_precision = 1.0 / (self.sigma ** 2 + 1e-6)
        posterior_precision = prior_precision + obs_precision
        
        new_mu = (prior_precision * self.mu + obs_precision * observation) / posterior_precision
        new_sigma = 1.0 / np.sqrt(posterior_precision)
        
        return BeliefState(
            target_id=self.target_id,
            mu=np.clip(new_mu, 0, 1),
            sigma=np.clip(new_sigma, 0.01, 1.0),
        )
    
    def to_dict(self) -> Dict:
        return {
            "target_id": self.target_id,
            "mu": self.mu,
            "sigma": self.sigma,
        }


@dataclass
class AgentState:
    """智能体的完整状态。"""
    agent_id: int
    class_code: str
    
    # 对所有同学的能力信念 B_j = {B_{jk}}_{k∈V}
    beliefs: Dict[int, BeliefState] = field(default_factory=dict)
    
    # 当前时间点
    current_epoch: int = 0
    current_round: int = 0
    
    def get_belief(self, target_id: int) -> BeliefState:
        """获取对某同学的信念，不存在则初始化。"""
        if target_id not in self.beliefs:
            self.beliefs[target_id] = BeliefState(target_id=target_id)
        return self.beliefs[target_id]
    
    def update_belief(
        self,
        target_id: int,
        observation: float,
        precision: float,
    ):
        """更新对某同学的信念。"""
        old_belief = self.get_belief(target_id)
        self.beliefs[target_id] = old_belief.update_bayesian(observation, precision)
    
    def set_self_anchor(self, ability: float, sigma: float = 0.05):
        """设置自我真值锚点。
        
        μ_{jj}^{(t)} ← h(Y_j^{(t)}), σ_{jj}^{(t)} ← σ_self << 1
        """
        self.beliefs[self.agent_id] = BeliefState(
            target_id=self.agent_id,
            mu=ability,
            sigma=sigma,
        )
    
    def get_belief_summary(self) -> Dict[int, Tuple[float, float]]:
        """获取信念摘要：{target_id: (mu, sigma)}"""
        return {k: (v.mu, v.sigma) for k, v in self.beliefs.items()}


# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    """LLM API 客户端（支持 OpenRouter，支持并行调用）。"""
    
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        timeout: float = 60.0,
        max_retries: int = 6,
        cache_dir: Optional[Path] = None,
        max_workers: int = 50,  # 并行线程数
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        # 缓存目录
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计（线程安全）
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _cache_key(self, messages: List[Dict]) -> str:
        """生成缓存键。"""
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self, key: str) -> Optional[str]:
        """从缓存加载（线程安全）。"""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    with self._lock:
                        self.cache_hits += 1
                    return data.get("response")
            except (json.JSONDecodeError, IOError):
                return None
        return None
    
    def _save_cache(self, key: str, response: str):
        """保存到缓存（线程安全）。"""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"response": response}, f, ensure_ascii=False)
        except IOError:
            pass  # 忽略写入错误
    
    def _update_stats(self, input_tokens: int, output_tokens: int):
        """线程安全地更新统计。"""
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """调用 LLM API（线程安全）。"""
        # 检查缓存
        cache_key = self._cache_key(messages)
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        # 如果没有 API key，返回模拟响应
        if not self.api_key:
            return self._mock_response(messages)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # 更新统计（线程安全）
                    usage = data.get("usage", {})
                    self._update_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    
                    result = data["choices"][0]["message"]["content"]
                    
                    # 保存缓存
                    self._save_cache(cache_key, result)
                    
                    return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                # 最后一次尝试失败，返回模拟响应而非抛异常
                return self._mock_response(messages)
        
        return self._mock_response(messages)
    
    def chat_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """批量并行调用 LLM API。
        
        Args:
            messages_list: 多个消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Returns:
            响应列表（与输入顺序对应）
        """
        results = [None] * len(messages_list)
        
        def process_one(idx: int, messages: List[Dict[str, str]]) -> Tuple[int, str]:
            result = self.chat(messages, temperature, max_tokens)
            return idx, result
        
        futures = []
        for idx, messages in enumerate(messages_list):
            future = self._executor.submit(process_one, idx, messages)
            futures.append(future)
        
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
        
        return results
    
    def _mock_response(self, messages: List[Dict]) -> str:
        """模拟响应（用于测试/无 API key 时）。"""
        # 检测是评估任务还是信任评估任务
        system_msg = messages[0]["content"] if messages else ""
        
        if "ability_score" in system_msg:
            # 评估任务 - 返回模拟的 JSON
            return json.dumps({
                "evaluations": [
                    {
                        "target_id": 12345,
                        "ability_score": 0.6,
                        "uncertainty": 0.3,
                        "rationale": "基于有限信息的估计"
                    }
                ]
            }, ensure_ascii=False)
        elif "trust_weight" in system_msg:
            # 信任评估任务
            return json.dumps({
                "trust_weight": 0.7,
                "uncertainty": 0.2
            }, ensure_ascii=False)
        else:
            return "{}"
    
    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cache_hits": self.cache_hits,
            "estimated_cost_usd": (
                self.total_input_tokens * 0.02 / 1_000_000 +
                self.total_output_tokens * 0.03 / 1_000_000
            ),
        }


# ============================================================
# 提示词模板
# ============================================================

EVALUATION_SYSTEM_PROMPT = """你是班级社交环境中的学生智能体。你只能依据给定的【检索证据】与【自身性格/焦虑】做出判断。

你的任务是评估若干同学的学业能力。请输出严格 JSON，不要输出任何额外文本。

输出格式:
{
  "evaluations": [
    {
      "target_id": <目标学生ID>,
      "ability_score": <0-1之间的能力评分，1表示最强>,
      "uncertainty": <0-1之间的不确定性，1表示完全不确定>,
      "rationale": "<≤40字的评价理由>"
    }
  ]
}

注意:
- ability_score 应基于你对该同学学业表现的主观感知
- uncertainty 反映你对这个评估的确信程度
- 如果你对某同学了解很少，uncertainty 应该较高
- 理由要简洁，不超过40个字"""

EVALUATION_USER_TEMPLATE = """【你的信息】
- 学生ID: {agent_id}
- 性格: {personality}
- 焦虑水平: {worry_level}

【检索证据】
{rag_context}

【当前信念摘要】
{belief_summary}

【任务】
请评估以下同学的学业能力: {target_ids}

请只输出JSON，不要有其他内容。"""


TRUST_SYSTEM_PROMPT = """你是信息可信度评估器。根据发送者/接收者性格、上下文一致性，评估收到的评价信息的可信度。

输出格式（只输出JSON）:
{
  "trust_weight": <0-1之间的信任度，1表示完全信任>,
  "uncertainty": <0-1之间的不确定性>
}

评估依据:
- 发送者的性格特点（情绪稳定的人可能更可靠）
- 评价与你当前认知的一致性
- 评价的不确定性（高不确定性的评价可能需要打折）"""

TRUST_USER_TEMPLATE = """【你的信息】
- 学生ID: {receiver_id}
- 性格: {receiver_personality}

【收到的评价】
- 发送者ID: {sender_id}
- 发送者性格: {sender_personality}
- 关于: 学生{target_id}
- 能力评分: {ability_score}
- 不确定性: {sender_uncertainty}
- 理由: {rationale}

【你对学生{target_id}的当前认知】
- 能力均值: {current_mu:.2f}
- 不确定性: {current_sigma:.2f}

请只输出JSON，不要有其他内容。"""


# ============================================================
# Agent 类
# ============================================================

class LLMAgent:
    """LLM 驱动的学生智能体。"""
    
    def __init__(
        self,
        student: StudentRecord,
        subjective_graph: SubjectiveGraph,
        rag_engine: RAGEngine,
        llm_client: LLMClient,
        class_data: ClassData,
    ):
        self.student = student
        self.graph = subjective_graph
        self.rag = rag_engine
        self.llm = llm_client
        self.class_data = class_data
        
        # 初始化状态
        self.state = AgentState(
            agent_id=student.student_id,
            class_code=student.class_code,
        )
        
        # 初始化对所有同学的信念（先验）
        for sid in class_data.student_ids:
            self.state.beliefs[sid] = BeliefState(target_id=sid)
    
    @property
    def agent_id(self) -> int:
        return self.student.student_id
    
    def set_self_anchor(self, epoch: int):
        """根据当前考试成绩设置自我真值锚点。"""
        if epoch >= len(self.student.exam_class_ranks):
            return
        
        rank = self.student.exam_class_ranks[epoch]
        if np.isnan(rank):
            return
        
        # 将班级排名映射到 [0, 1] 的能力值（排名越低越强）
        n_students = self.class_data.size
        ability = 1.0 - (rank - 1) / max(n_students - 1, 1)
        
        self.state.set_self_anchor(ability, sigma=0.05)
    
    def generate_evaluations(
        self,
        target_ids: List[int],
        epoch: int,
        round_: int,
        use_llm_message: bool = True,
    ) -> List[EvaluationMessage]:
        """生成对多个目标的评估（批处理）。"""
        if not target_ids:
            return []

        # 消融：No-LLM-Message -> 用简单随机/先验生成消息（不调用 LLM）
        if not use_llm_message:
            return self._generate_heuristic_evaluations(target_ids, epoch, round_)
        
        # RAG 检索
        rag_result = self.rag.retrieve_for_evaluation(
            agent_id=self.agent_id,
            target_ids=target_ids,
            exam_idx=epoch,
        )
        
        # 构造提示词
        personality_str = self._format_personality()
        worry_map = {0: "从不担心", 1: "有时担心", 2: "经常担心", 3: "非常担心"}
        worry_str = worry_map.get(self.student.worry_level, "未知")
        
        belief_summary = self._format_belief_summary(target_ids)
        
        user_message = EVALUATION_USER_TEMPLATE.format(
            agent_id=self.agent_id,
            personality=personality_str,
            worry_level=worry_str,
            rag_context=rag_result.to_prompt_context(),
            belief_summary=belief_summary,
            target_ids=", ".join(map(str, target_ids)),
        )
        
        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        # 调用 LLM
        response = self.llm.chat(messages)
        
        # 解析响应
        evaluations = self._parse_evaluation_response(response, target_ids, epoch, round_)
        
        return evaluations

    def _generate_heuristic_evaluations(
        self,
        target_ids: List[int],
        epoch: int,
        round_: int,
    ) -> List[EvaluationMessage]:
        """在不调用 LLM 的情况下生成评估消息（用于消融）。"""
        # 用可复现的 RNG（避免多线程下使用全局 RNG 造成竞争）
        seed = (hash((self.agent_id, epoch, round_)) & 0xFFFFFFFF)
        rng = np.random.default_rng(seed)

        evaluations: List[EvaluationMessage] = []
        for tid in target_ids:
            belief = self.state.get_belief(tid)
            # 围绕当前信念均值做小幅扰动，避免完全常数
            ability = float(np.clip(belief.mu + rng.normal(0.0, 0.05), 0, 1))
            # 不确定性：与 belief.sigma 同量级，并略偏高
            unc = float(np.clip(min(1.0, belief.sigma + 0.15 + abs(rng.normal(0.0, 0.02))), 0, 1))
            evaluations.append(EvaluationMessage(
                sender_id=self.agent_id,
                target_id=tid,
                ability_score=ability,
                uncertainty=unc,
                rationale="（消融）非LLM消息生成",
                timestamp=(epoch, round_),
            ))
        return evaluations
    
    def evaluate_trust(
        self,
        message: EvaluationMessage,
        sender: StudentRecord,
    ) -> TrustWeight:
        """评估收到消息的可信度。"""
        # 获取当前对目标的信念
        belief = self.state.get_belief(message.target_id)
        
        # 构造提示词
        receiver_personality = self._format_personality()
        sender_personality = self._format_personality_for(sender)
        
        user_message = TRUST_USER_TEMPLATE.format(
            receiver_id=self.agent_id,
            receiver_personality=receiver_personality,
            sender_id=message.sender_id,
            sender_personality=sender_personality,
            target_id=message.target_id,
            ability_score=message.ability_score,
            sender_uncertainty=message.uncertainty,
            rationale=message.rationale,
            current_mu=belief.mu,
            current_sigma=belief.sigma,
        )
        
        messages = [
            {"role": "system", "content": TRUST_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        # 调用 LLM
        response = self.llm.chat(messages)
        
        # 解析响应
        trust = self._parse_trust_response(response, message)
        
        return trust
    
    def receive_and_update(
        self,
        message: EvaluationMessage,
        sender: StudentRecord,
        use_llm_trust: bool = True,
    ):
        """接收消息并更新信念。"""
        # 获取可信度权重
        if use_llm_trust:
            trust = self.evaluate_trust(message, sender)
            omega = trust.trust
        else:
            # 简单的一致性函数
            belief = self.state.get_belief(message.target_id)
            diff = abs(belief.mu - message.ability_score)
            omega = max(0.1, 1 - diff)
        
        # 计算观测精度
        # τ = φ(ω, û)
        precision = self._compute_precision(omega, message.uncertainty)
        
        # 贝叶斯更新
        self.state.update_belief(
            target_id=message.target_id,
            observation=message.ability_score,
            precision=precision,
        )
    
    def _compute_precision(self, trust: float, uncertainty: float) -> float:
        """计算观测精度。
        
        τ = φ(ω, û) = ω * (1 - û) / σ_base
        """
        sigma_base = 0.3  # 基准标准差
        precision = trust * (1 - uncertainty) / (sigma_base ** 2)
        return max(0.01, precision)
    
    def _format_personality(self) -> str:
        """格式化自己的性格特点。"""
        return self._format_personality_for(self.student)
    
    def _format_personality_for(self, student: StudentRecord) -> str:
        """格式化指定学生的性格特点。"""
        trait_map = {
            "extroverted_lively": "外向活泼",
            "introverted_quiet": "内向安静",
            "emotionally_stable": "情绪稳定",
            "emotionally_volatile": "情绪波动",
            "optimistic_positive": "乐观积极",
            "pessimistic": "悲观",
            "impulsive": "冲动",
            "thoughtful": "深思熟虑",
        }
        traits = []
        for key, chinese in trait_map.items():
            if student.personality.get(key, 0) == 1:
                traits.append(chinese)
        return "、".join(traits) if traits else "未知"
    
    def _format_belief_summary(self, target_ids: List[int]) -> str:
        """格式化对目标的当前信念摘要。"""
        lines = []
        for tid in target_ids:
            belief = self.state.get_belief(tid)
            lines.append(f"- 学生{tid}: 能力估计={belief.mu:.2f}, 不确定性={belief.sigma:.2f}")
        return "\n".join(lines) if lines else "无先前记录"
    
    def _parse_evaluation_response(
        self,
        response: str,
        target_ids: List[int],
        epoch: int,
        round_: int,
    ) -> List[EvaluationMessage]:
        """解析 LLM 的评估响应。"""
        evaluations = []
        
        try:
            # 尝试解析 JSON
            # 处理可能的 markdown 代码块
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            # 兼容 LLM 返回 [{...}] 格式
            if isinstance(data, list):
                data = {"evaluations": data}
            
            # 确保 data 是 dict
            if not isinstance(data, dict):
                data = {}
            
            for item in data.get("evaluations", []):
                target_id = item.get("target_id")
                if target_id not in target_ids:
                    continue
                
                evaluations.append(EvaluationMessage(
                    sender_id=self.agent_id,
                    target_id=target_id,
                    ability_score=np.clip(float(item.get("ability_score", 0.5)), 0, 1),
                    uncertainty=np.clip(float(item.get("uncertainty", 0.5)), 0, 1),
                    rationale=str(item.get("rationale", ""))[:40],
                    timestamp=(epoch, round_),
                ))
        except (json.JSONDecodeError, KeyError, TypeError):
            # 解析失败，为每个目标生成默认评估
            for tid in target_ids:
                evaluations.append(EvaluationMessage(
                    sender_id=self.agent_id,
                    target_id=tid,
                    ability_score=0.5,
                    uncertainty=0.8,  # 高不确定性
                    rationale="信息不足",
                    timestamp=(epoch, round_),
                ))
        
        return evaluations
    
    def _parse_trust_response(
        self,
        response: str,
        message: EvaluationMessage,
    ) -> TrustWeight:
        """解析 LLM 的信任评估响应。"""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            # 兼容 LLM 返回 [{...}] 格式
            if isinstance(data, list):
                data = data[0] if data else {}
            
            # 确保 data 是 dict
            if not isinstance(data, dict):
                data = {}
            
            return TrustWeight(
                receiver_id=self.agent_id,
                sender_id=message.sender_id,
                target_id=message.target_id,
                trust=np.clip(float(data.get("trust_weight", 0.5)), 0, 1),
                uncertainty=np.clip(float(data.get("uncertainty", 0.5)), 0, 1),
            )
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            # 默认信任度
            return TrustWeight(
                receiver_id=self.agent_id,
                sender_id=message.sender_id,
                target_id=message.target_id,
                trust=0.5,
                uncertainty=0.5,
            )


