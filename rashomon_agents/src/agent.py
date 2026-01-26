"""Core LLM agent: message generation, trust evaluation, and Bayesian updates.

- `LLMClient`: Unified wrapper for OpenRouter chat/completions (supports parallel calls, caching, and mock mode without API key)
- `LLMAgent`: Generates evaluation messages in limited-scope evidence context, and performs trust gating and belief updates on received messages
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
    """Agent's evaluation message for a target.
    
    m_{i→j}^{(t,r)} = {ŝ, û, rationale}
    """
    sender_id: int
    target_id: int
    ability_score: float  # ŝ ∈ [0, 1]
    uncertainty: float  # û ∈ [0, 1]
    rationale: str  # Rationale (≤40 characters)
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
    """Trust weight.
    
    ω_j(i→k; t,r) ∈ [0, 1]
    """
    receiver_id: int
    sender_id: int
    target_id: int
    trust: float
    uncertainty: float
    
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
    """Agent's belief distribution about a student's ability.
    
    B_{jk}^{(t)}(θ) = N(μ, σ)
    """
    target_id: int
    mu: float = 0.5  # Mean
    sigma: float = 0.3  # Standard deviation
    
    def update_bayesian(
        self,
        observation: float,
        obs_precision: float,
    ) -> "BeliefState":
        """Bayesian update (precision-weighted fusion).
        
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
    """Complete state of an agent."""
    agent_id: int
    class_code: str
    
    # Beliefs about all students' abilities B_j = {B_{jk}}_{k∈V}
    beliefs: Dict[int, BeliefState] = field(default_factory=dict)
    
    # Current time point
    current_epoch: int = 0
    current_round: int = 0
    
    def get_belief(self, target_id: int) -> BeliefState:
        """Get belief about a student, initialize if not exists."""
        if target_id not in self.beliefs:
            self.beliefs[target_id] = BeliefState(target_id=target_id)
        return self.beliefs[target_id]
    
    def update_belief(
        self,
        target_id: int,
        observation: float,
        precision: float,
    ):

        old_belief = self.get_belief(target_id)
        self.beliefs[target_id] = old_belief.update_bayesian(observation, precision)
    
    def set_self_anchor(self, ability: float, sigma: float = 0.05):

        self.beliefs[self.agent_id] = BeliefState(
            target_id=self.agent_id,
            mu=ability,
            sigma=sigma,
        )
    
    def get_belief_summary(self) -> Dict[int, Tuple[float, float]]:
        """Get belief summary: {target_id: (mu, sigma)}"""
        return {k: (v.mu, v.sigma) for k, v in self.beliefs.items()}


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """LLM API client (supports OpenRouter, parallel calls)."""
    
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
        max_workers: int = 50,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key."""
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self, key: str) -> Optional[str]:
        """Load from cache (thread-safe)."""
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
        """Save to cache (thread-safe)."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"response": response}, f, ensure_ascii=False)
        except IOError:
            pass  
    
    def _update_stats(self, input_tokens: int, output_tokens: int):
        """Update statistics (thread-safe)."""
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
        """Call LLM API (thread-safe)."""
        cache_key = self._cache_key(messages)
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        if not self.api_key:
            return self._mock_response(messages)
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
                    
                    usage = data.get("usage", {})
                    self._update_stats(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0)
                    )
                    
                    result = data["choices"][0]["message"]["content"]
                    self._save_cache(cache_key, result)
                    
                    return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return self._mock_response(messages)
        
        return self._mock_response(messages)
    
    def chat_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Batch parallel LLM API calls.
        
        Args:
            messages_list: List of message lists
            temperature: Temperature parameter
            max_tokens: Maximum tokens
        
        Returns:
            List of responses (in same order as input)
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
        """Mock response (for testing/without API key)."""
        system_msg = messages[0]["content"] if messages else ""
        
        if "ability_score" in system_msg:
            return json.dumps({
                "evaluations": [
                    {
                        "target_id": 12345,
                        "ability_score": 0.6,
                        "uncertainty": 0.3,
                        "rationale": "Estimate based on limited information"
                    }
                ]
            }, ensure_ascii=False)
        elif "trust_weight" in system_msg:
            return json.dumps({
                "trust_weight": 0.7,
                "uncertainty": 0.2
            }, ensure_ascii=False)
        else:
            return "{}"
    
    def get_stats(self) -> Dict:
        """Get statistics."""
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
# Prompt Templates
# ============================================================

EVALUATION_SYSTEM_PROMPT = """You are a student agent in a classroom social environment. You can only make judgments based on the given [Retrieval Evidence] and [Your Personality/Anxiety].

Your task is to evaluate the academic ability of several students. Output strict JSON only, no additional text.

Output format:
{
  "evaluations": [
    {
      "target_id": <target student ID>,
      "ability_score": <ability score between 0-1, 1 means strongest>,
      "uncertainty": <uncertainty between 0-1, 1 means completely uncertain>,
      "rationale": "<evaluation rationale ≤40 characters>"
    }
  ]
}

Notes:
- ability_score should be based on your subjective perception of the student's academic performance
- uncertainty reflects your confidence in this evaluation
- If you know little about a student, uncertainty should be higher
- Rationale should be concise, no more than 40 characters"""

EVALUATION_USER_TEMPLATE = """[Your Information]
- Student ID: {agent_id}
- Personality: {personality}
- Anxiety Level: {worry_level}

[Retrieval Evidence]
{rag_context}

[Current Belief Summary]
{belief_summary}

[Task]
Please evaluate the academic ability of the following students: {target_ids}

Output JSON only, no other content."""


TRUST_SYSTEM_PROMPT = """You are an information trustworthiness evaluator. Evaluate the trustworthiness of received evaluation messages based on sender/receiver personality and contextual consistency.

Output format (JSON only):
{
  "trust_weight": <trust between 0-1, 1 means fully trusted>,
  "uncertainty": <uncertainty between 0-1>
}

Evaluation criteria:
- Sender's personality traits (emotionally stable people may be more reliable)
- Consistency of the evaluation with your current beliefs
- Uncertainty of the evaluation (high uncertainty evaluations may need discounting)"""

TRUST_USER_TEMPLATE = """[Your Information]
- Student ID: {receiver_id}
- Personality: {receiver_personality}

[Received Evaluation]
- Sender ID: {sender_id}
- Sender Personality: {sender_personality}
- About: Student {target_id}
- Ability Score: {ability_score}
- Uncertainty: {sender_uncertainty}
- Rationale: {rationale}

[Your Current Belief about Student {target_id}]
- Ability Mean: {current_mu:.2f}
- Uncertainty: {current_sigma:.2f}

Output JSON only, no other content."""


# ============================================================
# Agent Class
# ============================================================

class LLMAgent:
    """LLM-driven student agent."""
    
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
        
        self.state = AgentState(
            agent_id=student.student_id,
            class_code=student.class_code,
        )
        
        for sid in class_data.student_ids:
            self.state.beliefs[sid] = BeliefState(target_id=sid)
    
    @property
    def agent_id(self) -> int:
        return self.student.student_id
    
    def set_self_anchor(self, epoch: int):
        """Set self ground-truth anchor based on current exam score."""
        if epoch >= len(self.student.exam_class_ranks):
            return
        
        rank = self.student.exam_class_ranks[epoch]
        if np.isnan(rank):
            return
        
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
        """Generate evaluations for multiple targets (batch processing)."""
        if not target_ids:
            return []

        if not use_llm_message:
            return self._generate_heuristic_evaluations(target_ids, epoch, round_)
        
        rag_result = self.rag.retrieve_for_evaluation(
            agent_id=self.agent_id,
            target_ids=target_ids,
            exam_idx=epoch,
        )
        
        personality_str = self._format_personality()
        worry_map = {0: "Never worry", 1: "Sometimes worry", 2: "Often worry", 3: "Very concerned"}
        worry_str = worry_map.get(self.student.worry_level, "Unknown")
        
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
        
        response = self.llm.chat(messages)
        evaluations = self._parse_evaluation_response(response, target_ids, epoch, round_)
        
        return evaluations

    def _generate_heuristic_evaluations(
        self,
        target_ids: List[int],
        epoch: int,
        round_: int,
    ) -> List[EvaluationMessage]:
        """Generate evaluation messages without calling LLM (for ablation)."""
        seed = (hash((self.agent_id, epoch, round_)) & 0xFFFFFFFF)
        rng = np.random.default_rng(seed)

        evaluations: List[EvaluationMessage] = []
        for tid in target_ids:
            belief = self.state.get_belief(tid)
            ability = float(np.clip(belief.mu + rng.normal(0.0, 0.05), 0, 1))
            unc = float(np.clip(min(1.0, belief.sigma + 0.15 + abs(rng.normal(0.0, 0.02))), 0, 1))
            evaluations.append(EvaluationMessage(
                sender_id=self.agent_id,
                target_id=tid,
                ability_score=ability,
                uncertainty=unc,
                rationale="(Ablation) Non-LLM message generation",
                timestamp=(epoch, round_),
            ))
        return evaluations
    
    def evaluate_trust(
        self,
        message: EvaluationMessage,
        sender: StudentRecord,
    ) -> TrustWeight:
        """Evaluate trustworthiness of received message."""
        belief = self.state.get_belief(message.target_id)
        
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
        
        response = self.llm.chat(messages)
        trust = self._parse_trust_response(response, message)
        
        return trust
    
    def receive_and_update(
        self,
        message: EvaluationMessage,
        sender: StudentRecord,
        use_llm_trust: bool = True,
    ):
        """Receive message and update belief."""
        if use_llm_trust:
            trust = self.evaluate_trust(message, sender)
            omega = trust.trust
        else:
            belief = self.state.get_belief(message.target_id)
            diff = abs(belief.mu - message.ability_score)
            omega = max(0.1, 1 - diff)
        
        precision = self._compute_precision(omega, message.uncertainty)
        
        self.state.update_belief(
            target_id=message.target_id,
            observation=message.ability_score,
            precision=precision,
        )
    
    def _compute_precision(self, trust: float, uncertainty: float) -> float:
        """Compute observation precision.
        
        τ = φ(ω, û) = ω * (1 - û) / σ_base
        """
        sigma_base = 0.3
        precision = trust * (1 - uncertainty) / (sigma_base ** 2)
        return max(0.01, precision)
    
    def _format_personality(self) -> str:
        """Format own personality traits."""
        return self._format_personality_for(self.student)
    
    def _format_personality_for(self, student: StudentRecord) -> str:
        """Format personality traits for a student."""
        trait_map = {
            "extroverted_lively": "Extroverted and lively",
            "introverted_quiet": "Introverted and quiet",
            "emotionally_stable": "Emotionally stable",
            "emotionally_volatile": "Emotionally volatile",
            "optimistic_positive": "Optimistic and positive",
            "pessimistic": "Pessimistic",
            "impulsive": "Impulsive",
            "thoughtful": "Thoughtful",
        }
        traits = []
        for key, english in trait_map.items():
            if student.personality.get(key, 0) == 1:
                traits.append(english)
        return ", ".join(traits) if traits else "Unknown"
    
    def _format_belief_summary(self, target_ids: List[int]) -> str:
        """Format current belief summary for targets."""
        lines = []
        for tid in target_ids:
            belief = self.state.get_belief(tid)
            lines.append(f"- Student {tid}: Ability estimate={belief.mu:.2f}, Uncertainty={belief.sigma:.2f}")
        return "\n".join(lines) if lines else "No previous records"
    
    def _parse_evaluation_response(
        self,
        response: str,
        target_ids: List[int],
        epoch: int,
        round_: int,
    ) -> List[EvaluationMessage]:
        """Parse LLM evaluation response."""
        evaluations = []
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            if isinstance(data, list):
                data = {"evaluations": data}
            
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
            for tid in target_ids:
                evaluations.append(EvaluationMessage(
                    sender_id=self.agent_id,
                    target_id=tid,
                    ability_score=0.5,
                    uncertainty=0.8,
                    rationale="Insufficient information",
                    timestamp=(epoch, round_),
                ))
        
        return evaluations
    
    def _parse_trust_response(
        self,
        response: str,
        message: EvaluationMessage,
    ) -> TrustWeight:
        """Parse LLM trust evaluation response."""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            if isinstance(data, list):
                data = data[0] if data else {}
            
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
            return TrustWeight(
                receiver_id=self.agent_id,
                sender_id=message.sender_id,
                target_id=message.target_id,
                trust=0.5,
                uncertainty=0.5,
            )


