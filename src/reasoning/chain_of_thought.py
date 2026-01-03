"""
Chain-of-Thought (CoT) Reasoning
=================================

Research-grounded implementations of Chain-of-Thought prompting and training:

1. Zero-Shot CoT
   - Reference: "Large Language Models are Zero-Shot Reasoners" (Kojima et al., NeurIPS 2022)
   - Prompt: "Let's think step by step"
   - No examples needed

2. Few-Shot CoT
   - Reference: "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., NeurIPS 2022)
   - Provides reasoning examples
   - Significantly improves complex reasoning

3. Self-Consistency
   - Reference: "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., ICLR 2023)
   - Sample multiple reasoning paths
   - Aggregate answers via majority vote

4. Least-to-Most Prompting
   - Reference: "Least-to-Most Prompting" (Zhou et al., 2022)
   - Decompose complex problems
   - Solve subproblems sequentially

Mathematical Foundation:
-----------------------

Standard Prompting:
  P(answer|question) = LM(question → answer)

Chain-of-Thought:
  P(answer|question) = Σ_r P(r|question) · P(answer|question, r)

  where r = reasoning chain

  Approximation:
  1. Generate reasoning: r ~ P(r|question)
  2. Extract answer: answer ~ P(answer|question, r)

Self-Consistency:
  1. Sample K reasoning paths: r_1, ..., r_K ~ P(r|question)
  2. Extract answers: a_i ~ P(answer|question, r_i)
  3. Majority vote: answer* = argmax_a count(a in {a_1, ..., a_K})

Complexity:
- Zero-Shot CoT: 1 generation pass (but longer sequence)
- Few-Shot CoT: 1 pass with longer prompt
- Self-Consistency: K generation passes
- Least-to-Most: D passes where D = decomposition depth

Empirical Results (from papers):
- Zero-Shot CoT: +17.7% on MultiArith (Kojima et al.)
- Few-Shot CoT: +78% on GSM8K (Wei et al.)
- Self-Consistency: +17.9% over CoT on GSM8K (Wang et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning."""
    # Zero-shot CoT trigger
    zero_shot_trigger: str = "Let's think step by step."

    # Few-shot settings
    num_examples: int = 8

    # Self-consistency settings
    num_samples: int = 5
    temperature: float = 0.7

    # Answer extraction
    answer_pattern: str = r"(?:answer is|answer:)\s*(.+?)(?:\.|$)"

    # Decomposition (Least-to-Most)
    decomposition_prompt: str = "To solve this problem, we need to first:"
    max_decomposition_depth: int = 5


class ZeroShotCoT:
    """
    Zero-Shot Chain-of-Thought.

    Reference: "Large Language Models are Zero-Shot Reasoners"
               (Kojima et al., NeurIPS 2022)

    Key Insight:
    - Simply adding "Let's think step by step" after the question
      triggers step-by-step reasoning in LLMs
    - No examples needed
    - Works across diverse reasoning tasks

    Algorithm:
    1. Append trigger phrase to question
    2. Generate reasoning chain
    3. Extract answer from reasoning

    Example:
    Q: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
        Each can has 3 tennis balls. How many tennis balls does he have now?"

    With trigger: "... How many tennis balls does he have now? Let's think step by step."

    LLM Output:
    "Roger started with 5 balls.
     2 cans × 3 balls/can = 6 balls.
     5 + 6 = 11 balls.
     Therefore, the answer is 11."
    """

    def __init__(self, config: CoTConfig):
        self.config = config

    def create_prompt(self, question: str) -> str:
        """
        Create zero-shot CoT prompt.

        Args:
            question: Input question

        Returns:
            Prompt with trigger phrase
        """
        return f"{question}\n{self.config.zero_shot_trigger}\n"

    def extract_answer(self, reasoning: str) -> str:
        """
        Extract final answer from reasoning chain.

        Args:
            reasoning: Generated reasoning text

        Returns:
            Extracted answer
        """
        # Look for answer patterns
        match = re.search(self.config.answer_pattern, reasoning, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Fallback: Use last sentence
        sentences = reasoning.split('.')
        if sentences:
            return sentences[-1].strip()

        return reasoning.strip()

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str], str],
    ) -> Dict[str, str]:
        """
        Solve question with zero-shot CoT.

        Args:
            question: Input question
            generate_fn: Function to generate text from prompt

        Returns:
            Dictionary with reasoning and answer
        """
        # Create prompt
        prompt = self.create_prompt(question)

        # Generate reasoning
        reasoning = generate_fn(prompt)

        # Extract answer
        answer = self.extract_answer(reasoning)

        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        }


class FewShotCoT:
    """
    Few-Shot Chain-of-Thought.

    Reference: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
               (Wei et al., NeurIPS 2022)

    Key Insight:
    - Providing examples of step-by-step reasoning in the prompt
      teaches the model to reason similarly
    - Dramatically improves performance on complex reasoning

    Algorithm:
    1. Construct prompt with exemplars showing reasoning
    2. Append target question
    3. Generate reasoning and extract answer

    Example Format:
    Q: <example question 1>
    A: <step-by-step reasoning 1> Therefore, the answer is <answer 1>.

    Q: <example question 2>
    A: <step-by-step reasoning 2> Therefore, the answer is <answer 2>.

    Q: <target question>
    A: [model generates this]
    """

    def __init__(self, config: CoTConfig):
        self.config = config
        self.exemplars: List[Dict[str, str]] = []

    def add_exemplar(
        self,
        question: str,
        reasoning: str,
        answer: str,
    ):
        """
        Add an exemplar to the few-shot prompt.

        Args:
            question: Example question
            reasoning: Step-by-step reasoning
            answer: Final answer
        """
        self.exemplars.append({
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        })

    def create_prompt(self, question: str) -> str:
        """
        Create few-shot CoT prompt.

        Args:
            question: Target question

        Returns:
            Prompt with exemplars and target question
        """
        prompt_parts = []

        # Add exemplars
        for ex in self.exemplars[:self.config.num_examples]:
            prompt_parts.append(f"Q: {ex['question']}")
            prompt_parts.append(f"A: {ex['reasoning']} Therefore, the answer is {ex['answer']}.")
            prompt_parts.append("")  # Blank line

        # Add target question
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A:")

        return "\n".join(prompt_parts)

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str], str],
    ) -> Dict[str, str]:
        """
        Solve question with few-shot CoT.

        Args:
            question: Input question
            generate_fn: Function to generate text from prompt

        Returns:
            Dictionary with reasoning and answer
        """
        # Create prompt
        prompt = self.create_prompt(question)

        # Generate reasoning
        reasoning = generate_fn(prompt)

        # Extract answer
        answer_pattern = r"(?:answer is|answer:)\s*(.+?)(?:\.|$)"
        match = re.search(answer_pattern, reasoning, re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
        else:
            answer = reasoning.split('.')[-1].strip()

        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
        }


class SelfConsistencyCoT:
    """
    Self-Consistency with Chain-of-Thought.

    Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
               (Wang et al., ICLR 2023)

    Key Insight:
    - Sample multiple reasoning paths
    - Different paths may reach different answers
    - Majority vote gives more reliable answer

    Algorithm:
    1. Generate K diverse reasoning paths (using temperature > 0)
    2. Extract answer from each path
    3. Take majority vote

    Empirical Results:
    - GSM8K: 74.4% → 83.7% (+9.3%) over standard CoT
    - SVAMP: 76.0% → 83.7% (+7.7%)
    - Arithmetic: 89.3% → 94.0% (+4.7%)

    Why it works:
    - Model may make mistakes in individual reasoning paths
    - Correct reasoning is more likely to be consistent
    - Voting aggregates evidence across paths
    """

    def __init__(self, config: CoTConfig):
        self.config = config

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str, float], str],
        base_cot: Optional[FewShotCoT] = None,
    ) -> Dict[str, any]:
        """
        Solve question with self-consistency.

        Args:
            question: Input question
            generate_fn: Function to generate text (takes prompt and temperature)
            base_cot: Base CoT method (few-shot or zero-shot)

        Returns:
            Dictionary with all reasoning paths and final answer
        """
        if base_cot is None:
            base_cot = ZeroShotCoT(self.config)

        # Generate multiple reasoning paths
        reasoning_paths = []
        answers = []

        for _ in range(self.config.num_samples):
            # Create prompt
            if isinstance(base_cot, FewShotCoT):
                prompt = base_cot.create_prompt(question)
            else:
                prompt = base_cot.create_prompt(question)

            # Generate with temperature for diversity
            reasoning = generate_fn(prompt, self.config.temperature)

            # Extract answer
            answer = base_cot.extract_answer(reasoning)

            reasoning_paths.append(reasoning)
            answers.append(answer)

        # Majority vote
        answer_counts = Counter(answers)
        final_answer, count = answer_counts.most_common(1)[0]

        return {
            "question": question,
            "reasoning_paths": reasoning_paths,
            "all_answers": answers,
            "answer_counts": dict(answer_counts),
            "final_answer": final_answer,
            "confidence": count / len(answers),
        }


class LeastToMostPrompting:
    """
    Least-to-Most Prompting.

    Reference: "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"
               (Zhou et al., 2022)

    Key Insight:
    - Decompose complex problems into simpler subproblems
    - Solve subproblems sequentially, building on previous solutions
    - Enables solving problems beyond few-shot capabilities

    Algorithm:
    1. Decomposition: Break down question into subquestions
    2. Sequential Solving: Solve each subquestion using previous answers
    3. Final Answer: Synthesize subquestion answers

    Example:
    Q: "Alice has 3 brothers. Each brother has 2 sisters. How many sisters does Alice have?"

    Decomposition:
    1. How many brothers does Alice have?
    2. How many sisters does each brother have?
    3. How many sisters does Alice have?

    Sequential Solving:
    1. Alice has 3 brothers.
    2. Each brother has 2 sisters.
    3. Since Alice is one of the sisters, Alice has 2 - 1 = 1 sister.

    Actually, this is a trick question - the correct answer is 1, not derived from 3 brothers.
    """

    def __init__(self, config: CoTConfig):
        self.config = config

    def decompose(
        self,
        question: str,
        generate_fn: Callable[[str], str],
    ) -> List[str]:
        """
        Decompose question into subquestions.

        Args:
            question: Complex question
            generate_fn: Function to generate text

        Returns:
            List of subquestions
        """
        decomposition_prompt = f"{question}\n\n{self.config.decomposition_prompt}\n"

        # Generate decomposition
        decomposition_text = generate_fn(decomposition_prompt)

        # Parse subquestions (assuming numbered list)
        subquestions = []
        for line in decomposition_text.split('\n'):
            # Match patterns like "1. ", "2. ", etc.
            match = re.match(r'\d+\.\s*(.+)', line)
            if match:
                subquestions.append(match.group(1).strip())

        return subquestions[:self.config.max_decomposition_depth]

    def solve_sequential(
        self,
        subquestions: List[str],
        generate_fn: Callable[[str], str],
        base_cot: Optional[FewShotCoT] = None,
    ) -> List[Dict[str, str]]:
        """
        Solve subquestions sequentially.

        Args:
            subquestions: List of subquestions
            generate_fn: Function to generate text
            base_cot: Base CoT method

        Returns:
            List of solutions for each subquestion
        """
        if base_cot is None:
            base_cot = ZeroShotCoT(self.config)

        solutions = []
        context = ""

        for i, subq in enumerate(subquestions):
            # Create prompt with previous context
            prompt_parts = []

            if context:
                prompt_parts.append("Previous steps:")
                prompt_parts.append(context)
                prompt_parts.append("")

            prompt_parts.append(f"Question: {subq}")

            if isinstance(base_cot, FewShotCoT):
                # Use few-shot format
                full_prompt = base_cot.create_prompt(subq)
                if context:
                    full_prompt = f"Context:\n{context}\n\n{full_prompt}"
            else:
                prompt_parts.append(base_cot.config.zero_shot_trigger)
                full_prompt = "\n".join(prompt_parts)

            # Generate reasoning
            reasoning = generate_fn(full_prompt)

            # Extract answer
            answer = base_cot.extract_answer(reasoning)

            solutions.append({
                "subquestion": subq,
                "reasoning": reasoning,
                "answer": answer,
            })

            # Update context for next subquestion
            context += f"\nStep {i+1}: {subq}\nAnswer: {answer}\n"

        return solutions

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str], str],
        base_cot: Optional[FewShotCoT] = None,
    ) -> Dict[str, any]:
        """
        Solve question with Least-to-Most prompting.

        Args:
            question: Complex question
            generate_fn: Function to generate text
            base_cot: Base CoT method

        Returns:
            Dictionary with decomposition, solutions, and final answer
        """
        # Step 1: Decompose
        subquestions = self.decompose(question, generate_fn)

        # Step 2: Solve sequentially
        solutions = self.solve_sequential(subquestions, generate_fn, base_cot)

        # Step 3: Final answer (from last subquestion)
        final_answer = solutions[-1]["answer"] if solutions else "Unable to solve"

        return {
            "question": question,
            "decomposition": subquestions,
            "solutions": solutions,
            "final_answer": final_answer,
        }


class ComplexCoTReasoner:
    """
    Combined Chain-of-Thought reasoning system.

    Integrates multiple CoT strategies for robust reasoning:
    1. Attempts zero-shot CoT
    2. Falls back to few-shot if available
    3. Uses self-consistency for critical questions
    4. Decomposes very complex questions
    """

    def __init__(self, config: CoTConfig):
        self.config = config

        self.zero_shot = ZeroShotCoT(config)
        self.few_shot = FewShotCoT(config)
        self.self_consistency = SelfConsistencyCoT(config)
        self.least_to_most = LeastToMostPrompting(config)

    def assess_complexity(self, question: str) -> str:
        """
        Assess question complexity.

        Returns:
            "simple", "medium", or "complex"
        """
        # Heuristics for complexity
        word_count = len(question.split())
        has_multiple_conditions = question.count('and') + question.count('if') > 1
        has_nested_structure = '(' in question or '[' in question

        if word_count > 50 or has_nested_structure:
            return "complex"
        elif word_count > 20 or has_multiple_conditions:
            return "medium"
        else:
            return "simple"

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str, Optional[float]], str],
        use_self_consistency: bool = False,
    ) -> Dict[str, any]:
        """
        Solve question with appropriate strategy.

        Args:
            question: Input question
            generate_fn: Function to generate text
            use_self_consistency: Whether to use self-consistency

        Returns:
            Dictionary with solution and metadata
        """
        complexity = self.assess_complexity(question)

        # Strategy selection
        if complexity == "simple":
            # Use zero-shot CoT
            result = self.zero_shot.solve(question, generate_fn)
            result["strategy"] = "zero_shot"

        elif complexity == "medium":
            # Use few-shot if examples available, else zero-shot
            if self.few_shot.exemplars:
                if use_self_consistency:
                    result = self.self_consistency.solve(
                        question,
                        generate_fn,
                        self.few_shot,
                    )
                    result["strategy"] = "few_shot_self_consistency"
                else:
                    result = self.few_shot.solve(question, generate_fn)
                    result["strategy"] = "few_shot"
            else:
                result = self.zero_shot.solve(question, generate_fn)
                result["strategy"] = "zero_shot"

        else:  # complex
            # Use Least-to-Most
            result = self.least_to_most.solve(
                question,
                generate_fn,
                self.few_shot if self.few_shot.exemplars else None,
            )
            result["strategy"] = "least_to_most"

        result["complexity"] = complexity
        return result
