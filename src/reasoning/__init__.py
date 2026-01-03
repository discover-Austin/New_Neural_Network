"""
Advanced Reasoning Modules
===========================

Research-grounded implementations of advanced reasoning strategies:

1. Chain-of-Thought (CoT)
   - Zero-Shot CoT: "Let's think step by step"
   - Few-Shot CoT: Learning from examples
   - Self-Consistency: Majority voting over multiple paths
   - Least-to-Most: Problem decomposition

2. Tree-of-Thoughts (ToT)
   - Explores multiple reasoning paths
   - Uses search algorithms (BFS, DFS, Best-First)
   - Evaluates and prunes states
   - Enables backtracking and look-ahead

All components cite peer-reviewed research:
- Wei et al., NeurIPS 2022 (Chain-of-Thought)
- Kojima et al., NeurIPS 2022 (Zero-Shot CoT)
- Wang et al., ICLR 2023 (Self-Consistency)
- Zhou et al., 2022 (Least-to-Most)
- Yao et al., NeurIPS 2023 (Tree-of-Thoughts)
"""

from .chain_of_thought import (
    CoTConfig,
    ZeroShotCoT,
    FewShotCoT,
    SelfConsistencyCoT,
    LeastToMostPrompting,
    ComplexCoTReasoner,
)

from .tree_of_thought import (
    ToTConfig,
    ThoughtNode,
    ThoughtGenerator,
    StateEvaluator,
    TreeOfThoughts,
)

__all__ = [
    # CoT
    "CoTConfig",
    "ZeroShotCoT",
    "FewShotCoT",
    "SelfConsistencyCoT",
    "LeastToMostPrompting",
    "ComplexCoTReasoner",
    # ToT
    "ToTConfig",
    "ThoughtNode",
    "ThoughtGenerator",
    "StateEvaluator",
    "TreeOfThoughts",
]
