"""
Tree-of-Thoughts (ToT) Reasoning
=================================

Research-grounded implementation of Tree-of-Thought reasoning:

Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
           (Yao et al., NeurIPS 2023)

Key Innovation:
- Extends Chain-of-Thought to explore multiple reasoning paths
- Maintains a tree of partial solutions
- Uses search algorithms (BFS, DFS) to explore tree
- Evaluates intermediate states
- Enables look-ahead and backtracking

Comparison to Chain-of-Thought:
- CoT: Linear reasoning path (one chain)
- ToT: Tree exploration (multiple branches)
- ToT can recover from mistakes via backtracking

Mathematical Foundation:
-----------------------

State Space:
  S = set of all possible reasoning states
  s_0 = initial state (problem statement)
  s_goal = goal state (solution)

Thought Generation:
  For state s, generate k candidate next thoughts:
  {t_1, ..., t_k} ~ P(thought|s)

State Evaluation:
  V(s) = value of state s (how promising)
  Estimated by LLM or heuristic

Search Algorithm:
  BFS: Explore all states at depth d before depth d+1
       - Complete: finds optimal solution
       - Memory: O(b^d) where b=branching factor

  DFS: Explore depth-first
       - Memory efficient: O(d)
       - May miss optimal solution

  Best-First: Expand most promising state
       - Uses V(s) to prioritize
       - Greedy but effective

Complexity:
- Time: O(b^d) in worst case
- Space: O(b^d) for BFS, O(d) for DFS
- In practice: Prune unpromising branches

Empirical Results (from paper):
- Game of 24: 74% success (vs 4% with CoT)
- Creative Writing: 41% preference (vs 21% with CoT)
- 5x5 Crossword: Solvable (vs unsolvable with CoT)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue
import copy


@dataclass
class ThoughtNode:
    """
    Node in the Tree of Thoughts.

    Represents a partial solution state.
    """
    state: str  # Current reasoning state
    depth: int  # Depth in tree
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    value: float = 0.0  # Evaluated value of this state
    is_terminal: bool = False  # Whether this is a solution
    path: List[str] = field(default_factory=list)  # Path from root

    def __lt__(self, other):
        """For priority queue (higher value = higher priority)."""
        return self.value > other.value


@dataclass
class ToTConfig:
    """Configuration for Tree of Thoughts."""
    # Thought generation
    num_thoughts_per_step: int = 3  # Branching factor
    max_depth: int = 5  # Maximum tree depth

    # Search algorithm
    search_algorithm: str = "bfs"  # "bfs", "dfs", "best_first"
    beam_width: int = 5  # For beam search variant

    # Evaluation
    evaluation_strategy: str = "llm"  # "llm", "heuristic", "vote"
    evaluation_prompt: str = "Evaluate the following reasoning step on a scale of 1-10:\n{thought}\n\nScore:"

    # Pruning
    prune_threshold: float = 3.0  # Minimum value to keep exploring
    max_nodes: int = 100  # Maximum nodes to explore

    # Termination
    early_termination: bool = True  # Stop when solution found
    min_solutions: int = 1  # Minimum solutions before terminating


class ThoughtGenerator:
    """
    Generates candidate thoughts from current state.

    Uses LLM to propose next reasoning steps.
    """

    def __init__(self, config: ToTConfig):
        self.config = config

    def generate_thoughts(
        self,
        state: str,
        problem: str,
        generate_fn: Callable[[str], str],
    ) -> List[str]:
        """
        Generate candidate next thoughts.

        Args:
            state: Current reasoning state
            problem: Original problem
            generate_fn: LLM generation function

        Returns:
            List of candidate thoughts
        """
        # Create prompt for thought generation
        if state:
            prompt = f"""Problem: {problem}

Current reasoning:
{state}

Propose {self.config.num_thoughts_per_step} different next steps to solve this problem:
1."""
        else:
            prompt = f"""Problem: {problem}

Propose {self.config.num_thoughts_per_step} different approaches to solve this problem:
1."""

        # Generate thoughts
        response = generate_fn(prompt)

        # Parse thoughts (assuming numbered list)
        thoughts = []
        for line in response.split('\n'):
            # Remove numbering
            if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                thought = line.split('.', 1)[1].strip()
                if thought:
                    thoughts.append(thought)

        # Ensure we have the right number
        thoughts = thoughts[:self.config.num_thoughts_per_step]

        # Fill with variations if not enough
        while len(thoughts) < self.config.num_thoughts_per_step:
            thoughts.append(f"Alternative approach {len(thoughts) + 1}")

        return thoughts


class StateEvaluator:
    """
    Evaluates the value/promise of a reasoning state.

    Uses LLM to score how promising a state is.
    """

    def __init__(self, config: ToTConfig):
        self.config = config

    def evaluate_llm(
        self,
        state: str,
        problem: str,
        generate_fn: Callable[[str], str],
    ) -> float:
        """
        Evaluate state using LLM.

        Args:
            state: Reasoning state to evaluate
            problem: Original problem
            generate_fn: LLM generation function

        Returns:
            Value score (0-10)
        """
        prompt = self.config.evaluation_prompt.format(thought=state)
        prompt = f"Problem: {problem}\n\n{prompt}"

        response = generate_fn(prompt)

        # Extract score
        try:
            # Look for number
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 10.0)  # Clamp to [0, 10]
        except:
            pass

        return 5.0  # Default middle value

    def evaluate_vote(
        self,
        state: str,
        problem: str,
        generate_fn: Callable[[str], str],
        num_votes: int = 3,
    ) -> float:
        """
        Evaluate state using voting.

        Args:
            state: Reasoning state
            problem: Original problem
            generate_fn: LLM generation function
            num_votes: Number of votes to aggregate

        Returns:
            Average vote score
        """
        scores = []

        for _ in range(num_votes):
            score = self.evaluate_llm(state, problem, generate_fn)
            scores.append(score)

        return sum(scores) / len(scores)

    def evaluate_heuristic(
        self,
        state: str,
        problem: str,
    ) -> float:
        """
        Evaluate state using heuristics.

        Args:
            state: Reasoning state
            problem: Original problem

        Returns:
            Heuristic score
        """
        # Simple heuristics
        score = 5.0

        # Longer reasoning might be more detailed
        if len(state) > 100:
            score += 1.0

        # Presence of numbers/calculations
        import re
        if re.search(r'\d+', state):
            score += 1.0

        # Presence of logical connectives
        logical_words = ['therefore', 'thus', 'hence', 'because', 'since']
        for word in logical_words:
            if word in state.lower():
                score += 0.5

        return min(score, 10.0)

    def evaluate(
        self,
        state: str,
        problem: str,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> float:
        """
        Evaluate state using configured strategy.

        Args:
            state: Reasoning state
            problem: Original problem
            generate_fn: LLM generation function (optional)

        Returns:
            Value score
        """
        if self.config.evaluation_strategy == "llm":
            if generate_fn is None:
                raise ValueError("generate_fn required for LLM evaluation")
            return self.evaluate_llm(state, problem, generate_fn)

        elif self.config.evaluation_strategy == "vote":
            if generate_fn is None:
                raise ValueError("generate_fn required for vote evaluation")
            return self.evaluate_vote(state, problem, generate_fn)

        elif self.config.evaluation_strategy == "heuristic":
            return self.evaluate_heuristic(state, problem)

        else:
            raise ValueError(f"Unknown evaluation strategy: {self.config.evaluation_strategy}")


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning system.

    Reference: Yao et al., NeurIPS 2023

    Maintains a search tree of reasoning states and explores
    it using search algorithms (BFS, DFS, Best-First).
    """

    def __init__(self, config: ToTConfig):
        self.config = config
        self.thought_generator = ThoughtGenerator(config)
        self.state_evaluator = StateEvaluator(config)

    def is_terminal(
        self,
        state: str,
        problem: str,
        check_fn: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """
        Check if state is a terminal solution.

        Args:
            state: Current state
            problem: Original problem
            check_fn: Optional function to check if state is solution

        Returns:
            Whether state is terminal
        """
        if check_fn is not None:
            return check_fn(state)

        # Default: Check for answer indication
        answer_keywords = ['answer is', 'answer:', 'final answer', 'solution is']
        return any(keyword in state.lower() for keyword in answer_keywords)

    def bfs_search(
        self,
        problem: str,
        generate_fn: Callable[[str], str],
        check_fn: Optional[Callable[[str], bool]] = None,
    ) -> List[ThoughtNode]:
        """
        Breadth-First Search.

        Explores all states at depth d before depth d+1.
        Guarantees finding solution at minimum depth.

        Args:
            problem: Problem to solve
            generate_fn: LLM generation function
            check_fn: Function to check if state is solution

        Returns:
            List of solution nodes
        """
        # Initialize
        root = ThoughtNode(state="", depth=0, path=[])
        queue = Queue()
        queue.put(root)

        solutions = []
        nodes_explored = 0

        while not queue.empty() and nodes_explored < self.config.max_nodes:
            # Get next node
            node = queue.get()
            nodes_explored += 1

            # Check if terminal
            if node.depth > 0 and self.is_terminal(node.state, problem, check_fn):
                solutions.append(node)
                if self.config.early_termination and len(solutions) >= self.config.min_solutions:
                    break
                continue

            # Check depth limit
            if node.depth >= self.config.max_depth:
                continue

            # Generate thoughts
            thoughts = self.thought_generator.generate_thoughts(
                node.state,
                problem,
                generate_fn,
            )

            # Expand node
            for thought in thoughts:
                # Create new state
                if node.state:
                    new_state = f"{node.state}\n{thought}"
                else:
                    new_state = thought

                # Evaluate state
                value = self.state_evaluator.evaluate(new_state, problem, generate_fn)

                # Prune low-value states
                if value < self.config.prune_threshold:
                    continue

                # Create child node
                child = ThoughtNode(
                    state=new_state,
                    depth=node.depth + 1,
                    parent=node,
                    value=value,
                    path=node.path + [thought],
                )
                node.children.append(child)
                queue.put(child)

        return solutions

    def dfs_search(
        self,
        problem: str,
        generate_fn: Callable[[str], str],
        check_fn: Optional[Callable[[str], bool]] = None,
    ) -> List[ThoughtNode]:
        """
        Depth-First Search.

        Explores depth-first with backtracking.
        Memory efficient but may not find optimal solution.

        Args:
            problem: Problem to solve
            generate_fn: LLM generation function
            check_fn: Function to check if state is solution

        Returns:
            List of solution nodes
        """
        solutions = []
        nodes_explored = [0]  # Use list for mutability in nested function

        def dfs_recursive(node: ThoughtNode):
            # Check termination
            if nodes_explored[0] >= self.config.max_nodes:
                return

            nodes_explored[0] += 1

            # Check if terminal
            if node.depth > 0 and self.is_terminal(node.state, problem, check_fn):
                solutions.append(node)
                if self.config.early_termination and len(solutions) >= self.config.min_solutions:
                    return
                return

            # Check depth limit
            if node.depth >= self.config.max_depth:
                return

            # Generate thoughts
            thoughts = self.thought_generator.generate_thoughts(
                node.state,
                problem,
                generate_fn,
            )

            # Explore each thought
            for thought in thoughts:
                # Create new state
                if node.state:
                    new_state = f"{node.state}\n{thought}"
                else:
                    new_state = thought

                # Evaluate state
                value = self.state_evaluator.evaluate(new_state, problem, generate_fn)

                # Prune low-value states
                if value < self.config.prune_threshold:
                    continue

                # Create child node
                child = ThoughtNode(
                    state=new_state,
                    depth=node.depth + 1,
                    parent=node,
                    value=value,
                    path=node.path + [thought],
                )
                node.children.append(child)

                # Recurse
                dfs_recursive(child)

                # Check if should stop
                if self.config.early_termination and len(solutions) >= self.config.min_solutions:
                    return

        # Start search
        root = ThoughtNode(state="", depth=0, path=[])
        dfs_recursive(root)

        return solutions

    def best_first_search(
        self,
        problem: str,
        generate_fn: Callable[[str], str],
        check_fn: Optional[Callable[[str], bool]] = None,
    ) -> List[ThoughtNode]:
        """
        Best-First Search.

        Expands most promising state first (greedy).
        Uses value function to prioritize.

        Args:
            problem: Problem to solve
            generate_fn: LLM generation function
            check_fn: Function to check if state is solution

        Returns:
            List of solution nodes
        """
        # Initialize
        root = ThoughtNode(state="", depth=0, path=[], value=10.0)
        pqueue = PriorityQueue()
        pqueue.put(root)

        solutions = []
        nodes_explored = 0

        while not pqueue.empty() and nodes_explored < self.config.max_nodes:
            # Get most promising node
            node = pqueue.get()
            nodes_explored += 1

            # Check if terminal
            if node.depth > 0 and self.is_terminal(node.state, problem, check_fn):
                solutions.append(node)
                if self.config.early_termination and len(solutions) >= self.config.min_solutions:
                    break
                continue

            # Check depth limit
            if node.depth >= self.config.max_depth:
                continue

            # Generate thoughts
            thoughts = self.thought_generator.generate_thoughts(
                node.state,
                problem,
                generate_fn,
            )

            # Expand node
            for thought in thoughts:
                # Create new state
                if node.state:
                    new_state = f"{node.state}\n{thought}"
                else:
                    new_state = thought

                # Evaluate state
                value = self.state_evaluator.evaluate(new_state, problem, generate_fn)

                # Prune low-value states
                if value < self.config.prune_threshold:
                    continue

                # Create child node
                child = ThoughtNode(
                    state=new_state,
                    depth=node.depth + 1,
                    parent=node,
                    value=value,
                    path=node.path + [thought],
                )
                node.children.append(child)
                pqueue.put(child)

        return solutions

    def solve(
        self,
        problem: str,
        generate_fn: Callable[[str], str],
        check_fn: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Solve problem using Tree of Thoughts.

        Args:
            problem: Problem to solve
            generate_fn: LLM generation function
            check_fn: Optional function to verify solutions

        Returns:
            Dictionary with solutions and search metadata
        """
        # Select search algorithm
        if self.config.search_algorithm == "bfs":
            solutions = self.bfs_search(problem, generate_fn, check_fn)
        elif self.config.search_algorithm == "dfs":
            solutions = self.dfs_search(problem, generate_fn, check_fn)
        elif self.config.search_algorithm == "best_first":
            solutions = self.best_first_search(problem, generate_fn, check_fn)
        else:
            raise ValueError(f"Unknown search algorithm: {self.config.search_algorithm}")

        # Sort solutions by value
        solutions.sort(key=lambda x: x.value, reverse=True)

        return {
            "problem": problem,
            "num_solutions": len(solutions),
            "solutions": [
                {
                    "state": sol.state,
                    "value": sol.value,
                    "depth": sol.depth,
                    "path": sol.path,
                }
                for sol in solutions
            ],
            "best_solution": solutions[0].state if solutions else None,
            "search_algorithm": self.config.search_algorithm,
        }
