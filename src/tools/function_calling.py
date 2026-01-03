"""
Function Calling and Tool Use
==============================

Research-grounded implementations for LLM tool use:

1. Function Calling
   - Reference: "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
   - Reference: "Gorilla: Large Language Model Connected with Massive APIs" (Patil et al., 2023)
   - Enables LLMs to call external functions/APIs

2. ReAct (Reasoning + Acting)
   - Reference: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., ICLR 2023)
   - Interleaves reasoning and tool use
   - Thought → Action → Observation loop

3. Tool Selection and Execution
   - Function schema matching
   - Parameter extraction
   - Result integration

Mathematical Foundation:
-----------------------

Function Calling:
  Given: User query Q, Available tools T = {t_1, ..., t_n}

  1. Tool Selection: P(t_i|Q) = softmax(score(Q, t_i))
  2. Parameter Extraction: args = extract_params(Q, schema(t_i))
  3. Execution: result = t_i(**args)
  4. Integration: response = LM(Q, result)

ReAct Loop:
  Loop:
    1. Thought: Generate reasoning step
    2. Action: Select and execute tool
    3. Observation: Receive tool output
    4. Update context
  Until: Final answer or max iterations

Complexity:
- Tool selection: O(n × d) where n=num tools, d=embedding dim
- Parameter extraction: O(m) where m=num parameters
- ReAct: O(k × (thought + action)) where k=num iterations

Empirical Results (from papers):
- Toolformer: +16.4% on math tasks with calculator
- ReAct: 34% → 69% on HotpotQA (+35%)
- Gorilla: 97.7% accuracy on API selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import json
import re
from abc import ABC, abstractmethod


@dataclass
class FunctionSchema:
    """Schema for a callable function."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # {param_name: {type, description, required}}
    returns: Dict[str, str]  # {type, description}
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolUseConfig:
    """Configuration for tool use."""
    max_iterations: int = 5  # For ReAct loop
    max_tools_per_call: int = 3  # Maximum tools to use in one call
    confidence_threshold: float = 0.7  # Minimum confidence for tool use
    use_react: bool = True  # Whether to use ReAct pattern


class Tool(ABC):
    """
    Abstract base class for tools.

    Tools are functions that LLMs can call.
    """

    def __init__(self, schema: FunctionSchema):
        self.schema = schema

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool output
        """
        pass

    def get_signature(self) -> str:
        """
        Get function signature as string.

        Returns:
            Signature for prompting
        """
        params = []
        for name, info in self.schema.parameters.items():
            param_type = info.get("type", "any")
            required = "*" if info.get("required", False) else ""
            params.append(f"{name}: {param_type}{required}")

        return f"{self.schema.name}({', '.join(params)}) -> {self.schema.returns.get('type', 'any')}"


class CalculatorTool(Tool):
    """
    Calculator tool for arithmetic operations.

    Reference: Used in Toolformer paper.
    """

    def __init__(self):
        schema = FunctionSchema(
            name="calculator",
            description="Performs arithmetic calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "required": True,
                }
            },
            returns={"type": "float", "description": "Result of calculation"},
            examples=[
                {"input": "2 + 2", "output": 4},
                {"input": "10 * 5", "output": 50},
                {"input": "100 / 4", "output": 25},
            ],
        )
        super().__init__(schema)

    def execute(self, expression: str) -> float:
        """Execute calculation."""
        try:
            # Safe eval (only allow basic arithmetic)
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")

            result = eval(expression)
            return float(result)
        except Exception as e:
            raise ValueError(f"Calculation error: {str(e)}")


class SearchTool(Tool):
    """
    Search tool for information retrieval.

    Reference: Used in ReAct and Toolformer.
    """

    def __init__(self, search_fn: Optional[Callable[[str], List[str]]] = None):
        schema = FunctionSchema(
            name="search",
            description="Search for information on a query",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True,
                },
                "num_results": {
                    "type": "int",
                    "description": "Number of results to return",
                    "required": False,
                },
            },
            returns={"type": "List[str]", "description": "Search results"},
            examples=[
                {"input": "capital of France", "output": ["Paris"]},
            ],
        )
        super().__init__(schema)
        self.search_fn = search_fn

    def execute(self, query: str, num_results: int = 5) -> List[str]:
        """Execute search."""
        if self.search_fn is not None:
            return self.search_fn(query)[:num_results]
        else:
            # Dummy implementation
            return [f"Search result {i+1} for: {query}" for i in range(num_results)]


class FunctionCallParser:
    """
    Parses function calls from LLM output.

    Supports multiple formats:
    1. JSON: {"name": "func", "args": {...}}
    2. Python-style: func(arg1=val1, arg2=val2)
    3. Natural language (with extraction)
    """

    def __init__(self):
        pass

    def parse_json_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON-formatted function call."""
        try:
            # Extract JSON object
            match = re.search(r'\{[^}]+\}', text)
            if match:
                call_dict = json.loads(match.group())
                if "name" in call_dict and "args" in call_dict:
                    return call_dict
        except:
            pass
        return None

    def parse_python_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse Python-style function call."""
        # Match: function_name(arg1=val1, arg2=val2)
        pattern = r'(\w+)\((.*?)\)'
        match = re.search(pattern, text)

        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = {}
            if args_str.strip():
                # Split by comma (naive, doesn't handle nested commas)
                for arg in args_str.split(','):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        args[key] = value

            return {"name": func_name, "args": args}

        return None

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from text.

        Args:
            text: LLM output containing function calls

        Returns:
            List of parsed function calls
        """
        calls = []

        # Try JSON format
        json_call = self.parse_json_call(text)
        if json_call:
            calls.append(json_call)

        # Try Python format
        python_call = self.parse_python_call(text)
        if python_call:
            calls.append(python_call)

        return calls


class ToolExecutor:
    """
    Executes tools and manages tool registry.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.schema.name] = tool

    def get_tool_descriptions(self) -> str:
        """
        Get descriptions of all available tools.

        Returns:
            Formatted tool descriptions for prompting
        """
        descriptions = []
        descriptions.append("Available tools:")

        for name, tool in self.tools.items():
            sig = tool.get_signature()
            desc = tool.schema.description
            descriptions.append(f"\n{name}: {desc}")
            descriptions.append(f"  Signature: {sig}")

            if tool.schema.examples:
                descriptions.append("  Examples:")
                for ex in tool.schema.examples[:2]:  # Show max 2 examples
                    descriptions.append(f"    {ex}")

        return "\n".join(descriptions)

    def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """
        Execute a tool.

        Args:
            tool_name: Name of tool to execute
            args: Tool arguments

        Returns:
            (success, result) tuple
        """
        if tool_name not in self.tools:
            return False, f"Unknown tool: {tool_name}"

        tool = self.tools[tool_name]

        try:
            result = tool.execute(**args)
            return True, result
        except Exception as e:
            return False, f"Tool execution error: {str(e)}"


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.

    Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
               (Yao et al., ICLR 2023)

    Pattern:
    Thought: [reasoning about what to do]
    Action: [tool to use and arguments]
    Observation: [tool output]
    ... (repeat)
    Thought: [final reasoning]
    Answer: [final response]

    Empirical Results (from paper):
    - HotpotQA: 34% → 69% (+35%)
    - FEVER: 56% → 72% (+16%)
    - Enables complex multi-step reasoning
    """

    def __init__(self, config: ToolUseConfig):
        self.config = config
        self.tool_executor = ToolExecutor()
        self.parser = FunctionCallParser()

    def create_react_prompt(
        self,
        question: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Create ReAct prompt.

        Args:
            question: User question
            history: Interaction history

        Returns:
            Prompt for LLM
        """
        prompt_parts = []

        # Tool descriptions
        prompt_parts.append(self.tool_executor.get_tool_descriptions())
        prompt_parts.append("")

        # Instructions
        prompt_parts.append("Answer the question using the following format:")
        prompt_parts.append("Thought: [your reasoning about what to do next]")
        prompt_parts.append("Action: tool_name(arg1=val1, arg2=val2)")
        prompt_parts.append("Observation: [tool output will be provided]")
        prompt_parts.append("... (repeat Thought/Action/Observation as needed)")
        prompt_parts.append("Thought: [final reasoning]")
        prompt_parts.append("Answer: [final answer to the question]")
        prompt_parts.append("")

        # Question
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("")

        # History
        for entry in history:
            if "thought" in entry:
                prompt_parts.append(f"Thought: {entry['thought']}")
            if "action" in entry:
                prompt_parts.append(f"Action: {entry['action']}")
            if "observation" in entry:
                prompt_parts.append(f"Observation: {entry['observation']}")

        # Next step
        if not history or "observation" in history[-1]:
            prompt_parts.append("Thought:")
        else:
            prompt_parts.append("Action:")

        return "\n".join(prompt_parts)

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse LLM response.

        Args:
            response: LLM output

        Returns:
            Dictionary with thought/action/answer
        """
        result = {}

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # Extract answer
        answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()

        return result

    def solve(
        self,
        question: str,
        generate_fn: Callable[[str], str],
    ) -> Dict[str, Any]:
        """
        Solve question using ReAct.

        Args:
            question: User question
            generate_fn: LLM generation function

        Returns:
            Dictionary with answer and interaction history
        """
        history = []

        for iteration in range(self.config.max_iterations):
            # Create prompt
            prompt = self.create_react_prompt(question, history)

            # Generate response
            response = generate_fn(prompt)

            # Parse response
            parsed = self.parse_response(response)

            # Add thought to history
            if "thought" in parsed:
                history.append({"thought": parsed["thought"]})

            # Check for final answer
            if "answer" in parsed:
                return {
                    "question": question,
                    "answer": parsed["answer"],
                    "history": history,
                    "iterations": iteration + 1,
                }

            # Execute action
            if "action" in parsed:
                history[-1]["action"] = parsed["action"]

                # Parse function call
                calls = self.parser.parse(parsed["action"])

                if calls:
                    call = calls[0]
                    success, result = self.tool_executor.execute_tool(
                        call["name"],
                        call["args"],
                    )

                    observation = str(result)
                    history[-1]["observation"] = observation
                else:
                    history[-1]["observation"] = "Error: Could not parse action"

        # Max iterations reached
        return {
            "question": question,
            "answer": "Max iterations reached without final answer",
            "history": history,
            "iterations": self.config.max_iterations,
        }


class ToolAugmentedLM(nn.Module):
    """
    Tool-Augmented Language Model.

    Reference: "Toolformer: Language Models Can Teach Themselves to Use Tools"
               (Schick et al., 2023)

    Approach:
    1. Detect when to use tools (via special tokens)
    2. Generate tool calls
    3. Integrate tool outputs into generation

    Training:
    - Self-supervised: LM annotates its own data with tool calls
    - Filter: Keep annotations that improve predictions
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: ToolUseConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tool_executor = ToolExecutor()

        # Special tokens for tool use
        self.tool_start_token = "<|tool|>"
        self.tool_end_token = "<|/tool|>"
        self.result_start_token = "<|result|>"
        self.result_end_token = "<|/result|>"

    def forward(
        self,
        input_ids: torch.Tensor,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with tool integration.

        Args:
            input_ids: Input token IDs
            tool_calls: Optional pre-specified tool calls
            **kwargs: Other model arguments

        Returns:
            Model outputs
        """
        # Standard forward pass
        outputs = self.base_model(input_ids, **kwargs)

        # TODO: In full implementation, would:
        # 1. Detect tool call tokens in input
        # 2. Execute tools
        # 3. Inject results into generation

        return outputs

    def generate_with_tools(
        self,
        prompt: str,
        tokenizer,
        max_length: int = 512,
    ) -> str:
        """
        Generate text with tool augmentation.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer
            max_length: Maximum generation length

        Returns:
            Generated text with tool results integrated
        """
        # TODO: Full implementation would:
        # 1. Generate text token-by-token
        # 2. Detect tool call tokens
        # 3. Pause generation, execute tool
        # 4. Insert result, continue generation

        # Simplified: Just do standard generation
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.base_model.generate(input_ids, max_length=max_length)
        return tokenizer.decode(outputs[0])
