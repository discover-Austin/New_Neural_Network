"""
Tool Use and Function Calling
==============================

Research-grounded implementations for LLM tool use:

1. Function Calling
   - Function schema and parameter extraction
   - Tool execution and result integration

2. ReAct (Reasoning + Acting)
   - Thought → Action → Observation loop
   - Interleaves reasoning and tool use

3. Toolformer
   - Self-supervised tool use learning
   - Automatic tool call annotation

Citations:
- Toolformer (Schick et al., 2023)
- ReAct (Yao et al., ICLR 2023)
- Gorilla (Patil et al., 2023)
"""

from .function_calling import (
    FunctionSchema,
    ToolUseConfig,
    Tool,
    CalculatorTool,
    SearchTool,
    FunctionCallParser,
    ToolExecutor,
    ReActAgent,
    ToolAugmentedLM,
)

__all__ = [
    "FunctionSchema",
    "ToolUseConfig",
    "Tool",
    "CalculatorTool",
    "SearchTool",
    "FunctionCallParser",
    "ToolExecutor",
    "ReActAgent",
    "ToolAugmentedLM",
]
