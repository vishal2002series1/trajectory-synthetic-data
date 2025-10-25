"""
Decision Engine for CALL/ASK/ANSWER Logic

Determines whether LLM should:
- CALL: Request more tools/data
- ASK: Request user clarification
- ANSWER: Provide final response
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Three decision types from the mathematical framework."""
    CALL = "CALL"      # Need more tools/data
    ASK = "ASK"        # Need user clarification
    ANSWER = "ANSWER"  # Ready to provide answer


@dataclass
class Decision:
    """
    Represents a decision made by the LLM.
    
    δ ∈ {CALL, ASK, ANSWER}
    """
    type: DecisionType
    reasoning: str                    # Chain of thought (COT)
    tools: Optional[List[str]] = None  # For CALL: which tools to call
    clarification: Optional[str] = None  # For ASK: question to user
    answer: Optional[str] = None      # For ANSWER: final response
    
    def __str__(self) -> str:
        if self.type == DecisionType.CALL:
            return f"CALL({self.tools})"
        elif self.type == DecisionType.ASK:
            return f"ASK({self.clarification[:50]}...)"
        elif self.type == DecisionType.ANSWER:
            return f"ANSWER({self.answer[:50]}...)"


class DecisionEngine:
    """
    Determines what action the LLM should take at each iteration.
    
    Uses prompt-based approach to let LLM decide: CALL, ASK, or ANSWER
    """
    
    # Prompt templates
    ITERATION_0_PROMPT = """You are a financial advisor assistant helping users with portfolio and investment questions.

User Query: "{query}"

This is your first time seeing this query. You need to decide what to do:

1. **CALL** - Call tools to get information needed to answer
   - Use this when you need data from tools (portfolio info, market data, etc.)
   
2. **ASK** - Ask the user for clarification
   - Use this when the query is ambiguous or missing critical information
   - Example: "Which account?" if user has multiple accounts
   
3. **ANSWER** - Answer immediately without tools
   - Use this ONLY for general knowledge questions that don't require user-specific data
   - Example: "What is diversification?" - can answer without tools

Available Tools:
{tools_list}

Analyze the query and decide:
- Does it need user-specific data? → CALL
- Is it ambiguous or unclear? → ASK
- Can you answer with general knowledge? → ANSWER

Respond in this EXACT format:
DECISION: [CALL/ASK/ANSWER]
REASONING: [Explain your thought process in 1-2 sentences]
[If CALL] TOOLS: ["tool1", "tool2", ...]
[If ASK] CLARIFICATION: "Your question to the user"
[If ANSWER] RESPONSE: "Your complete answer"

Your decision:"""

    ITERATION_N_PROMPT = """You are a financial advisor assistant helping users with portfolio and investment questions.

Original User Query: "{query}"

You have already called tools and received data:
{context}

Now you need to decide what to do next:

1. **CALL** - Call additional tools if you need more information
   - Use this if the data you have is insufficient to answer
   
2. **ANSWER** - Provide the final answer
   - Use this if you have all the information needed

Available Tools (not yet called):
{remaining_tools}

Analyze what you have and decide:
- Do you have enough information to answer the user's question? → ANSWER
- Do you need more data? → CALL additional tools

Respond in this EXACT format:
DECISION: [CALL/ANSWER]
REASONING: [Explain your thought process in 1-2 sentences]
[If CALL] TOOLS: ["tool1", "tool2", ...]
[If ANSWER] RESPONSE: "Your complete answer based on the data"

Your decision:"""

    def __init__(self, bedrock_provider):
        """
        Initialize decision engine.
        
        Args:
            bedrock_provider: BedrockProvider instance for LLM calls
        """
        self.provider = bedrock_provider
        logger.info("Initialized DecisionEngine")
    
    def decide(
        self,
        query: str,
        context: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        iteration: int,
        max_iterations: int = 3
    ) -> Decision:
        """
        Determine what decision to make: CALL, ASK, or ANSWER.
        
        Args:
            query: Original user query
            context: Tool results from previous iterations
            available_tools: List of tools that can be called
            iteration: Current iteration number (0-based)
            max_iterations: Maximum allowed iterations
            
        Returns:
            Decision object with type and details
        """
        logger.info(f"Making decision for iteration {iteration}")
        
        # Special case: Force ANSWER at max iterations
        if iteration >= max_iterations - 1 and context:
            logger.warning(f"Max iterations reached. Forcing ANSWER.")
            return self._force_answer(query, context)
        
        # Choose prompt based on iteration
        if iteration == 0:
            prompt = self._build_iteration_0_prompt(query, available_tools)
        else:
            # Filter out already-called tools
            called_tools = self._extract_called_tools(context)
            remaining_tools = [t for t in available_tools if t["name"] not in called_tools]
            prompt = self._build_iteration_n_prompt(query, context, remaining_tools)
        
        # Call LLM to make decision
        try:
            response = self.provider.generate_text(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3  # Lower temp for more consistent decisions
            )
            
            # Parse response into Decision object
            decision = self._parse_decision(response, iteration, context)
            
            logger.info(f"Decision: {decision.type.value}")
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            # Fallback: CALL search_knowledge_base if iteration 0, else ANSWER
            if iteration == 0:
                return Decision(
                    type=DecisionType.CALL,
                    reasoning="Error occurred. Defaulting to search knowledge base.",
                    tools=["search_knowledge_base"]
                )
            else:
                return Decision(
                    type=DecisionType.ANSWER,
                    reasoning="Error occurred. Providing answer based on available context.",
                    answer="I apologize, but I encountered an error. Based on the available information, I cannot provide a complete answer at this time."
                )
    
    def _build_iteration_0_prompt(
        self,
        query: str,
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for iteration 0 (no context yet)."""
        # Format tools list
        tools_list = self._format_tools_list(available_tools)
        
        return self.ITERATION_0_PROMPT.format(
            query=query,
            tools_list=tools_list
        )
    
    def _build_iteration_n_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        remaining_tools: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for iteration N (with context)."""
        # Format context
        context_str = self._format_context(context)
        
        # Format remaining tools
        tools_list = self._format_tools_list(remaining_tools)
        
        return self.ITERATION_N_PROMPT.format(
            query=query,
            context=context_str,
            remaining_tools=tools_list
        )
    
    def _format_tools_list(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools into readable list."""
        lines = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) if lines else "No tools available"
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context (tool results) into readable text."""
        if not context:
            return "No data retrieved yet."
        
        lines = []
        for idx, result in enumerate(context, 1):
            tool = result.get("tool", "unknown")
            data = result.get("data", {})
            lines.append(f"{idx}. Tool '{tool}' returned:")
            lines.append(f"   {json.dumps(data, indent=2)}")
        
        return "\n".join(lines)
    
    def _extract_called_tools(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract names of tools that have already been called."""
        return [result.get("tool", "") for result in context]
    
    def _parse_decision(
        self,
        llm_response: str,
        iteration: int,
        context: List[Dict[str, Any]]
    ) -> Decision:
        """
        Parse LLM response into Decision object.
        
        Expected format:
        DECISION: CALL/ASK/ANSWER
        REASONING: ...
        [TOOLS/CLARIFICATION/RESPONSE]: ...
        """
        lines = llm_response.strip().split("\n")
        
        # Extract decision type
        decision_line = next((l for l in lines if l.startswith("DECISION:")), "")
        decision_type_str = decision_line.replace("DECISION:", "").strip().upper()
        
        # Map to DecisionType
        if "CALL" in decision_type_str:
            decision_type = DecisionType.CALL
        elif "ASK" in decision_type_str:
            decision_type = DecisionType.ASK
        elif "ANSWER" in decision_type_str:
            decision_type = DecisionType.ANSWER
        else:
            # Default based on iteration
            decision_type = DecisionType.CALL if iteration == 0 else DecisionType.ANSWER
            logger.warning(f"Could not parse decision type from: {decision_type_str}. Using {decision_type.value}")
        
        # Extract reasoning
        reasoning_line = next((l for l in lines if l.startswith("REASONING:")), "")
        reasoning = reasoning_line.replace("REASONING:", "").strip()
        
        # Extract type-specific content
        if decision_type == DecisionType.CALL:
            tools = self._extract_tools(llm_response)
            return Decision(
                type=DecisionType.CALL,
                reasoning=reasoning or "Need to retrieve data from tools.",
                tools=tools or ["search_knowledge_base"]  # Default to search
            )
            
        elif decision_type == DecisionType.ASK:
            clarification = self._extract_clarification(llm_response)
            return Decision(
                type=DecisionType.ASK,
                reasoning=reasoning or "Query is ambiguous and needs clarification.",
                clarification=clarification or "Could you please provide more details?"
            )
            
        elif decision_type == DecisionType.ANSWER:
            answer = self._extract_answer(llm_response, context)
            return Decision(
                type=DecisionType.ANSWER,
                reasoning=reasoning or "Have sufficient information to answer.",
                answer=answer or "Based on the available information, here is your answer."
            )
    
    def _extract_tools(self, llm_response: str) -> List[str]:
        """Extract tool names from TOOLS: line."""
        tools_line = next((l for l in llm_response.split("\n") if l.startswith("TOOLS:")), "")
        if not tools_line:
            return ["search_knowledge_base"]  # Default
        
        tools_str = tools_line.replace("TOOLS:", "").strip()
        
        try:
            # Try to parse as JSON array
            tools = json.loads(tools_str)
            return tools if isinstance(tools, list) else [tools]
        except:
            # Fallback: split by comma
            tools = [t.strip().strip('"\'[]') for t in tools_str.split(",")]
            return [t for t in tools if t]
    
    def _extract_clarification(self, llm_response: str) -> str:
        """Extract clarification question from CLARIFICATION: line."""
        clarification_line = next(
            (l for l in llm_response.split("\n") if l.startswith("CLARIFICATION:")), 
            ""
        )
        if clarification_line:
            return clarification_line.replace("CLARIFICATION:", "").strip().strip('"')
        return "Could you please clarify your question?"
    
    def _extract_answer(self, llm_response: str, context: List[Dict[str, Any]]) -> str:
        """Extract answer from RESPONSE: line."""
        # Find RESPONSE: line and everything after it
        lines = llm_response.split("\n")
        response_idx = next(
            (i for i, l in enumerate(lines) if l.startswith("RESPONSE:")),
            None
        )
        
        if response_idx is not None:
            # Join all lines from RESPONSE: onward
            answer_lines = lines[response_idx:]
            answer = "\n".join(answer_lines).replace("RESPONSE:", "").strip()
            return answer
        
        return "Based on the available information, I can provide an answer."
    
    def _force_answer(self, query: str, context: List[Dict[str, Any]]) -> Decision:
        """Force an ANSWER decision when max iterations reached."""
        # Use a simple prompt to generate answer from context
        context_str = self._format_context(context)
        
        prompt = f"""Based on the following data, answer this question: "{query}"

Data:
{context_str}

Provide a clear, concise answer:"""
        
        try:
            answer = self.provider.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.5
            )
        except Exception as e:
            logger.error(f"Error forcing answer: {e}")
            answer = "I apologize, but I cannot provide a complete answer based on the available information."
        
        return Decision(
            type=DecisionType.ANSWER,
            reasoning="Maximum iterations reached. Providing answer based on available data.",
            answer=answer
        )


# Example usage and testing
if __name__ == "__main__":
    from src.core.bedrock_provider import BedrockProvider
    from src.utils import load_config, setup_logger
    
    setup_logger("INFO")
    config = load_config()
    
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        region=config.bedrock.region
    )
    
    engine = DecisionEngine(provider)
    
    # Test: Iteration 0
    print("\n" + "="*80)
    print("TEST: ITERATION 0 DECISION")
    print("="*80)
    
    tools = [
        {
            "name": "search_knowledge_base",
            "description": "Search for relevant information in documents"
        },
        {
            "name": "get_allocation",
            "description": "Get user's portfolio allocation"
        }
    ]
    
    decision = engine.decide(
        query="What is my current portfolio allocation?",
        context=[],
        available_tools=tools,
        iteration=0,
        max_iterations=3
    )
    
    print(f"\nDecision: {decision}")
    print(f"Type: {decision.type.value}")
    print(f"Reasoning: {decision.reasoning}")
    if decision.tools:
        print(f"Tools: {decision.tools}")
    
    print("\n✅ DecisionEngine test complete!")