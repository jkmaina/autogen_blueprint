"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 4: Mixture of Agents

Description:
Demonstrates heterogeneous agent architecture combining multiple agent types:
LLM-powered agents, rule-based agents, and human-in-the-loop agents working
together in a coordinated system with intelligent routing and fallback patterns.

Prerequisites:
- OpenAI API key set in .env file (optional for demo)
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Understanding of AutoGen Core messaging

Usage:
```bash
python -m chapter11.04_mixture_of_agents
```

Expected Output:
Mixture of agents demonstration:
1. Interactive query interface
2. Rule-based pattern matching for simple queries
3. LLM processing for complex queries
4. Human-in-the-loop for subjective questions
5. Intelligent routing and fallback mechanisms
6. Coordinated response aggregation

Key Concepts:
- Heterogeneous agent architecture
- Agent specialization and routing
- Rule-based vs LLM-powered processing
- Human-in-the-loop integration
- Fallback and escalation patterns
- Topic-based message coordination
- Interactive agent orchestration
- Multi-modal response generation

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Third-party imports
from autogen_core import (
    Agent,
    RoutedAgent,
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import SystemMessage, UserMessage

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from utils.config import get_openai_config
    HAS_OPENAI = True
except ImportError:
    print("OpenAI not available. Using dummy LLM agent.")
    HAS_OPENAI = False

# Define message types
@dataclass
class UserQuery:
    """A query from the user to the system."""
    content: str

@dataclass
class SystemResponse:
    """A response from the system to the user."""
    content: str
    source: str  # Which agent generated this response

@dataclass
class InternalMessage:
    """An internal message between agents."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HumanFeedbackRequest:
    """A request for human feedback."""
    query: str
    context: str

@dataclass
class HumanFeedback:
    """Feedback provided by a human."""
    content: str

# 1. Rule-based Agent
class RuleBasedAgent(RoutedAgent):
    """
    A rule-based agent that uses predefined patterns and rules to respond.
    """
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
        # Define patterns and responses
        self.patterns = [
            (r"(?i)hello|hi|hey", "Hello! I'm the rule-based agent. I can help with greetings and basic information."),
            (r"(?i)what is your name", "I'm a rule-based agent in the AutoGen framework."),
            (r"(?i)how are you", "I'm functioning as expected, thank you for asking!"),
            (r"(?i)bye|goodbye", "Goodbye! Have a great day!"),
            (r"(?i)thank you|thanks", "You're welcome!"),
            (r"(?i)help", "I can respond to basic greetings and questions. For more complex queries, I'll pass to other agents."),
        ]
        print(f"Rule-based agent {self.id.type} initialized")
    
    @message_handler
    async def handle_user_query(self, message: UserQuery, ctx: MessageContext) -> None:
        """Handle user queries using pattern matching."""
        query = message.content
        
        # Try to match the query against our patterns
        for pattern, response in self.patterns:
            if re.search(pattern, query):
                print(f"Rule-based agent matched pattern: {pattern}")
                # Send response back to user
                await self.publish_message(
                    SystemResponse(content=response, source=self.id.type),
                    topic_id=TopicId(type="user.response", source=self.id.key)
                )
                return
        
        # If no pattern matches, forward to the LLM agent
        print(f"Rule-based agent couldn't handle: '{query}'. Forwarding to LLM agent.")
        await self.publish_message(
            InternalMessage(
                content=query,
                metadata={"source": self.id.type, "requires": "llm_processing"}
            ),
            topic_id=TopicId(type="llm.query", source=self.id.key)
        )

# 2. LLM-powered Agent
class LLMAgent(RoutedAgent):
    """
    An agent powered by a Large Language Model.
    """
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
        
        # Initialize LLM client if available
        self.model_client = None
        if HAS_OPENAI:
            # Use configuration from utils.config instead of hardcoded values
            openai_config = get_openai_config()
            print(f"Using model: {openai_config.get('model', 'default')}")
            self.model_client = OpenAIChatCompletionClient(**openai_config)
        
        # System prompt for the LLM
        self.system_prompt = """
        You are a helpful AI assistant in a mixture-of-agents system.
        You handle complex queries that rule-based agents cannot process.
        Keep your responses concise, informative, and helpful.
        If you're unsure about something, indicate that you need human input.
        """
        
        print(f"LLM agent {self.id.type} initialized")
    
    @message_handler
    async def handle_internal_message(self, message: InternalMessage, ctx: MessageContext) -> None:
        """Handle messages forwarded from other agents."""
        query = message.content
        print(f"LLM agent processing: '{query}'")
        
        # Check if we need human input for this query
        if self._needs_human_input(query):
            print("LLM agent requesting human input")
            await self.publish_message(
                HumanFeedbackRequest(
                    query=query,
                    context="The LLM agent needs your input on this query."
                ),
                topic_id=TopicId(type="human.request", source=self.id.key)
            )
            return
        
        # Process with LLM
        response = await self._get_llm_response(query)
        
        # Send response back to user
        await self.publish_message(
            SystemResponse(content=response, source=self.id.type),
            topic_id=TopicId(type="user.response", source=self.id.key)
        )
    
    @message_handler
    async def handle_human_feedback(self, message: HumanFeedback, ctx: MessageContext) -> None:
        """Handle feedback received from the human agent."""
        print(f"LLM agent received human feedback: '{message.content}'")
        
        # Incorporate human feedback into response
        response = f"Based on human input: {message.content}"
        
        # Send response back to user
        await self.publish_message(
            SystemResponse(content=response, source=f"{self.id.type} (with human input)"),
            topic_id=TopicId(type="user.response", source=self.id.key)
        )
    
    def _needs_human_input(self, query: str) -> bool:
        """Determine if a query needs human input."""
        # Simple heuristic: check for specific keywords
        human_input_triggers = [
            "opinion", "judgment", "preference", "ethical", "moral", 
            "controversial", "personal", "subjective", "human input"
        ]
        return any(trigger in query.lower() for trigger in human_input_triggers)
    
    async def _get_llm_response(self, query: str) -> str:
        """Get a response from the LLM."""
        if self.model_client:
            try:
                # Use the actual LLM client with proper message types
                messages = [
                    SystemMessage(content=self.system_prompt, source="system"),
                    UserMessage(content=query, source="user")
                ]
                response = await self.model_client.create(messages=messages)
                return response.content
            except Exception as e:
                print(f"Error calling LLM: {e}")
                return f"I encountered an error processing your request: {str(e)}"
        else:
            # Dummy response for when OpenAI is not available
            return f"This is a simulated LLM response to: '{query}'. In a real implementation, this would use an actual language model."

# 3. Human-in-the-loop Agent
class HumanAgent(RoutedAgent):
    """
    An agent that represents a human in the loop, providing oversight and input.
    """
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
        print(f"Human agent {self.id.type} initialized")
    
    @message_handler
    async def handle_feedback_request(self, message: HumanFeedbackRequest, ctx: MessageContext) -> None:
        """Handle requests for human feedback."""
        query = message.query
        context = message.context
        
        print(f"\n--- Human Input Requested ---")
        print(f"Query: {query}")
        print(f"Context: {context}")
        
        # Get input from the human
        human_input = input("Please provide your input: ")
        
        # Send the human feedback back to the requesting agent
        await self.publish_message(
            HumanFeedback(content=human_input),
            topic_id=TopicId(type="llm.feedback", source=self.id.key)
        )

# 4. Coordinator Agent
class CoordinatorAgent(RoutedAgent):
    """
    A coordinator agent that manages the flow of messages between agents and the user.
    """
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
        print(f"Coordinator agent {self.id.type} initialized")
    
    @message_handler
    async def handle_user_query(self, message: UserQuery, ctx: MessageContext) -> None:
        """Handle initial user queries and route them appropriately."""
        query = message.content
        print(f"Coordinator received user query: '{query}'")
        
        # First, try the rule-based agent
        await self.publish_message(
            UserQuery(content=query),
            topic_id=TopicId(type="rule.query", source=self.id.key)
        )
    
    @message_handler
    async def handle_system_response(self, message: SystemResponse, ctx: MessageContext) -> None:
        """Handle responses from other agents and present them to the user."""
        response = message.content
        source = message.source
        
        print(f"\n--- Response from {source} ---")
        print(response)
        print("----------------------------\n")

async def main():
    """Main function to demonstrate heterogeneous agent architecture patterns."""
    print("\n=== Mixture of Agents Example ===\n")
    
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Register agents
    coordinator_type = await CoordinatorAgent.register(
        runtime,
        type="coordinator",
        factory=lambda: CoordinatorAgent("System Coordinator")
    )
    
    rule_agent_type = await RuleBasedAgent.register(
        runtime,
        type="rule_agent",
        factory=lambda: RuleBasedAgent("Rule-based Pattern Matcher")
    )
    
    llm_agent_type = await LLMAgent.register(
        runtime,
        type="llm_agent",
        factory=lambda: LLMAgent("LLM-powered Assistant")
    )
    
    human_agent_type = await HumanAgent.register(
        runtime,
        type="human_agent",
        factory=lambda: HumanAgent("Human-in-the-loop")
    )
    
    # Add subscriptions
    # Coordinator subscriptions
    await runtime.add_subscription(TypeSubscription(topic_type="user.query", agent_type=coordinator_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type="user.response", agent_type=coordinator_type.type))
    
    # Rule-based agent subscriptions
    await runtime.add_subscription(TypeSubscription(topic_type="rule.query", agent_type=rule_agent_type.type))
    
    # LLM agent subscriptions
    await runtime.add_subscription(TypeSubscription(topic_type="llm.query", agent_type=llm_agent_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type="llm.feedback", agent_type=llm_agent_type.type))
    
    # Human agent subscriptions
    await runtime.add_subscription(TypeSubscription(topic_type="human.request", agent_type=human_agent_type.type))
    
    # Start the runtime
    runtime.start()
    
    # Interactive loop for user queries
    print("Enter your queries (type 'exit' to quit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        if not user_input.strip():
            continue  # Skip empty inputs
        
        # Send user query to the coordinator
        await runtime.publish_message(
            UserQuery(content=user_input),
            topic_id=TopicId(type="user.query", source="user")
        )
        
        # Wait longer for processing and give visual feedback
        print("Processing...")
        await asyncio.sleep(3)  # Increase wait time to allow for LLM response
    
    # Clean up resources
    if HAS_OPENAI:
        llm_agent = await runtime.get_agent_by_type(llm_agent_type.type)
        if llm_agent and hasattr(llm_agent, "model_client"):
            await llm_agent.model_client.close()
    
    # Stop the runtime
    await runtime.stop_when_idle()
    print("\nRuntime stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMixture of agents demo interrupted by user")
    except Exception as e:
        print(f"Error running mixture of agents demo: {e}")
        raise
