"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 2: Agent Handoffs and Transfer System

Description:
Demonstrates sophisticated agent handoff patterns using AutoGen Core with
a multi-agent customer service system. Shows triage routing, specialized
agent workflows, tool integration, and human escalation patterns.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Understanding of AutoGen Core messaging

Usage:
```bash
python -m chapter11.02_handoffs
```

Expected Output:
Agent handoff system demonstration:
1. User initiates conversation through triage agent
2. Triage routes to specialized agents (Sales, Refund, Human)
3. Each agent performs specialized workflows with tools
4. Seamless handoffs between agents based on context
5. Human escalation when needed
6. Complete conversation flow with proper context passing

Key Concepts:
- Multi-agent orchestration patterns
- Topic-based message routing
- Agent specialization and handoff protocols
- Tool integration per agent role
- Context preservation across handoffs
- Human-in-the-loop escalation
- Event-driven agent communication
- Delegate tools vs regular tools

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

# Third-party imports
from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# ------------------------ Message Protocol Definition ------------------------
# Define message types used for event-driven communication between agents.

class UserLogin(BaseModel):
    """Event triggered when a user logs in to start a new session."""
    pass

class UserTask(BaseModel):
    """Event representing the user's current task or query (contains chat context)."""
    context: List[LLMMessage]

class AgentResponse(BaseModel):
    """Event for agents' replies to the user (contains updated chat context and a topic for the user reply)."""
    reply_to_topic_type: str   # which agent topic the user should reply to next
    context: List[LLMMessage]

# ------------------------ AI Agent Base Class ------------------------
class AIAgent(RoutedAgent):
    """
    Base class for AI agents (Triage, Sales, Refund). Uses an LLM to handle tasks.
    Subscribes to its own topic (agent_topic_type) and publishes responses to the user_topic_type.
    Can call regular tools for actions or delegate_tools to hand off tasks to other agents.
    """
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
    ):
        super().__init__(description)
        self._system_message = system_message           # System prompt for this agent's role/persona
        self._model_client = model_client               # LLM client (e.g., OpenAI API)
        self._tools = {tool.name: tool for tool in tools}  # Regular tools by name
        self._tool_schema = [tool.schema for tool in tools]  # JSON schemas for regular tools
        self._delegate_tools = {tool.name: tool for tool in delegate_tools}       # Delegate tools by name
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]     # JSON schemas for delegate tools
        self._agent_topic_type = agent_topic_type       # Topic type this agent listens to (its own type)
        self._user_topic_type = user_topic_type         # Topic type for sending messages to the user

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        """
        Handle an incoming user task (chat context). The agent generates a response using the LLM.
        If the response contains function calls:
          - For a regular tool, execute it and continue the conversation.
          - For a delegate tool (handoff), publish a UserTask to the appropriate agent's topic.
        If the response is a normal message, send it back to the user as AgentResponse.
        """
        # 1. Generate a response from the LLM, including possible tool calls.
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,  # allow both regular and delegate tools
            cancellation_token=ctx.cancellation_token,
        )
        print("-" * 80 + f"\n{self.id.type} (LLM):\n{llm_result.content}", flush=True)
        
        # 2. If the LLM returned function calls (could be multiple), handle them:
        while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []

            # Process each function call in sequence
            for call in llm_result.content:
                arguments = json.loads(call.arguments) if call.arguments else {}  # parse arguments for the function call
                if call.name in self._tools:
                    # Regular tool call: execute the tool and collect the result.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    result_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools:
                    # Delegate tool call: determine which agent/topic to hand off to.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                    target_topic = self._delegate_tools[call.name].return_value_as_string(result)
                    # Prepare a new UserTask message for the target agent, including the current conversation plus this handoff.
                    delegate_messages = list(message.context) + [
                        AssistantMessage(content=[call], source=self.id.type),               # include the function call as an assistant message
                        FunctionExecutionResultMessage(content=[FunctionExecutionResult(
                            call_id=call.id,
                            content=f"Transferred to {target_topic}. Adopt persona immediately.",
                            is_error=False,
                            name=call.name,
                        )])
                    ]
                    delegate_targets.append((target_topic, UserTask(context=delegate_messages)))
                else:
                    raise ValueError(f"Unknown tool called: {call.name}")

            # If any delegate handoffs were prepared, publish those tasks to the respective agent topics
            if delegate_targets:
                for topic_type, task in delegate_targets:
                    print("-" * 80 + f"\n{self.id.type}: Delegating to {topic_type}", flush=True)
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
                # Once handed off, this agent's work on the task is done.
                return

            # If we had regular tool results (no delegate), feed those results back into the LLM for a follow-up response.
            if tool_call_results:
                print("-" * 80 + f"\n{self.id.type} (Tool Results):\n{tool_call_results}", flush=True)
                # Extend the context with the tool call and its result before asking the LLM to continue.
                message.context.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    FunctionExecutionResultMessage(content=tool_call_results),
                ])
                llm_result = await self._model_client.create(
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    cancellation_token=ctx.cancellation_token,
                )
                print("-" * 80 + f"\n{self.id.type} (LLM follow-up):\n{llm_result.content}", flush=True)
            # Loop will continue if the new llm_result is again a list of FunctionCalls.
        # 3. If we exit the loop, the LLM result is a final answer (string). Send it back to the user.
        assert isinstance(llm_result.content, str)
        # Append the assistant's final answer to the context:
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
        # Publish the response as an AgentResponse event for the user, indicating which agent will handle the next user reply.
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
        # (The reply_to_topic_type tells the user agent which agent topic to send the next UserTask to.)

# ------------------------ Human Agent ------------------------
class HumanAgent(RoutedAgent):
    """
    Proxy for a human support agent. If an AI agent cannot handle the query, it hands off here.
    The HumanAgent subscribes to its own topic and will prompt a real human (via console input) for a response.
    """
    def __init__(self, description: str, agent_topic_type: str, user_topic_type: str):
        super().__init__(description)
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_user_task(self, message: UserTask, ctx: MessageContext) -> None:
        """
        Handle a task handed off to the human agent by prompting the user (human operator) for input.
        """
        # In a real app, this could notify a human via UI. Here we use console input for the demo.
        human_reply = input("👤 [Human agent input]: ")
        print("-" * 80 + f"\n{self.id.type} (Human reply):\n{human_reply}", flush=True)
        # Append the human's reply to the conversation context:
        message.context.append(AssistantMessage(content=human_reply, source=self.id.type))
        # Send the human's answer back to the user as an AgentResponse, directing future replies back to the originating AI agent.
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )

# ------------------------ User Agent ------------------------
class UserAgent(RoutedAgent):
    """
    Represents the end-user interface. It listens for AgentResponse events (from any agent) and prompts the user for input.
    It also initiates the session on UserLogin by asking the user’s first question.
    """
    def __init__(self, description: str, user_topic_type: str, agent_topic_type: str):
        super().__init__(description)
        self._user_topic_type = user_topic_type       # The topic type for user messages (usually "User")
        self._agent_topic_type = agent_topic_type     # The initial agent topic to send user queries to (e.g., triage agent)

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        """Handles a new user session start by prompting for the first user input."""
        print("-" * 80 + f"\n[Session started] User session ID: {self.id.key}", flush=True)
        # Prompt the user for their initial query upon login:
        user_input = input("🗣️ User: ")
        print("-" * 80 + f"\n{self.id.type} (UserAgent received):\n{user_input}")
        # Publish the user's query as a UserTask to the triage agent (or initial agent).
        await self.publish_message(
            UserTask(context=[UserMessage(content=user_input, source="User")]),
            topic_id=TopicId(self._agent_topic_type, source=self.id.key),
        )

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        """
        Receives an agent's response and prompts the user for the next input, then hands it off to the appropriate agent.
        """
        # Display the agent's response to the user and prompt for next input:
        next_input = input("🗣️ User (type 'exit' to quit): ")
        print("-" * 80 + f"\n{self.id.type} (UserAgent received):\n{next_input}", flush=True)
        if next_input.strip().lower() == "exit":
            print("-" * 80 + f"\n[Session ended] Session ID: {self.id.key}", flush=True)
            return  # End the session loop.
        # Append the new user message to context and route it to the agent that should handle the reply (based on reply_to_topic_type).
        message.context.append(UserMessage(content=next_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context),
            topic_id=TopicId(message.reply_to_topic_type, source=self.id.key),
        )

# ------------------------ Define Tools ------------------------
# Regular tools (functions) that agents can use to fulfill tasks without delegating.
def execute_order(product: str, price: int) -> str:
    """Process an order for a product at a given price (simulated with confirmation prompt)."""
    print("\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=====================\n")
    confirm = input("✅ Confirm order? (y/n): ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled.")
        return "User cancelled order."

def look_up_item(search_query: str) -> str:
    """Lookup an item ID based on a search query (simulated)."""
    item_id = "item_132612938"
    print(f"🔍 Found item ID: {item_id}")
    return item_id

def execute_refund(item_id: str, reason: str = "not provided") -> str:
    """Process a refund for a given item ID (simulated)."""
    print("\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("======================\n")
    print("Refund execution successful!")
    return "success"

# Wrap these functions as FunctionTool objects so agents can call them.
execute_order_tool = FunctionTool(execute_order, description="Execute a product order (confirmation required).")
look_up_item_tool = FunctionTool(look_up_item, description="Lookup an item ID by a search query (for refunds/sales).")
execute_refund_tool = FunctionTool(execute_refund, description="Execute a refund for a given item ID and reason.")

# ------------------------ Define Delegate Tools (for Handoff) ------------------------
# Delegate tools are used by agents to hand off the conversation to another agent.
# Each returns the topic type string of the target agent, which triggers a UserTask to that agent.

# Topic type names for each specialized agent:
triage_agent_topic_type = "TriageAgent"
refund_agent_topic_type = "RefundAgent"       # handles refund requests (issues & repairs)
sales_agent_topic_type = "SalesAgent"         # handles sales requests
human_agent_topic_type = "HumanAgent"         # handles escalations to human
user_topic_type = "User"                     # topic type for the user (customer interface)

def transfer_to_sales_agent() -> str:
    """Handoff tool: signal that the SalesAgent should handle the next part of the conversation."""
    return sales_agent_topic_type

def transfer_to_refund_agent() -> str:
    """Handoff tool: signal that the RefundAgent (issues/support) should handle the next part."""
    return refund_agent_topic_type

def transfer_back_to_triage() -> str:
    """Handoff tool: return control to the TriageAgent (if topic goes out of scope or needs escalation)."""
    return triage_agent_topic_type

def escalate_to_human() -> str:
    """Handoff tool: escalate the conversation to a human agent."""
    return human_agent_topic_type

# Wrap delegate functions as tools (no arguments needed for these).
transfer_to_sales_agent_tool = FunctionTool(transfer_to_sales_agent, description="Use for sales or buying-related inquiries.")
transfer_to_refund_agent_tool = FunctionTool(transfer_to_refund_agent, description="Use for issues, repairs, or refund requests.")
transfer_back_to_triage_tool = FunctionTool(transfer_back_to_triage, description="Use if the query falls outside your scope; hands back to triage (or escalate).")
escalate_to_human_tool = FunctionTool(escalate_to_human, description="Use to escalate to a human agent (for complex or requested cases).")

# ------------------------ Assemble and Run the Agent Team ------------------------
async def main():
    """Main function to demonstrate agent handoff patterns and workflows."""
    runtime = SingleThreadedAgentRuntime()

    # Initialize the model client using configuration
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)

    # Register and configure each agent in the runtime:

    # 1. Triage Agent – first contact point that decides where to route queries.
    triage_agent_type = await AIAgent.register(
        runtime,
        type=triage_agent_topic_type,  # Using topic type string as agent type identifier.
        factory=lambda: AIAgent(
            description="Triage Agent: directs user requests to the appropriate department.",
            system_message=SystemMessage(content=(
                "You are a customer service triage bot for ACME Inc. "
                "Your goal is to briefly greet the customer and figure out their need. "
                "Decide whether their request is about a refund/issue or a new product purchase, or needs human help. "
                "Use the appropriate handoff tool: 'transfer_to_refund_agent', 'transfer_to_sales_agent', or 'escalate_to_human'."
            )),
            model_client=model_client,
            tools=[],  # Triage agent doesn't perform actions itself, only delegates.
            delegate_tools=[transfer_to_refund_agent_tool, transfer_to_sales_agent_tool, escalate_to_human_tool],
            agent_topic_type=triage_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    # Subscribe the triage agent to messages of type "TriageAgent" (its own topic).
    await runtime.add_subscription(TypeSubscription(topic_type=triage_agent_topic_type, agent_type=triage_agent_type.type))

    # 2. Sales Agent – handles product purchase conversations.
    sales_agent_type = await AIAgent.register(
        runtime,
        type=sales_agent_topic_type,
        factory=lambda: AIAgent(
            description="Sales Agent: assists with product inquiries and orders.",
            system_message=SystemMessage(content=(
                "You are a sales agent for ACME Inc. Answer very briefly. "
                "Follow this workflow with the user:\n"
                "1. Ask about any problem the user has (especially related to catching roadrunners).\n"
                "2. Casually mention a fictional ACME product that could help (don't mention price yet).\n"
                "3. If the user seems interested, reveal a ridiculous price for the product.\n"
                "4. If the user agrees, finalize the sale by executing their order (use execute_order tool).\n"
                "If the conversation goes off sales topic or they ask for something else, use 'transfer_back_to_triage'."
            )),
            model_client=model_client,
            tools=[execute_order_tool],             # Sales agent can execute orders.
            delegate_tools=[transfer_back_to_triage_tool],  # Can hand back to triage if needed.
            agent_topic_type=sales_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=sales_agent_topic_type, agent_type=sales_agent_type.type))

    # 3. Refund Agent (Issues & Repairs Agent) – handles customer support issues and refunds.
    refund_agent_type = await AIAgent.register(
        runtime,
        type=refund_agent_topic_type,
        factory=lambda: AIAgent(
            description="Refund Agent: handles support issues and refund requests.",
            system_message=SystemMessage(content=(
                "You are a customer support agent for ACME Inc (issues and refunds). Answer very briefly. "
                "Follow this workflow:\n"
                "1. If the user hasn't explained the issue, ask a probing question to understand the problem.\n"
                "2. Propose a possible fix or solution first (make something up if needed).\n"
                "3. If the user is not satisfied with the fix, offer a refund.\n"
                "4. If refund accepted, find the item ID (use look_up_item tool) and then process refund (use execute_refund tool)."
            )),
            model_client=model_client,
            tools=[look_up_item_tool, execute_refund_tool],  # Refund agent can look up items and execute refunds.
            delegate_tools=[transfer_back_to_triage_tool],   # Can hand back to triage if issue goes beyond its scope.
            agent_topic_type=refund_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=refund_agent_topic_type, agent_type=refund_agent_type.type))

    # 4. Human Agent – handles escalated conversations that AI agents can’t resolve.
    human_agent_type = await HumanAgent.register(
        runtime,
        type=human_agent_topic_type,
        factory=lambda: HumanAgent(
            description="Human Agent: a human operator for complex queries.",
            agent_topic_type=human_agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=human_agent_topic_type, agent_type=human_agent_type.type))

    # 5. User Agent – simulates the end-user interface connecting to the chatbot.
    #    It starts the interaction by sending a UserTask to the triage agent and handles agent responses.
    user_agent_type = await UserAgent.register(
        runtime,
        type=user_topic_type,
        factory=lambda: UserAgent(
            description="User Agent: interface for the customer.",
            user_topic_type=user_topic_type,
            agent_topic_type=triage_agent_topic_type,  # initial handoff goes to triage agent
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))

    # Start the runtime loop and initiate a new user session:
    runtime.start()
    session_id = str(uuid.uuid4())  # unique session ID for this conversation
    # Publish a UserLogin event to start the session. This triggers the UserAgent to ask for user input.
    await runtime.publish_message(UserLogin(), topic_id=TopicId(user_topic_type, source=session_id))

    # Wait for the conversation to complete (the runtime becomes idle when the user exits or all tasks done).
    await runtime.stop_when_idle()
    await model_client.close()  # Cleanly shut down the model client

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHandoff system interrupted by user")
    except Exception as e:
        print(f"Error running handoff system: {e}")
        raise
