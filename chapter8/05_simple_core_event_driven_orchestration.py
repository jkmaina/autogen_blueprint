"""
Chapter 8: Advanced Orchestration Patterns
Example 5: Simple Core Event-Driven Orchestration

Description:
Demonstrates event-driven orchestration using AutoGen Core with message-based
coordination patterns. Shows how to build reactive systems with event handlers,
message routing, and asynchronous agent communication.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.05_simple_core_event_driven_orchestration
```

Expected Output:
Event-driven orchestration demonstration:
1. Event-based message system initialization
2. Reactive agent response to events
3. Asynchronous message handling
4. Event-driven workflow coordination
5. Message routing and distribution
6. Reactive system patterns

Key Concepts:
- Event-driven architecture
- Reactive agent systems
- Message-based coordination
- Asynchronous event handling
- Event routing patterns
- Core-level orchestration
- Publish-subscribe patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)

@dataclass
class TextMessage:
    content: str
    source: str

@dataclass
class GetResultsRequest:
    sender: AgentId

@dataclass
class GetResultsResponse:
    results: list

@default_subscription
class Coordinator(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A coordinator agent.")

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> None:
        if message.source != "user":
            return
        print(f"Coordinator: Starting task - {message.content}")
        # Decide who should handle the task
        if "data" in message.content.lower():
            target = AgentId("data_specialist", "default")
        elif "code" in message.content.lower():
            target = AgentId("code_specialist", "default")
        else:
            target = AgentId("generalist", "default")
        await self.runtime.send_message(
            TextMessage(content=message.content, source="coordinator"),
            target
        )

@default_subscription
class DataSpecialist(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A data specialist agent.")

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> None:
        if "data" not in message.content.lower():
            return
        print(f"DataSpecialist: Processing data task - {message.content}")
        result = f"Data analysis result for: {message.content}"
        await self.runtime.send_message(
            TextMessage(content=result, source="data_specialist"),
            AgentId("result_collector", "default")
        )

@default_subscription
class CodeSpecialist(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A code specialist agent.")

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> None:
        if "code" not in message.content.lower():
            return
        print(f"CodeSpecialist: Processing code task - {message.content}")
        result = f"Code implementation for: {message.content}"
        await self.runtime.send_message(
            TextMessage(content=result, source="code_specialist"),
            AgentId("result_collector", "default")
        )

@default_subscription
class ResultCollector(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A result collector agent.")
        self.results = []

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> None:
        if message.source not in ["data_specialist", "code_specialist"]:
            return
        print(f"ResultCollector: Received result - {message.content}")
        self.results.append(message.content)

    @message_handler
    async def handle_get_results(self, message: GetResultsRequest, ctx: MessageContext) -> None:
        await self.runtime.send_message(
            GetResultsResponse(results=self.results),
            message.sender
        )

# Temporary agent to catch the GetResultsResponse
@default_subscription
class ResponseCatcher(RoutedAgent):
    def __init__(self):
        super().__init__("A response catcher agent.")
        self.future = None

    @message_handler
    async def handle_response(self, message: GetResultsResponse, ctx: MessageContext) -> None:
        if self.future and not self.future.done():
            self.future.set_result(message.results)

async def main():
    runtime = SingleThreadedAgentRuntime()
    await Coordinator.register(runtime, "coordinator", lambda: Coordinator())
    await DataSpecialist.register(runtime, "data_specialist", lambda: DataSpecialist())
    await CodeSpecialist.register(runtime, "code_specialist", lambda: CodeSpecialist())
    await ResultCollector.register(runtime, "result_collector", lambda: ResultCollector())
    catcher_agent = ResponseCatcher()
    await ResponseCatcher.register(runtime, "response_catcher", lambda: catcher_agent)
    runtime.start()
    # Send the initial message to the coordinator agent via the runtime
    await runtime.send_message(
        TextMessage(content="Analyze the sales data and implement a visualization function", source="user"),
        AgentId("coordinator", "default")
    )
    await asyncio.sleep(2)
    # Prepare to catch the response
    catcher_agent.future = asyncio.get_event_loop().create_future()
    await runtime.send_message(
        GetResultsRequest(sender=catcher_agent.id),
        AgentId("result_collector", "default")
    )
    results = await catcher_agent.future
    print("\nFinal Results:")
    for result in results:
        print(f"- {result}")
    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())