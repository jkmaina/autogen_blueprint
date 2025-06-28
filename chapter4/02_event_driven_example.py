"""
Chapter 4: AutoGen Core Fundamentals
Example 2: Event-Driven Architecture

Description:
Demonstrates advanced event-driven architecture using AutoGen Core with custom topic
subscriptions and multi-agent coordination. Shows how agents can communicate through
different topic channels and monitor processing events in real-time.

Prerequisites:
- AutoGen Core v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter4.02_event_driven_example
```

Expected Output:
A demonstration of event-driven processing flow:
1. QueryProcessor receives text query and publishes progress events
2. EventMonitor tracks processing status and completion percentage
3. ResponseHandler receives and stores the final response
Shows real-time event monitoring and multi-channel communication.

Key Concepts:
- Event-driven architecture patterns
- Custom topic subscriptions with @type_subscription
- Multi-channel agent communication
- Processing event monitoring
- Topic-based message routing
- Real-time status tracking
- Advanced message handling workflows

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Third-party imports
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    default_subscription,
    type_subscription,
    message_handler,
)

# Define message types
@dataclass
class TextQuery:
    """A text query message."""
    content: str

@dataclass
class TextResponse:
    """A text response message."""
    content: str
    source: str

@dataclass
class ProcessingEvent:
    """An event indicating processing status."""
    status: str
    progress: float  # 0.0 to 1.0

# Define agent classes
@default_subscription
class QueryProcessor(RoutedAgent):
    """An agent that processes text queries."""

    def __init__(self, description: str) -> None:
        super().__init__(description)

    @message_handler
    async def handle_query(self, message: TextQuery, ctx: MessageContext) -> None:
        """Handle a text query message."""
        print(f"QueryProcessor: Received query: {message.content}")
        
        # Publish a processing event to indicate progress
        await self.publish_message(
            ProcessingEvent(status="Processing", progress=0.5),
            TopicId(type="processing_events", source=self.id.key)
        )
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Generate a response
        response = f"Response to: {message.content}"
        
        # Publish the response
        await self.publish_message(
            TextResponse(content=response, source=self.id.type),
            DefaultTopicId()
        )
        
        # Publish a processing event to indicate completion
        await self.publish_message(
            ProcessingEvent(status="Completed", progress=1.0),
            TopicId(type="processing_events", source=self.id.key)
        )

# FIX: Subscribe to the processing_events topic instead of default
@type_subscription(topic_type="processing_events")
class EventMonitor(RoutedAgent):
    """An agent that monitors processing events."""

    def __init__(self, description: str) -> None:
        super().__init__(description)

    @message_handler
    async def handle_event(self, message: ProcessingEvent, ctx: MessageContext) -> None:
        """Handle a processing event message."""
        print(f"EventMonitor: {message.status} - {message.progress * 100:.0f}% complete")

@default_subscription
class ResponseHandler(RoutedAgent):
    """An agent that handles text responses."""

    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.responses: List[TextResponse] = []

    @message_handler
    async def handle_response(self, message: TextResponse, ctx: MessageContext) -> None:
        """Handle a text response message."""
        print(f"ResponseHandler: Received response from {message.source}: {message.content}")
        self.responses.append(message)

async def main() -> None:
    """
    Main function to demonstrate the advanced event-driven architecture.
    """
    # Create a local embedded runtime
    runtime = SingleThreadedAgentRuntime()
    
    try:
        # Register the agents
        await QueryProcessor.register(
            runtime,
            "query_processor",
            lambda: QueryProcessor("An agent that processes text queries.")
        )
        
        await EventMonitor.register(
            runtime,
            "event_monitor",
            lambda: EventMonitor("An agent that monitors processing events.")
        )
        
        await ResponseHandler.register(
            runtime,
            "response_handler",
            lambda: ResponseHandler("An agent that handles text responses.")
        )
        
        # Start the runtime
        runtime.start()
        
        # Send a query message to the query processor
        print("\nSending query to query processor...")
        await runtime.send_message(
            TextQuery(content="What is the capital of France?"),
            AgentId("query_processor", "default")
        )
        
        # Wait for the runtime to process all messages
        await runtime.stop_when_idle()
        
        print("\nEvent-driven processing complete!")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())