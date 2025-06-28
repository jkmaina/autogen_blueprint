"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 3: Agent Lifecycle Management

Description:
Demonstrates complete agent lifecycle patterns including creation, initialization,
state management, monitoring, and graceful shutdown. Shows agent registration,
message handling, health monitoring, and cleanup procedures.

Prerequisites:
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- Understanding of AutoGen Core concepts

Usage:
```bash
python -m chapter11.03_agent_lifecycle
```

Expected Output:
Agent lifecycle demonstration:
1. Agent creation and registration
2. Initialization with configuration
3. Heartbeat monitoring and health checks
4. Message processing and state tracking
5. Status monitoring across multiple agents
6. Graceful shutdown and cleanup procedures

Key Concepts:
- Agent lifecycle management
- State tracking and monitoring
- Message-based health checks
- Runtime agent registration
- Topic-based communication patterns
- Graceful shutdown procedures
- Resource cleanup patterns
- Agent monitoring systems

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

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

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Define message types
@dataclass
class InitializeMessage:
    """Message to initialize an agent with configuration."""
    config: Dict[str, Any]

@dataclass
class HeartbeatMessage:
    """Periodic heartbeat message to check agent health."""
    timestamp: float = field(default_factory=time.time)

@dataclass
class StatusRequestMessage:
    """Message to request agent status."""
    request_id: str

@dataclass
class StatusResponseMessage:
    """Message with agent status information."""
    request_id: str
    agent_id: str
    status: str
    uptime: float
    message_count: int
    last_activity: float

@dataclass
class ShutdownMessage:
    """Message to request agent shutdown."""
    reason: str

@dataclass
class CheckAgentsMessage:
    """Message to trigger the monitor to check all agents."""
    pass

class LifecycleAgent(RoutedAgent):
    """An agent that demonstrates the complete lifecycle."""
    
    def __init__(self, description: str) -> None:
        """Initialize the agent."""
        super().__init__(description)
        self.initialized = False
        self.start_time = time.time()
        self.message_count = 0
        self.last_activity = self.start_time
        self.config = {}
        
        print(f"Agent {self.id.type} created with ID {self.id}")
    
    @message_handler
    async def handle_initialize(self, message: InitializeMessage, ctx: MessageContext) -> None:
        """Handle initialization message."""
        print(f"Agent {self.id.type} initializing with config: {message.config}")
        self.config = message.config
        self.initialized = True
        self.message_count += 1
        self.last_activity = time.time()
    
    @message_handler
    async def handle_heartbeat(self, message: HeartbeatMessage, ctx: MessageContext) -> None:
        """Handle heartbeat message."""
        print(f"Agent {self.id.type} received heartbeat at {message.timestamp}")
        self.message_count += 1
        self.last_activity = time.time()
        
        # Respond with current status
        status_topic = TopicId(type="agent.status", source=self.id.key)
        await self.publish_message(
            StatusResponseMessage(
                request_id="heartbeat",
                agent_id=self.id.type,
                status="active" if self.initialized else "initializing",
                uptime=time.time() - self.start_time,
                message_count=self.message_count,
                last_activity=self.last_activity
            ),
            topic_id=status_topic
        )
    
    @message_handler
    async def handle_status_request(self, message: StatusRequestMessage, ctx: MessageContext) -> None:
        """Handle status request message."""
        print(f"Agent {self.id.type} received status request: {message.request_id}")
        self.message_count += 1
        self.last_activity = time.time()
        
        # Respond with current status
        status_topic = TopicId(type="agent.status", source=self.id.key)
        await self.publish_message(
            StatusResponseMessage(
                request_id=message.request_id,
                agent_id=self.id.type,
                status="active" if self.initialized else "initializing",
                uptime=time.time() - self.start_time,
                message_count=self.message_count,
                last_activity=self.last_activity
            ),
            topic_id=status_topic
        )
    
    @message_handler
    async def handle_shutdown(self, message: ShutdownMessage, ctx: MessageContext) -> None:
        """Handle shutdown message."""
        print(f"Agent {self.id.type} shutting down: {message.reason}")
        self.message_count += 1
        self.last_activity = time.time()
        
        # Perform cleanup tasks
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Perform cleanup tasks before shutdown."""
        print(f"Agent {self.id.type} cleaning up resources")
        # In a real implementation, this would release resources,
        # close connections, save state, etc.
        await asyncio.sleep(0.5)  # Simulate cleanup work
        print(f"Agent {self.id.type} cleanup complete")

class MonitorAgent(RoutedAgent):
    """An agent that monitors other agents."""
    
    def __init__(self, description: str) -> None:
        """Initialize the monitor agent."""
        super().__init__(description)
        self.agent_statuses = {}
        
        print(f"Monitor agent {self.id.type} created with ID {self.id}")
    
    @message_handler
    async def handle_status(self, message: StatusResponseMessage, ctx: MessageContext) -> None:
        """Handle status response messages."""
        print(f"Monitor received status from {message.agent_id}: {message.status}")
        self.agent_statuses[message.agent_id] = {
            "status": message.status,
            "uptime": message.uptime,
            "message_count": message.message_count,
            "last_activity": message.last_activity
        }
    
    @message_handler
    async def handle_check_request(self, message: CheckAgentsMessage, ctx: MessageContext) -> None:
        """Handle a request to check all agents."""
        await self.check_agents(ctx)
    
    async def check_agents(self, ctx: MessageContext) -> None:
        """Check all agents by sending status requests."""
        print("\nMonitor checking all agents...")
        
        # Send status requests to all agent types
        for agent_type in ["agent_1", "agent_2", "agent_3"]:
            agent_topic = TopicId(type=f"agent.{agent_type}", source=self.id.key)
            await self.publish_message(
                StatusRequestMessage(request_id=f"check_{int(time.time())}"),
                topic_id=agent_topic
            )
        
        # Wait for responses
        await asyncio.sleep(1)
        
        # Print current status of all agents
        print("\nCurrent agent statuses:")
        for agent_id, status in self.agent_statuses.items():
            print(f"  {agent_id}: {status['status']}, uptime: {status['uptime']:.1f}s, messages: {status['message_count']}")
        print()

async def main() -> None:
    """Main function to demonstrate comprehensive agent lifecycle management."""
    print("\n=== Agent Lifecycle Example ===\n")
    
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Register the monitor agent type
    monitor_type = await MonitorAgent.register(
        runtime,
        type="monitor",
        factory=lambda: MonitorAgent("System Monitor")
    )
    
    # Add subscription for the monitor agent to receive status messages
    await runtime.add_subscription(TypeSubscription(topic_type="agent.status", agent_type=monitor_type.type))
    await runtime.add_subscription(TypeSubscription(topic_type="monitor.check", agent_type=monitor_type.type))
    
    # Register lifecycle agent types
    agent_types = []
    for i in range(3):
        agent_type_name = f"agent_{i+1}"
        
        # Use a factory function that creates a new agent instance when needed
        agent_type = await LifecycleAgent.register(
            runtime,
            type=agent_type_name,
            factory=lambda name=agent_type_name: LifecycleAgent(f"Agent {name}")
        )
        
        # Add subscriptions for this agent type
        await runtime.add_subscription(TypeSubscription(topic_type=f"agent.{agent_type_name}", agent_type=agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type="broadcast", agent_type=agent_type.type))
        await runtime.add_subscription(TypeSubscription(topic_type=f"topic_{i+1}", agent_type=agent_type.type))
        
        agent_types.append(agent_type)
    
    # Start the runtime
    runtime.start()
    
    # Initialize the agents with different configurations
    for i, agent_type in enumerate(agent_types):
        agent_topic = TopicId(type=f"agent.{agent_type.type}", source="main")
        config = {
            "parameters": {
                "priority": i+1,
                "max_messages": 100 * (i+1)
            }
        }
        await runtime.publish_message(
            InitializeMessage(config=config),
            topic_id=agent_topic
        )
    
    # Wait for initialization
    await asyncio.sleep(1)
    
    # Send heartbeats to all agents
    print("\nSending heartbeats to all agents...")
    broadcast_topic = TopicId(type="broadcast", source="main")
    await runtime.publish_message(
        HeartbeatMessage(),
        topic_id=broadcast_topic
    )
    
    # Wait for heartbeat responses
    await asyncio.sleep(1)
    
    # Trigger the monitor to check agent statuses
    monitor_topic = TopicId(type="monitor.check", source="main")
    await runtime.publish_message(
        CheckAgentsMessage(),
        topic_id=monitor_topic
    )
    
    # Wait for check to complete
    await asyncio.sleep(1)
    
    # Simulate some activity
    for i in range(2):
        print(f"\nSimulating activity cycle {i+1}...")
        
        # Send messages to specific topics
        for j in range(3):
            if i % 2 == 0 or j == 0:  # Either first cycle or first agent
                topic = TopicId(type=f"topic_{j+1}", source="main")
                await runtime.publish_message(
                    HeartbeatMessage(),
                    topic_id=topic
                )
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Trigger the monitor to check agent statuses
        await runtime.publish_message(
            CheckAgentsMessage(),
            topic_id=monitor_topic
        )
        
        # Wait for check to complete
        await asyncio.sleep(1)
    
    # Shutdown one agent
    print("\nShutting down Agent 2...")
    agent2_topic = TopicId(type="agent.agent_2", source="main")
    await runtime.publish_message(
        ShutdownMessage(reason="Maintenance"),
        topic_id=agent2_topic
    )
    
    # Wait for shutdown
    await asyncio.sleep(1)
    
    # Trigger the monitor to check agent statuses
    await runtime.publish_message(
        CheckAgentsMessage(),
        topic_id=monitor_topic
    )
    
    # Wait for check to complete
    await asyncio.sleep(1)
    
    # Shutdown all remaining agents
    print("\nShutting down all remaining agents...")
    broadcast_topic = TopicId(type="broadcast", source="main")
    await runtime.publish_message(
        ShutdownMessage(reason="System shutdown"),
        topic_id=broadcast_topic
    )
    
    # Wait for shutdown
    await asyncio.sleep(1)
    
    # Stop the runtime when all tasks are complete
    await runtime.stop_when_idle()
    print("\nRuntime stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent lifecycle demo interrupted by user")
    except Exception as e:
        print(f"Error running agent lifecycle demo: {e}")
        raise
