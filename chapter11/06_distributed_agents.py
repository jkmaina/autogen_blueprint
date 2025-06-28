"""
Chapter 11: Concurrent Agents and Distributed Workflows
Example 6: Distributed Agents with gRPC

Description:
Demonstrates distributed agent architectures using gRPC for cross-process
communication. Shows how to create multi-process agent systems with
host/worker patterns, distributed message passing, and coordinated execution.

Prerequisites:
- AutoGen v0.5+ installed with gRPC extensions
- Python 3.9+ with asyncio support
- Understanding of distributed systems concepts
- Multiple terminal sessions for multi-process demo

Usage:
Run in separate terminals:
```bash
# Terminal 1 - Start host
python -m chapter11.06_distributed_agents host --port 50051

# Terminal 2 - Start worker
python -m chapter11.06_distributed_agents worker --id worker1

# Terminal 3 - Start collector
python -m chapter11.06_distributed_agents collector

# Terminal 4 - Send queries
python -m chapter11.06_distributed_agents client --queries 3
```

Expected Output:
Distributed agents demonstration:
1. Host service coordination of distributed workers
2. Worker agents processing queries across processes
3. Collector agent aggregating responses
4. Client sending queries to distributed system
5. Cross-process message passing via gRPC
6. Distributed state management

Key Concepts:
- Distributed agent architecture
- gRPC-based inter-process communication
- Host/worker coordination patterns
- Cross-process message passing
- Distributed agent registration
- Network-based agent discovery
- Scalable agent deployment
- Multi-process orchestration

AutoGen Version: 0.5+
"""

# Standard library imports
import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler
)
from autogen_ext.runtimes.grpc import (
    GrpcWorkerAgentRuntime,
    GrpcWorkerAgentRuntimeHost
)

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Define message types
@dataclass
class QueryMessage:
    """A query message."""
    query_id: str
    content: str

@dataclass
class ResponseMessage:
    """A response message."""
    query_id: str
    content: str
    source: str

# Define agent classes
@default_subscription
class QueryProcessorAgent(RoutedAgent):
    """An agent that processes queries."""
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
    
    @message_handler
    async def handle_query(self, message: QueryMessage, ctx: MessageContext) -> None:
        """Handle a query message."""
        print(f"QueryProcessor: Processing query {message.query_id}: {message.content}")
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Generate a response
        response = f"Response to query {message.query_id}: Processed '{message.content}'"
        
        # Publish the response
        await self.publish_message(
            ResponseMessage(message.query_id, response, self.id.type),
            DefaultTopicId()
        )
        
        print(f"QueryProcessor: Published response for query {message.query_id}")

@default_subscription
class ResponseCollectorAgent(RoutedAgent):
    """An agent that collects responses."""
    
    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.responses: Dict[str, str] = {}
    
    @message_handler
    async def handle_response(self, message: ResponseMessage, ctx: MessageContext) -> None:
        """Handle a response message."""
        print(f"ResponseCollector: Received response for query {message.query_id} from {message.source}")
        self.responses[message.query_id] = message.content
        print(f"ResponseCollector: Current responses: {self.responses}")

async def run_host(port: int) -> None:
    """Run the host service that coordinates distributed agents."""
    print(f"Starting host service on port {port}")
    
    # Create a host service
    host = GrpcWorkerAgentRuntimeHost(address=f"localhost:{port}")
    
    # Start the host
    host.start()
    
    # Keep the host running
    try:
        print(f"Host service started on port {port}. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping host service...")
    finally:
        await host.stop()

async def run_worker(host_address: str, worker_id: str) -> None:
    """Run a worker runtime with a query processor agent."""
    print(f"Starting worker runtime {worker_id} connecting to host at {host_address}")
    
    # Create a worker runtime
    worker_runtime = GrpcWorkerAgentRuntime(host_address=host_address)
    
    # Start the worker
    await worker_runtime.start()
    
    # Register the query processor agent
    await QueryProcessorAgent.register(
        worker_runtime,
        f"query_processor_{worker_id}",
        lambda: QueryProcessorAgent(f"A query processor agent on worker {worker_id}.")
    )
    
    # Keep the worker running
    try:
        print(f"Worker {worker_id} runtime started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"Stopping worker {worker_id} runtime...")
    finally:
        await worker_runtime.stop()

async def run_collector(host_address: str) -> None:
    """Run a worker runtime with a response collector agent."""
    print(f"Starting collector runtime connecting to host at {host_address}")
    
    # Create a worker runtime
    collector_runtime = GrpcWorkerAgentRuntime(host_address=host_address)
    
    # Start the worker
    await collector_runtime.start()
    
    # Register the response collector agent
    await ResponseCollectorAgent.register(
        collector_runtime,
        "response_collector",
        lambda: ResponseCollectorAgent("An agent that collects responses.")
    )
    
    # Keep the worker running
    try:
        print("Collector runtime started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping collector runtime...")
    finally:
        await collector_runtime.stop()

async def send_queries(host_address: str, num_queries: int) -> None:
    """Send queries to the distributed system."""
    print(f"Connecting to host at {host_address} to send {num_queries} queries")
    
    # Create a client runtime to connect to the host
    client_runtime = GrpcWorkerAgentRuntime(host_address=host_address)
    
    # Start the client
    await client_runtime.start()
    
    # Send queries
    for i in range(num_queries):
        query_id = f"query_{i+1}"
        query_content = f"This is test query {i+1}"
        
        print(f"Sending {query_id}: {query_content}")
        
        # Send the query to all query processor agents
        await client_runtime.publish_message(
            QueryMessage(query_id=query_id, content=query_content),
            DefaultTopicId()
        )
        
        # Wait a bit between queries
        await asyncio.sleep(1)
    
    # Wait for responses to be processed
    await asyncio.sleep(5)
    
    # Stop the client runtime
    await client_runtime.stop()

def main() -> None:
    """Main function to run the distributed agents example."""
    parser = argparse.ArgumentParser(description="Run distributed agents example")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Host mode
    host_parser = subparsers.add_parser("host", help="Run as host")
    host_parser.add_argument("--port", type=int, default=50051, help="Port to run the host on")
    
    # Worker mode
    worker_parser = subparsers.add_parser("worker", help="Run as worker")
    worker_parser.add_argument("--host", type=str, default="localhost:50051", help="Host address")
    worker_parser.add_argument("--id", type=str, required=True, help="Worker ID")
    
    # Collector mode
    collector_parser = subparsers.add_parser("collector", help="Run as collector")
    collector_parser.add_argument("--host", type=str, default="localhost:50051", help="Host address")
    
    # Client mode
    client_parser = subparsers.add_parser("client", help="Run as client to send queries")
    client_parser.add_argument("--host", type=str, default="localhost:50051", help="Host address")
    client_parser.add_argument("--queries", type=int, default=5, help="Number of queries to send")
    
    args = parser.parse_args()
    
    if args.mode == "host":
        asyncio.run(run_host(args.port))
    elif args.mode == "worker":
        asyncio.run(run_worker(args.host, args.id))
    elif args.mode == "collector":
        asyncio.run(run_collector(args.host))
    elif args.mode == "client":
        asyncio.run(send_queries(args.host, args.queries))
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDistributed agents demo interrupted by user")
    except Exception as e:
        print(f"Error running distributed agents demo: {e}")
        raise
