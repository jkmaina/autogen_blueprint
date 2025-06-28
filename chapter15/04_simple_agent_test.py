"""
Chapter 15: Testing Frameworks
Example 4: Simple Agent Testing

Description:
Demonstrates simplified agent testing approaches for AutoGen v0.5 with
straightforward mock implementations, basic test scenarios, and minimal
setup requirements for quick testing and validation workflows.

Prerequisites:
- Python 3.9+ with asyncio support
- AutoGen v0.5+ installed
- Basic understanding of testing concepts
- No external testing frameworks required

Usage:
```bash
python -m chapter15.04_simple_agent_test
```

Expected Output:
Simple agent testing demonstration:
1. Mock client implementation and usage
2. Agent initialization validation
3. Basic response testing
4. Multiple interaction testing
5. Call tracking and verification
6. Simplified test pattern examples

Key Concepts:
- Simplified mock implementations
- Basic agent testing patterns
- Minimal setup testing approaches
- Response validation techniques
- Call tracking mechanisms
- Quick testing workflows
- Lightweight test frameworks
- Development testing strategies

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports (None required for this simple testing approach)

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import (
    ChatCompletionClient, 
    CreateResult, 
    LLMMessage, 
    RequestUsage
)

# Local imports
sys.path.append(str(Path(__file__).parent.parent))


class MockChatCompletionClient:
    """A simplified mock model client for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["This is a mock response."]
        self.response_index = 0
        self.create_calls = []
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async def create(self, messages, **kwargs):
        """Mock create method that returns predefined responses."""
        self.create_calls.append((messages, kwargs))
        
        # Get the next response
        response_content = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        
        # Update usage
        usage = RequestUsage(prompt_tokens=10, completion_tokens=5)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        
        # Return a properly formatted CreateResult
        return CreateResult(
            finish_reason="stop",
            content=response_content,
            usage=usage,
            cached=False
        )
    
    async def generate(self, messages, **kwargs):
        """Mock generate method that returns predefined responses."""
        result = await self.create(messages, **kwargs)
        return result.content
    
    async def close(self):
        """Mock close method."""
        pass
    
    @property
    def model_info(self):
        """Return model info."""
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mock"
        }


async def run_agent_tests():
    """Run tests for the agent."""
    print("\n=== Running Agent Tests ===")
    
    # Create a mock client with predefined responses
    mock_client = MockChatCompletionClient(["Response 1", "Response 2", "Response 3"])
    
    # Create an agent with the mock client
    agent = AssistantAgent(
        name="test_agent",
        system_message="You are a test agent.",
        model_client=mock_client,
    )
    
    # Test 1: Check agent initialization
    print("\nTest 1: Agent Initialization")
    print(f"Agent name: {agent.name}")
    print(f"System message: {agent._system_messages[0].content}")
    assert agent.name == "test_agent"
    assert agent._system_messages[0].content == "You are a test agent."
    print("✓ Agent initialization test passed")
    
    # Test 2: Basic response
    print("\nTest 2: Basic Response")
    response1 = await agent.run(task="Test task 1")
    print(f"Response: {response1}")
    print(f"Create calls: {len(mock_client.create_calls)}")
    assert len(mock_client.create_calls) == 1
    print("✓ Basic response test passed")
    
    # Test 3: Second response
    print("\nTest 3: Second Response")
    response2 = await agent.run(task="Test task 2")
    print(f"Response: {response2}")
    print(f"Create calls: {len(mock_client.create_calls)}")
    assert len(mock_client.create_calls) == 2
    print("✓ Second response test passed")
    
    # Test 4: Third response
    print("\nTest 4: Third Response")
    response3 = await agent.run(task="Test task 3")
    print(f"Response: {response3}")
    print(f"Create calls: {len(mock_client.create_calls)}")
    assert len(mock_client.create_calls) == 3
    print("✓ Third response test passed")
    
    # Close the model client
    await mock_client.close()
    print("\n=== Agent Tests Completed ===")


async def main():
    """Main function to demonstrate simple agent testing approaches."""
    print("AutoGen v0.5 Agent Testing Example")
    print("="*50)
    
    # Run agent tests
    await run_agent_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimple agent testing demo interrupted by user")
    except Exception as e:
        print(f"Error running simple agent testing demo: {e}")
        raise
