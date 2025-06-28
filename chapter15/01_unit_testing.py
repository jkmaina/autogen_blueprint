"""
Chapter 15: Testing Frameworks
Example 1: Unit Testing

Description:
Demonstrates comprehensive unit testing strategies for AutoGen v0.5 agents
including mock LLM implementations, agent behavior validation, configuration
testing, and isolated component testing with pytest framework.

Prerequisites:
- Python 3.9+ with pytest framework
- AutoGen v0.5+ installed
- unittest.mock for test mocking
- Basic understanding of unit testing principles

Usage:
```bash
python -m chapter15.01_unit_testing
# Or run tests with pytest:
pytest chapter15/01_unit_testing.py -v
```

Expected Output:
Unit testing framework demonstration:
1. Mock LLM client creation and usage
2. Agent creation and configuration testing
3. Message generation validation
4. Tool integration testing
5. System message handling verification
6. Test coverage examples

Key Concepts:
- Mock LLM implementations
- Agent configuration testing
- Behavior validation strategies
- Tool integration testing
- System message verification
- Test isolation techniques
- Pytest fixture usage
- Mocking external dependencies

AutoGen Version: 0.5+
"""

# Standard library imports
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest

# AutoGen imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Mock LLM implementation for testing
class MockLLM:
    """Mock LLM client for testing agents without making API calls"""
    
    def __init__(self, responses=None):
        self.responses = responses or ["Default mock response"]
        self.calls = []
        self.call_count = 0
    
    def create(self, messages, **kwargs):
        """Mock the create method of LLM clients"""
        self.calls.append({"messages": messages, "kwargs": kwargs})
        self.call_count += 1
        
        # Get the next response or use default if we've run out
        response = self.responses.pop(0) if self.responses else "Default mock response"
        
        # Return in the format expected by AutoGen
        return {
            "choices": [
                {
                    "message": {
                        "content": response,
                        "role": "assistant"
                    }
                }
            ]
        }
    
    # For async testing
    async def acreate(self, messages, **kwargs):
        return self.create(messages, **kwargs)

# Fixture to create a mock LLM config
@pytest.fixture
def mock_llm_config():
    mock_llm = MockLLM(["I'll help with that", "Here's the solution", "Task completed"])
    return {"config_list": [{"model": "mock"}], "cache_seed": None}

# Fixture to patch the get_llm_client function
@pytest.fixture
def patch_llm_client(mock_llm_config):
    mock_client = mock_llm_config["mock_client"] = MockLLM(["I'll help with that", "Here's the solution"])
    
    with patch("autogen.agentchat.conversable_agent.get_llm_client") as mock_get_client:
        mock_get_client.return_value = mock_client
        yield mock_client

# Test basic agent creation and configuration
def test_agent_creation():
    """Test that agents can be created with different configurations"""
    # Create agent with default config
    default_agent = AssistantAgent("default_agent")
    
    # Create agent with custom config
    custom_agent = AssistantAgent(
        "custom_agent",
        system_message="You are a specialized assistant.",
        llm_config={"config_list": [{"model": "gpt-4"}]},
        max_consecutive_auto_reply=5
    )
    
    # Verify configurations
    assert default_agent.name == "default_agent"
    assert custom_agent.name == "custom_agent"
    assert custom_agent.system_message == "You are a specialized assistant."
    assert custom_agent.max_consecutive_auto_reply == 5
    assert custom_agent.llm_config["config_list"][0]["model"] == "gpt-4"

# Test agent message generation with mock LLM
def test_agent_generate_reply(patch_llm_client):
    """Test that agents can generate replies using the mock LLM"""
    # Create agent with the patched LLM client
    agent = AssistantAgent("test_agent", llm_config={"config_list": [{"model": "mock"}]})
    
    # Generate a reply
    reply = agent.generate_reply([
        {"role": "user", "content": "Hello, can you help me?"}
    ])
    
    # Verify the reply and LLM client usage
    assert reply == "I'll help with that"
    assert patch_llm_client.call_count == 1
    assert "Hello, can you help me?" in str(patch_llm_client.calls[0]["messages"])

# Test agent with tools
def test_agent_with_tools(patch_llm_client):
    """Test that agents can use tools correctly"""
    # Create a mock tool
    mock_calculator = MagicMock(return_value="4")
    
    # Create agent with the tool
    agent = AssistantAgent(
        "tool_agent",
        llm_config={"config_list": [{"model": "mock"}]},
        tools=[{"name": "calculator", "function": mock_calculator}]
    )
    
    # Set up the mock LLM to return a response that uses the tool
    patch_llm_client.responses = [
        '{"tool_calls": [{"name": "calculator", "arguments": {"a": 2, "b": 2}}]}',
        "The result is 4"
    ]
    
    # Generate a reply that should use the tool
    reply = agent.generate_reply([
        {"role": "user", "content": "Calculate 2+2"}
    ])
    
    # Verify tool was called and response includes tool result
    mock_calculator.assert_called_once()
    assert "4" in reply

# Test user proxy agent
def test_user_proxy_agent():
    """Test UserProxyAgent functionality"""
    # Create user proxy with auto-reply mode
    user_proxy = UserProxyAgent(
        "user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config=False
    )
    
    # Verify configuration
    assert user_proxy.human_input_mode == "NEVER"
    assert user_proxy.max_consecutive_auto_reply == 2
    assert user_proxy.code_execution_config is False

# Test system message handling
def test_system_message():
    """Test that system messages are handled correctly"""
    # Create agent with custom system message
    agent = AssistantAgent(
        "test_agent",
        system_message="You are a helpful assistant specialized in Python programming."
    )
    
    # Verify system message
    assert agent.system_message == "You are a helpful assistant specialized in Python programming."
    
    # Update system message
    agent.update_system_message("You are now specialized in data analysis.")
    assert agent.system_message == "You are now specialized in data analysis."

# Main function to demonstrate usage
def main():
    """Main function to demonstrate unit testing framework for AutoGen agents."""
    # Create a mock LLM
    mock_llm = MockLLM(["I'll help you with Python programming", "Here's a sample code:"])
    
    # Create an agent with the mock LLM
    agent = AssistantAgent(
        "demo_agent",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # Use the agent with the mock LLM
    with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=mock_llm):
        response = agent.generate_reply([
            {"role": "user", "content": "Can you help me with Python?"}
        ])
    
    print(f"Agent response: {response}")
    print(f"LLM was called {mock_llm.call_count} times")
    print(f"LLM received prompt: {mock_llm.calls[0]['messages']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUnit testing demo interrupted by user")
    except Exception as e:
        print(f"Error running unit testing demo: {e}")
        raise
