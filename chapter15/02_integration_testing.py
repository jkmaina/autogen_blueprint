"""
Chapter 15: Testing Frameworks
Example 2: Integration Testing

Description:
Demonstrates comprehensive integration testing strategies for AutoGen v0.5 agents
including multi-agent conversation flows, group chat dynamics, workflow
orchestration, and end-to-end interaction validation.

Prerequisites:
- Python 3.9+ with pytest framework
- AutoGen v0.5+ installed
- unittest.mock for test mocking
- Understanding of integration testing concepts
- Chapter 15 Example 1 (unit testing) for MockLLM

Usage:
```bash
python -m chapter15.02_integration_testing
# Or run tests with pytest:
pytest chapter15/02_integration_testing.py -v
```

Expected Output:
Integration testing framework demonstration:
1. Agent conversation flow testing
2. Group chat dynamics validation
3. Message routing verification
4. Workflow orchestration testing
5. Termination condition validation
6. Multi-agent interaction patterns

Key Concepts:
- Multi-agent conversation testing
- Group chat orchestration
- Message flow validation
- Workflow testing patterns
- Agent interaction verification
- Conversation termination testing
- Speaker selection testing
- End-to-end integration validation

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
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the MockLLM from the unit testing example
from chapter15.01_unit_testing import MockLLM

# Fixture to create a set of agents for testing
@pytest.fixture
def test_agents():
    """Create a set of agents with mock LLMs for testing"""
    # Create mock LLMs with predefined responses
    user_mock = MockLLM(["I need help", "Thanks"])
    assistant_mock = MockLLM(["I'll help you", "Here's the solution", "You're welcome"])
    expert_mock = MockLLM(["From my expertise, I suggest...", "The technical details are..."])
    
    # Create the agents
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config=False
    )
    
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    expert = AssistantAgent(
        name="expert",
        system_message="You are a technical expert.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # Return the agents and their mock LLMs
    return {
        "user": user_proxy,
        "assistant": assistant,
        "expert": expert,
        "mocks": {
            "user": user_mock,
            "assistant": assistant_mock,
            "expert": expert_mock
        }
    }

# Test basic conversation between two agents
def test_agent_conversation(test_agents):
    """Test a basic conversation between user and assistant agents"""
    user = test_agents["user"]
    assistant = test_agents["assistant"]
    assistant_mock = test_agents["mocks"]["assistant"]
    
    # Patch the LLM client for the assistant
    with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=assistant_mock):
        # Initiate a conversation
        with patch.object(user, "get_human_input", return_value="exit"):
            chat_result = user.initiate_chat(
                assistant,
                message="Can you help me with a problem?"
            )
    
    # Verify the conversation structure
    assert len(chat_result.chat_history) >= 2
    assert chat_result.chat_history[0]["role"] == "user"
    assert chat_result.chat_history[0]["content"] == "Can you help me with a problem?"
    assert chat_result.chat_history[1]["role"] == "assistant"
    assert "I'll help you" in chat_result.chat_history[1]["content"]
    
    # Verify the LLM was called
    assert assistant_mock.call_count >= 1

# Test conversation with termination condition
def test_conversation_termination(test_agents):
    """Test that conversations terminate correctly"""
    user = test_agents["user"]
    assistant = test_agents["assistant"]
    assistant_mock = test_agents["mocks"]["assistant"]
    
    # Set up the mock to return a terminating message
    assistant_mock.responses = ["I'll help you. TERMINATE"]
    
    # Patch the LLM client for the assistant
    with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=assistant_mock):
        # Initiate a conversation that should terminate
        chat_result = user.initiate_chat(
            assistant,
            message="Help me and then terminate."
        )
    
    # Verify the conversation terminated correctly
    assert "TERMINATE" in chat_result.chat_history[-1]["content"]
    assert len(chat_result.chat_history) == 2  # Just the request and terminating response

# Test group chat with multiple agents
def test_group_chat(test_agents):
    """Test group chat functionality with multiple agents"""
    user = test_agents["user"]
    assistant = test_agents["assistant"]
    expert = test_agents["expert"]
    
    assistant_mock = test_agents["mocks"]["assistant"]
    expert_mock = test_agents["mocks"]["expert"]
    
    # Create a group chat
    group_chat = GroupChat(
        agents=[user, assistant, expert],
        messages=[],
        max_round=3
    )
    
    # Create a group chat manager
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # Mock the speaker selection to follow a predetermined order
    original_select_speaker = group_chat.select_speaker
    speaker_order = [assistant, expert, assistant]
    speaker_index = 0
    
    def mock_select_speaker(messages):
        nonlocal speaker_index
        if speaker_index < len(speaker_order):
            next_speaker = speaker_order[speaker_index]
            speaker_index += 1
            return next_speaker
        return None
    
    group_chat.select_speaker = mock_select_speaker
    
    # Patch the LLM clients
    with patch("autogen.agentchat.conversable_agent.get_llm_client") as mock_get_client:
        def get_mock_client(config):
            if config["config_list"][0]["model"] == "mock":
                # Return the appropriate mock based on the current speaker
                if speaker_index > 0 and speaker_order[speaker_index-1] == assistant:
                    return assistant_mock
                elif speaker_index > 0 and speaker_order[speaker_index-1] == expert:
                    return expert_mock
            return MockLLM(["Default response"])
        
        mock_get_client.side_effect = get_mock_client
        
        # Run the group chat
        with patch.object(user, "get_human_input", return_value="exit"):
            chat_result = manager.run(
                message="I need help with a technical problem."
            )
    
    # Restore the original select_speaker method
    group_chat.select_speaker = original_select_speaker
    
    # Verify the group chat behavior
    assert len(chat_result.chat_history) >= 4  # Initial message + at least 3 responses
    
    # Check that both agents participated
    agent_roles = [msg["role"] for msg in chat_result.chat_history]
    assert "assistant" in agent_roles
    assert "expert" in agent_roles

# Test message routing in a workflow
def test_message_routing():
    """Test that messages are correctly routed between agents in a workflow"""
    # Create mock LLMs
    planner_mock = MockLLM(["First, analyze the data. Second, create visualizations."])
    analyst_mock = MockLLM(["Data analysis complete. Key findings: ..."])
    visualizer_mock = MockLLM(["Visualizations created: [Chart 1], [Chart 2]"])
    
    # Create agents
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False
    )
    
    planner = AssistantAgent(
        name="planner",
        system_message="You create step-by-step plans.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    analyst = AssistantAgent(
        name="analyst",
        system_message="You analyze data.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    visualizer = AssistantAgent(
        name="visualizer",
        system_message="You create data visualizations.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # Create a simple workflow manager
    class WorkflowManager:
        def __init__(self, agents):
            self.agents = agents
            self.messages = []
        
        def run(self, initial_message):
            # Start with user message to planner
            self.messages.append({"role": "user", "content": initial_message})
            
            # Get plan from planner
            with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=planner_mock):
                plan_response = self.agents["planner"].generate_reply(self.messages)
            
            self.messages.append({"role": "planner", "content": plan_response})
            
            # Send to analyst
            with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=analyst_mock):
                analysis_response = self.agents["analyst"].generate_reply(self.messages)
            
            self.messages.append({"role": "analyst", "content": analysis_response})
            
            # Send to visualizer
            with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=visualizer_mock):
                viz_response = self.agents["visualizer"].generate_reply(self.messages)
            
            self.messages.append({"role": "visualizer", "content": viz_response})
            
            return self.messages
    
    # Create and run the workflow
    workflow = WorkflowManager({
        "planner": planner,
        "analyst": analyst,
        "visualizer": visualizer
    })
    
    result = workflow.run("Create a data analysis report with visualizations.")
    
    # Verify the workflow execution
    assert len(result) == 4  # Initial message + 3 agent responses
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "planner"
    assert result[2]["role"] == "analyst"
    assert result[3]["role"] == "visualizer"
    
    # Verify message content
    assert "analyze the data" in result[1]["content"]
    assert "Data analysis complete" in result[2]["content"]
    assert "Visualizations created" in result[3]["content"]

# Main function to demonstrate usage
def main():
    """Main function to demonstrate integration testing framework for AutoGen agents."""
    # Create mock LLMs
    assistant_mock = MockLLM(["I'll help you with that", "Here's the solution"])
    expert_mock = MockLLM(["From my expertise, I recommend..."])
    
    # Create agents
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False
    )
    
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    expert = AssistantAgent(
        name="expert",
        system_message="You are a technical expert.",
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # Test a simple conversation
    print("Testing simple conversation...")
    with patch("autogen.agentchat.conversable_agent.get_llm_client", return_value=assistant_mock):
        chat_result = user.initiate_chat(
            assistant,
            message="Can you help me with a problem?"
        )
    
    print(f"Conversation history: {chat_result.chat_history}")
    
    # Test a group chat
    print("\nTesting group chat...")
    group_chat = GroupChat(
        agents=[user, assistant, expert],
        messages=[],
        max_round=2
    )
    
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "mock"}]}
    )
    
    # This is a simplified demonstration - in a real test, you would need
    # more sophisticated mocking of the group chat dynamics

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nIntegration testing demo interrupted by user")
    except Exception as e:
        print(f"Error running integration testing demo: {e}")
        raise
