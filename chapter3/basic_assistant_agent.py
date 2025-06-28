"""
# chapter3/basic_ssistant_agent.py
Example 8: AgentChat Application
This demonstrates a basic Hello World AgentChat.
"""
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import asyncio

# Add the parent directory to the path so we can import the utils module
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import get_openai_config

async def main():
    # Create a model client
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config) # or OpenAIChatCompletionClient(model="gpt-4o", api_key="")
    # Create an AssistantAgent
    agent = AssistantAgent(name="assistant", model_client=client)
    # Run the agent with a task
    result = await agent.run(task="Say Hello World!")
    # Print the result from the agent
    print(result.messages[-1].content)
    await client.close()

asyncio.run(main())
#outpouts “Hello World” 
