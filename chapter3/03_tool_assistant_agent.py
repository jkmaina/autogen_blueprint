"""
Chapter 3: Agent Communication Patterns
Example 3: Tool-Enhanced Assistant Agent

Description:
Demonstrates advanced tool integration with AgentChat, including time retrieval,
mathematical operations, Wikipedia lookups, and country information queries.
Shows how to equip agents with multiple specialized tools for complex tasks.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed
- httpx package for HTTP requests

Usage:
```bash
python -m chapter3.03_tool_assistant_agent
```

Expected Output:
The agent will use multiple tools to help with a travel planning query:
- Get current time
- Look up country information for Japan
- Perform mathematical calculations for trip costs
- Provide Wikipedia information about topics
Demonstrates comprehensive tool usage in a real-world scenario.

Key Concepts:
- Multi-tool agent configuration
- External API integration (Wikipedia, REST Countries)
- HTTP client usage within tools
- Mathematical operation tools
- Real-world application scenarios
- Tool error handling and fallbacks

AutoGen Version: 0.5+
"""
# Standard library imports
import asyncio
import sys
import urllib.parse
from pathlib import Path

# Third-party imports
import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Tool 1: Get the current time   
async def get_time() -> str:
    """Get the current time"""
    from datetime import datetime
    current_time = datetime.now().strftime("%I:%M %p")
    return current_time

# Tool 2: Perform a simple math operation
async def add_numbers(a: int, b: int) -> int:  
    """Add two numbers together."""                      
    return a + b

# Tool 3: Simple fact lookup using WikiPedia (using a reliable free API)
async def lookup_topic(topic: str) -> str:
    """Look up information about a topic using Wikipedia search"""
    # Use a simple HTTP client to get information
    async with httpx.AsyncClient() as client:
        try:
            # Wikipedia has a simple API that usually works
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', topic)
                summary = data.get('extract', 'No information found')            
                
                return f"{title}: {summary}"
            else:
                return f"Could not find information about '{topic}'"
                
        except:
            return f"Sorry, couldn't look up '{topic}' right now"

# Tool 4: Country information lookup
async def get_country_info(country: str) -> str:
    """Get detailed information about a country"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"https://restcountries.com/v3.1/name/{urllib.parse.quote(country)}")
            response.raise_for_status()
            data = response.json()
            
            if data:
                country_info = data[0]
                name = country_info['name']['common']
                capital = country_info.get('capital', ['N/A'])[0] if country_info.get('capital') else 'N/A'
                population = country_info.get('population', 0)
                region = country_info.get('region', 'Unknown')
                languages = list(country_info.get('languages', {}).values())
                
                return (f"**{name}**\n"
                       f"Capital: {capital}\n" 
                       f"Population: {population:,}\n"
                       f"Region: {region}\n"
                       f"Languages: {', '.join(languages[:3]) if languages else 'N/A'}")
            else:
                return f"No information found for country '{country}'"
        except Exception as e:
            return f"Error fetching country info for '{country}': {str(e)}"

async def main():
    # Create a model client
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config) 
    
    # Create the Assistant Agent with tools
    assistant = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=[get_time, add_numbers, lookup_topic, get_country_info],
        description="A helpful assistant that can get the current time, perform math operations, look up topics on Wikipedia, and provide country information."
    )
    
    # Run the agent with a task that requires tool usage
    print("Assistant is responding...\n")
    travel_query = """
        I'm planning a trip to Japan and need help with some research. 
        What time is it right now? I want to know basic information about Japan as a country. 
        Also, can you tell me about Japan's capital'? 
        Finally, if a round-trip flight costs $1,200 and hotel is $150 per night for 7 nights, 
        what's the total cost? (1200 + 150 * 7)
        """
    stream = assistant.run_stream( 
        task=travel_query        
    )
    
    # Display streamed output using Console
    await Console(stream)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())