import logging
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_assistant")

# Example tool function with error handling
def search_academic_papers(query: str) -> str:
    """Search for academic papers with error handling."""
    try:
        if not query or "fail" in query:
            raise ValueError("Invalid or failing query.")
        return f"Results for '{query}'"
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}")
        return f"Error searching for papers on '{query}'. Please try a different query or approach."

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Create a minimal agent with the tool
topic = "ai agents"
agent = AssistantAgent(
    name="research_assistant",
    model_client=model_client,
    tools=[search_academic_papers],
    system_message="You are a research assistant. Use the 'search_academic_papers' tool to find papers. Handle errors gracefully and report them to the user."
)

async def main():
    logger.info(f"Starting research on: {topic}")
    try:
        response = await agent.run(task=f"Use the search_academic_papers tool to find recent papers on '{topic}'. Also try searching for 'fail this search' to demonstrate error handling.")
        logger.info("Research complete.")
        print(response)
    except Exception as e:
        logger.error(f"Agent failed to complete the task: {e}")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
