from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
# Define the tool function
def search_academic_papers(query: str) -> str:
    """Search for academic papers related to the query."""
    # Return a list of three faux papers
    papers = [
        f"1. Advances in {query}: A Comprehensive Review",
        f"2. {query} Techniques and Applications in Modern Research",
        f"3. Challenges and Future Directions in {query}"
    ]
    return "\n".join(papers)
# Create a model client (replace with your actual model and API key if needed)
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Add the tool to the agent
literature_reviewer = AssistantAgent(
    name="literature_reviewer",
    system_message="You are a literature reviewer. Use your tools to help find and summarize academic papers.",
    model_client=model_client,
    tools=[search_academic_papers],
)
# Example usage
response = asyncio.run(literature_reviewer.run(task="Find recent papers on reinforcement learning."))
print("\nLiterature Reviewer's response:")
print(response)
