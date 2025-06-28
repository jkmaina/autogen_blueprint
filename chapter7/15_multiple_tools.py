from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# Tool 1: Search for academic papers
def search_academic_papers(query: str) -> str:
    """Search for academic papers related to the query."""
    papers = [
        f"1. Advances in {query}: A Comprehensive Review",
        f"2. {query} Techniques and Applications in Modern Research",
        f"3. Challenges and Future Directions in {query}"
    ]
    return "\n".join(papers)

# Tool 2: Extract metadata from a paper title
def extract_paper_metadata(paper_title: str) -> str:
    """Extract metadata from a given paper title."""
    # Faux metadata
    return f"Title: {paper_title}\nAuthors: Jane Doe, John Smith\nYear: 2024\nJournal: Journal of {paper_title.split()[0]} Studies"

# Create a model client (replace with your actual model and API key if needed)
model_client = OpenAIChatCompletionClient(model="gpt-4o")

literature_reviewer = AssistantAgent(
    name="literature_reviewer",
    system_message="""You are an expert
literature reviewer specializing in finding and analyzing academic papers.
Your responsibilities include:
1. Searching for relevant papers using academic search tools
2. Extracting key findings and methodologies from papers
3. Identifying connections between different papers
4. Evaluating the quality and relevance of sources
5. Providing concise summaries of important papers
Always cite your sources properly with author names, publication year, and titles.
When analyzing papers, focus on methodology, key findings, limitations, and implications.
""",
    model_client=model_client,
    tools=[search_academic_papers, extract_paper_metadata],
)

# Example usage
response = asyncio.run(literature_reviewer.run(task="Find and summarize recent papers on reinforcement learning."))
print("\nLiterature Reviewer's response:")
print(response)
