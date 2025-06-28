"""
Chapter 8: Advanced Orchestration Patterns
Example 6: Simplified Research Assistant

Description:
Demonstrates a comprehensive research assistant system using multiple
orchestration patterns including graph flows, selectors, and swarms.
Shows integration of different coordination approaches in a unified system.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.06_simplified_research_assistant
```

Expected Output:
Comprehensive research workflow demonstration:
1. Research manager coordinates overall process
2. Literature reviewer handles academic sources
3. Data analyst processes quantitative information
4. Domain expert provides specialized knowledge
5. Integrated multi-pattern orchestration
6. Complete research workflow execution

Key Concepts:
- Multi-pattern orchestration integration
- Research workflow coordination
- Specialized agent roles
- Graph flow and selector combination
- Complex research pipelines
- Unified orchestration systems
- Advanced workflow patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow, SelectorGroupChat, Swarm
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Create agents
research_manager = AssistantAgent(
    name="research_manager",
    system_message="""You are the Research Manager, responsible for coordinating the research process.
    Your responsibilities include:
    1. Creating a research plan based on the user's query
    2. Delegating specific tasks to specialized agents
    3. Tracking progress and ensuring all aspects of the research are covered
    4. Synthesizing findings into a coherent whole
    5. Identifying gaps in the research and requesting additional information
    
    When delegating tasks, use these specific phrases to trigger the appropriate agents:
    - Use "LITERATURE_REVIEW" to delegate to the literature reviewer
    - Use "DATA_ANALYSIS" to delegate to the data analyst  
    - Use "DOMAIN_EXPERTISE" to delegate to the domain expert
    - Use "USER_INPUT" when you need user clarification
    - Use "WRITE_REPORT" when ready to create the final report
    
    When the research is complete, include "RESEARCH COMPLETE" in your message.""",
    model_client=model_client,
)

literature_reviewer = AssistantAgent(
    name="literature_reviewer",
    system_message="""You are a Literature Reviewer specializing in academic research.
    Your responsibilities include:
    1. Finding relevant academic papers and sources
    2. Analyzing the methodology and findings of papers
    3. Evaluating the credibility and relevance of sources
    4. Extracting key information and quotes
    5. Identifying connections between different sources""",
    model_client=model_client,
)

data_analyst = AssistantAgent(
    name="data_analyst",
    system_message="""You are a Data Analyst specializing in research data.
    Your responsibilities include:
    1. Analyzing quantitative and qualitative data
    2. Creating visualizations and summaries
    3. Identifying trends and patterns
    4. Interpreting statistical findings
    5. Evaluating the validity and reliability of data""",
    model_client=model_client,
)

domain_expert = AssistantAgent(
    name="domain_expert",
    system_message="""You are a Domain Expert with specialized knowledge.
    Your responsibilities include:
    1. Providing context and background information
    2. Explaining domain-specific concepts and terminology
    3. Identifying important factors and considerations
    4. Evaluating the significance of findings
    5. Suggesting directions for further research""",
    model_client=model_client,
)

content_writer = AssistantAgent(
    name="content_writer",
    system_message="""You are a Content Writer specializing in research reports.
    Your responsibilities include:
    1. Synthesizing information from multiple sources
    2. Organizing content logically with clear structure
    3. Writing in a clear, concise, and engaging style
    4. Adapting tone and complexity to the target audience
    5. Ensuring proper citation and attribution""",
    model_client=model_client,
)

fact_checker = AssistantAgent(
    name="fact_checker",
    system_message="""You are a Fact Checker specializing in verifying information.
    Your responsibilities include:
    1. Verifying claims against reliable sources
    2. Identifying unsupported assertions or logical fallacies
    3. Checking statistical interpretations for accuracy
    4. Ensuring proper representation of source material
    5. Flagging potential misinformation or biased reporting
    
    When fact-checking is complete, include "RESEARCH COMPLETE" in your message.""",
    model_client=model_client,
)

user_proxy = UserProxyAgent(
    name="user_proxy",    
)

# Create the high-level research workflow using GraphFlow
builder = DiGraphBuilder()

# Add nodes
builder.add_node(research_manager)
builder.add_node(literature_reviewer)
builder.add_node(data_analyst)
builder.add_node(domain_expert)
builder.add_node(content_writer)
builder.add_node(fact_checker)
builder.add_node(user_proxy)

# Set entry point
builder.set_entry_point(research_manager)

# Define the workflow
builder.add_edge(research_manager, literature_reviewer, condition="LITERATURE_REVIEW")
builder.add_edge(research_manager, data_analyst, condition="DATA_ANALYSIS")
builder.add_edge(research_manager, domain_expert, condition="DOMAIN_EXPERTISE")
builder.add_edge(research_manager, user_proxy, condition="USER_INPUT")

builder.add_edge(literature_reviewer, research_manager)
builder.add_edge(data_analyst, research_manager)
builder.add_edge(domain_expert, research_manager)
builder.add_edge(user_proxy, research_manager)

builder.add_edge(research_manager, content_writer, condition="WRITE_REPORT")
builder.add_edge(content_writer, fact_checker)
# Removed the edge from fact_checker back to content_writer to create a leaf node

# Build the graph
graph = builder.build()

# Create the research flow
research_flow = GraphFlow(
    participants=[research_manager, literature_reviewer, data_analyst, 
                 domain_expert, content_writer, fact_checker, user_proxy],
    graph=graph,
    termination_condition=TextMentionTermination("RESEARCH COMPLETE"),
)

# Create a mid-level research team for collaborative research phases
def research_selector(messages):
    """Selector function for the research team."""
    if not messages:
        return "research_manager"
    
    # Check for termination
    if "PHASE COMPLETE" in messages[-1].content:
        return None
    
    # If the last message is from the user_proxy, the research_manager should respond
    if messages[-1].source == "user_proxy":
        return "research_manager"
    
    # If the last message is from the research_manager, check for delegations
    if messages[-1].source == "research_manager":
        content = messages[-1].content.lower()
        if "literature" in content or "papers" in content or "sources" in content:
            return "literature_reviewer"
        elif "data" in content or "analysis" in content or "statistics" in content:
            return "data_analyst"
        elif "domain" in content or "expert" in content or "specialized" in content:
            return "domain_expert"
        elif "user" in content or "question" in content or "clarify" in content:
            return "user_proxy"
    
    # Default to the research manager
    return "research_manager"

# Create the research team
research_team = SelectorGroupChat(
    [research_manager, literature_reviewer, data_analyst, domain_expert, user_proxy],
    selector_func=research_selector,
    termination_condition=TextMentionTermination("PHASE COMPLETE"),
    model_client=model_client
)

# Create a specialized writing team using Swarm
# writing_team = Swarm(
#     participants=[content_writer, fact_checker, research_manager],
#     termination_condition=TextMentionTermination("WRITING COMPLETE"),
# )

# Main function to run the research assistant
async def run_research_assistant(topic):
    """Run the research assistant on a given topic."""
    print(f"Starting research on: {topic}")
    
    # Run the high-level research flow
    result = await research_flow.run(task=f"Research the topic: {topic}")
    
    print("\nResearch complete!")
    print(result)
    
    # Close the model client
    await model_client.close()

# Run the research assistant
asyncio.run(run_research_assistant("The impact of artificial intelligence on education"))