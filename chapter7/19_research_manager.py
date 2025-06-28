import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define agents with better system messages that align with the selector
research_manager = AssistantAgent(
    name="research_manager",
    model_client=model_client,
    system_message="""You are the research manager. Coordinate all aspects of the research process. 
    When delegating tasks, use these EXACT phrases:
    - "I need the literature_reviewer to find papers"
    - "I need the data_analyst to analyze data" 
    - "I need the content_writer to write the summary"
    - "I need the fact_checker to verify information"
    - "I need the user_proxy to provide more questions"
    Say TERMINATE when the final paper is complete."""
)

literature_reviewer = AssistantAgent(
    name="literature_reviewer",
    model_client=model_client,
    system_message="""You are the literature reviewer. Find and summarize relevant papers. 
    When finished, always say: 'Research manager, I have completed the literature review.'"""
)

data_analyst = AssistantAgent(
    name="data_analyst", 
    model_client=model_client,
    system_message="""You are the data analyst. Analyze data and report findings.
    When finished, always say: 'Research manager, I have completed the data analysis.'"""
)

content_writer = AssistantAgent(
    name="content_writer",
    model_client=model_client,
    system_message="""You are the content writer. Write research summaries and papers.
    When finished, always say: 'Research manager, I have completed the writing task.'"""
)

fact_checker = AssistantAgent(
    name="fact_checker",
    model_client=model_client,
    system_message="""You are the fact checker. Verify accuracy of information.
    When finished, always say: 'Research manager, I have completed fact checking.'"""
)

# Use actual UserProxyAgent for real user interaction
user_proxy = UserProxyAgent(
    name="user_proxy"
)

def research_selector(messages):
    if not messages:
        return "research_manager"
    
    last_message = messages[-1]
    last_speaker = last_message.source
    content = last_message.content.lower()
    
    # If user spoke, research manager coordinates
    if last_speaker == "user_proxy":
        return "research_manager"
    
    # If research manager spoke, look for delegation keywords
    elif last_speaker == "research_manager":
        if "literature_reviewer" in content:
            return "literature_reviewer"
        elif "data_analyst" in content:
            return "data_analyst"
        elif "content_writer" in content:
            return "content_writer"
        elif "fact_checker" in content:
            return "fact_checker"
        elif "user_proxy" in content:
            return "user_proxy"
        else:
            # If no clear delegation, continue with research manager
            return "research_manager"
    
    # If any specialist spoke, return to research manager for coordination
    elif last_speaker in ["literature_reviewer", "data_analyst", "content_writer", "fact_checker"]:
        return "research_manager"
    
    # Default fallback
    return "research_manager"

termination_condition = TextMentionTermination("TERMINATE") | MaxMessageTermination(15)

# Create the research team
research_team = SelectorGroupChat(
    [research_manager, literature_reviewer, data_analyst, content_writer, fact_checker, user_proxy],
    selector_func=research_selector,
    model_client=model_client,
    termination_condition=termination_condition
)

async def main():
    stream = research_team.run_stream(
        task="I need a summary of recent research on reinforcement learning. Please coordinate the process."
    )
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())