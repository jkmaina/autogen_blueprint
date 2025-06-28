from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Define a selector function that prioritizes the Research Manager
def research_selector(messages):
    # If the last message contains "RESEARCH COMPLETE", end the conversation
    if messages and "RESEARCH COMPLETE" in messages[-1].content:
        return None
    
    # If the last message is from the user_proxy, the research_manager should respond
    if messages[-1].source == "user_proxy":
        return "research_manager"
    
    # If the last message is from the research_manager, let it specify who should speak next
    if messages[-1].source == "research_manager":
        content = messages[-1].content.lower()
        if "literature reviewer" in content or "find papers" in content:
            return "literature_reviewer"
        elif "data analyst" in content or "analyze data" in content:
            return "data_analyst"
        elif "content writer" in content or "write" in content:
            return "content_writer"
        elif "fact checker" in content or "verify" in content:
            return "fact_checker"
        elif "user" in content or "question" in content:
            return "user_proxy"
    
    # Default to the research manager to keep things moving
    return "research_manager"

# Create termination conditions
termination = TextMentionTermination("RESEARCH COMPLETE") | MaxMessageTermination(50)

# Create the research team
research_team = SelectorGroupChat(
    [research_manager, literature_reviewer, data_analyst, content_writer, fact_checker, user_proxy],
    selector_func=research_selector,
    termination_condition=termination,
)