import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    try:
        # Define specialized agents for the tutorial
        researcher = AssistantAgent(
            name="researcher",
            description="Conducts research and gathers information on topics",
            model_client=model_client,
            system_message="""You are a researcher. Your job is to gather information and provide comprehensive research on the given topic. 
            Focus on current applications, benefits, and challenges. 
            When you finish your research, say 'Research complete. Ready for analysis.'"""
        )

        analyst = AssistantAgent(
            name="analyst", 
            description="Analyzes information and identifies key insights",
            model_client=model_client,
            system_message="""You are an analyst. Your job is to analyze the research provided and identify key insights, trends, and implications.
            Wait for the researcher to complete their work before starting your analysis.
            When done, say 'Analysis complete. Ready for writing.'"""
        )

        writer = AssistantAgent(
            name="writer",
            description="Creates final reports and documentation", 
            model_client=model_client,
            system_message="""You are a writer. Your job is to take research and analysis and create a well-structured final report.
            Only start writing after research and analysis are complete.
            End your report with 'TASK_COMPLETE' when finished."""
        )

        # Simple, effective termination conditions
        text_termination = TextMentionTermination("TASK_COMPLETE")
        max_termination = MaxMessageTermination(12)
        termination = text_termination | max_termination

        # Selector prompt - keep it simple and clear (key for model-based selection)
        selector_prompt = """You are selecting the next speaker in a research team workflow.

Team members:
{roles}

Conversation history:
{history}

SELECTION RULES:
1. If no research has been done, select: researcher
2. If research is done but no analysis exists, select: analyst  
3. If research and analysis are done but no final report exists, select: writer
4. Follow the workflow: researcher → analyst → writer

Select ONE agent from {participants}. 
Respond with only the agent name (e.g., researcher)."""

        # Create the SelectorGroupChat team
        team = SelectorGroupChat(
            participants=[researcher, analyst, writer],
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,  # Allow agents to continue if needed
            max_turns=3  # Limit selector retries
        )

        print("🎯 Tutorial: Model-Based Speaker Selection")
        print("=" * 50)
        print("Demonstrating how the model selects the next speaker...")
        print()

        # Run the team with a clear task
        task = "Create a report on AI applications in healthcare, covering current uses, benefits, and challenges."
        
        print(f"📝 Task: {task}")
        print("\n🔄 Team Collaboration Starting...")
        print("-" * 50)

        result = await team.run(task=task)

        print("\n" + "=" * 50)
        print("📊 TUTORIAL RESULTS")
        print("=" * 50)
        
        # Analyze the conversation flow for tutorial purposes
        speakers = []
        for msg in result.messages:
            if hasattr(msg, 'source') and msg.source != 'user':
                speakers.append(msg.source)
        
        print(f"🎬 Speaker sequence: {' → '.join(speakers)}")
        print(f"💬 Total messages: {len([m for m in result.messages if hasattr(m, 'source') and m.source != 'user'])}")
        print(f"⏹️ Stop reason: {result.stop_reason}")
        
        # Show how model-based selection worked
        print(f"\n🤖 Model-Based Selection Summary:")
        print(f"• The selector model analyzed context and chose speakers based on:")
        print(f"  - Agent descriptions and roles")
        print(f"  - Current conversation state") 
        print(f"  - Workflow requirements in the selector prompt")
        
        if len(set(speakers)) == 3:
            print(f"✅ SUCCESS: All three agents participated as expected!")
        else:
            print(f"⚠️ NOTE: {len(set(speakers))} of 3 agents participated")

        print(f"\n📋 Final Output Preview:")
        print("-" * 30)
        final_content = result.messages[-1].content if result.messages else "No content"
        print(final_content[:200] + "..." if len(final_content) > 200 else final_content)

        return result

    except Exception as e:
        print(f"❌ Tutorial Error: {e}")
        print("\n🔧 Common Issues & Solutions:")
        print("• API Key: Ensure OPENAI_API_KEY is set correctly")
        print("• Model Access: Verify you have access to gpt-4o")
        print("• Rate Limits: Try reducing MaxMessageTermination if hitting limits")
        print("• Selector Prompt: The model needs clear, simple instructions")
        
    finally:
        await model_client.close()
        print(f"\n🎓 Tutorial Complete!")

# Additional tutorial function to show selector prompt variations
async def tutorial_selector_variations():
    """Shows different selector prompt approaches for educational purposes"""
    print("\n📚 TUTORIAL: Selector Prompt Variations")
    print("=" * 50)
    
    # Example 1: Simple approach
    simple_prompt = """Select the next speaker from {participants}.
    
Current conversation:
{history}

Choose: researcher, analyst, or writer"""

    # Example 2: Rule-based approach  
    rule_based_prompt = """Select an agent to perform the next task.

{roles}

Current conversation:
{history}

RULES:
- researcher: When information gathering is needed
- analyst: When analysis of research is needed  
- writer: When final report creation is needed

Select from {participants}. Respond with agent name only."""

    # Example 3: Workflow approach (recommended)
    workflow_prompt = """You are coordinating a research workflow.

Team: {roles}

Context: {history}

Workflow: researcher → analyst → writer

Select the next appropriate agent from {participants}.
Respond with only the agent name."""

    examples = [
        ("Simple", simple_prompt),
        ("Rule-Based", rule_based_prompt), 
        ("Workflow", workflow_prompt)
    ]
    
    for name, prompt in examples:
        print(f"\n{name} Selector Prompt:")
        print("-" * 20)
        print(prompt[:150] + "..." if len(prompt) > 150 else prompt)

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(tutorial_selector_variations())