"""
Chapter 8: Advanced Orchestration Patterns
Example 2: Dynamic Graph Flow

Description:
Demonstrates dynamic graph construction and modification during runtime.
Shows how to build adaptive workflows that can add participants and edges
dynamically based on evolving requirements and conditions.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter8.02_dynamic_graph_flow
```

Expected Output:
Dynamic graph adaptation demonstration:
1. Initial workflow graph construction
2. Runtime addition of new participants
3. Dynamic edge creation and modification
4. Adaptive workflow execution
5. Real-time graph structure changes
6. Flexible orchestration patterns

Key Concepts:
- Runtime graph modification
- Dynamic participant addition
- Adaptive workflow patterns
- Real-time graph restructuring
- Flexible orchestration systems
- Dynamic edge management
- Evolutionary workflow design

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

class DynamicGraphFlow:
    def __init__(self, initial_participants):
        self.participants = list(initial_participants)
        self.edges = []  # Keep track of edges manually
        self.entry_point = None
        
    def set_entry_point(self, agent):
        """Set the entry point for the graph."""
        self.entry_point = agent
    
    def add_edge(self, source, target, condition=None):
        """Add an edge to the internal edge list."""
        self.edges.append({
            'source': source,
            'target': target, 
            'condition': condition
        })
    
    async def add_participant(self, participant):
        """Add a new participant to the flow."""
        if participant not in self.participants:
            self.participants.append(participant)
    
    async def add_edge_dynamic(self, source, target, condition=None):
        """Add a new edge dynamically during runtime."""
        self.add_edge(source, target, condition)
    
    def _build_graph(self):
        """Build the graph from current participants and edges."""
        builder = DiGraphBuilder()
        
        # Add all participants
        for participant in self.participants:
            builder.add_node(participant)
        
        # Set entry point
        if self.entry_point:
            builder.set_entry_point(self.entry_point)
        
        # Add all edges
        for edge in self.edges:
            builder.add_edge(
                edge['source'],
                edge['target'],
                condition=edge['condition']
            )
        
        return builder.build()
    
    async def run(self, task):
        """Run the flow with the current graph."""
        graph = self._build_graph()
        
        flow = GraphFlow(
            participants=self.participants,
            graph=graph,
        )
        
        return await flow.run(task=task)

# --- Example usage following AutoGen patterns ---

async def main():
    # Create a model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    try:
        # Define agents
        planner = AssistantAgent(
            name="planner",
            model_client=model_client,
            system_message="You are the planner. Start the process and assign tasks. When ready for research, say 'START_RESEARCH'."
        )
        researcher = AssistantAgent(
            name="researcher",
            model_client=model_client,
            system_message="You are the researcher. Gather information. When research is complete, say 'RESEARCH_COMPLETE'."
        )
        writer = AssistantAgent(
            name="writer",
            model_client=model_client,
            system_message="You are the writer. Draft the report. When draft is ready, say 'DRAFT_COMPLETE'."
        )

        # Create dynamic flow with initial participants
        dyn_flow = DynamicGraphFlow([planner, researcher, writer])
        
        # Set entry point
        dyn_flow.set_entry_point(planner)
        
        # Add initial edges
        dyn_flow.add_edge(planner, researcher, condition="START_RESEARCH")
        dyn_flow.add_edge(researcher, writer, condition="RESEARCH_COMPLETE")

        print("🚀 Initial workflow: planner → researcher → writer")
        
        # Optionally add a new participant and edge dynamically
        reviewer = AssistantAgent(
            name="reviewer",
            model_client=model_client,
            system_message="You are the reviewer. Review the draft and provide feedback. Say 'REVIEW_COMPLETE' when done."
        )
        
        print("➕ Adding reviewer dynamically...")
        await dyn_flow.add_participant(reviewer)
        await dyn_flow.add_edge_dynamic(writer, reviewer, condition="DRAFT_COMPLETE")

        print("🔄 Enhanced workflow: planner → researcher → writer → reviewer")
        
        # Run the flow
        print("\n🎯 Running enhanced workflow...")
        result = await dyn_flow.run("Produce a short research report on AI in education.")
        
        print("\n✅ Workflow completed!")
        print(f"📊 Final message count: {len(result.messages)}")
        print(f"⏹️ Stop reason: {result.stop_reason}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Check your OpenAI API key and model access {e}")
        
    finally:
        await model_client.close()

# Alternative: Simpler approach using direct DiGraphBuilder
async def simple_dynamic_example():
    """Simpler example that rebuilds the entire graph each time."""
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    try:
        # Define agents
        planner = AssistantAgent(name="planner", model_client=model_client, 
                               system_message="Plan the work.")
        researcher = AssistantAgent(name="researcher", model_client=model_client,
                                  system_message="Research the topic.")
        writer = AssistantAgent(name="writer", model_client=model_client,
                              system_message="Write the report.")
        
        # Build initial graph
        def build_graph(participants, include_reviewer=False):
            builder = DiGraphBuilder()
            
            # Add nodes
            for participant in participants:
                builder.add_node(participant)
            
            # Set entry point
            builder.set_entry_point(planner)
            
            # Add edges
            builder.add_edge(planner, researcher)
            builder.add_edge(researcher, writer)
            
            # Conditionally add reviewer
            if include_reviewer and len(participants) > 3:
                reviewer = participants[3]  # Assume reviewer is 4th participant
                builder.add_edge(writer, reviewer)
            
            return builder.build()
        
        # Initial run
        participants = [planner, researcher, writer]
        graph = build_graph(participants)
        
        flow = GraphFlow(participants=participants, graph=graph)
        print("🚀 Running initial workflow...")
        
        # For demonstration, we'll just show the setup
        print("✅ Initial graph built successfully")
        print("📊 Participants:", [p.name for p in participants])
        
        # Add reviewer dynamically
        reviewer = AssistantAgent(name="reviewer", model_client=model_client,
                                system_message="Review and approve.")
        
        participants.append(reviewer)
        enhanced_graph = build_graph(participants, include_reviewer=True)
        
        enhanced_flow = GraphFlow(participants=participants, graph=enhanced_graph)
        print("➕ Enhanced graph built with reviewer")
        print("📊 Enhanced participants:", [p.name for p in participants])
        
    finally:
        await model_client.close()

if __name__ == "__main__":
    print("=== Dynamic GraphFlow Example ===")
    asyncio.run(main())
    
    print("\n=== Simple Dynamic Example ===") 
    asyncio.run(simple_dynamic_example())