
import json
import os
import boto3
import asyncio
from typing import Dict, Any, List, Optional

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
agents_table = dynamodb.Table(os.environ.get('AGENTS_TABLE', 'AutoGenAgents'))

# Helper function to create a response
def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': True,
        },
        'body': json.dumps(body)
    }

# Create a new agent
def create_agent(event, context):
    try:
        # Parse request body
        body = json.loads(event['body'])
        agent_id = body.get('agent_id')
        system_message = body.get('system_message')
        model_name = body.get('model_name', 'gpt-4o')
        
        if not agent_id or not system_message:
            return create_response(400, {'error': 'Missing required fields'})
        
        # Check if agent already exists
        response = agents_table.get_item(Key={'agent_id': agent_id})
        if 'Item' in response:
            return create_response(400, {'error': f'Agent {agent_id} already exists'})
        
        # Store agent configuration in DynamoDB
        agents_table.put_item(
            Item={
                'agent_id': agent_id,
                'system_message': system_message,
                'model_name': model_name,
                'created_at': int(context.timestamp_millis)
            }
        )
        
        return create_response(201, {'message': f'Agent {agent_id} created successfully'})
    
    except Exception as e:
        return create_response(500, {'error': str(e)})

# Delete an agent
def delete_agent(event, context):
    try:
        # Get agent_id from path parameters
        agent_id = event['pathParameters']['agent_id']
        
        # Check if agent exists
        response = agents_table.get_item(Key={'agent_id': agent_id})
        if 'Item' not in response:
            return create_response(404, {'error': f'Agent {agent_id} not found'})
        
        # Delete agent from DynamoDB
        agents_table.delete_item(Key={'agent_id': agent_id})
        
        return create_response(200, {'message': f'Agent {agent_id} deleted successfully'})
    
    except Exception as e:
        return create_response(500, {'error': str(e)})

# List all agents
def list_agents(event, context):
    try:
        # Scan DynamoDB for all agents
        response = agents_table.scan(
            ProjectionExpression='agent_id, system_message, model_name, created_at'
        )
        
        return create_response(200, {'agents': response.get('Items', [])})
    
    except Exception as e:
        return create_response(500, {'error': str(e)})

# Run a task with an agent
def run_task(event, context):
    try:
        # Parse request body
        body = json.loads(event['body'])
        agent_id = body.get('agent_id')
        task = body.get('task')
        
        if not agent_id or not task:
            return create_response(400, {'error': 'Missing required fields'})
        
        # Get agent configuration from DynamoDB
        response = agents_table.get_item(Key={'agent_id': agent_id})
        if 'Item' not in response:
            return create_response(404, {'error': f'Agent {agent_id} not found'})
        
        agent_config = response['Item']
        
        # Create model client
        model_client = OpenAIChatCompletionClient(
            model=agent_config['model_name'],
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        
        # Create agent
        agent = AssistantAgent(
            name=agent_id,
            system_message=agent_config['system_message'],
            model_client=model_client,
        )
        
        # Run the task
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(agent.run(task=task))
        
        return create_response(200, {
            'agent_id': agent_id,
            'task': task,
            'response': response
        })
    
    except Exception as e:
        return create_response(500, {'error': str(e)})
