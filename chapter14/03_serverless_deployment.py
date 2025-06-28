"""
Chapter 14: Deployment Patterns
Example 3: Serverless Deployment

Description:
Demonstrates comprehensive serverless deployment patterns for AutoGen v0.5 agents
using AWS Lambda, API Gateway, and DynamoDB for scalable, cost-effective
cloud deployment with automatic scaling and pay-per-use pricing.

Prerequisites:
- AWS account with appropriate permissions
- AWS CLI configured with credentials
- Serverless Framework installed globally
- OpenAI API key for agent services
- Node.js and npm for serverless plugins

Usage:
```bash
python -m chapter14.03_serverless_deployment
```

Expected Output:
Serverless deployment setup demonstration:
1. Serverless.yml configuration generation
2. Lambda function handler creation
3. DynamoDB CloudFormation template
4. AWS API Gateway integration
5. Complete deployment instructions
6. Testing and cleanup procedures

Key Concepts:
- Serverless architecture patterns
- AWS Lambda function deployment
- API Gateway integration
- DynamoDB state management
- Pay-per-use cost optimization
- Automatic scaling capabilities
- CloudFormation infrastructure
- Event-driven processing

AutoGen Version: 0.5+
"""

# Standard library imports
import argparse
import os
from typing import Any, Dict

# Third-party imports (None required for this deployment script)

# Local imports (None required for this deployment script)

# Define the serverless.yml content
SERVERLESS_YML = """
service: autogen-agent-service

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.10
  region: us-east-1
  memorySize: 1024
  timeout: 30
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
  
  httpApi:
    cors: true

functions:
  create_agent:
    handler: handler.create_agent
    events:
      - httpApi:
          path: /agents
          method: post
  
  delete_agent:
    handler: handler.delete_agent
    events:
      - httpApi:
          path: /agents/{agent_id}
          method: delete
  
  list_agents:
    handler: handler.list_agents
    events:
      - httpApi:
          path: /agents
          method: get
  
  run_task:
    handler: handler.run_task
    events:
      - httpApi:
          path: /tasks
          method: post
    timeout: 60

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
    layer: true
"""

# Define the handler.py content
HANDLER_PY = """
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
"""

# Define the requirements.txt content
REQUIREMENTS_TXT = """
autogen-core>=0.5.0
autogen-agentchat>=0.5.0
autogen-ext>=0.5.0
autogen-ext[openai]>=0.5.0
boto3>=1.26.0
"""

# Define the CloudFormation template for DynamoDB
CLOUDFORMATION_YML = """
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AutoGen Agent Service Infrastructure'

Resources:
  AgentsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: AutoGenAgents
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: agent_id
          AttributeType: S
      KeySchema:
        - AttributeName: agent_id
          KeyType: HASH

Outputs:
  AgentsTableName:
    Description: Name of the DynamoDB table for storing agent configurations
    Value: !Ref AgentsTable
"""

# Define the README.md content
README_MD = """# AutoGen Serverless Deployment

This directory contains files for deploying AutoGen agents using AWS Lambda and API Gateway.

## Prerequisites

- AWS account
- AWS CLI configured
- Node.js and npm installed
- Serverless Framework installed (`npm install -g serverless`)

## Deployment Steps

1. Install the Serverless Framework and plugins:
   ```bash
   npm install -g serverless
   npm install --save-dev serverless-python-requirements
   ```

2. Set up your AWS credentials:
   ```bash
   aws configure
   ```

3. Deploy the DynamoDB table:
   ```bash
   aws cloudformation deploy --template-file cloudformation.yml --stack-name autogen-infrastructure
   ```

4. Deploy the serverless application:
   ```bash
   export OPENAI_API_KEY=your-openai-api-key
   serverless deploy
   ```

5. Test the API:
   ```bash
   # Create an agent
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/agents \\
     -H "Content-Type: application/json" \\
     -d '{"agent_id": "test-agent", "system_message": "You are a helpful assistant.", "model_name": "gpt-4o"}'
   
   # Run a task
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/tasks \\
     -H "Content-Type: application/json" \\
     -d '{"agent_id": "test-agent", "task": "Tell me a joke about AI."}'
   ```

## API Endpoints

- `POST /agents` - Create a new agent
- `DELETE /agents/{agent_id}` - Delete an agent
- `GET /agents` - List all agents
- `POST /tasks` - Run a task with an agent

## Cleanup

To remove all deployed resources:

```bash
serverless remove
aws cloudformation delete-stack --stack-name autogen-infrastructure
```
"""

def create_serverless_files(output_dir: str) -> None:
    """
    Create the serverless deployment files in the specified directory.
    
    Args:
        output_dir: The directory to create the files in
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the serverless.yml file
    with open(os.path.join(output_dir, "serverless.yml"), "w") as f:
        f.write(SERVERLESS_YML)
    
    # Create the handler.py file
    with open(os.path.join(output_dir, "handler.py"), "w") as f:
        f.write(HANDLER_PY)
    
    # Create the requirements.txt file
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS_TXT)
    
    # Create the CloudFormation template
    with open(os.path.join(output_dir, "cloudformation.yml"), "w") as f:
        f.write(CLOUDFORMATION_YML)
    
    # Create the README.md file
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(README_MD)
    
    print(f"Serverless deployment files created in {output_dir}")
    print("\nFollow the instructions in README.md to deploy to AWS Lambda.")

def main() -> None:
    """Main function to demonstrate serverless deployment pattern creation."""
    parser = argparse.ArgumentParser(description="Create serverless deployment files for AutoGen")
    parser.add_argument("--output", type=str, default="./serverless-deployment", help="Output directory for deployment files")
    
    args = parser.parse_args()
    create_serverless_files(args.output)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServerless deployment demo interrupted by user")
    except Exception as e:
        print(f"Error running serverless deployment demo: {e}")
        raise
