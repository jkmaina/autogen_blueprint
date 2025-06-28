# AutoGen Serverless Deployment

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
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/agents \
     -H "Content-Type: application/json" \
     -d '{"agent_id": "test-agent", "system_message": "You are a helpful assistant.", "model_name": "gpt-4o"}'
   
   # Run a task
   curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/tasks \
     -H "Content-Type: application/json" \
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
