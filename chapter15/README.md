# Chapter 11: Testing Strategies

This directory contains code examples for Chapter 11 of "The Complete AutoGen v0.5 Blueprint" book, focusing on testing strategies for AutoGen applications.

## Examples Overview

1. **Unit Testing** (`01_unit_testing.py`)
   - Creating mock LLM responses
   - Testing agent behavior
   - Testing agent configuration

2. **Integration Testing** (`02_integration_testing.py`)
   - Testing agent interactions
   - Testing conversation flows
   - Testing group chat dynamics

3. **Evaluation Framework** (`03_evaluation_framework.py`)
   - Defining performance metrics
   - Creating a benchmarking system
   - Analyzing and visualizing results

## Requirements

- AutoGen v0.5+
- Python 3.10+
- Required packages:
  - `autogen`
  - `pytest`
  - `matplotlib` (for visualization)

## Running the Examples

1. Install the required dependencies:
   ```bash
   pip install "autogen>=0.5.7" pytest matplotlib
   ```

2. Run the unit tests:
   ```bash
   pytest 01_unit_testing.py -v
   ```

3. Run the integration tests:
   ```bash
   pytest 02_integration_testing.py -v
   ```

4. Run the evaluation framework:
   ```bash
   python 03_evaluation_framework.py
   ```

## Key Testing Concepts

### Unit Testing
- **Mock LLM Responses**: Create predictable responses for testing without API calls
- **Agent Behavior Testing**: Verify that agents process messages and use tools correctly
- **Configuration Testing**: Ensure agents handle configuration parameters properly

### Integration Testing
- **Agent Interaction Testing**: Verify that agents communicate correctly
- **Conversation Flow Testing**: Test specific conversation patterns and termination
- **Group Chat Testing**: Validate speaker selection and message routing

### Evaluation and Benchmarking
- **Performance Metrics**: Track response time, token usage, and conversation turns
- **Quality Evaluation**: Assess relevance, correctness, and completeness of responses
- **Benchmarking**: Compare different agent configurations systematically

## Directory Structure

A recommended structure for organizing tests in your AutoGen projects:

```
project/
├── src/
│   └── agent_system/
├── tests/
│   ├── unit/
│   │   ├── test_agents.py
│   │   └── test_tools.py
│   ├── integration/
│   │   ├── test_agent_interactions.py
│   │   └── test_group_chat.py
│   ├── system/
│   │   └── test_end_to_end.py
│   ├── conftest.py
│   └── test_data/
└── benchmarks/
    ├── benchmark_runner.py
    └── test_cases/
```

## Best Practices

1. **Use Mocks**: Mock LLM responses and external dependencies
2. **Deterministic Testing**: Set random seeds and control randomness
3. **Test Edge Cases**: Verify behavior with invalid inputs and error conditions
4. **Continuous Testing**: Integrate tests into CI/CD pipelines
5. **Benchmark Regularly**: Track performance metrics over time
6. **Separate Test Data**: Keep test cases and expected outputs in separate files
7. **Test Real-World Scenarios**: Include tests that mimic actual usage patterns
