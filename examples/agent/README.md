# Agent Examples

This directory contains examples demonstrating various agent patterns in go-llamaindex.

## Examples

### 1. ReAct Agent (`react_agent/`)

Demonstrates the ReAct (Reasoning and Acting) agent pattern with calculator tools.

**Key Concepts:**
- Thought-Action-Observation loop
- Tool execution
- Multi-step reasoning

**Run:**
```bash
cd react_agent && go run main.go
```

### 2. ReAct Agent with Query Engine (`react_agent_with_query_engine/`)

Shows how to combine a ReAct agent with a query engine for RAG-enhanced reasoning.

**Key Concepts:**
- QueryEngineTool integration
- Knowledge-augmented agents
- Combining retrieval with reasoning

**Run:**
```bash
cd react_agent_with_query_engine && go run main.go
```

### 3. Function Calling Agent (`function_calling_agent/`)

Demonstrates using OpenAI's function calling capabilities for tool use.

**Key Concepts:**
- Native LLM tool calling
- FunctionCallingReActAgent
- Structured tool definitions

**Run:**
```bash
cd function_calling_agent && go run main.go
```

### 4. Agent with Retrieval (`agent_retrieval/`)

Shows how to use a retriever tool with an agent for document retrieval.

**Key Concepts:**
- RetrieverTool integration
- Document retrieval in agent loop
- Combining search with reasoning

**Run:**
```bash
cd agent_retrieval && go run main.go
```

### 5. Workflow Agent (`workflow_agent/`)

Demonstrates implementing an agent using the workflow system.

**Key Concepts:**
- Event-driven agent execution
- Workflow-based tool calling
- Stateful agent workflows

**Run:**
```bash
cd workflow_agent && go run main.go
```

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Agent Types

### ReActAgent
Uses text-based reasoning with explicit Thought/Action/Observation format. Works with any LLM.

### FunctionCallingReActAgent
Uses native LLM tool calling (e.g., OpenAI function calling). More reliable tool invocation.

### SimpleAgent
Basic agent without tools. Just LLM + memory for conversation.

## Related Python Examples

These examples correspond to the following Python notebooks:
- `agent/react_agent.ipynb`
- `agent/react_agent_with_query_engine.ipynb`
- `agent/openai_agent_with_query_engine.ipynb`
- `agent/openai_agent_retrieval.ipynb`
- `workflow/function_calling_agent.ipynb`
- `workflow/react_agent.ipynb`
