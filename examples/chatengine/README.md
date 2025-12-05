# Chat Engine Examples

This example demonstrates different chat engine modes in Go, corresponding to the Python chat engine examples.

## Overview

Learn how to use different chat engines:
1. **SimpleChatEngine** - Direct LLM chat without knowledge base
2. **ContextChatEngine** - RAG-enhanced chat with retrieval
3. **CondensePlusContextChatEngine** - Query condensation with context

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/chatengine
go run main.go
```

## Chat Engine Types

### SimpleChatEngine
- Direct conversation with LLM
- Maintains conversation history in memory
- No external knowledge base
- Best for: General conversation, creative tasks

### ContextChatEngine
- Retrieves relevant context for each message
- Augments LLM responses with retrieved information
- Best for: Q&A over documents, knowledge-grounded chat

### CondensePlusContextChatEngine
- Condenses conversation history into standalone questions
- Better retrieval for follow-up questions
- Best for: Multi-turn conversations with context

## Components Used

- `chatengine.SimpleChatEngine` - Direct LLM chat
- `chatengine.ContextChatEngine` - RAG-enhanced chat
- `chatengine.CondensePlusContextChatEngine` - Condense + context
- `memory.ChatMemoryBuffer` - Conversation memory
- `rag/retriever.Retriever` - Context retrieval
