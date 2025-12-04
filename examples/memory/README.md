# Memory Examples

This directory contains examples demonstrating chat memory management in go-llamaindex.

## Examples

### 1. Basic Memory (`basic_memory/`)

Demonstrates fundamental memory types and operations.

**Features:**
- SimpleMemory: Store all messages without limits
- ChatMemoryBuffer: Token-limited message storage
- Custom tokenizers
- Memory operations (Put, Get, Set, Reset)
- Multiple conversations with different keys

**Run:**
```bash
cd basic_memory && go run main.go
```

### 2. Chat Summary Memory (`chat_summary_memory/`)

Shows how to use summary memory for long conversations.

**Features:**
- Automatic summarization of older messages
- Custom summarization prompts
- Token limit management
- Comparison with regular buffer
- Auto-configuration from LLM context window

**Run:**
```bash
cd chat_summary_memory && go run main.go
```

**Requires:** `OPENAI_API_KEY` environment variable

### 3. Custom Memory (`custom_memory/`)

Demonstrates implementing custom memory types.

**Custom Implementations:**
- SlidingWindowMemory: Fixed-size window of recent messages
- TimestampedMemory: Track message timing, filter by recency
- TaggedMemory: Categorize and filter by tags
- PriorityMemory: Keep highest priority messages
- ConversationTurnMemory: Group by user-assistant turns

**Run:**
```bash
cd custom_memory && go run main.go
```

## Key Concepts

### Memory Interface

All memory types implement the `memory.Memory` interface:

```go
type Memory interface {
    Get(ctx context.Context, input string) ([]llm.ChatMessage, error)
    GetAll(ctx context.Context) ([]llm.ChatMessage, error)
    Put(ctx context.Context, message llm.ChatMessage) error
    PutMessages(ctx context.Context, messages []llm.ChatMessage) error
    Set(ctx context.Context, messages []llm.ChatMessage) error
    Reset(ctx context.Context) error
}
```

### Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| SimpleMemory | Stores all messages | Short conversations |
| ChatMemoryBuffer | Token-limited storage | Context window management |
| ChatSummaryMemoryBuffer | Summarizes old messages | Long conversations |

### Usage with Agents

```go
// Create memory
chatMemory := memory.NewChatMemoryBuffer(
    memory.WithTokenLimit(2000),
)

// Use with agent
agent := agent.NewReActAgentFromDefaults(
    llmInstance,
    tools,
    agent.WithAgentMemory(chatMemory),
)

// Memory persists across chat calls
response1, _ := agent.Chat(ctx, "Hello")
response2, _ := agent.Chat(ctx, "What did I just say?") // Remembers previous message
```

### Token Management

```go
// Fixed token limit
buffer := memory.NewChatMemoryBuffer(
    memory.WithTokenLimit(3000),
)

// Auto-detect from LLM
buffer, _ := memory.NewChatMemoryBufferFromDefaults(
    nil,        // No initial history
    llmInstance, // LLM for context window detection
    0,          // 0 = auto-detect
)
```

### Summary Memory

```go
summaryMemory := memory.NewChatSummaryMemoryBuffer(
    memory.WithSummaryLLM(llmInstance),
    memory.WithSummaryTokenLimit(2000),
    memory.WithSummarizePrompt("Summarize the key points of this conversation."),
)
```

## Environment Variables

- `OPENAI_API_KEY` - Required for chat_summary_memory example
