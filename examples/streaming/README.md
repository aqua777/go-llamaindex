# Streaming Examples

This directory contains examples demonstrating streaming responses in `go-llamaindex`.

## Examples

### 1. Basic Streaming (`basic_streaming/`)

Demonstrates basic streaming with LLM.

**Features:**
- `Stream()` for prompt streaming
- `StreamChat()` for chat message streaming
- Token counting during streaming
- Timeout handling with context
- Progress indicators
- Multi-turn streaming conversations

**Run:**
```bash
cd basic_streaming && go run main.go
```

### 2. Chat Engine Streaming (`chat_streaming/`)

Demonstrates streaming with various chat engines.

**Features:**
- SimpleChatEngine streaming
- ContextChatEngine (RAG) streaming
- CondensePlusContextChatEngine streaming
- `Consume()` for blocking full response
- Source nodes from RAG retrieval
- Multi-turn conversations with memory

**Run:**
```bash
cd chat_streaming && go run main.go
```

## Key Concepts

### LLM Streaming

```go
// Basic prompt streaming
streamChan, err := llm.Stream(ctx, "Your prompt here")
for token := range streamChan {
    fmt.Print(token)
}

// Chat message streaming (returns StreamToken)
tokenChan, err := llm.StreamChat(ctx, messages)
for token := range tokenChan {
    fmt.Print(token.Delta)
    if token.FinishReason != "" {
        // Generation complete
    }
}
```

### StreamToken

```go
type StreamToken struct {
    Delta        string  // New content in this token
    FinishReason string  // Why generation stopped (if applicable)
}
```

### Chat Engine Streaming

```go
// Stream from chat engine
streamResp, err := chatEngine.StreamChat(ctx, "Your message")

// Option 1: Process tokens as they arrive
for token := range streamResp.ResponseChan {
    fmt.Print(token)
}

// Option 2: Block and get full response
fullResponse := streamResp.Consume()

// Check completion status
if streamResp.IsDone() {
    // Streaming complete
}

// Access source nodes (for RAG engines)
for _, node := range streamResp.SourceNodes {
    fmt.Printf("Source: %s (score: %.2f)\n", node.Node.Text, node.Score)
}
```

### StreamingChatResponse

```go
type StreamingChatResponse struct {
    ResponseChan <-chan string          // Channel for streaming tokens
    SourceNodes  []schema.NodeWithScore // Retrieved context (RAG)
    Sources      []ToolSource           // Tool outputs used
}

// Methods
func (r *StreamingChatResponse) Response() string   // Get accumulated response
func (r *StreamingChatResponse) IsDone() bool       // Check if complete
func (r *StreamingChatResponse) Consume() string    // Block and get full response
```

### Streaming with Timeout

```go
// Create context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Stream will respect the timeout
streamChan, err := llm.Stream(ctx, prompt)
for token := range streamChan {
    fmt.Print(token)
}
```

### Streaming with Progress

```go
streamChan, err := llm.Stream(ctx, prompt)

var response strings.Builder
tokenCount := 0
startTime := time.Now()

for token := range streamChan {
    response.WriteString(token)
    tokenCount++
    
    // Update progress
    elapsed := time.Since(startTime)
    fmt.Printf("\r[Tokens: %d, Rate: %.1f/s]", tokenCount, float64(tokenCount)/elapsed.Seconds())
}
```

### Chat Engine Types with Streaming

| Engine | Description | Streaming Support |
|--------|-------------|-------------------|
| `SimpleChatEngine` | Basic chat with memory | ✅ `StreamChat()` |
| `ContextChatEngine` | RAG-based chat | ✅ `StreamChat()` + SourceNodes |
| `CondensePlusContextChatEngine` | Question condensing + RAG | ✅ `StreamChat()` + SourceNodes |

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI LLM streaming
