# LLM Provider Examples

This directory contains examples demonstrating LLM provider usage in `go-llamaindex`.

## Examples

### 1. OpenAI LLM (`openai_llm/`)

Demonstrates OpenAI LLM capabilities.

**Features:**
- Basic and custom model initialization
- Complete (single prompt)
- Chat (multi-turn conversation)
- Streaming (Stream and StreamChat)
- Model metadata
- Tool/function calling
- Structured output (JSON mode)
- Model comparison

**Run:**
```bash
export OPENAI_API_KEY="your-api-key"
cd openai_llm && go run main.go
```

### 2. AWS Bedrock (`bedrock/`)

Demonstrates AWS Bedrock LLM and Embedding capabilities.

**Features:**
- LLM: Claude, Nova, Llama, Mistral, Cohere models
- Embeddings: Titan, Cohere models
- Complete, Chat, Streaming
- Tool/function calling
- Batch embeddings
- Model comparison

**Run:**
```bash
# Configure AWS credentials (via env vars, profile, or IAM role)
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
cd bedrock && go run main.go
```

### 3. Azure OpenAI (`azure_openai/`)

Demonstrates Azure OpenAI LLM capabilities.

**Features:**
- Environment variable initialization
- Options pattern configuration
- Explicit configuration
- All LLM operations (Complete, Chat, Stream, etc.)
- Tool calling
- JSON mode output
- Azure-specific considerations

**Run:**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
cd azure_openai && go run main.go
```

## Key Concepts

### LLM Interface

```go
type LLM interface {
    Complete(ctx context.Context, prompt string) (string, error)
    Chat(ctx context.Context, messages []ChatMessage) (string, error)
    Stream(ctx context.Context, prompt string) (<-chan string, error)
}
```

### OpenAI LLM

```go
// Default initialization (from env vars)
llm := llm.NewOpenAILLM("", "", "")

// Custom model
llm := llm.NewOpenAILLM("", "gpt-4", "")

// Custom base URL (for proxies or compatible APIs)
llm := llm.NewOpenAILLM("https://api.example.com/v1", "gpt-3.5-turbo", "api-key")

// With existing client
llm := llm.NewOpenAILLMWithClient(client, "gpt-4")
```

### Azure OpenAI LLM

```go
// From environment variables
llm := llm.NewAzureOpenAILLM()

// With options
llm := llm.NewAzureOpenAILLM(
    llm.WithAzureDeployment("gpt-4"),
    llm.WithAzureAPIVersion("2024-02-15-preview"),
)

// Explicit configuration
llm := llm.NewAzureOpenAILLMWithConfig(
    "https://your-resource.openai.azure.com/",
    "your-api-key",
    "gpt-35-turbo",
    "2024-02-15-preview",
)
```

### AWS Bedrock LLM

```go
import "github.com/aqua777/go-llamaindex/llm/bedrock"

// Default (Claude 3.5 Sonnet)
llm := bedrock.New()

// With specific model
llm := bedrock.New(
    bedrock.WithModel(bedrock.Claude3Haiku),
    bedrock.WithRegion("us-east-1"),
    bedrock.WithMaxTokens(2048),
)

// Available models: Claude35SonnetV2, Claude3Haiku, NovaProV1, Llama33_70BInstruct, etc.
```

### AWS Bedrock Embeddings

```go
import "github.com/aqua777/go-llamaindex/llm/bedrock"

// Default (Titan Embed V2)
emb := bedrock.NewEmbedding()

// With specific model and dimensions
emb := bedrock.NewEmbedding(
    bedrock.WithEmbeddingModel(bedrock.TitanEmbedTextV2),
    bedrock.WithEmbeddingDimensions(512),
)

// Get embedding
vector, err := emb.GetTextEmbedding(ctx, "Hello world")

// Batch embeddings
vectors, err := emb.GetTextEmbeddingsBatch(ctx, texts, nil)
```

### Chat Messages

```go
messages := []llm.ChatMessage{
    llm.NewSystemMessage("You are a helpful assistant."),
    llm.NewUserMessage("Hello!"),
    llm.NewAssistantMessage("Hi there!"),
}

response, err := llm.Chat(ctx, messages)
```

### Streaming

```go
// Basic streaming
streamChan, err := llm.Stream(ctx, "Your prompt")
for token := range streamChan {
    fmt.Print(token)
}

// Chat streaming (returns StreamToken)
tokenChan, err := llm.StreamChat(ctx, messages)
for token := range tokenChan {
    fmt.Print(token.Delta)
    if token.FinishReason != "" {
        // Generation complete
    }
}
```

### Tool Calling

```go
tools := []*llm.ToolMetadata{
    {
        Name:        "get_weather",
        Description: "Get weather for a location",
        Parameters: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "location": map[string]interface{}{
                    "type": "string",
                },
            },
            "required": []string{"location"},
        },
    },
}

resp, err := llm.ChatWithTools(ctx, messages, tools, nil)
if resp.Message != nil {
    for _, block := range resp.Message.Blocks {
        if block.Type == llm.ContentBlockTypeToolCall {
            fmt.Printf("Tool: %s, Args: %s\n", block.ToolCall.Name, block.ToolCall.Arguments)
        }
    }
}
```

### Structured Output (JSON Mode)

```go
format := &llm.ResponseFormat{
    Type: "json_object",
}

jsonResp, err := llm.ChatWithFormat(ctx, messages, format)
```

### Model Metadata

```go
metadata := llm.Metadata()
fmt.Printf("Model: %s\n", metadata.ModelName)
fmt.Printf("Context Window: %d\n", metadata.ContextWindow)
fmt.Printf("Function Calling: %v\n", metadata.IsFunctionCalling)
```

## Supported LLM Providers

| Provider | Package | Status |
|----------|---------|--------|
| OpenAI | `llm.NewOpenAILLM()` | ✅ |
| Azure OpenAI | `llm.NewAzureOpenAILLM()` | ✅ |
| Anthropic | `llm.NewAnthropicLLM()` | ✅ |
| AWS Bedrock | `bedrock.New()` | ✅ |
| Ollama | `llm.NewOllamaLLM()` | ✅ |
| Cohere | `llm.NewCohereLLM()` | ✅ |
| Groq | `llm.NewGroqLLM()` | ✅ |
| DeepSeek | `llm.NewDeepSeekLLM()` | ✅ |
| Mistral AI | `llm.NewMistralLLM()` | ✅ |

## Environment Variables

### OpenAI
- `OPENAI_API_KEY` - API key
- `OPENAI_URL` - Custom base URL (optional)

### Azure OpenAI
- `AZURE_OPENAI_ENDPOINT` - Azure resource endpoint
- `AZURE_OPENAI_API_KEY` - API key
- `AZURE_OPENAI_DEPLOYMENT` - Deployment name

### Anthropic
- `ANTHROPIC_API_KEY` - API key

### Ollama
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)

### Cohere
- `COHERE_API_KEY` - API key

### AWS Bedrock
- `AWS_REGION` / `AWS_DEFAULT_REGION` - AWS region
- `AWS_ACCESS_KEY_ID` - Access key (optional if using IAM role)
- `AWS_SECRET_ACCESS_KEY` - Secret key (optional if using IAM role)
- `AWS_SESSION_TOKEN` - Session token (optional)

### Groq
- `GROQ_API_KEY` - API key

### DeepSeek
- `DEEPSEEK_API_KEY` - API key

### Mistral AI
- `MISTRAL_API_KEY` - API key
