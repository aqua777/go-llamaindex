# Prompt Examples

This directory contains examples demonstrating prompt template features in `go-llamaindex`.

## Examples

### 1. Advanced Prompts (`advanced_prompts/`)

Demonstrates advanced prompt template features and customization.

**Features:**
- Basic PromptTemplate usage
- ChatPromptTemplate for multi-message prompts
- Partial formatting (pre-filling variables)
- Prompt metadata
- Default prompts
- Custom RAG prompts

**Run:**
```bash
cd advanced_prompts && go run main.go
```

### 2. Prompt Mixin (`prompt_mixin/`)

Demonstrates hierarchical prompt management with PromptMixin.

**Features:**
- BasePromptMixin for prompt management
- Hierarchical prompt organization
- Sub-module prompt access
- Runtime prompt updates
- Prompt inspection

**Run:**
```bash
cd prompt_mixin && go run main.go
```

### 3. Template Features (`template_features/`)

Demonstrates rich prompt template features and patterns.

**Features:**
- Variable extraction
- String formatting
- Prompt types
- ChatPromptTemplate construction
- Partial formatting patterns
- Template composition
- Metadata usage

**Run:**
```bash
cd template_features && go run main.go
```

## Key Concepts

### PromptTemplate

```go
// Create a simple prompt template
prompt := prompts.NewPromptTemplate(
    `Context: {context}
Question: {question}
Answer:`,
    prompts.PromptTypeQuestionAnswer,
)

// Format with variables
formatted := prompt.Format(map[string]string{
    "context":  "Go is a programming language.",
    "question": "What is Go?",
})
```

### ChatPromptTemplate

```go
// Create a chat prompt template
chatPrompt := prompts.NewChatPromptTemplate(
    []llm.ChatMessage{
        llm.NewSystemMessage("You are a {role} assistant."),
        llm.NewUserMessage("{question}"),
    },
    prompts.PromptTypeConversation,
)

// Format as messages
messages := chatPrompt.FormatMessages(map[string]string{
    "role":     "helpful",
    "question": "What is Go?",
})
```

### Partial Formatting

```go
// Create template with many variables
template := prompts.NewPromptTemplate(
    "You are a {language} expert. Explain {topic} at {level} level.",
    prompts.PromptTypeCustom,
)

// Pre-fill some variables
partial := template.PartialFormat(map[string]string{
    "language": "Go",
    "level":    "beginner",
})

// Later, only need to provide remaining variables
final := partial.Format(map[string]string{
    "topic": "goroutines",
})
```

### PromptMixin

```go
type MyComponent struct {
    *prompts.BasePromptMixin
}

func NewMyComponent() *MyComponent {
    c := &MyComponent{
        BasePromptMixin: prompts.NewBasePromptMixin(),
    }
    
    // Set prompts
    c.SetPrompt("main_prompt", prompts.NewPromptTemplate(...))
    
    // Add sub-modules
    c.AddModule("retriever", retrieverComponent)
    
    return c
}

// Get all prompts (including sub-modules)
allPrompts := component.GetPrompts()

// Update prompts (including sub-modules via "module:prompt" syntax)
component.UpdatePrompts(prompts.PromptDictType{
    "main_prompt":           newPrompt,
    "retriever:query_prompt": newQueryPrompt,
})
```

### Prompt Types

| Type | Description |
|------|-------------|
| `PromptTypeSummary` | Summarization prompts |
| `PromptTypeQuestionAnswer` | Q&A prompts |
| `PromptTypeRefine` | Answer refinement |
| `PromptTypeKeywordExtract` | Keyword extraction |
| `PromptTypeKnowledgeTripletExtract` | KG triplet extraction |
| `PromptTypeChoiceSelect` | Choice selection |
| `PromptTypeConversation` | Multi-turn conversation |
| `PromptTypeCustom` | Custom prompts |

### Default Prompts

```go
// Get default prompts
qaPrompt := prompts.DefaultTextQAPrompt
summaryPrompt := prompts.DefaultSummaryPrompt
refinePrompt := prompts.DefaultRefinePrompt

// Or by type
prompt := prompts.GetDefaultPrompt(prompts.PromptTypeQuestionAnswer)
```

### Metadata

```go
prompt := prompts.NewPromptTemplateWithMetadata(
    "Analyze {text}",
    prompts.PromptTypeCustom,
    map[string]interface{}{
        "version": "1.0",
        "author":  "team",
        "tags":    []string{"analysis"},
    },
)

metadata := prompt.GetMetadata()
```

## Environment Variables

Examples using LLM require:
- `OPENAI_API_KEY` - OpenAI API key
