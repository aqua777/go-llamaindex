package prompts

import (
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/stretchr/testify/assert"
)

func TestGetTemplateVars(t *testing.T) {
	tests := []struct {
		template string
		expected []string
	}{
		{"Hello {name}!", []string{"name"}},
		{"Hello {name}, you are {age} years old.", []string{"name", "age"}},
		{"{a} {b} {a}", []string{"a", "b"}}, // duplicates removed
		{"No variables here", []string{}},
		{"{query_str}\n{context_str}", []string{"query_str", "context_str"}},
	}

	for _, tt := range tests {
		vars := GetTemplateVars(tt.template)
		assert.Equal(t, tt.expected, vars)
	}
}

func TestFormatString(t *testing.T) {
	template := "Hello {name}, you are {age} years old."
	vars := map[string]string{
		"name": "Alice",
		"age":  "30",
	}

	result := FormatString(template, vars)
	assert.Equal(t, "Hello Alice, you are 30 years old.", result)
}

func TestPromptTemplate(t *testing.T) {
	template := "Query: {query_str}\nContext: {context_str}"
	pt := NewPromptTemplate(template, PromptTypeQuestionAnswer)

	assert.Equal(t, template, pt.GetTemplate())
	assert.Equal(t, PromptTypeQuestionAnswer, pt.GetPromptType())
	assert.ElementsMatch(t, []string{"query_str", "context_str"}, pt.GetTemplateVars())
}

func TestPromptTemplateFormat(t *testing.T) {
	template := "Query: {query_str}\nContext: {context_str}"
	pt := NewPromptTemplate(template, PromptTypeQuestionAnswer)

	result := pt.Format(map[string]string{
		"query_str":   "What is AI?",
		"context_str": "AI is artificial intelligence.",
	})

	assert.Equal(t, "Query: What is AI?\nContext: AI is artificial intelligence.", result)
}

func TestPromptTemplatePartialFormat(t *testing.T) {
	template := "Query: {query_str}\nContext: {context_str}"
	pt := NewPromptTemplate(template, PromptTypeQuestionAnswer)

	// Partial format with context
	partial := pt.PartialFormat(map[string]string{
		"context_str": "AI is artificial intelligence.",
	})

	// Now format with just query
	result := partial.Format(map[string]string{
		"query_str": "What is AI?",
	})

	assert.Equal(t, "Query: What is AI?\nContext: AI is artificial intelligence.", result)
}

func TestPromptTemplateFormatMessages(t *testing.T) {
	template := "What is {topic}?"
	pt := NewPromptTemplate(template, PromptTypeSimpleInput)

	messages := pt.FormatMessages(map[string]string{"topic": "AI"})

	assert.Len(t, messages, 1)
	assert.Equal(t, llm.MessageRoleUser, messages[0].Role)
	assert.Equal(t, "What is AI?", messages[0].Content)
}

func TestChatPromptTemplate(t *testing.T) {
	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a helpful assistant."),
		llm.NewUserMessage("Query: {query_str}"),
	}
	cpt := NewChatPromptTemplate(messages, PromptTypeQuestionAnswer)

	assert.Equal(t, PromptTypeQuestionAnswer, cpt.GetPromptType())
	assert.ElementsMatch(t, []string{"query_str"}, cpt.GetTemplateVars())
}

func TestChatPromptTemplateFormatMessages(t *testing.T) {
	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a {role}."),
		llm.NewUserMessage("Query: {query_str}"),
	}
	cpt := NewChatPromptTemplate(messages, PromptTypeQuestionAnswer)

	formatted := cpt.FormatMessages(map[string]string{
		"role":      "helpful assistant",
		"query_str": "What is AI?",
	})

	assert.Len(t, formatted, 2)
	assert.Equal(t, llm.MessageRoleSystem, formatted[0].Role)
	assert.Equal(t, "You are a helpful assistant.", formatted[0].Content)
	assert.Equal(t, llm.MessageRoleUser, formatted[1].Role)
	assert.Equal(t, "Query: What is AI?", formatted[1].Content)
}

func TestChatPromptTemplatePartialFormat(t *testing.T) {
	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are a {role}."),
		llm.NewUserMessage("Query: {query_str}"),
	}
	cpt := NewChatPromptTemplate(messages, PromptTypeQuestionAnswer)

	// Partial format with role
	partial := cpt.PartialFormat(map[string]string{"role": "helpful assistant"})

	// Now format with just query
	formatted := partial.FormatMessages(map[string]string{"query_str": "What is AI?"})

	assert.Equal(t, "You are a helpful assistant.", formatted[0].Content)
	assert.Equal(t, "Query: What is AI?", formatted[1].Content)
}

func TestPromptType(t *testing.T) {
	assert.Equal(t, "summary", PromptTypeSummary.String())
	assert.Equal(t, "text_qa", PromptTypeQuestionAnswer.String())
	assert.Equal(t, "refine", PromptTypeRefine.String())
	assert.Equal(t, "custom", PromptTypeCustom.String())
}

func TestBasePromptMixin(t *testing.T) {
	mixin := NewBasePromptMixin()

	// Set a prompt
	prompt := NewPromptTemplate("Hello {name}", PromptTypeCustom)
	mixin.SetPrompt("greeting", prompt)

	// Get the prompt
	retrieved := mixin.GetPrompt("greeting")
	assert.NotNil(t, retrieved)
	assert.Equal(t, "Hello {name}", retrieved.GetTemplate())

	// Get all prompts
	allPrompts := mixin.GetPrompts()
	assert.Len(t, allPrompts, 1)
	assert.Contains(t, allPrompts, "greeting")
}

func TestBasePromptMixinWithModules(t *testing.T) {
	// Create parent mixin
	parent := NewBasePromptMixin()
	parent.SetPrompt("parent_prompt", NewPromptTemplate("Parent: {text}", PromptTypeCustom))

	// Create child mixin
	child := NewBasePromptMixin()
	child.SetPrompt("child_prompt", NewPromptTemplate("Child: {text}", PromptTypeCustom))

	// Add child as module
	parent.AddModule("child", child)

	// Get all prompts (should include child prompts with prefix)
	allPrompts := parent.GetPrompts()
	assert.Len(t, allPrompts, 2)
	assert.Contains(t, allPrompts, "parent_prompt")
	assert.Contains(t, allPrompts, "child:child_prompt")
}

func TestBasePromptMixinUpdatePrompts(t *testing.T) {
	parent := NewBasePromptMixin()
	child := NewBasePromptMixin()
	child.SetPrompt("greeting", NewPromptTemplate("Old greeting", PromptTypeCustom))
	parent.AddModule("child", child)

	// Update child prompt via parent
	parent.UpdatePrompts(PromptDictType{
		"child:greeting": NewPromptTemplate("New greeting", PromptTypeCustom),
	})

	// Verify update
	childPrompt := child.GetPrompt("greeting")
	assert.Equal(t, "New greeting", childPrompt.GetTemplate())
}

func TestDefaultPrompts(t *testing.T) {
	// Test that default prompts are properly initialized
	assert.NotNil(t, DefaultSummaryPrompt)
	assert.NotNil(t, DefaultTextQAPrompt)
	assert.NotNil(t, DefaultRefinePrompt)

	// Test GetDefaultPrompt
	summaryPrompt := GetDefaultPrompt(PromptTypeSummary)
	assert.NotNil(t, summaryPrompt)
	assert.Equal(t, PromptTypeSummary, summaryPrompt.GetPromptType())

	qaPrompt := GetDefaultPrompt(PromptTypeQuestionAnswer)
	assert.NotNil(t, qaPrompt)
	assert.Equal(t, PromptTypeQuestionAnswer, qaPrompt.GetPromptType())
}

func TestDefaultTextQAPromptFormat(t *testing.T) {
	result := DefaultTextQAPrompt.Format(map[string]string{
		"context_str": "AI is artificial intelligence.",
		"query_str":   "What is AI?",
	})

	assert.Contains(t, result, "AI is artificial intelligence.")
	assert.Contains(t, result, "What is AI?")
	assert.Contains(t, result, "Given the context information")
}

func TestDefaultRefinePromptFormat(t *testing.T) {
	result := DefaultRefinePrompt.Format(map[string]string{
		"query_str":       "What is AI?",
		"existing_answer": "AI is technology.",
		"context_msg":     "More context here.",
	})

	assert.Contains(t, result, "What is AI?")
	assert.Contains(t, result, "AI is technology.")
	assert.Contains(t, result, "More context here.")
	assert.Contains(t, result, "refine")
}

func TestPromptTemplateMetadata(t *testing.T) {
	metadata := map[string]interface{}{
		"version": "1.0",
		"author":  "test",
	}
	pt := NewPromptTemplateWithMetadata("Hello {name}", PromptTypeCustom, metadata)

	assert.Equal(t, "1.0", pt.GetMetadata()["version"])
	assert.Equal(t, "test", pt.GetMetadata()["author"])
}
