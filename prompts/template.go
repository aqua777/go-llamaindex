package prompts

import (
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

// templateVarRegex matches {variable} placeholders in templates.
var templateVarRegex = regexp.MustCompile(`\{(\w+)\}`)

// GetTemplateVars extracts variable names from a template string.
func GetTemplateVars(template string) []string {
	matches := templateVarRegex.FindAllStringSubmatch(template, -1)
	vars := make([]string, 0, len(matches))
	seen := make(map[string]bool)
	for _, match := range matches {
		if len(match) > 1 && !seen[match[1]] {
			vars = append(vars, match[1])
			seen[match[1]] = true
		}
	}
	return vars
}

// FormatString formats a template string with the given variables.
func FormatString(template string, vars map[string]string) string {
	result := template
	for key, value := range vars {
		placeholder := "{" + key + "}"
		result = strings.ReplaceAll(result, placeholder, value)
	}
	return result
}

// BasePromptTemplate is the interface for all prompt templates.
type BasePromptTemplate interface {
	// Format formats the prompt into a string.
	Format(vars map[string]string) string
	// FormatMessages formats the prompt into chat messages.
	FormatMessages(vars map[string]string) []llm.ChatMessage
	// GetTemplate returns the raw template string.
	GetTemplate() string
	// GetTemplateVars returns the variable names in the template.
	GetTemplateVars() []string
	// PartialFormat creates a new template with some variables pre-filled.
	PartialFormat(vars map[string]string) BasePromptTemplate
	// GetPromptType returns the prompt type.
	GetPromptType() PromptType
	// GetMetadata returns the prompt metadata.
	GetMetadata() map[string]interface{}
}

// PromptTemplate is a simple string-based prompt template.
type PromptTemplate struct {
	// Template is the template string with {variable} placeholders.
	Template string
	// TemplateVars are the variable names extracted from the template.
	TemplateVars []string
	// PromptType is the type/category of this prompt.
	PromptType PromptType
	// Metadata contains additional prompt metadata.
	Metadata map[string]interface{}
	// PartialVars are pre-filled variables.
	PartialVars map[string]string
}

// NewPromptTemplate creates a new PromptTemplate.
func NewPromptTemplate(template string, promptType PromptType) *PromptTemplate {
	return &PromptTemplate{
		Template:     template,
		TemplateVars: GetTemplateVars(template),
		PromptType:   promptType,
		Metadata:     make(map[string]interface{}),
		PartialVars:  make(map[string]string),
	}
}

// NewPromptTemplateWithMetadata creates a new PromptTemplate with metadata.
func NewPromptTemplateWithMetadata(template string, promptType PromptType, metadata map[string]interface{}) *PromptTemplate {
	pt := NewPromptTemplate(template, promptType)
	pt.Metadata = metadata
	return pt
}

// Format formats the prompt into a string.
func (pt *PromptTemplate) Format(vars map[string]string) string {
	// Merge partial vars with provided vars (provided vars take precedence)
	allVars := make(map[string]string)
	for k, v := range pt.PartialVars {
		allVars[k] = v
	}
	for k, v := range vars {
		allVars[k] = v
	}
	return FormatString(pt.Template, allVars)
}

// FormatMessages formats the prompt into chat messages.
func (pt *PromptTemplate) FormatMessages(vars map[string]string) []llm.ChatMessage {
	formatted := pt.Format(vars)
	return []llm.ChatMessage{
		llm.NewUserMessage(formatted),
	}
}

// GetTemplate returns the raw template string.
func (pt *PromptTemplate) GetTemplate() string {
	return pt.Template
}

// GetTemplateVars returns the variable names in the template.
func (pt *PromptTemplate) GetTemplateVars() []string {
	return pt.TemplateVars
}

// PartialFormat creates a new template with some variables pre-filled.
func (pt *PromptTemplate) PartialFormat(vars map[string]string) BasePromptTemplate {
	newPT := &PromptTemplate{
		Template:     pt.Template,
		TemplateVars: pt.TemplateVars,
		PromptType:   pt.PromptType,
		Metadata:     pt.Metadata,
		PartialVars:  make(map[string]string),
	}
	// Copy existing partial vars
	for k, v := range pt.PartialVars {
		newPT.PartialVars[k] = v
	}
	// Add new partial vars
	for k, v := range vars {
		newPT.PartialVars[k] = v
	}
	return newPT
}

// GetPromptType returns the prompt type.
func (pt *PromptTemplate) GetPromptType() PromptType {
	return pt.PromptType
}

// GetMetadata returns the prompt metadata.
func (pt *PromptTemplate) GetMetadata() map[string]interface{} {
	return pt.Metadata
}

// ChatPromptTemplate is a message-based prompt template.
type ChatPromptTemplate struct {
	// MessageTemplates are the chat message templates.
	MessageTemplates []llm.ChatMessage
	// TemplateVars are the variable names extracted from all messages.
	TemplateVars []string
	// PromptType is the type/category of this prompt.
	PromptType PromptType
	// Metadata contains additional prompt metadata.
	Metadata map[string]interface{}
	// PartialVars are pre-filled variables.
	PartialVars map[string]string
}

// NewChatPromptTemplate creates a new ChatPromptTemplate.
func NewChatPromptTemplate(messages []llm.ChatMessage, promptType PromptType) *ChatPromptTemplate {
	// Extract all template vars from all messages
	var allVars []string
	seen := make(map[string]bool)
	for _, msg := range messages {
		vars := GetTemplateVars(msg.Content)
		for _, v := range vars {
			if !seen[v] {
				allVars = append(allVars, v)
				seen[v] = true
			}
		}
	}

	return &ChatPromptTemplate{
		MessageTemplates: messages,
		TemplateVars:     allVars,
		PromptType:       promptType,
		Metadata:         make(map[string]interface{}),
		PartialVars:      make(map[string]string),
	}
}

// ChatPromptTemplateFromMessages creates a ChatPromptTemplate from role-content pairs.
func ChatPromptTemplateFromMessages(messages []struct {
	Role    llm.MessageRole
	Content string
}, promptType PromptType) *ChatPromptTemplate {
	chatMessages := make([]llm.ChatMessage, len(messages))
	for i, m := range messages {
		chatMessages[i] = llm.NewChatMessage(m.Role, m.Content)
	}
	return NewChatPromptTemplate(chatMessages, promptType)
}

// Format formats the prompt into a string (concatenates all messages).
func (cpt *ChatPromptTemplate) Format(vars map[string]string) string {
	messages := cpt.FormatMessages(vars)
	var parts []string
	for _, msg := range messages {
		parts = append(parts, string(msg.Role)+": "+msg.Content)
	}
	return strings.Join(parts, "\n\n")
}

// FormatMessages formats the prompt into chat messages.
func (cpt *ChatPromptTemplate) FormatMessages(vars map[string]string) []llm.ChatMessage {
	// Merge partial vars with provided vars
	allVars := make(map[string]string)
	for k, v := range cpt.PartialVars {
		allVars[k] = v
	}
	for k, v := range vars {
		allVars[k] = v
	}

	messages := make([]llm.ChatMessage, len(cpt.MessageTemplates))
	for i, tmpl := range cpt.MessageTemplates {
		messages[i] = llm.ChatMessage{
			Role:    tmpl.Role,
			Content: FormatString(tmpl.Content, allVars),
			Name:    tmpl.Name,
		}
	}
	return messages
}

// GetTemplate returns the raw template (concatenated messages).
func (cpt *ChatPromptTemplate) GetTemplate() string {
	var parts []string
	for _, msg := range cpt.MessageTemplates {
		parts = append(parts, string(msg.Role)+": "+msg.Content)
	}
	return strings.Join(parts, "\n\n")
}

// GetTemplateVars returns the variable names in the template.
func (cpt *ChatPromptTemplate) GetTemplateVars() []string {
	return cpt.TemplateVars
}

// PartialFormat creates a new template with some variables pre-filled.
func (cpt *ChatPromptTemplate) PartialFormat(vars map[string]string) BasePromptTemplate {
	newCPT := &ChatPromptTemplate{
		MessageTemplates: cpt.MessageTemplates,
		TemplateVars:     cpt.TemplateVars,
		PromptType:       cpt.PromptType,
		Metadata:         cpt.Metadata,
		PartialVars:      make(map[string]string),
	}
	// Copy existing partial vars
	for k, v := range cpt.PartialVars {
		newCPT.PartialVars[k] = v
	}
	// Add new partial vars
	for k, v := range vars {
		newCPT.PartialVars[k] = v
	}
	return newCPT
}

// GetPromptType returns the prompt type.
func (cpt *ChatPromptTemplate) GetPromptType() PromptType {
	return cpt.PromptType
}

// GetMetadata returns the prompt metadata.
func (cpt *ChatPromptTemplate) GetMetadata() map[string]interface{} {
	return cpt.Metadata
}

// Ensure implementations satisfy the interface.
var _ BasePromptTemplate = (*PromptTemplate)(nil)
var _ BasePromptTemplate = (*ChatPromptTemplate)(nil)
