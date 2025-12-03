// Package selector provides query routing functionality.
package selector

import (
	"context"
	"fmt"
)

// ToolMetadata describes a tool or choice for selection.
type ToolMetadata struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// SingleSelection represents a single selection with index and reason.
type SingleSelection struct {
	Index  int    `json:"index"`
	Reason string `json:"reason"`
}

// SelectorResult contains the selection results.
type SelectorResult struct {
	Selections []SingleSelection `json:"selections"`
}

// Ind returns the index if there's exactly one selection.
func (r *SelectorResult) Ind() (int, error) {
	if len(r.Selections) != 1 {
		return 0, fmt.Errorf("there are %d selections, use Inds() instead", len(r.Selections))
	}
	return r.Selections[0].Index, nil
}

// Reason returns the reason if there's exactly one selection.
func (r *SelectorResult) Reason() (string, error) {
	if len(r.Selections) != 1 {
		return "", fmt.Errorf("there are %d selections, use Reasons() instead", len(r.Selections))
	}
	return r.Selections[0].Reason, nil
}

// Inds returns all selection indices.
func (r *SelectorResult) Inds() []int {
	inds := make([]int, len(r.Selections))
	for i, s := range r.Selections {
		inds[i] = s.Index
	}
	return inds
}

// Reasons returns all selection reasons.
func (r *SelectorResult) Reasons() []string {
	reasons := make([]string, len(r.Selections))
	for i, s := range r.Selections {
		reasons[i] = s.Reason
	}
	return reasons
}

// Selector is the interface for query routing selectors.
type Selector interface {
	// Select chooses from the given choices based on the query.
	Select(ctx context.Context, choices []ToolMetadata, query string) (*SelectorResult, error)
	// Name returns the name of the selector.
	Name() string
}

// BaseSelector provides a base implementation of Selector.
type BaseSelector struct {
	name string
}

// BaseSelectorOption configures a BaseSelector.
type BaseSelectorOption func(*BaseSelector)

// WithSelectorName sets the selector name.
func WithSelectorName(name string) BaseSelectorOption {
	return func(s *BaseSelector) {
		s.name = name
	}
}

// NewBaseSelector creates a new BaseSelector.
func NewBaseSelector(opts ...BaseSelectorOption) *BaseSelector {
	s := &BaseSelector{
		name: "BaseSelector",
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Name returns the name of the selector.
func (s *BaseSelector) Name() string {
	return s.name
}

// Select is a no-op implementation.
func (s *BaseSelector) Select(ctx context.Context, choices []ToolMetadata, query string) (*SelectorResult, error) {
	return &SelectorResult{Selections: []SingleSelection{}}, nil
}

// Ensure BaseSelector implements Selector.
var _ Selector = (*BaseSelector)(nil)

// BuildChoicesText converts choices to enumerated text format.
func BuildChoicesText(choices []ToolMetadata) string {
	var result string
	for i, choice := range choices {
		text := fmt.Sprintf("(%d) %s", i+1, choice.Description)
		if i > 0 {
			result += "\n\n"
		}
		result += text
	}
	return result
}
