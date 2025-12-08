package postprocessor

import (
	"context"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// KeywordPostprocessor filters nodes by required and excluded keywords.
type KeywordPostprocessor struct {
	*BaseNodePostprocessor
	requiredKeywords []string
	excludeKeywords  []string
	caseSensitive    bool
}

// KeywordPostprocessorOption configures a KeywordPostprocessor.
type KeywordPostprocessorOption func(*KeywordPostprocessor)

// WithRequiredKeywords sets the required keywords.
func WithRequiredKeywords(keywords []string) KeywordPostprocessorOption {
	return func(p *KeywordPostprocessor) {
		p.requiredKeywords = keywords
	}
}

// WithExcludeKeywords sets the excluded keywords.
func WithExcludeKeywords(keywords []string) KeywordPostprocessorOption {
	return func(p *KeywordPostprocessor) {
		p.excludeKeywords = keywords
	}
}

// WithCaseSensitive sets whether keyword matching is case sensitive.
func WithCaseSensitive(caseSensitive bool) KeywordPostprocessorOption {
	return func(p *KeywordPostprocessor) {
		p.caseSensitive = caseSensitive
	}
}

// NewKeywordPostprocessor creates a new KeywordPostprocessor.
func NewKeywordPostprocessor(opts ...KeywordPostprocessorOption) *KeywordPostprocessor {
	p := &KeywordPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(
			WithPostprocessorName("KeywordPostprocessor"),
		),
		requiredKeywords: []string{},
		excludeKeywords:  []string{},
		caseSensitive:    false,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes filters nodes based on keyword requirements.
func (p *KeywordPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	result := make([]schema.NodeWithScore, 0, len(nodes))

	for _, nodeWithScore := range nodes {
		content := nodeWithScore.Node.GetContent(schema.MetadataModeNone)

		// Check required keywords
		if len(p.requiredKeywords) > 0 {
			if !p.containsAllKeywords(content, p.requiredKeywords) {
				continue
			}
		}

		// Check excluded keywords
		if len(p.excludeKeywords) > 0 {
			if p.containsAnyKeyword(content, p.excludeKeywords) {
				continue
			}
		}

		result = append(result, nodeWithScore)
	}

	return result, nil
}

// containsAllKeywords checks if content contains all keywords.
func (p *KeywordPostprocessor) containsAllKeywords(content string, keywords []string) bool {
	checkContent := content
	if !p.caseSensitive {
		checkContent = strings.ToLower(content)
	}

	for _, keyword := range keywords {
		checkKeyword := keyword
		if !p.caseSensitive {
			checkKeyword = strings.ToLower(keyword)
		}

		if !strings.Contains(checkContent, checkKeyword) {
			return false
		}
	}

	return true
}

// containsAnyKeyword checks if content contains any of the keywords.
func (p *KeywordPostprocessor) containsAnyKeyword(content string, keywords []string) bool {
	checkContent := content
	if !p.caseSensitive {
		checkContent = strings.ToLower(content)
	}

	for _, keyword := range keywords {
		checkKeyword := keyword
		if !p.caseSensitive {
			checkKeyword = strings.ToLower(keyword)
		}

		if strings.Contains(checkContent, checkKeyword) {
			return true
		}
	}

	return false
}

// RequiredKeywords returns the required keywords.
func (p *KeywordPostprocessor) RequiredKeywords() []string {
	return p.requiredKeywords
}

// ExcludeKeywords returns the excluded keywords.
func (p *KeywordPostprocessor) ExcludeKeywords() []string {
	return p.excludeKeywords
}

// Ensure KeywordPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*KeywordPostprocessor)(nil)
