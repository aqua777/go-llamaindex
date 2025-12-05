package postprocessor

import (
	"context"
	"regexp"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// PIIType represents a type of personally identifiable information.
type PIIType string

const (
	// PIITypeEmail represents email addresses.
	PIITypeEmail PIIType = "email"
	// PIITypePhone represents phone numbers.
	PIITypePhone PIIType = "phone"
	// PIITypeSSN represents social security numbers.
	PIITypeSSN PIIType = "ssn"
	// PIITypeCreditCard represents credit card numbers.
	PIITypeCreditCard PIIType = "credit_card"
	// PIITypeIPAddress represents IP addresses.
	PIITypeIPAddress PIIType = "ip_address"
	// PIITypeName represents person names (requires NER, basic implementation).
	PIITypeName PIIType = "name"
	// PIITypeAddress represents physical addresses (basic implementation).
	PIITypeAddress PIIType = "address"
)

// PIIPattern defines a regex pattern for detecting PII.
type PIIPattern struct {
	Type    PIIType
	Pattern *regexp.Regexp
	Mask    string
}

// DefaultPIIPatterns returns the default PII detection patterns.
func DefaultPIIPatterns() []PIIPattern {
	return []PIIPattern{
		{
			Type:    PIITypeEmail,
			Pattern: regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`),
			Mask:    "[EMAIL]",
		},
		{
			Type:    PIITypePhone,
			Pattern: regexp.MustCompile(`(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`),
			Mask:    "[PHONE]",
		},
		{
			Type:    PIITypeSSN,
			Pattern: regexp.MustCompile(`\d{3}[-\s]?\d{2}[-\s]?\d{4}`),
			Mask:    "[SSN]",
		},
		{
			Type:    PIITypeCreditCard,
			Pattern: regexp.MustCompile(`\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}`),
			Mask:    "[CREDIT_CARD]",
		},
		{
			Type:    PIITypeIPAddress,
			Pattern: regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`),
			Mask:    "[IP_ADDRESS]",
		},
	}
}

// PIIPostprocessor detects and masks PII in node content.
type PIIPostprocessor struct {
	*BaseNodePostprocessor
	// Patterns are the PII patterns to detect.
	Patterns []PIIPattern
	// PIITypes specifies which PII types to detect. If empty, all types are detected.
	PIITypes []PIIType
	// MaskPII determines if PII should be masked (true) or nodes filtered (false).
	MaskPII bool
	// CustomMask is a custom mask string to use for all PII types.
	CustomMask string
	// StoreOriginal stores the original text in metadata before masking.
	StoreOriginal bool
}

// PIIPostprocessorOption configures a PIIPostprocessor.
type PIIPostprocessorOption func(*PIIPostprocessor)

// WithPIIPatterns sets custom PII patterns.
func WithPIIPatterns(patterns []PIIPattern) PIIPostprocessorOption {
	return func(p *PIIPostprocessor) {
		p.Patterns = patterns
	}
}

// WithPIITypes sets which PII types to detect.
func WithPIITypes(types ...PIIType) PIIPostprocessorOption {
	return func(p *PIIPostprocessor) {
		p.PIITypes = types
	}
}

// WithPIIMask enables PII masking instead of filtering.
func WithPIIMask(mask bool) PIIPostprocessorOption {
	return func(p *PIIPostprocessor) {
		p.MaskPII = mask
	}
}

// WithPIICustomMask sets a custom mask string.
func WithPIICustomMask(mask string) PIIPostprocessorOption {
	return func(p *PIIPostprocessor) {
		p.CustomMask = mask
	}
}

// WithPIIStoreOriginal enables storing original text in metadata.
func WithPIIStoreOriginal(store bool) PIIPostprocessorOption {
	return func(p *PIIPostprocessor) {
		p.StoreOriginal = store
	}
}

// NewPIIPostprocessor creates a new PIIPostprocessor.
func NewPIIPostprocessor(opts ...PIIPostprocessorOption) *PIIPostprocessor {
	p := &PIIPostprocessor{
		BaseNodePostprocessor: NewBaseNodePostprocessor(WithPostprocessorName("PIIPostprocessor")),
		Patterns:              DefaultPIIPatterns(),
		MaskPII:               true,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PostprocessNodes processes nodes to detect/mask PII.
func (p *PIIPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if p.MaskPII {
		return p.maskPIIInNodes(nodes), nil
	}
	return p.filterNodesWithPII(nodes), nil
}

// maskPIIInNodes masks PII in node content.
func (p *PIIPostprocessor) maskPIIInNodes(nodes []schema.NodeWithScore) []schema.NodeWithScore {
	result := make([]schema.NodeWithScore, len(nodes))

	for i, nodeWithScore := range nodes {
		node := nodeWithScore.Node
		originalText := node.Text

		// Mask PII in text
		maskedText := p.maskPII(originalText)

		// Create new node with masked text
		newNode := node
		newNode.Text = maskedText

		// Store original if requested
		if p.StoreOriginal && maskedText != originalText {
			if newNode.Metadata == nil {
				newNode.Metadata = make(map[string]interface{})
			}
			newNode.Metadata["original_text"] = originalText
			newNode.Metadata["pii_masked"] = true
		}

		result[i] = schema.NodeWithScore{
			Node:  newNode,
			Score: nodeWithScore.Score,
		}
	}

	return result
}

// filterNodesWithPII removes nodes that contain PII.
func (p *PIIPostprocessor) filterNodesWithPII(nodes []schema.NodeWithScore) []schema.NodeWithScore {
	var result []schema.NodeWithScore

	for _, nodeWithScore := range nodes {
		if !p.containsPII(nodeWithScore.Node.Text) {
			result = append(result, nodeWithScore)
		}
	}

	return result
}

// maskPII masks PII in text.
func (p *PIIPostprocessor) maskPII(text string) string {
	result := text

	for _, pattern := range p.Patterns {
		if !p.shouldDetect(pattern.Type) {
			continue
		}

		mask := pattern.Mask
		if p.CustomMask != "" {
			mask = p.CustomMask
		}

		result = pattern.Pattern.ReplaceAllString(result, mask)
	}

	return result
}

// containsPII checks if text contains PII.
func (p *PIIPostprocessor) containsPII(text string) bool {
	for _, pattern := range p.Patterns {
		if !p.shouldDetect(pattern.Type) {
			continue
		}

		if pattern.Pattern.MatchString(text) {
			return true
		}
	}

	return false
}

// shouldDetect checks if a PII type should be detected.
func (p *PIIPostprocessor) shouldDetect(piiType PIIType) bool {
	if len(p.PIITypes) == 0 {
		return true
	}

	for _, t := range p.PIITypes {
		if t == piiType {
			return true
		}
	}

	return false
}

// DetectPII returns all PII found in text.
func (p *PIIPostprocessor) DetectPII(text string) []PIIMatch {
	var matches []PIIMatch

	for _, pattern := range p.Patterns {
		if !p.shouldDetect(pattern.Type) {
			continue
		}

		found := pattern.Pattern.FindAllStringIndex(text, -1)
		for _, loc := range found {
			matches = append(matches, PIIMatch{
				Type:  pattern.Type,
				Value: text[loc[0]:loc[1]],
				Start: loc[0],
				End:   loc[1],
			})
		}
	}

	return matches
}

// PIIMatch represents a PII match in text.
type PIIMatch struct {
	Type  PIIType
	Value string
	Start int
	End   int
}

// AddPattern adds a custom PII pattern.
func (p *PIIPostprocessor) AddPattern(piiType PIIType, pattern string, mask string) error {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}

	p.Patterns = append(p.Patterns, PIIPattern{
		Type:    piiType,
		Pattern: re,
		Mask:    mask,
	})

	return nil
}

// NERBasedPIIPostprocessor uses simple heuristics for name detection.
// For production use, consider integrating with a proper NER library.
type NERBasedPIIPostprocessor struct {
	*PIIPostprocessor
	// NamePrefixes are common name prefixes to detect.
	NamePrefixes []string
}

// NewNERBasedPIIPostprocessor creates a NER-based PII postprocessor.
func NewNERBasedPIIPostprocessor(opts ...PIIPostprocessorOption) *NERBasedPIIPostprocessor {
	return &NERBasedPIIPostprocessor{
		PIIPostprocessor: NewPIIPostprocessor(opts...),
		NamePrefixes:     []string{"Mr.", "Mrs.", "Ms.", "Dr.", "Prof."},
	}
}

// DetectNames attempts to detect person names using simple heuristics.
// This is a basic implementation - for production, use a proper NER library.
func (p *NERBasedPIIPostprocessor) DetectNames(text string) []string {
	var names []string

	// Look for name prefixes followed by capitalized words
	for _, prefix := range p.NamePrefixes {
		pattern := regexp.MustCompile(regexp.QuoteMeta(prefix) + `\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)`)
		matches := pattern.FindAllStringSubmatch(text, -1)
		for _, match := range matches {
			if len(match) > 1 {
				names = append(names, strings.TrimSpace(match[0]))
			}
		}
	}

	return names
}

// Ensure PIIPostprocessor implements NodePostprocessor.
var _ NodePostprocessor = (*PIIPostprocessor)(nil)
