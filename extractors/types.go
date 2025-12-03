// Package extractors provides metadata extraction functionality for nodes.
// Supported metadata extractors:
// - TitleExtractor: Document title extraction
// - SummaryExtractor: Section summaries with adjacent context
// - KeywordsExtractor: Keyword extraction
// - QuestionsAnsweredExtractor: Questions the content can answer
package extractors

import (
	"context"
	"strings"
	"sync"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// MetadataMode specifies how metadata should be included.
type MetadataMode string

const (
	// MetadataModeAll includes all metadata.
	MetadataModeAll MetadataMode = "all"
	// MetadataModeNone excludes all metadata.
	MetadataModeNone MetadataMode = "none"
	// MetadataModeEmbed includes only embed metadata.
	MetadataModeEmbed MetadataMode = "embed"
	// MetadataModeLLM includes only LLM metadata.
	MetadataModeLLM MetadataMode = "llm"
)

// ExtractedMetadata represents metadata extracted from a node.
type ExtractedMetadata map[string]interface{}

// MetadataExtractor is the interface for all metadata extractors.
type MetadataExtractor interface {
	// Extract extracts metadata from a sequence of nodes.
	// Returns a list of metadata dictionaries corresponding to each node.
	Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error)

	// ProcessNodes extracts metadata and updates nodes in place or returns copies.
	ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error)

	// Name returns the name of the extractor.
	Name() string
}

// BaseExtractor provides common functionality for metadata extractors.
type BaseExtractor struct {
	name                   string
	isTextNodeOnly         bool
	metadataMode           MetadataMode
	inPlace                bool
	numWorkers             int
	showProgress           bool
	nodeTextTemplate       string
	disableTemplateRewrite bool
}

// BaseExtractorOption configures a BaseExtractor.
type BaseExtractorOption func(*BaseExtractor)

// DefaultNodeTextTemplate is the default template for node text.
const DefaultNodeTextTemplate = `[Excerpt from document]
{metadata_str}
Excerpt:
-----
{content}
-----
`

// WithExtractorName sets the extractor name.
func WithExtractorName(name string) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.name = name
	}
}

// WithTextNodeOnly sets whether to only process text nodes.
func WithTextNodeOnly(only bool) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.isTextNodeOnly = only
	}
}

// WithMetadataMode sets the metadata mode.
func WithMetadataMode(mode MetadataMode) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.metadataMode = mode
	}
}

// WithInPlace sets whether to modify nodes in place.
func WithInPlace(inPlace bool) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.inPlace = inPlace
	}
}

// WithNumWorkers sets the number of concurrent workers.
func WithNumWorkers(n int) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.numWorkers = n
	}
}

// WithShowProgress sets whether to show progress.
func WithShowProgress(show bool) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.showProgress = show
	}
}

// WithNodeTextTemplate sets the node text template.
func WithNodeTextTemplate(template string) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.nodeTextTemplate = template
	}
}

// WithDisableTemplateRewrite disables template rewriting.
func WithDisableTemplateRewrite(disable bool) BaseExtractorOption {
	return func(e *BaseExtractor) {
		e.disableTemplateRewrite = disable
	}
}

// NewBaseExtractor creates a new BaseExtractor.
func NewBaseExtractor(opts ...BaseExtractorOption) *BaseExtractor {
	e := &BaseExtractor{
		name:             "BaseExtractor",
		isTextNodeOnly:   true,
		metadataMode:     MetadataModeAll,
		inPlace:          true,
		numWorkers:       4,
		showProgress:     false,
		nodeTextTemplate: DefaultNodeTextTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Name returns the name of the extractor.
func (e *BaseExtractor) Name() string {
	return e.name
}

// IsTextNodeOnly returns whether to only process text nodes.
func (e *BaseExtractor) IsTextNodeOnly() bool {
	return e.isTextNodeOnly
}

// MetadataMode returns the metadata mode.
func (e *BaseExtractor) MetadataMode() MetadataMode {
	return e.metadataMode
}

// InPlace returns whether to modify nodes in place.
func (e *BaseExtractor) InPlace() bool {
	return e.inPlace
}

// NumWorkers returns the number of concurrent workers.
func (e *BaseExtractor) NumWorkers() int {
	return e.numWorkers
}

// GetNodeContent gets the content of a node based on metadata mode.
func (e *BaseExtractor) GetNodeContent(node *schema.Node) string {
	switch e.metadataMode {
	case MetadataModeNone:
		return node.GetContent(schema.MetadataModeNone)
	case MetadataModeEmbed:
		return node.GetContent(schema.MetadataModeEmbed)
	case MetadataModeLLM:
		return node.GetContent(schema.MetadataModeLLM)
	default:
		return node.GetContent(schema.MetadataModeAll)
	}
}

// Extract is a no-op implementation that should be overridden.
func (e *BaseExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	result := make([]ExtractedMetadata, len(nodes))
	for i := range result {
		result[i] = make(ExtractedMetadata)
	}
	return result, nil
}

// ProcessNodes extracts metadata and updates nodes.
func (e *BaseExtractor) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
	var newNodes []*schema.Node
	if e.inPlace {
		newNodes = nodes
	} else {
		newNodes = make([]*schema.Node, len(nodes))
		for i, node := range nodes {
			nodeCopy := *node
			newNodes[i] = &nodeCopy
		}
	}

	metadataList, err := e.Extract(ctx, newNodes)
	if err != nil {
		return nil, err
	}

	for i, node := range newNodes {
		if node.Metadata == nil {
			node.Metadata = make(map[string]interface{})
		}
		for k, v := range metadataList[i] {
			node.Metadata[k] = v
		}
	}

	return newNodes, nil
}

// ExtractorChain chains multiple extractors together.
type ExtractorChain struct {
	extractors []MetadataExtractor
}

// NewExtractorChain creates a new ExtractorChain.
func NewExtractorChain(extractors ...MetadataExtractor) *ExtractorChain {
	return &ExtractorChain{
		extractors: extractors,
	}
}

// Name returns the name of the chain.
func (c *ExtractorChain) Name() string {
	return "ExtractorChain"
}

// Extract runs all extractors and merges metadata.
func (c *ExtractorChain) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	result := make([]ExtractedMetadata, len(nodes))
	for i := range result {
		result[i] = make(ExtractedMetadata)
	}

	for _, extractor := range c.extractors {
		metadata, err := extractor.Extract(ctx, nodes)
		if err != nil {
			return nil, err
		}

		for i, m := range metadata {
			for k, v := range m {
				result[i][k] = v
			}
		}
	}

	return result, nil
}

// ProcessNodes runs all extractors on nodes.
func (c *ExtractorChain) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
	currentNodes := nodes

	for _, extractor := range c.extractors {
		var err error
		currentNodes, err = extractor.ProcessNodes(ctx, currentNodes)
		if err != nil {
			return nil, err
		}
	}

	return currentNodes, nil
}

// Add adds an extractor to the chain.
func (c *ExtractorChain) Add(extractor MetadataExtractor) {
	c.extractors = append(c.extractors, extractor)
}

// Extractors returns the extractors in the chain.
func (c *ExtractorChain) Extractors() []MetadataExtractor {
	return c.extractors
}

// Ensure ExtractorChain implements MetadataExtractor.
var _ MetadataExtractor = (*ExtractorChain)(nil)

// LLMExtractor is a base for extractors that use an LLM.
type LLMExtractor struct {
	*BaseExtractor
	llm llm.LLM
}

// LLMExtractorOption configures an LLMExtractor.
type LLMExtractorOption func(*LLMExtractor)

// WithLLM sets the LLM.
func WithLLM(l llm.LLM) LLMExtractorOption {
	return func(e *LLMExtractor) {
		e.llm = l
	}
}

// NewLLMExtractor creates a new LLMExtractor.
func NewLLMExtractor(baseOpts []BaseExtractorOption, opts ...LLMExtractorOption) *LLMExtractor {
	e := &LLMExtractor{
		BaseExtractor: NewBaseExtractor(baseOpts...),
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// LLM returns the LLM.
func (e *LLMExtractor) LLM() llm.LLM {
	return e.llm
}

// runConcurrent runs a function concurrently on nodes.
func runConcurrent[T any](
	ctx context.Context,
	nodes []*schema.Node,
	numWorkers int,
	fn func(ctx context.Context, node *schema.Node, index int) (T, error),
) ([]T, error) {
	if numWorkers <= 0 {
		numWorkers = 1
	}

	results := make([]T, len(nodes))
	errors := make([]error, len(nodes))

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, numWorkers)

	for i, node := range nodes {
		wg.Add(1)
		go func(idx int, n *schema.Node) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			result, err := fn(ctx, n, idx)
			results[idx] = result
			errors[idx] = err
		}(i, node)
	}

	wg.Wait()

	// Return first error encountered
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// formatPrompt replaces placeholders in a prompt template.
func formatPrompt(template string, vars map[string]string) string {
	result := template
	for k, v := range vars {
		result = strings.ReplaceAll(result, "{"+k+"}", v)
	}
	return result
}
