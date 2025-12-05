package synthesizer

import (
	"context"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/schema"
)

// Synthesizer is the interface for response synthesizers.
type Synthesizer interface {
	// Synthesize generates a response from the query and source nodes.
	Synthesize(ctx context.Context, query string, nodes []schema.NodeWithScore) (*Response, error)

	// GetResponse generates a response from query and text chunks.
	GetResponse(ctx context.Context, query string, textChunks []string) (string, error)
}

// BaseSynthesizer provides common functionality for synthesizers.
type BaseSynthesizer struct {
	// LLM is the language model for generating responses.
	LLM llm.LLM
	// Streaming enables streaming responses.
	Streaming bool
	// Verbose enables verbose logging.
	Verbose bool
	// PromptMixin for prompt management.
	*prompts.BasePromptMixin
}

// NewBaseSynthesizer creates a new BaseSynthesizer.
func NewBaseSynthesizer(llmModel llm.LLM) *BaseSynthesizer {
	return &BaseSynthesizer{
		LLM:             llmModel,
		Streaming:       false,
		Verbose:         false,
		BasePromptMixin: prompts.NewBasePromptMixin(),
	}
}

// BaseSynthesizerOption is a functional option for BaseSynthesizer.
type BaseSynthesizerOption func(*BaseSynthesizer)

// WithStreaming enables streaming responses.
func WithStreaming(streaming bool) BaseSynthesizerOption {
	return func(bs *BaseSynthesizer) {
		bs.Streaming = streaming
	}
}

// WithSynthesizerVerbose enables verbose logging.
func WithSynthesizerVerbose(verbose bool) BaseSynthesizerOption {
	return func(bs *BaseSynthesizer) {
		bs.Verbose = verbose
	}
}

// NewBaseSynthesizerWithOptions creates a new BaseSynthesizer with options.
func NewBaseSynthesizerWithOptions(llmModel llm.LLM, opts ...BaseSynthesizerOption) *BaseSynthesizer {
	bs := NewBaseSynthesizer(llmModel)
	for _, opt := range opts {
		opt(bs)
	}
	return bs
}

// GetMetadataForResponse extracts metadata from nodes.
func (bs *BaseSynthesizer) GetMetadataForResponse(nodes []schema.NodeWithScore) map[string]interface{} {
	metadata := make(map[string]interface{})
	for _, node := range nodes {
		metadata[node.Node.ID] = node.Node.Metadata
	}
	return metadata
}

// PrepareResponseOutput creates a Response from response string and source nodes.
func (bs *BaseSynthesizer) PrepareResponseOutput(responseStr string, sourceNodes []schema.NodeWithScore) *Response {
	metadata := bs.GetMetadataForResponse(sourceNodes)
	return &Response{
		Response:    responseStr,
		SourceNodes: sourceNodes,
		Metadata:    metadata,
	}
}

// GetTextChunksFromNodes extracts text content from nodes.
func GetTextChunksFromNodes(nodes []schema.NodeWithScore, mode schema.MetadataMode) []string {
	chunks := make([]string, len(nodes))
	for i, node := range nodes {
		chunks[i] = node.Node.GetContent(mode)
	}
	return chunks
}

// CompactTextChunks combines text chunks to better utilize context window.
func CompactTextChunks(chunks []string, maxChunkSize int, separator string) []string {
	if len(chunks) == 0 {
		return chunks
	}
	if separator == "" {
		separator = "\n\n"
	}

	var compacted []string
	var current strings.Builder
	currentSize := 0

	for _, chunk := range chunks {
		chunkSize := len(chunk)

		// If adding this chunk would exceed max size, start a new compacted chunk
		if currentSize > 0 && currentSize+len(separator)+chunkSize > maxChunkSize {
			compacted = append(compacted, current.String())
			current.Reset()
			currentSize = 0
		}

		// Add separator if not first chunk in current
		if currentSize > 0 {
			current.WriteString(separator)
			currentSize += len(separator)
		}

		current.WriteString(chunk)
		currentSize += chunkSize
	}

	// Add remaining content
	if currentSize > 0 {
		compacted = append(compacted, current.String())
	}

	return compacted
}
