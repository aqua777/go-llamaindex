package synthesizer

import (
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
)

// Response represents a standard response with metadata.
type Response struct {
	// Response is the response text.
	Response string
	// SourceNodes are the source nodes used to generate the response.
	SourceNodes []schema.NodeWithScore
	// Metadata contains additional response metadata.
	Metadata map[string]interface{}
}

// NewResponse creates a new Response.
func NewResponse(response string, sourceNodes []schema.NodeWithScore) *Response {
	return &Response{
		Response:    response,
		SourceNodes: sourceNodes,
		Metadata:    make(map[string]interface{}),
	}
}

// NewResponseWithMetadata creates a new Response with metadata.
func NewResponseWithMetadata(response string, sourceNodes []schema.NodeWithScore, metadata map[string]interface{}) *Response {
	return &Response{
		Response:    response,
		SourceNodes: sourceNodes,
		Metadata:    metadata,
	}
}

// String returns the response text.
func (r *Response) String() string {
	if r.Response == "" {
		return "None"
	}
	return r.Response
}

// GetFormattedSources returns formatted source text.
func (r *Response) GetFormattedSources(length int) string {
	var texts []string
	for _, sourceNode := range r.SourceNodes {
		content := sourceNode.Node.GetContent(schema.MetadataModeLLM)
		if len(content) > length {
			content = content[:length] + "..."
		}
		docID := sourceNode.Node.ID
		if docID == "" {
			docID = "None"
		}
		sourceText := "> Source (Doc id: " + docID + "): " + content
		texts = append(texts, sourceText)
	}
	return strings.Join(texts, "\n\n")
}

// StreamingResponse represents a streaming response.
type StreamingResponse struct {
	// ResponseChan is the channel for streaming response tokens.
	ResponseChan <-chan string
	// SourceNodes are the source nodes used to generate the response.
	SourceNodes []schema.NodeWithScore
	// Metadata contains additional response metadata.
	Metadata map[string]interface{}
	// responseTxt caches the full response after streaming completes.
	responseTxt string
}

// NewStreamingResponse creates a new StreamingResponse.
func NewStreamingResponse(responseChan <-chan string, sourceNodes []schema.NodeWithScore) *StreamingResponse {
	return &StreamingResponse{
		ResponseChan: responseChan,
		SourceNodes:  sourceNodes,
		Metadata:     make(map[string]interface{}),
	}
}

// String consumes the stream and returns the full response.
func (sr *StreamingResponse) String() string {
	if sr.responseTxt != "" {
		return sr.responseTxt
	}
	if sr.ResponseChan == nil {
		return "None"
	}

	var builder strings.Builder
	for token := range sr.ResponseChan {
		builder.WriteString(token)
	}
	sr.responseTxt = builder.String()
	return sr.responseTxt
}

// GetResponse returns a standard Response after consuming the stream.
func (sr *StreamingResponse) GetResponse() *Response {
	return &Response{
		Response:    sr.String(),
		SourceNodes: sr.SourceNodes,
		Metadata:    sr.Metadata,
	}
}

// GetFormattedSources returns formatted source text.
func (sr *StreamingResponse) GetFormattedSources(length int) string {
	var texts []string
	for _, sourceNode := range sr.SourceNodes {
		content := sourceNode.Node.GetContent(schema.MetadataModeLLM)
		if len(content) > length {
			content = content[:length] + "..."
		}
		nodeID := sourceNode.Node.ID
		if nodeID == "" {
			nodeID = "None"
		}
		sourceText := "> Source (Node id: " + nodeID + "): " + content
		texts = append(texts, sourceText)
	}
	return strings.Join(texts, "\n\n")
}

// ResponseType is an interface for all response types.
type ResponseType interface {
	String() string
	GetFormattedSources(length int) string
}

// Ensure implementations satisfy the interface.
var _ ResponseType = (*Response)(nil)
var _ ResponseType = (*StreamingResponse)(nil)
