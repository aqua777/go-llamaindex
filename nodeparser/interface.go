package nodeparser

import (
	"github.com/aqua777/go-llamaindex/schema"
)

// NodeParser is the interface for parsing documents into nodes.
// It takes documents or nodes and splits them into smaller chunks,
// establishing relationships between the resulting nodes.
type NodeParser interface {
	// GetNodesFromDocuments parses documents into nodes.
	GetNodesFromDocuments(documents []schema.Document) []*schema.Node

	// ParseNodes parses nodes into smaller nodes.
	// This is useful for recursive parsing or re-chunking existing nodes.
	ParseNodes(nodes []*schema.Node) []*schema.Node
}

// NodeParserWithOptions extends NodeParser with configuration options.
type NodeParserWithOptions interface {
	NodeParser

	// WithIncludeMetadata sets whether to include parent metadata in child nodes.
	WithIncludeMetadata(include bool) NodeParserWithOptions

	// WithIncludePrevNextRel sets whether to establish PREVIOUS/NEXT relationships.
	WithIncludePrevNextRel(include bool) NodeParserWithOptions
}

// NodeParserOptions contains configuration options for node parsers.
type NodeParserOptions struct {
	// IncludeMetadata determines whether parent metadata is copied to child nodes.
	IncludeMetadata bool
	// IncludePrevNextRel determines whether PREVIOUS/NEXT relationships are established.
	IncludePrevNextRel bool
	// IDFunc is a function to generate node IDs. If nil, UUIDs are used.
	IDFunc func() string
}

// DefaultNodeParserOptions returns the default options.
func DefaultNodeParserOptions() NodeParserOptions {
	return NodeParserOptions{
		IncludeMetadata:    true,
		IncludePrevNextRel: true,
		IDFunc:             nil, // Use UUID by default
	}
}

// NodeParserCallback is called during parsing to report progress or events.
type NodeParserCallback func(event NodeParserEvent)

// NodeParserEvent represents an event during parsing.
type NodeParserEvent struct {
	// Type is the type of event.
	Type NodeParserEventType
	// DocumentID is the ID of the document being processed.
	DocumentID string
	// NodeCount is the number of nodes created so far.
	NodeCount int
	// Message is an optional message.
	Message string
}

// NodeParserEventType represents the type of parser event.
type NodeParserEventType string

const (
	// EventTypeStart indicates parsing has started.
	EventTypeStart NodeParserEventType = "start"
	// EventTypeProgress indicates parsing progress.
	EventTypeProgress NodeParserEventType = "progress"
	// EventTypeComplete indicates parsing has completed.
	EventTypeComplete NodeParserEventType = "complete"
	// EventTypeError indicates an error occurred.
	EventTypeError NodeParserEventType = "error"
)
