package schema

// MetadataMode controls how metadata is included in different contexts.
type MetadataMode string

const (
	// MetadataModeAll includes all metadata.
	MetadataModeAll MetadataMode = "all"
	// MetadataModeEmbed includes metadata for embedding (excludes ExcludedEmbedMetadataKeys).
	MetadataModeEmbed MetadataMode = "embed"
	// MetadataModeLLM includes metadata for LLM context (excludes ExcludedLLMMetadataKeys).
	MetadataModeLLM MetadataMode = "llm"
	// MetadataModeNone excludes all metadata.
	MetadataModeNone MetadataMode = "none"
)
