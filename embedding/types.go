package embedding

// EmbeddingInfo contains metadata about an embedding model's capabilities.
type EmbeddingInfo struct {
	// ModelName is the name/identifier of the model.
	ModelName string `json:"model_name"`
	// Dimensions is the number of dimensions in the embedding vector.
	Dimensions int `json:"dimensions"`
	// MaxTokens is the maximum number of tokens the model can process.
	MaxTokens int `json:"max_tokens"`
	// IsMultiModal indicates if the model supports multi-modal inputs (e.g., images).
	IsMultiModal bool `json:"is_multi_modal"`
	// TokenizerName is the name of the tokenizer used by the model.
	TokenizerName string `json:"tokenizer_name,omitempty"`
}

// DefaultEmbeddingInfo returns default info for unknown models.
func DefaultEmbeddingInfo(modelName string) EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:  modelName,
		Dimensions: 1536, // Common default
		MaxTokens:  8191,
	}
}

// OpenAISmallEmbedding3Info returns info for text-embedding-3-small.
func OpenAISmallEmbedding3Info() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:     "text-embedding-3-small",
		Dimensions:    1536,
		MaxTokens:     8191,
		TokenizerName: "cl100k_base",
	}
}

// OpenAILargeEmbedding3Info returns info for text-embedding-3-large.
func OpenAILargeEmbedding3Info() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:     "text-embedding-3-large",
		Dimensions:    3072,
		MaxTokens:     8191,
		TokenizerName: "cl100k_base",
	}
}

// OpenAIAdaEmbeddingInfo returns info for text-embedding-ada-002.
func OpenAIAdaEmbeddingInfo() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:     "text-embedding-ada-002",
		Dimensions:    1536,
		MaxTokens:     8191,
		TokenizerName: "cl100k_base",
	}
}

// MxbaiEmbedLargeInfo returns info for mxbai-embed-large (Ollama).
func MxbaiEmbedLargeInfo() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:  "mxbai-embed-large",
		Dimensions: 1024,
		MaxTokens:  512,
	}
}

// AllMiniLMInfo returns info for all-minilm (Ollama).
func AllMiniLMInfo() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:  "all-minilm",
		Dimensions: 384,
		MaxTokens:  256,
	}
}

// NomicEmbedTextInfo returns info for nomic-embed-text (Ollama).
func NomicEmbedTextInfo() EmbeddingInfo {
	return EmbeddingInfo{
		ModelName:  "nomic-embed-text",
		Dimensions: 768,
		MaxTokens:  8192,
	}
}

// EmbeddingResult represents the result of an embedding operation.
type EmbeddingResult struct {
	// Embedding is the vector representation.
	Embedding []float64 `json:"embedding"`
	// Text is the original text that was embedded.
	Text string `json:"text,omitempty"`
	// TokenCount is the number of tokens in the text.
	TokenCount int `json:"token_count,omitempty"`
}

// BatchEmbeddingResult represents the result of a batch embedding operation.
type BatchEmbeddingResult struct {
	// Embeddings contains all embedding results.
	Embeddings []EmbeddingResult `json:"embeddings"`
	// TotalTokens is the total number of tokens processed.
	TotalTokens int `json:"total_tokens,omitempty"`
}

// ProgressCallback is called during batch operations to report progress.
// current is the number of items processed, total is the total number of items.
type ProgressCallback func(current, total int)

// ImageType represents different ways to specify an image for embedding.
type ImageType struct {
	// URL is the URL of the image.
	URL string `json:"url,omitempty"`
	// Base64 is the base64-encoded image data.
	Base64 string `json:"base64,omitempty"`
	// Path is the local filesystem path to the image.
	Path string `json:"path,omitempty"`
	// MimeType is the MIME type of the image.
	MimeType string `json:"mime_type,omitempty"`
}

// NewImageFromURL creates an ImageType from a URL.
func NewImageFromURL(url string) ImageType {
	return ImageType{URL: url}
}

// NewImageFromBase64 creates an ImageType from base64 data.
func NewImageFromBase64(data, mimeType string) ImageType {
	return ImageType{Base64: data, MimeType: mimeType}
}

// NewImageFromPath creates an ImageType from a file path.
func NewImageFromPath(path string) ImageType {
	return ImageType{Path: path}
}

// IsEmpty returns true if no image data is specified.
func (i ImageType) IsEmpty() bool {
	return i.URL == "" && i.Base64 == "" && i.Path == ""
}
