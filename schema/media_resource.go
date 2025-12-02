package schema

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"net/url"
)

// EmbeddingKind represents the type of embedding (sparse or dense).
type EmbeddingKind string

const (
	// EmbeddingKindSparse represents sparse embeddings.
	EmbeddingKindSparse EmbeddingKind = "sparse"
	// EmbeddingKindDense represents dense embeddings.
	EmbeddingKindDense EmbeddingKind = "dense"
)

// MediaResource is a container class for media content.
// It represents a generic media resource that can be stored and accessed
// in multiple ways - as raw bytes, on the filesystem, or via URL.
// It also supports storing vector embeddings for the media content.
type MediaResource struct {
	// Embeddings is a multi-vector dict representation for embedding-based search/retrieval.
	Embeddings map[EmbeddingKind][]float64 `json:"embeddings,omitempty"`
	// Data is the raw binary data of the media content (base64 encoded).
	Data []byte `json:"data,omitempty"`
	// Text is the plain text representation of this resource.
	Text string `json:"text,omitempty"`
	// Path is the local filesystem path where the media content can be accessed.
	Path string `json:"path,omitempty"`
	// URL is the URL where the media content can be accessed remotely.
	URL string `json:"url,omitempty"`
	// MimeType is the MIME type indicating the format/type of the media content.
	MimeType string `json:"mimetype,omitempty"`
}

// NewMediaResource creates a new empty MediaResource.
func NewMediaResource() *MediaResource {
	return &MediaResource{
		Embeddings: make(map[EmbeddingKind][]float64),
	}
}

// NewMediaResourceFromText creates a MediaResource from text.
func NewMediaResourceFromText(text string) *MediaResource {
	return &MediaResource{
		Text:       text,
		MimeType:   "text/plain",
		Embeddings: make(map[EmbeddingKind][]float64),
	}
}

// NewMediaResourceFromData creates a MediaResource from binary data.
func NewMediaResourceFromData(data []byte, mimeType string) *MediaResource {
	return &MediaResource{
		Data:       data,
		MimeType:   mimeType,
		Embeddings: make(map[EmbeddingKind][]float64),
	}
}

// NewMediaResourceFromPath creates a MediaResource from a file path.
func NewMediaResourceFromPath(path string, mimeType string) *MediaResource {
	return &MediaResource{
		Path:       path,
		MimeType:   mimeType,
		Embeddings: make(map[EmbeddingKind][]float64),
	}
}

// NewMediaResourceFromURL creates a MediaResource from a URL.
func NewMediaResourceFromURL(urlStr string, mimeType string) *MediaResource {
	return &MediaResource{
		URL:        urlStr,
		MimeType:   mimeType,
		Embeddings: make(map[EmbeddingKind][]float64),
	}
}

// GetDataBase64 returns the data as a base64 encoded string.
func (m *MediaResource) GetDataBase64() string {
	if m.Data == nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(m.Data)
}

// SetDataFromBase64 sets the data from a base64 encoded string.
func (m *MediaResource) SetDataFromBase64(b64 string) error {
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return err
	}
	m.Data = data
	return nil
}

// Hash generates a hash to uniquely identify the media resource.
// The hash is generated based on the available content (data, path, text or url).
// Returns an empty string if no content is available.
func (m *MediaResource) Hash() string {
	var bits []string

	if m.Text != "" {
		bits = append(bits, m.Text)
	}
	if m.Data != nil {
		h := sha256.Sum256(m.Data)
		bits = append(bits, hex.EncodeToString(h[:]))
	}
	if m.Path != "" {
		h := sha256.Sum256([]byte(m.Path))
		bits = append(bits, hex.EncodeToString(h[:]))
	}
	if m.URL != "" {
		h := sha256.Sum256([]byte(m.URL))
		bits = append(bits, hex.EncodeToString(h[:]))
	}

	if len(bits) == 0 {
		return ""
	}

	// Combine all bits and hash
	combined := ""
	for _, bit := range bits {
		combined += bit
	}
	h := sha256.Sum256([]byte(combined))
	return hex.EncodeToString(h[:])
}

// IsEmpty returns true if the media resource has no content.
func (m *MediaResource) IsEmpty() bool {
	return m.Text == "" && m.Data == nil && m.Path == "" && m.URL == ""
}

// HasData returns true if the media resource has binary data.
func (m *MediaResource) HasData() bool {
	return m.Data != nil && len(m.Data) > 0
}

// HasPath returns true if the media resource has a file path.
func (m *MediaResource) HasPath() bool {
	return m.Path != ""
}

// HasURL returns true if the media resource has a URL.
func (m *MediaResource) HasURL() bool {
	return m.URL != ""
}

// HasText returns true if the media resource has text content.
func (m *MediaResource) HasText() bool {
	return m.Text != ""
}

// GetParsedURL returns the URL as a parsed url.URL, or nil if not set or invalid.
func (m *MediaResource) GetParsedURL() *url.URL {
	if m.URL == "" {
		return nil
	}
	parsed, err := url.Parse(m.URL)
	if err != nil {
		return nil
	}
	return parsed
}

// SetEmbedding sets an embedding for the given kind.
func (m *MediaResource) SetEmbedding(kind EmbeddingKind, embedding []float64) {
	if m.Embeddings == nil {
		m.Embeddings = make(map[EmbeddingKind][]float64)
	}
	m.Embeddings[kind] = embedding
}

// GetEmbedding returns the embedding for the given kind, or nil if not set.
func (m *MediaResource) GetEmbedding(kind EmbeddingKind) []float64 {
	if m.Embeddings == nil {
		return nil
	}
	return m.Embeddings[kind]
}

// GetDenseEmbedding returns the dense embedding, or nil if not set.
func (m *MediaResource) GetDenseEmbedding() []float64 {
	return m.GetEmbedding(EmbeddingKindDense)
}

// GetSparseEmbedding returns the sparse embedding, or nil if not set.
func (m *MediaResource) GetSparseEmbedding() []float64 {
	return m.GetEmbedding(EmbeddingKindSparse)
}
