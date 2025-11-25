package settings

import (
	"sync"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/llm"
)

const (
	DefaultChunkSize    = 1024
	DefaultChunkOverlap = 200
)

var (
	mu              sync.RWMutex
	globalLLM       llm.LLM
	globalEmbed     embedding.EmbeddingModel
	globalChunkSz   int
	globalChunkOvlp int
)

func init() {
	// Initialize with defaults
	// Note: Providers might need API keys from env, handled by their constructors
	globalLLM = llm.NewOpenAILLM("", "", "")
	globalEmbed = embedding.NewOpenAIEmbedding("", "")
	globalChunkSz = DefaultChunkSize
	globalChunkOvlp = DefaultChunkOverlap
}

// SetLLM sets the global LLM.
func SetLLM(l llm.LLM) {
	mu.Lock()
	defer mu.Unlock()
	globalLLM = l
}

// GetLLM gets the global LLM.
func GetLLM() llm.LLM {
	mu.RLock()
	defer mu.RUnlock()
	return globalLLM
}

// SetEmbedModel sets the global embedding model.
func SetEmbedModel(e embedding.EmbeddingModel) {
	mu.Lock()
	defer mu.Unlock()
	globalEmbed = e
}

// GetEmbedModel gets the global embedding model.
func GetEmbedModel() embedding.EmbeddingModel {
	mu.RLock()
	defer mu.RUnlock()
	return globalEmbed
}

// SetChunkSize sets the global chunk size.
func SetChunkSize(size int) {
	mu.Lock()
	defer mu.Unlock()
	globalChunkSz = size
}

// GetChunkSize gets the global chunk size.
func GetChunkSize() int {
	mu.RLock()
	defer mu.RUnlock()
	return globalChunkSz
}

// SetChunkOverlap sets the global chunk overlap.
func SetChunkOverlap(overlap int) {
	mu.Lock()
	defer mu.Unlock()
	globalChunkOvlp = overlap
}

// GetChunkOverlap gets the global chunk overlap.
func GetChunkOverlap() int {
	mu.RLock()
	defer mu.RUnlock()
	return globalChunkOvlp
}
