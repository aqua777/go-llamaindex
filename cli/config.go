package main

import (
	"os"
	"path/filepath"
)

const (
	GoLlamaIndex = "go-llamaindex"
	GoLlamaIndexCli = "go-llamaindex-cli"
)

// Default configuration values
const (
	DefaultOllamaURL        = "http://localhost:11434"
	DefaultOllamaModel      = "jan-v1:q6_k"
	DefaultOllamaEmbedModel = "bge-large"
	DefaultCollection       = "default"
	DefaultChunkSize        = 1024
	DefaultChunkOverlap     = 200
	DefaultTopK             = 5
)

// Config keys for krait
const (
	KeyCacheDir         = "cache.dir"
	KeyOllamaURL        = "ollama.url"
	KeyOllamaModel      = "ollama.model"
	KeyOllamaEmbedModel = "ollama.embed-model"
	KeyCollection       = "rag.collection"
	KeyChunkSize        = "rag.chunk-size"
	KeyChunkOverlap     = "rag.chunk-overlap"
	KeyTopK             = "rag.top-k"
	KeyVerbose          = "verbose"
	KeyStream           = "stream"
)

// DefaultCacheDir returns the default cache directory.
func DefaultCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "." + GoLlamaIndexCli
	}
	return filepath.Join(home, ".cache", GoLlamaIndexCli)
}

// ChromemPersistPath returns the path for chromem persistence.
func ChromemPersistPath(cacheDir string) string {
	return filepath.Join(cacheDir, "chromem")
}

// ChatHistoryPath returns the path for chat history.
func ChatHistoryPath(cacheDir string) string {
	return filepath.Join(cacheDir, "chat_history.json")
}

// FilesHistoryPath returns the path for tracking ingested files.
func FilesHistoryPath(cacheDir string) string {
	return filepath.Join(cacheDir, "files_history.txt")
}
