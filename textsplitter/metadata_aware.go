package textsplitter

import "fmt"

// MinEffectiveContentChunkTokens is the minimum usable chunk size (in tokenizer units)
// after reserving space for metadata.
const MinEffectiveContentChunkTokens = 50

// MetadataTokenCount returns the tokenizer token count for metadata.
func MetadataTokenCount(tokenizer Tokenizer, metadata string) int {
	if tokenizer == nil {
		return 0
	}
	return len(tokenizer.Encode(metadata))
}

// EffectiveChunkSizeAfterMetadata returns chunkSize minus metadataTokenCount, or an error
// if the remainder is below MinEffectiveContentChunkTokens.
func EffectiveChunkSizeAfterMetadata(chunkSize, metadataTokenCount int) (int, error) {
	if metadataTokenCount < 0 {
		metadataTokenCount = 0
	}
	effective := chunkSize - metadataTokenCount
	if effective < MinEffectiveContentChunkTokens {
		return 0, fmt.Errorf(
			"metadata length (%d tokens) is too large for chunk size (%d), resulting in insufficient content window (<%d tokens)",
			metadataTokenCount, chunkSize, MinEffectiveContentChunkTokens,
		)
	}
	return effective, nil
}

// EffectiveChunkSizeForMetadataAwareSplit returns the content chunk size (in tokenizer units)
// after reserving space for metadata. It is shared by splitters that clone themselves with a
// reduced ChunkSize for metadata-aware splitting.
func EffectiveChunkSizeForMetadataAwareSplit(chunkSize int, tokenizer Tokenizer, metadata string) (int, error) {
	mt := MetadataTokenCount(tokenizer, metadata)
	return EffectiveChunkSizeAfterMetadata(chunkSize, mt)
}
