package main

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	longBody := strings.Repeat("The quick brown fox jumps. ", 40)
	metadata := strings.Repeat("doc_id=chapter-7 section=appendix tags=long,list,of,keys ", 3)

	splitter := textsplitter.NewSentenceSplitter(120, 10, nil, nil)

	fmt.Println("Metadata-aware splitting (large metadata reserves chunk budget)")
	fmt.Printf("ChunkSize=%d metadata tokens=%d\n", splitter.ChunkSize, textsplitter.MetadataTokenCount(splitter.Tokenizer, metadata))

	chunks, err := splitter.SplitTextMetadataAware(longBody, metadata)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}

	plain := splitter.SplitText(longBody)
	fmt.Printf("Chunks without metadata accounting: %d\n", len(plain))
	fmt.Printf("Chunks with metadata accounting: %d\n", len(chunks))
	if len(chunks) > 0 {
		fmt.Printf("First chunk length (chars): %d\n", len(chunks[0]))
	}
	fmt.Println("SUCCESS: metadata-aware split completed")
}
