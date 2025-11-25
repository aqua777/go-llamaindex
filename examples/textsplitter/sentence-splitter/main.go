package main

import (
	"fmt"
	"os"

	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	// Read the example file
	content, err := os.ReadFile("example.txt")
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}
	text := string(content)

	fmt.Printf("Original Text Length: %d characters\n", len(text))
	fmt.Println("--------------------------------------------------")

	// 1. Default Splitter (SimpleTokenizer, RegexStrategy)
	fmt.Println("Scenario 1: Default Splitter (SimpleTokenizer, RegexStrategy)")
	splitterDefault := textsplitter.NewSentenceSplitter(200, 20, nil, nil)
	chunksDefault := splitterDefault.SplitText(text)
	fmt.Printf("Generated %d chunks.\n", len(chunksDefault))
	fmt.Println("--------------------------------------------------")

	// 2. TikToken Splitter
	fmt.Println("Scenario 2: TikToken Tokenizer (gpt-3.5-turbo)")
	tikTokenizer, err := textsplitter.NewTikTokenTokenizer("gpt-3.5-turbo")
	if err != nil {
		fmt.Printf("Failed to init tiktoken: %v\n", err)
	} else {
		splitterTik := textsplitter.NewSentenceSplitter(200, 20, tikTokenizer, nil)
		chunksTik := splitterTik.SplitText(text)
		fmt.Printf("Generated %d chunks.\n", len(chunksTik))
		// Print first chunk sample
		if len(chunksTik) > 0 {
			fmt.Printf("Sample Chunk 1:\n%s\n", chunksTik[0])
		}
	}
	fmt.Println("--------------------------------------------------")

	// 3. Neurosnap Splitter
	fmt.Println("Scenario 3: Neurosnap Sentence Splitter (Embedded English Data)")
	// We use nil to trigger the default embedded english.json data.
	neuroStrategy, err := textsplitter.NewNeurosnapSplitterStrategy(nil)
	if err != nil {
		fmt.Printf("Failed to init neurosnap strategy: %v\n", err)
	} else {
		splitterNeuro := textsplitter.NewSentenceSplitter(200, 20, nil, neuroStrategy)
		chunksNeuro := splitterNeuro.SplitText(text)
		fmt.Printf("Generated %d chunks.\n", len(chunksNeuro))
		if len(chunksNeuro) > 0 {
			// Print first few chunks to see how sentences are respected
			for i := 0; i < 2 && i < len(chunksNeuro); i++ {
				fmt.Printf("Chunk %d:\n%s\n---\n", i+1, chunksNeuro[i])
			}
		}
	}
	fmt.Println("--------------------------------------------------")
}
