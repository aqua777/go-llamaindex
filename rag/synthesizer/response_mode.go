// Package synthesizer provides response synthesis implementations for RAG systems.
package synthesizer

// ResponseMode represents the mode of response synthesis.
type ResponseMode string

const (
	// ResponseModeRefine iteratively refines the response across text chunks.
	// First generates an initial answer from the first chunk, then refines it
	// using subsequent chunks.
	ResponseModeRefine ResponseMode = "refine"

	// ResponseModeCompact combines text chunks into larger consolidated chunks
	// that better utilize the context window, then refines across them.
	// Faster than refine as it makes fewer LLM calls.
	ResponseModeCompact ResponseMode = "compact"

	// ResponseModeSimpleSummarize merges all text chunks into one and makes
	// a single LLM call. Fails if merged text exceeds context window.
	ResponseModeSimpleSummarize ResponseMode = "simple_summarize"

	// ResponseModeTreeSummarize builds a tree index over candidate nodes with
	// a summary prompt seeded with the query. Built bottom-up, returns root.
	ResponseModeTreeSummarize ResponseMode = "tree_summarize"

	// ResponseModeGeneration ignores context and uses LLM to generate response.
	ResponseModeGeneration ResponseMode = "generation"

	// ResponseModeNoText returns retrieved context nodes without synthesizing.
	ResponseModeNoText ResponseMode = "no_text"

	// ResponseModeContextOnly returns concatenated string of all text chunks.
	ResponseModeContextOnly ResponseMode = "context_only"

	// ResponseModeAccumulate synthesizes a response for each chunk, then
	// concatenates all responses.
	ResponseModeAccumulate ResponseMode = "accumulate"

	// ResponseModeCompactAccumulate combines chunks into larger consolidated
	// chunks, accumulates answers for each, then concatenates.
	ResponseModeCompactAccumulate ResponseMode = "compact_accumulate"
)

// String returns the string representation of the response mode.
func (rm ResponseMode) String() string {
	return string(rm)
}

// IsValid checks if the response mode is valid.
func (rm ResponseMode) IsValid() bool {
	switch rm {
	case ResponseModeRefine, ResponseModeCompact, ResponseModeSimpleSummarize,
		ResponseModeTreeSummarize, ResponseModeGeneration, ResponseModeNoText,
		ResponseModeContextOnly, ResponseModeAccumulate, ResponseModeCompactAccumulate:
		return true
	default:
		return false
	}
}
