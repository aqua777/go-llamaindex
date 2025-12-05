package synthesizer

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
)

// GetSynthesizer returns a synthesizer for the given response mode.
func GetSynthesizer(mode ResponseMode, llmModel llm.LLM) (Synthesizer, error) {
	switch mode {
	case ResponseModeSimpleSummarize:
		return NewSimpleSynthesizer(llmModel), nil
	case ResponseModeRefine:
		return NewRefineSynthesizer(llmModel), nil
	case ResponseModeCompact:
		return NewCompactAndRefineSynthesizer(llmModel), nil
	case ResponseModeTreeSummarize:
		return NewTreeSummarizeSynthesizer(llmModel), nil
	case ResponseModeAccumulate:
		return NewAccumulateSynthesizer(llmModel), nil
	case ResponseModeCompactAccumulate:
		return NewCompactAccumulateSynthesizer(llmModel), nil
	case ResponseModeGeneration, ResponseModeNoText, ResponseModeContextOnly:
		// These modes don't require a full synthesizer
		return NewSimpleSynthesizer(llmModel), nil
	default:
		return nil, fmt.Errorf("unsupported response mode: %s", mode)
	}
}
