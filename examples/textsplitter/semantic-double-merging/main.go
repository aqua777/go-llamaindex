package main

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	doc := `Neural networks learn representations from data. Gradient descent updates weights to minimize loss.
	Classical optimization theory applies to convex problems. Stock markets react to earnings and macro news.
	Weather patterns influence agricultural commodity prices. Deep learning excels at pattern recognition tasks.
	Investors diversify portfolios across asset classes. Seasonal rainfall affects crop yields in many regions.`

	lc := textsplitter.LanguageConfig{Language: "english", SpacyModel: "en_core_web_md"}
	strategy, err := textsplitter.NewNeurosnapSplitterStrategy(nil)
	if err != nil {
		fmt.Printf("neurosnap: %v\n", err)
		return
	}

	runs := []struct {
		label              string
		initial, app, merge float64
	}{
		{"tighter thresholds (more chunks)", 0.45, 0.45, 0.45},
		{"looser thresholds (fewer, larger chunks)", 0.25, 0.25, 0.25},
	}

	for _, r := range runs {
		sp := textsplitter.NewSemanticDoubleMergingSplitter(
			lc,
			r.initial, r.app, r.merge,
			2500,
			2,
			" ",
			strategy,
		)
		chunks := sp.SplitText(doc)
		fmt.Printf("SemanticDoubleMergingSplitter — %s: %d chunk(s)\n", r.label, len(chunks))
		for i, c := range chunks {
			fmt.Printf("--- Chunk %d (%d runes) ---\n%s\n", i+1, len([]rune(c)), strings.TrimSpace(c))
		}
		fmt.Println()
	}
	fmt.Println("SUCCESS: semantic double merging demo completed")
}
