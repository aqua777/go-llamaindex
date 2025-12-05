// Package main demonstrates the SubQuestionQueryEngine for decomposing complex queries.
// This example corresponds to Python's query_engine/sub_question_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Sub-Question Query Engine Demo ===")
	fmt.Println("LLM initialized")

	// 2. Create domain-specific query engines
	// In a real scenario, these would be backed by vector stores with actual documents
	paulGrahamEngine := NewDomainQueryEngine(
		"paul_graham",
		"Paul Graham's essays and startup advice",
		getPaulGrahamKnowledge(),
	)

	techNewsEngine := NewDomainQueryEngine(
		"tech_news",
		"Recent technology news and trends",
		getTechNewsKnowledge(),
	)

	financeEngine := NewDomainQueryEngine(
		"finance",
		"Financial markets and investment information",
		getFinanceKnowledge(),
	)

	// 3. Create query engine tools
	tools := []*queryengine.QueryEngineTool{
		queryengine.NewQueryEngineTool(
			paulGrahamEngine,
			"paul_graham_essays",
			"Contains Paul Graham's essays about startups, programming, and life advice. Use this for questions about startup philosophy, Y Combinator, or Paul Graham's views.",
		),
		queryengine.NewQueryEngineTool(
			techNewsEngine,
			"tech_news",
			"Contains recent technology news about AI, software, and tech companies. Use this for questions about current tech trends and news.",
		),
		queryengine.NewQueryEngineTool(
			financeEngine,
			"finance_data",
			"Contains financial market data and investment information. Use this for questions about stocks, markets, and investing.",
		),
	}

	fmt.Printf("\nCreated %d query engine tools:\n", len(tools))
	for _, tool := range tools {
		fmt.Printf("  - %s: %s\n", tool.Name, truncate(tool.Description, 60))
	}

	// 4. Create question generator
	questionGen := queryengine.NewLLMQuestionGenerator(llmInstance)

	// 5. Create synthesizer for final response
	synth := synthesizer.NewTreeSummarizeSynthesizer(llmInstance)

	// 6. Create SubQuestionQueryEngine
	subQuestionEngine := queryengine.NewSubQuestionQueryEngine(
		questionGen,
		synth,
		tools,
		queryengine.WithSubQuestionVerbose(true),
	)

	separator := strings.Repeat("=", 70)

	// 7. Test with complex queries that span multiple domains
	fmt.Println("\n" + separator)
	fmt.Println("=== Testing Sub-Question Decomposition ===")
	fmt.Println(separator)

	complexQueries := []string{
		"Compare Paul Graham's startup advice with current AI trends in the tech industry.",
		"How do Paul Graham's views on wealth creation relate to current stock market conditions?",
		"What can we learn from Paul Graham about building successful tech companies in today's market?",
	}

	for i, query := range complexQueries {
		fmt.Printf("\n--- Query %d ---\n", i+1)
		fmt.Printf("Query: %s\n\n", query)

		response, err := subQuestionEngine.Query(ctx, query)
		if err != nil {
			log.Printf("Query failed: %v\n", err)
			continue
		}

		fmt.Printf("Response:\n%s\n", response.Response)

		if len(response.SourceNodes) > 0 {
			fmt.Printf("\nSource nodes: %d\n", len(response.SourceNodes))
		}
	}

	// 8. Demonstrate question generation separately
	fmt.Println("\n" + separator)
	fmt.Println("=== Question Generation Demo ===")
	fmt.Println(separator)

	testQuery := "What are the key differences between Paul Graham's startup philosophy and modern tech company valuations?"
	fmt.Printf("\nOriginal Query: %s\n\n", testQuery)

	subQuestions, err := questionGen.Generate(ctx, tools, testQuery)
	if err != nil {
		log.Printf("Question generation failed: %v\n", err)
	} else {
		fmt.Println("Generated Sub-Questions:")
		for i, sq := range subQuestions {
			fmt.Printf("  %d. [%s] %s\n", i+1, sq.ToolName, sq.SubQuestion)
		}
	}

	// 9. Single-domain query for comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Single Domain Query (for comparison) ===")
	fmt.Println(separator)

	singleQuery := "What does Paul Graham say about starting a startup?"
	fmt.Printf("\nQuery: %s\n\n", singleQuery)

	response, err := paulGrahamEngine.Query(ctx, singleQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Direct Response: %s\n", response.Response)
	}

	fmt.Println("\n=== Sub-Question Query Engine Demo Complete ===")
}

// DomainQueryEngine is a simple query engine for a specific domain.
type DomainQueryEngine struct {
	name        string
	description string
	knowledge   map[string]string
}

// NewDomainQueryEngine creates a new DomainQueryEngine.
func NewDomainQueryEngine(name, description string, knowledge map[string]string) *DomainQueryEngine {
	return &DomainQueryEngine{
		name:        name,
		description: description,
		knowledge:   knowledge,
	}
}

// Query returns a response based on keyword matching.
func (d *DomainQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	queryLower := strings.ToLower(query)

	// Find best matching response
	bestMatch := ""
	bestScore := 0

	for keywords, response := range d.knowledge {
		score := 0
		for _, keyword := range strings.Split(keywords, ",") {
			if strings.Contains(queryLower, strings.TrimSpace(strings.ToLower(keyword))) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestMatch = response
		}
	}

	if bestMatch == "" {
		bestMatch = fmt.Sprintf("Based on %s knowledge: I don't have specific information about that query.", d.description)
	}

	// Create source node
	sourceNode := schema.NewTextNode(bestMatch)
	sourceNode.Metadata = map[string]interface{}{
		"source": d.name,
	}

	return &synthesizer.Response{
		Response: bestMatch,
		SourceNodes: []schema.NodeWithScore{
			{Node: *sourceNode, Score: 1.0},
		},
		Metadata: map[string]interface{}{
			"domain": d.name,
		},
	}, nil
}

// Knowledge bases

func getPaulGrahamKnowledge() map[string]string {
	return map[string]string{
		"startup,start,company,business": "Paul Graham emphasizes that the best startups solve problems the founders themselves have. He advocates for making something people want, launching early, and iterating based on user feedback. The key is to find a small group of users who love your product rather than a large group who are indifferent.",
		"wealth,money,rich":              "According to Paul Graham, wealth is not a zero-sum game. You can create wealth by creating things people want. Startups are a way to compress your working life into a few intense years. The key insight is that you can get rich by creating value, not just by capturing it.",
		"programming,code,hacker":        "Paul Graham believes that great hackers are artists. Programming is a creative act, and the best programmers are motivated by the desire to create beautiful things. He advocates for using powerful languages like Lisp and thinking about problems from first principles.",
		"y combinator,yc,accelerator":    "Y Combinator, founded by Paul Graham, revolutionized startup funding by providing small amounts of seed money, mentorship, and a network of founders. The YC model emphasizes batch funding, demo days, and building a community of founders who help each other.",
		"essay,writing,ideas":            "Paul Graham's essays cover topics from startups to philosophy. He believes in writing clearly and thinking independently. His essays often challenge conventional wisdom and encourage readers to question assumptions about work, success, and life.",
	}
}

func getTechNewsKnowledge() map[string]string {
	return map[string]string{
		"ai,artificial intelligence,machine learning,ml": "AI continues to transform industries. Large language models like GPT-4 and Claude are being integrated into enterprise software. Companies are racing to build AI assistants, and there's significant investment in AI infrastructure and chips.",
		"startup,funding,venture capital,vc":             "The tech startup ecosystem has seen a correction after the 2021 boom. Valuations have normalized, and investors are focusing more on profitability and sustainable growth. AI startups continue to attract significant funding despite the overall market cooldown.",
		"big tech,faang,google,meta,apple,amazon":        "Big tech companies are investing heavily in AI while also facing regulatory scrutiny. There's ongoing competition in cloud services, AI, and consumer devices. Layoffs have affected the sector as companies focus on efficiency.",
		"software,saas,cloud":                            "The software industry is increasingly AI-powered. SaaS companies are integrating AI features to stay competitive. Cloud infrastructure continues to grow, with major providers expanding their AI and ML offerings.",
		"crypto,blockchain,web3":                         "The crypto market has experienced significant volatility. Regulatory clarity is emerging in various jurisdictions. Enterprise blockchain adoption continues for specific use cases like supply chain and finance.",
	}
}

func getFinanceKnowledge() map[string]string {
	return map[string]string{
		"stock,market,equity,shares":              "Stock markets have shown resilience despite economic uncertainty. Tech stocks, particularly AI-related companies, have performed well. Diversification and long-term investing remain key principles for individual investors.",
		"invest,investment,portfolio":             "Investment strategies should align with individual goals and risk tolerance. Index funds offer broad market exposure with low fees. Dollar-cost averaging helps reduce timing risk. Consider both growth and value investments for balance.",
		"valuation,price,worth,multiple":          "Tech company valuations are influenced by growth rates, market size, and profitability potential. AI companies often command premium valuations due to perceived transformative potential. Traditional metrics like P/E ratios should be considered alongside growth metrics.",
		"interest rate,fed,federal reserve,rates": "Interest rates significantly impact stock valuations and investment decisions. Higher rates generally pressure growth stock valuations while benefiting savers. The Federal Reserve's policy decisions are closely watched by markets.",
		"economy,gdp,inflation,recession":         "Economic indicators suggest mixed signals. Inflation has moderated but remains above historical targets. Employment remains strong in many sectors. Consumer spending patterns continue to evolve post-pandemic.",
	}
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
