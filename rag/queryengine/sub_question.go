package queryengine

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

// SubQuestion represents a generated sub-question.
type SubQuestion struct {
	// SubQuestion is the sub-question text.
	SubQuestion string
	// ToolName is the name of the tool to use.
	ToolName string
}

// SubQuestionAnswerPair pairs a sub-question with its answer.
type SubQuestionAnswerPair struct {
	SubQ    SubQuestion
	Answer  string
	Sources []schema.NodeWithScore
}

// QuestionGenerator generates sub-questions from a complex query.
type QuestionGenerator interface {
	// Generate generates sub-questions for the given query and tools.
	Generate(ctx context.Context, tools []*QueryEngineTool, query string) ([]SubQuestion, error)
}

// LLMQuestionGenerator uses an LLM to generate sub-questions.
type LLMQuestionGenerator struct {
	LLM    llm.LLM
	Prompt prompts.BasePromptTemplate
}

// Default prompt for question generation.
const defaultQuestionGenPrompt = `You are a helpful assistant that generates search queries based on multiple tools.
You have access to the following tools:
{tools_str}

Given the following question, generate {num_questions} sub-questions that can be answered using the tools above.
Each sub-question should be on a new line in the format: [tool_name] sub-question

Question: {query_str}
Sub-questions:`

// NewLLMQuestionGenerator creates a new LLMQuestionGenerator.
func NewLLMQuestionGenerator(llmModel llm.LLM) *LLMQuestionGenerator {
	return &LLMQuestionGenerator{
		LLM:    llmModel,
		Prompt: prompts.NewPromptTemplate(defaultQuestionGenPrompt, prompts.PromptTypeCustom),
	}
}

// Generate generates sub-questions for the given query and tools.
func (qg *LLMQuestionGenerator) Generate(ctx context.Context, tools []*QueryEngineTool, query string) ([]SubQuestion, error) {
	// Build tools string
	var toolsStr strings.Builder
	for _, tool := range tools {
		toolsStr.WriteString(fmt.Sprintf("- %s: %s\n", tool.Name, tool.Description))
	}

	// Format prompt
	prompt := qg.Prompt.Format(map[string]string{
		"tools_str":     toolsStr.String(),
		"num_questions": fmt.Sprintf("%d", len(tools)),
		"query_str":     query,
	})

	// Get LLM response
	response, err := qg.LLM.Complete(ctx, prompt)
	if err != nil {
		return nil, err
	}

	// Parse response into sub-questions
	return qg.parseSubQuestions(response, tools)
}

// parseSubQuestions parses the LLM response into sub-questions.
func (qg *LLMQuestionGenerator) parseSubQuestions(response string, tools []*QueryEngineTool) ([]SubQuestion, error) {
	var subQuestions []SubQuestion

	// Create tool name set for validation
	toolNames := make(map[string]bool)
	for _, tool := range tools {
		toolNames[tool.Name] = true
	}

	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Try to parse [tool_name] question format
		if strings.HasPrefix(line, "[") {
			endBracket := strings.Index(line, "]")
			if endBracket > 0 {
				toolName := strings.TrimSpace(line[1:endBracket])
				question := strings.TrimSpace(line[endBracket+1:])

				if toolNames[toolName] && question != "" {
					subQuestions = append(subQuestions, SubQuestion{
						SubQuestion: question,
						ToolName:    toolName,
					})
				}
			}
		}
	}

	// If no sub-questions parsed, create one for each tool with original query
	if len(subQuestions) == 0 {
		for _, tool := range tools {
			subQuestions = append(subQuestions, SubQuestion{
				SubQuestion: response, // Use full response as question
				ToolName:    tool.Name,
			})
			break // Just use first tool
		}
	}

	return subQuestions, nil
}

// SubQuestionQueryEngine decomposes complex queries into sub-questions.
type SubQuestionQueryEngine struct {
	*BaseQueryEngine
	// QuestionGen generates sub-questions.
	QuestionGen QuestionGenerator
	// Synthesizer synthesizes the final response.
	Synthesizer synthesizer.Synthesizer
	// QueryEngines maps tool names to query engines.
	QueryEngines map[string]QueryEngine
	// Tools are the available query engine tools.
	Tools []*QueryEngineTool
}

// SubQuestionQueryEngineOption is a functional option.
type SubQuestionQueryEngineOption func(*SubQuestionQueryEngine)

// WithSubQuestionVerbose enables verbose logging.
func WithSubQuestionVerbose(verbose bool) SubQuestionQueryEngineOption {
	return func(sqe *SubQuestionQueryEngine) {
		sqe.Verbose = verbose
	}
}

// NewSubQuestionQueryEngine creates a new SubQuestionQueryEngine.
func NewSubQuestionQueryEngine(
	questionGen QuestionGenerator,
	synth synthesizer.Synthesizer,
	tools []*QueryEngineTool,
	opts ...SubQuestionQueryEngineOption,
) *SubQuestionQueryEngine {
	// Build query engine map
	queryEngines := make(map[string]QueryEngine)
	for _, tool := range tools {
		queryEngines[tool.Name] = tool.QueryEngine
	}

	sqe := &SubQuestionQueryEngine{
		BaseQueryEngine: NewBaseQueryEngine(),
		QuestionGen:     questionGen,
		Synthesizer:     synth,
		QueryEngines:    queryEngines,
		Tools:           tools,
	}

	for _, opt := range opts {
		opt(sqe)
	}

	return sqe
}

// Query executes a query by decomposing it into sub-questions.
func (sqe *SubQuestionQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Generate sub-questions
	subQuestions, err := sqe.QuestionGen.Generate(ctx, sqe.Tools, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate sub-questions: %w", err)
	}

	// Answer each sub-question
	var qaPairs []SubQuestionAnswerPair
	for _, subQ := range subQuestions {
		qaPair, err := sqe.querySubQuestion(ctx, subQ)
		if err != nil {
			// Log error but continue with other sub-questions
			continue
		}
		if qaPair != nil {
			qaPairs = append(qaPairs, *qaPair)
		}
	}

	// Build nodes from QA pairs
	var nodes []schema.NodeWithScore
	var sourceNodes []schema.NodeWithScore

	for _, pair := range qaPairs {
		// Create node from QA pair
		nodeText := fmt.Sprintf("Sub question: %s\nResponse: %s", pair.SubQ.SubQuestion, pair.Answer)
		node := schema.NewTextNode(nodeText)
		nodes = append(nodes, schema.NodeWithScore{Node: *node, Score: 1.0})

		// Collect source nodes
		sourceNodes = append(sourceNodes, pair.Sources...)
	}

	// Synthesize final response
	response, err := sqe.Synthesizer.Synthesize(ctx, query, nodes)
	if err != nil {
		return nil, err
	}

	// Add source nodes to response
	response.SourceNodes = append(response.SourceNodes, sourceNodes...)

	return response, nil
}

// querySubQuestion queries a single sub-question.
func (sqe *SubQuestionQueryEngine) querySubQuestion(ctx context.Context, subQ SubQuestion) (*SubQuestionAnswerPair, error) {
	engine, ok := sqe.QueryEngines[subQ.ToolName]
	if !ok {
		return nil, fmt.Errorf("query engine not found for tool: %s", subQ.ToolName)
	}

	response, err := engine.Query(ctx, subQ.SubQuestion)
	if err != nil {
		return nil, err
	}

	return &SubQuestionAnswerPair{
		SubQ:    subQ,
		Answer:  response.Response,
		Sources: response.SourceNodes,
	}, nil
}

// Ensure SubQuestionQueryEngine implements QueryEngine.
var _ QueryEngine = (*SubQuestionQueryEngine)(nil)
