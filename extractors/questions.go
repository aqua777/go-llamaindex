package extractors

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// DefaultQuestionGenTemplate is the default template for question generation.
const DefaultQuestionGenTemplate = `Here is the context:
{context_str}

Given the contextual information, generate {num_questions} questions this context can provide specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided as well. Try using these summaries to generate better questions that this context can answer.

`

// QuestionsAnsweredExtractor extracts questions that a node can answer.
// It uses an LLM to generate questions that the content can specifically answer,
// which is useful for improving retrieval through question-based matching.
type QuestionsAnsweredExtractor struct {
	*LLMExtractor
	questions      int    // number of questions to generate
	promptTemplate string // template for question generation
	embeddingOnly  bool   // whether to use metadata for embeddings only
}

// QuestionsAnsweredExtractorOption configures a QuestionsAnsweredExtractor.
type QuestionsAnsweredExtractorOption func(*QuestionsAnsweredExtractor)

// WithQuestionsCount sets the number of questions to generate.
func WithQuestionsCount(n int) QuestionsAnsweredExtractorOption {
	return func(e *QuestionsAnsweredExtractor) {
		if n > 0 {
			e.questions = n
		}
	}
}

// WithQuestionsPromptTemplate sets the question generation template.
func WithQuestionsPromptTemplate(template string) QuestionsAnsweredExtractorOption {
	return func(e *QuestionsAnsweredExtractor) {
		e.promptTemplate = template
	}
}

// WithQuestionsLLM sets the LLM for question generation.
func WithQuestionsLLM(l llm.LLM) QuestionsAnsweredExtractorOption {
	return func(e *QuestionsAnsweredExtractor) {
		e.llm = l
	}
}

// WithEmbeddingOnly sets whether to use metadata for embeddings only.
func WithEmbeddingOnly(embeddingOnly bool) QuestionsAnsweredExtractorOption {
	return func(e *QuestionsAnsweredExtractor) {
		e.embeddingOnly = embeddingOnly
	}
}

// NewQuestionsAnsweredExtractor creates a new QuestionsAnsweredExtractor.
func NewQuestionsAnsweredExtractor(opts ...QuestionsAnsweredExtractorOption) *QuestionsAnsweredExtractor {
	e := &QuestionsAnsweredExtractor{
		LLMExtractor: NewLLMExtractor(
			[]BaseExtractorOption{
				WithExtractorName("QuestionsAnsweredExtractor"),
				WithTextNodeOnly(true),
			},
		),
		questions:      5,
		promptTemplate: DefaultQuestionGenTemplate,
		embeddingOnly:  true,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Extract extracts questions from nodes.
// Returns metadata with "questions_this_excerpt_can_answer" field for each node.
func (e *QuestionsAnsweredExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for QuestionsAnsweredExtractor")
	}

	if len(nodes) == 0 {
		return []ExtractedMetadata{}, nil
	}

	return runConcurrent(ctx, nodes, e.numWorkers, func(ctx context.Context, node *schema.Node, _ int) (ExtractedMetadata, error) {
		return e.extractQuestionsFromNode(ctx, node)
	})
}

// extractQuestionsFromNode extracts questions from a single node.
func (e *QuestionsAnsweredExtractor) extractQuestionsFromNode(ctx context.Context, node *schema.Node) (ExtractedMetadata, error) {
	content := e.GetNodeContent(node)
	prompt := formatPrompt(e.promptTemplate, map[string]string{
		"context_str":   content,
		"num_questions": fmt.Sprintf("%d", e.questions),
	})

	questions, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate questions: %w", err)
	}

	return ExtractedMetadata{
		"questions_this_excerpt_can_answer": strings.TrimSpace(questions),
	}, nil
}

// IsEmbeddingOnly returns whether metadata is for embeddings only.
func (e *QuestionsAnsweredExtractor) IsEmbeddingOnly() bool {
	return e.embeddingOnly
}

// ParseQuestions parses a newline-separated questions string into a slice.
func ParseQuestions(questionsStr string) []string {
	lines := strings.Split(questionsStr, "\n")
	questions := make([]string, 0, len(lines))
	for _, line := range lines {
		question := strings.TrimSpace(line)
		// Remove common prefixes like "1.", "- ", etc.
		question = strings.TrimLeft(question, "0123456789.-) ")
		question = strings.TrimSpace(question)
		if question != "" {
			questions = append(questions, question)
		}
	}
	return questions
}

// ProcessNodes extracts questions and updates nodes.
func (e *QuestionsAnsweredExtractor) ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error) {
	var newNodes []*schema.Node
	if e.inPlace {
		newNodes = nodes
	} else {
		newNodes = make([]*schema.Node, len(nodes))
		for i, node := range nodes {
			nodeCopy := *node
			newNodes[i] = &nodeCopy
		}
	}

	metadataList, err := e.Extract(ctx, newNodes)
	if err != nil {
		return nil, err
	}

	for i, node := range newNodes {
		if node.Metadata == nil {
			node.Metadata = make(map[string]interface{})
		}
		for k, v := range metadataList[i] {
			node.Metadata[k] = v
		}
	}

	return newNodes, nil
}

// Ensure QuestionsAnsweredExtractor implements MetadataExtractor.
var _ MetadataExtractor = (*QuestionsAnsweredExtractor)(nil)
