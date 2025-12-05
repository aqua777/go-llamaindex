// Package prompts provides prompt templates and utilities for LLM interactions.
package prompts

// PromptType represents the type/category of a prompt.
type PromptType string

const (
	// Summarization prompts
	PromptTypeSummary       PromptType = "summary"
	PromptTypeTreeSummarize PromptType = "tree_summarize"

	// Tree operations
	PromptTypeTreeInsert         PromptType = "insert"
	PromptTypeTreeSelect         PromptType = "tree_select"
	PromptTypeTreeSelectMultiple PromptType = "tree_select_multiple"

	// Question answering
	PromptTypeQuestionAnswer PromptType = "text_qa"
	PromptTypeRefine         PromptType = "refine"

	// Keyword extraction
	PromptTypeKeywordExtract      PromptType = "keyword_extract"
	PromptTypeQueryKeywordExtract PromptType = "query_keyword_extract"

	// Schema and SQL
	PromptTypeSchemaExtract        PromptType = "schema_extract"
	PromptTypeTextToSQL            PromptType = "text_to_sql"
	PromptTypeTextToGraphQuery     PromptType = "text_to_graph_query"
	PromptTypeTableContext         PromptType = "table_context"
	PromptTypeSQLResponseSynthesis PromptType = "sql_response_synthesis"

	// Knowledge graph
	PromptTypeKnowledgeTripletExtract PromptType = "knowledge_triplet_extract"

	// Input/Output
	PromptTypeSimpleInput PromptType = "simple_input"
	PromptTypeJSONPath    PromptType = "json_path"

	// Selection
	PromptTypeSingleSelect PromptType = "single_select"
	PromptTypeMultiSelect  PromptType = "multi_select"
	PromptTypeChoiceSelect PromptType = "choice_select"

	// Vector store
	PromptTypeVectorStoreQuery PromptType = "vector_store_query"

	// Sub-question
	PromptTypeSubQuestion PromptType = "sub_question"

	// Conversation
	PromptTypeConversation PromptType = "conversation"

	// Query transformation
	PromptTypeDecompose PromptType = "decompose"

	// Reranking
	PromptTypeRankGPTRerank PromptType = "rankgpt_rerank"

	// Custom (default)
	PromptTypeCustom PromptType = "custom"
)

// String returns the string representation of the prompt type.
func (pt PromptType) String() string {
	return string(pt)
}
