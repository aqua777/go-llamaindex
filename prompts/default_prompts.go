package prompts

// Default prompt templates for common use cases.

// Summary prompts
const (
	DefaultSummaryPromptTmpl = `Write a summary of the following. Try to use only the information provided. Try to include as many key details as possible.

{context_str}

SUMMARY:`

	DefaultTreeSummarizeTmpl = `Context information from multiple sources is below.
---------------------
{context_str}
---------------------
Given the information from multiple sources and not prior knowledge, answer the query.
Query: {query_str}
Answer: `
)

// Question-Answer prompts
const (
	DefaultTextQAPromptTmpl = `Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: `

	DefaultRefinePromptTmpl = `The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer: `
)

// Tree prompts
const (
	DefaultInsertPromptTmpl = `Context information is below. It is provided in a numbered list (1 to {num_chunks}), where each item in the list corresponds to a summary.
---------------------
{context_list}
---------------------
Given the context information, here is a new piece of information: {new_chunk_text}
Answer with the number corresponding to the summary that should be updated. The answer should be the number corresponding to the summary that is most relevant to the question.
`

	DefaultQueryPromptTmpl = `Some choices are given below. It is provided in a numbered list (1 to {num_chunks}), where each item in the list corresponds to a summary.
---------------------
{context_list}
---------------------
Using only the choices above and not prior knowledge, return the choice that is most relevant to the question: '{query_str}'
Provide choice in the following format: 'ANSWER: <number>' and explain why this summary was selected in relation to the question.
`

	DefaultQueryPromptMultipleTmpl = `Some choices are given below. It is provided in a numbered list (1 to {num_chunks}), where each item in the list corresponds to a summary.
---------------------
{context_list}
---------------------
Using only the choices above and not prior knowledge, return the top choices (no more than {branching_factor}, ranked by most relevant to least) that are most relevant to the question: '{query_str}'
Provide choices in the following format: 'ANSWER: <numbers>' and explain why these summaries were selected in relation to the question.
`
)

// Keyword extraction prompts
const (
	DefaultKeywordExtractTmpl = `Some text is provided below. Given the text, extract up to {max_keywords} keywords from the text. Avoid stopwords.
---------------------
{text}
---------------------
Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'
`

	DefaultQueryKeywordExtractTmpl = `A question is provided below. Given the question, extract up to {max_keywords} keywords from the text. Focus on extracting the keywords that we can use to best lookup answers to the question. Avoid stopwords.
---------------------
{question}
---------------------
Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'
`
)

// Knowledge graph prompts
const (
	DefaultKGTripletExtractTmpl = `Some text is provided below. Given the text, extract up to {max_knowledge_triplets} knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.
---------------------
Example:
Text: Alice is Bob's mother.
Triplets:
(Alice, is mother of, Bob)
Text: Philz is a coffee shop founded in Berkeley in 1982.
Triplets:
(Philz, is, coffee shop)
(Philz, founded in, Berkeley)
(Philz, founded in, 1982)
---------------------
Text: {text}
Triplets:
`
)

// Simple prompts
const (
	DefaultSimpleInputTmpl = `{query_str}`

	DefaultHYDEPromptTmpl = `Please write a passage to answer the question
Try to include as many key details as possible.

{context_str}

Passage:`
)

// Choice selection prompts
const (
	DefaultChoiceSelectPromptTmpl = `A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Do not include any documents that are not relevant to the question.
Example format:
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{context_str}
Question: {query_str}
Answer:
`
)

// Reranking prompts
const (
	DefaultRankGPTRerankTmpl = `Search Query: {query}. 
Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.`
)

// Default prompt instances
var (
	// Summary prompts
	DefaultSummaryPrompt       = NewPromptTemplate(DefaultSummaryPromptTmpl, PromptTypeSummary)
	DefaultTreeSummarizePrompt = NewPromptTemplate(DefaultTreeSummarizeTmpl, PromptTypeSummary)

	// QA prompts
	DefaultTextQAPrompt = NewPromptTemplate(DefaultTextQAPromptTmpl, PromptTypeQuestionAnswer)
	DefaultRefinePrompt = NewPromptTemplate(DefaultRefinePromptTmpl, PromptTypeRefine)

	// Tree prompts
	DefaultInsertPrompt        = NewPromptTemplate(DefaultInsertPromptTmpl, PromptTypeTreeInsert)
	DefaultQueryPrompt         = NewPromptTemplate(DefaultQueryPromptTmpl, PromptTypeTreeSelect)
	DefaultQueryPromptMultiple = NewPromptTemplate(DefaultQueryPromptMultipleTmpl, PromptTypeTreeSelectMultiple)

	// Keyword prompts
	DefaultKeywordExtractPrompt      = NewPromptTemplate(DefaultKeywordExtractTmpl, PromptTypeKeywordExtract)
	DefaultQueryKeywordExtractPrompt = NewPromptTemplate(DefaultQueryKeywordExtractTmpl, PromptTypeQueryKeywordExtract)

	// KG prompts
	DefaultKGTripletExtractPrompt = NewPromptTemplate(DefaultKGTripletExtractTmpl, PromptTypeKnowledgeTripletExtract)

	// Simple prompts
	DefaultSimpleInputPrompt = NewPromptTemplate(DefaultSimpleInputTmpl, PromptTypeSimpleInput)
	DefaultHYDEPrompt        = NewPromptTemplate(DefaultHYDEPromptTmpl, PromptTypeSummary)

	// Choice prompts
	DefaultChoiceSelectPrompt = NewPromptTemplate(DefaultChoiceSelectPromptTmpl, PromptTypeChoiceSelect)

	// Rerank prompts
	DefaultRankGPTRerankPrompt = NewPromptTemplate(DefaultRankGPTRerankTmpl, PromptTypeRankGPTRerank)
)

// GetDefaultPrompt returns a default prompt by type.
func GetDefaultPrompt(promptType PromptType) BasePromptTemplate {
	switch promptType {
	case PromptTypeSummary:
		return DefaultSummaryPrompt
	case PromptTypeTreeSummarize:
		return DefaultTreeSummarizePrompt
	case PromptTypeQuestionAnswer:
		return DefaultTextQAPrompt
	case PromptTypeRefine:
		return DefaultRefinePrompt
	case PromptTypeTreeInsert:
		return DefaultInsertPrompt
	case PromptTypeTreeSelect:
		return DefaultQueryPrompt
	case PromptTypeTreeSelectMultiple:
		return DefaultQueryPromptMultiple
	case PromptTypeKeywordExtract:
		return DefaultKeywordExtractPrompt
	case PromptTypeQueryKeywordExtract:
		return DefaultQueryKeywordExtractPrompt
	case PromptTypeKnowledgeTripletExtract:
		return DefaultKGTripletExtractPrompt
	case PromptTypeSimpleInput:
		return DefaultSimpleInputPrompt
	case PromptTypeChoiceSelect:
		return DefaultChoiceSelectPrompt
	case PromptTypeRankGPTRerank:
		return DefaultRankGPTRerankPrompt
	default:
		return nil
	}
}
