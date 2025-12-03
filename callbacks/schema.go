// Package callbacks provides callback and instrumentation support for LlamaIndex.
package callbacks

import (
	"time"

	"github.com/google/uuid"
)

// TimestampFormat is the format for callback event timestamps.
const TimestampFormat = "01/02/2006, 15:04:05.000000"

// BaseTraceEvent is the base trace ID for the trace map.
const BaseTraceEvent = "root"

// CBEventType represents callback manager event types.
type CBEventType string

const (
	// CBEventTypeChunking logs for text splitting.
	CBEventTypeChunking CBEventType = "chunking"
	// CBEventTypeNodeParsing logs for document to node parsing.
	CBEventTypeNodeParsing CBEventType = "node_parsing"
	// CBEventTypeEmbedding logs for embedding operations.
	CBEventTypeEmbedding CBEventType = "embedding"
	// CBEventTypeLLM logs for LLM calls.
	CBEventTypeLLM CBEventType = "llm"
	// CBEventTypeQuery logs for query operations.
	CBEventTypeQuery CBEventType = "query"
	// CBEventTypeRetrieve logs for retrieval operations.
	CBEventTypeRetrieve CBEventType = "retrieve"
	// CBEventTypeSynthesize logs for synthesis operations.
	CBEventTypeSynthesize CBEventType = "synthesize"
	// CBEventTypeTree logs for tree summarization.
	CBEventTypeTree CBEventType = "tree"
	// CBEventTypeSubQuestion logs for sub-question generation.
	CBEventTypeSubQuestion CBEventType = "sub_question"
	// CBEventTypeTemplating logs for template operations.
	CBEventTypeTemplating CBEventType = "templating"
	// CBEventTypeFunctionCall logs for function calls.
	CBEventTypeFunctionCall CBEventType = "function_call"
	// CBEventTypeReranking logs for reranking operations.
	CBEventTypeReranking CBEventType = "reranking"
	// CBEventTypeException logs for exceptions.
	CBEventTypeException CBEventType = "exception"
	// CBEventTypeAgentStep logs for agent steps.
	CBEventTypeAgentStep CBEventType = "agent_step"
)

// LeafEvents are events that will never have children events.
var LeafEvents = []CBEventType{
	CBEventTypeChunking,
	CBEventTypeLLM,
	CBEventTypeEmbedding,
}

// IsLeafEvent checks if an event type is a leaf event.
func IsLeafEvent(eventType CBEventType) bool {
	for _, leaf := range LeafEvents {
		if eventType == leaf {
			return true
		}
	}
	return false
}

// EventPayload represents payload keys for events.
type EventPayload string

const (
	// EventPayloadDocuments is a list of documents before parsing.
	EventPayloadDocuments EventPayload = "documents"
	// EventPayloadChunks is a list of text chunks.
	EventPayloadChunks EventPayload = "chunks"
	// EventPayloadNodes is a list of nodes.
	EventPayloadNodes EventPayload = "nodes"
	// EventPayloadPrompt is the formatted prompt sent to LLM.
	EventPayloadPrompt EventPayload = "formatted_prompt"
	// EventPayloadMessages is a list of messages sent to LLM.
	EventPayloadMessages EventPayload = "messages"
	// EventPayloadCompletion is the completion from LLM.
	EventPayloadCompletion EventPayload = "completion"
	// EventPayloadResponse is the message response from LLM.
	EventPayloadResponse EventPayload = "response"
	// EventPayloadQueryStr is the query used for query engine.
	EventPayloadQueryStr EventPayload = "query_str"
	// EventPayloadSubQuestion is a sub question & answer + sources.
	EventPayloadSubQuestion EventPayload = "sub_question"
	// EventPayloadEmbeddings is a list of embeddings.
	EventPayloadEmbeddings EventPayload = "embeddings"
	// EventPayloadTopK is the top k nodes retrieved.
	EventPayloadTopK EventPayload = "top_k"
	// EventPayloadAdditionalKwargs is additional kwargs for event call.
	EventPayloadAdditionalKwargs EventPayload = "additional_kwargs"
	// EventPayloadSerialized is the serialized object for event caller.
	EventPayloadSerialized EventPayload = "serialized"
	// EventPayloadFunctionCall is the function call for the LLM.
	EventPayloadFunctionCall EventPayload = "function_call"
	// EventPayloadFunctionOutput is the function call output.
	EventPayloadFunctionOutput EventPayload = "function_call_response"
	// EventPayloadTool is the tool used in LLM call.
	EventPayloadTool EventPayload = "tool"
	// EventPayloadModelName is the model name used in an event.
	EventPayloadModelName EventPayload = "model_name"
	// EventPayloadTemplate is the template used in LLM call.
	EventPayloadTemplate EventPayload = "template"
	// EventPayloadTemplateVars is the template variables used in LLM call.
	EventPayloadTemplateVars EventPayload = "template_vars"
	// EventPayloadSystemPrompt is the system prompt used in LLM call.
	EventPayloadSystemPrompt EventPayload = "system_prompt"
	// EventPayloadQueryWrapperPrompt is the query wrapper prompt used in LLM.
	EventPayloadQueryWrapperPrompt EventPayload = "query_wrapper_prompt"
	// EventPayloadException is the exception raised in an event.
	EventPayloadException EventPayload = "exception"
)

// CBEvent is a generic class to store event information.
type CBEvent struct {
	// EventType is the type of the event.
	EventType CBEventType
	// Payload contains event-specific data.
	Payload map[string]interface{}
	// Time is the timestamp of the event.
	Time string
	// ID is the unique identifier for the event.
	ID string
}

// NewCBEvent creates a new CBEvent.
func NewCBEvent(eventType CBEventType, payload map[string]interface{}) *CBEvent {
	return &CBEvent{
		EventType: eventType,
		Payload:   payload,
		Time:      time.Now().Format(TimestampFormat),
		ID:        uuid.New().String(),
	}
}

// EventStats contains time-based statistics for events.
type EventStats struct {
	// TotalSecs is the total time in seconds.
	TotalSecs float64
	// AverageSecs is the average time in seconds.
	AverageSecs float64
	// TotalCount is the total number of events.
	TotalCount int
}

// NewEventStats creates a new EventStats.
func NewEventStats(totalSecs float64, count int) *EventStats {
	avgSecs := 0.0
	if count > 0 {
		avgSecs = totalSecs / float64(count)
	}
	return &EventStats{
		TotalSecs:   totalSecs,
		AverageSecs: avgSecs,
		TotalCount:  count,
	}
}
