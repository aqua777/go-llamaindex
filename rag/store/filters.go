package store

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/schema"
)

// MatchesFilters checks if node metadata matches the filter criteria.
// Supports AND/OR conditions and the Eq, Ne, and IsEmpty comparison operators.
func MatchesFilters(metadata map[string]interface{}, filters *schema.MetadataFilters) bool {
	if filters == nil || len(filters.Filters) == 0 {
		return true
	}

	condition := filters.Condition
	if condition == "" {
		condition = schema.FilterConditionAnd
	}

	for _, filter := range filters.Filters {
		match := evaluateFilter(metadata, filter)

		if condition == schema.FilterConditionAnd && !match {
			return false
		}
		if condition == schema.FilterConditionOr && match {
			return true
		}
	}

	// For AND: all passed; for OR: none matched.
	return condition == schema.FilterConditionAnd
}

// evaluateFilter checks if a single filter matches the metadata.
func evaluateFilter(metadata map[string]interface{}, f schema.MetadataFilter) bool {
	val, ok := metadata[f.Key]
	if !ok {
		return f.Operator == schema.FilterOperatorIsEmpty
	}

	valStr := fmt.Sprintf("%v", val)
	filterValStr := fmt.Sprintf("%v", f.Value)

	switch f.Operator {
	case schema.FilterOperatorEq:
		return valStr == filterValStr
	case schema.FilterOperatorNe:
		return valStr != filterValStr
	case schema.FilterOperatorIsEmpty:
		return false // Key exists, so not empty.
	default:
		return false
	}
}
