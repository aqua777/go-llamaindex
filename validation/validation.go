// Package validation provides input validation utilities for go-llamaindex.
package validation

import (
	"errors"
	"fmt"
	"strings"
)

// ValidationError represents a validation error with field context.
type ValidationError struct {
	Field   string
	Message string
	Value   interface{}
}

func (e *ValidationError) Error() string {
	if e.Value != nil {
		return fmt.Sprintf("%s: %s (got: %v)", e.Field, e.Message, e.Value)
	}
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

// ValidationErrors is a collection of validation errors.
type ValidationErrors []ValidationError

func (e ValidationErrors) Error() string {
	if len(e) == 0 {
		return ""
	}
	var msgs []string
	for _, err := range e {
		msgs = append(msgs, err.Error())
	}
	return "validation failed: " + strings.Join(msgs, "; ")
}

// HasErrors returns true if there are any validation errors.
func (e ValidationErrors) HasErrors() bool {
	return len(e) > 0
}

// ToError returns nil if no errors, otherwise returns the ValidationErrors.
func (e ValidationErrors) ToError() error {
	if len(e) == 0 {
		return nil
	}
	return e
}

// Validator collects validation errors.
type Validator struct {
	errors ValidationErrors
}

// NewValidator creates a new Validator.
func NewValidator() *Validator {
	return &Validator{}
}

// AddError adds a validation error.
func (v *Validator) AddError(field, message string, value interface{}) {
	v.errors = append(v.errors, ValidationError{
		Field:   field,
		Message: message,
		Value:   value,
	})
}

// Require checks that a condition is true.
func (v *Validator) Require(condition bool, field, message string) {
	if !condition {
		v.AddError(field, message, nil)
	}
}

// RequirePositive checks that an integer is positive (> 0).
func (v *Validator) RequirePositive(value int, field string) {
	if value <= 0 {
		v.AddError(field, "must be positive", value)
	}
}

// RequireNonNegative checks that an integer is non-negative (>= 0).
func (v *Validator) RequireNonNegative(value int, field string) {
	if value < 0 {
		v.AddError(field, "must be non-negative", value)
	}
}

// RequireNotEmpty checks that a string is not empty.
func (v *Validator) RequireNotEmpty(value, field string) {
	if value == "" {
		v.AddError(field, "must not be empty", nil)
	}
}

// RequireNotNil checks that a value is not nil.
func (v *Validator) RequireNotNil(value interface{}, field string) {
	if value == nil {
		v.AddError(field, "must not be nil", nil)
	}
}

// RequireLessThan checks that a < b.
func (v *Validator) RequireLessThan(a, b int, fieldA, fieldB string) {
	if a >= b {
		v.AddError(fieldA, fmt.Sprintf("must be less than %s", fieldB), a)
	}
}

// RequireLessOrEqual checks that a <= b.
func (v *Validator) RequireLessOrEqual(a, b int, fieldA, fieldB string) {
	if a > b {
		v.AddError(fieldA, fmt.Sprintf("must be less than or equal to %s", fieldB), a)
	}
}

// Errors returns the collected validation errors.
func (v *Validator) Errors() ValidationErrors {
	return v.errors
}

// Error returns an error if there are validation errors, nil otherwise.
func (v *Validator) Error() error {
	return v.errors.ToError()
}

// HasErrors returns true if there are any validation errors.
func (v *Validator) HasErrors() bool {
	return v.errors.HasErrors()
}

// Common validation errors
var (
	ErrChunkSizeNotPositive    = errors.New("chunk_size must be positive")
	ErrChunkOverlapNegative    = errors.New("chunk_overlap must be non-negative")
	ErrChunkOverlapTooLarge    = errors.New("chunk_overlap must be less than chunk_size")
	ErrEmptyInput              = errors.New("input must not be empty")
	ErrNilTokenizer            = errors.New("tokenizer must not be nil")
	ErrInvalidDirectory        = errors.New("directory path is invalid")
	ErrInvalidFilePath         = errors.New("file path is invalid")
)

// ValidateChunkParams validates chunk_size and chunk_overlap parameters.
func ValidateChunkParams(chunkSize, chunkOverlap int) error {
	v := NewValidator()
	v.RequirePositive(chunkSize, "chunk_size")
	v.RequireNonNegative(chunkOverlap, "chunk_overlap")
	if chunkOverlap >= chunkSize && chunkSize > 0 {
		v.AddError("chunk_overlap", "must be less than chunk_size", chunkOverlap)
	}
	return v.Error()
}

// ValidateChunkParamsWithMinSize validates chunk parameters with a minimum effective size.
func ValidateChunkParamsWithMinSize(chunkSize, chunkOverlap, minEffectiveSize int) error {
	if err := ValidateChunkParams(chunkSize, chunkOverlap); err != nil {
		return err
	}
	effectiveSize := chunkSize - chunkOverlap
	if effectiveSize < minEffectiveSize {
		return fmt.Errorf("effective chunk size (%d) must be at least %d", effectiveSize, minEffectiveSize)
	}
	return nil
}
