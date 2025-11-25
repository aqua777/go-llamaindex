package textsplitter

import (
	"regexp"
	"strings"
)

// SplitTextKeepSeparator splits text with separator and keeps the separator at the start of each split (except the first).
func SplitTextKeepSeparator(text string, separator string) []string {
	if separator == "" {
		// Avoid infinite loop or weird behavior with empty separator
		if text == "" {
			return []string{}
		}
		return []string{text}
	}
	parts := strings.Split(text, separator)
	var result []string
	for i, part := range parts {
		if i > 0 {
			part = separator + part
		}
		if part != "" {
			result = append(result, part)
		}
	}
	return result
}

// SplitBySep returns a function that splits text by a separator.
func SplitBySep(sep string) func(string) []string {
	return func(text string) []string {
		return SplitTextKeepSeparator(text, sep)
	}
}

// SplitByRegex returns a function that splits text using a regex.
func SplitByRegex(regexStr string) func(string) []string {
	// We panic here if regex is invalid because this is configuration time mostly
	// In a real library we might return error, but matching Python's style of functional composition.
	re := regexp.MustCompile(regexStr)
	return func(text string) []string {
		return re.FindAllString(text, -1)
	}
}

// SplitByChar returns a function that splits text into characters.
func SplitByChar() func(string) []string {
	return func(text string) []string {
		return strings.Split(text, "")
	}
}

