// Package main demonstrates PII (Personally Identifiable Information) masking.
// This example corresponds to Python's node_postprocessor/PII.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/postprocessor"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== PII Masking Demo ===")
	fmt.Println("\nDetects and masks personally identifiable information in retrieved content.")

	separator := strings.Repeat("=", 60)

	// 1. Create nodes with PII
	nodes := createNodesWithPII()

	fmt.Println("\nOriginal nodes containing PII:")
	for i, node := range nodes {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 2. Basic PII masking
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic PII Masking ===")
	fmt.Println(separator)

	piiMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
	)

	fmt.Println("\nMasking all detected PII types...")

	maskedNodes, err := piiMasker.PostprocessNodes(ctx, nodes, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("\nAfter PII masking:")
	for i, node := range maskedNodes {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 3. Selective PII types
	fmt.Println("\n" + separator)
	fmt.Println("=== Selective PII Masking ===")
	fmt.Println(separator)

	emailOnlyMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
		postprocessor.WithPIITypes(postprocessor.PIITypeEmail),
	)

	fmt.Println("\nMasking only email addresses...")

	emailMasked, _ := emailOnlyMasker.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter email-only masking:")
	for i, node := range emailMasked {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// Phone only
	phoneOnlyMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
		postprocessor.WithPIITypes(postprocessor.PIITypePhone),
	)

	fmt.Println("\nMasking only phone numbers...")

	phoneMasked, _ := phoneOnlyMasker.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter phone-only masking:")
	for i, node := range phoneMasked {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 4. Custom mask strings
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Mask Strings ===")
	fmt.Println(separator)

	customMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
		postprocessor.WithPIICustomMask("[REDACTED]"),
	)

	fmt.Println("\nUsing custom mask: [REDACTED]")

	customMasked, _ := customMasker.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter custom masking:")
	for i, node := range customMasked {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 5. Filter mode (remove nodes with PII)
	fmt.Println("\n" + separator)
	fmt.Println("=== Filter Mode ===")
	fmt.Println(separator)

	filterMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(false), // Filter instead of mask
	)

	fmt.Println("\nFiltering out nodes containing PII (instead of masking)...")

	filteredNodes, _ := filterMasker.PostprocessNodes(ctx, nodes, nil)

	fmt.Printf("\nOriginal count: %d, After filtering: %d\n", len(nodes), len(filteredNodes))
	fmt.Println("\nRemaining nodes:")
	for i, node := range filteredNodes {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 6. Store original text
	fmt.Println("\n" + separator)
	fmt.Println("=== Preserving Original Text ===")
	fmt.Println(separator)

	preserveMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
		postprocessor.WithPIIStoreOriginal(true),
	)

	fmt.Println("\nMasking PII while storing original in metadata...")

	preservedNodes, _ := preserveMasker.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter masking with preservation:")
	for i, node := range preservedNodes {
		fmt.Printf("  %d. Masked: %s\n", i+1, truncate(node.Node.Text, 50))
		if original, ok := node.Node.Metadata["original_text"]; ok {
			fmt.Printf("     Original: %s\n", truncate(original.(string), 50))
		}
		if masked, ok := node.Node.Metadata["pii_masked"]; ok && masked.(bool) {
			fmt.Println("     [PII was detected and masked]")
		}
	}

	// 7. Detect PII without masking
	fmt.Println("\n" + separator)
	fmt.Println("=== PII Detection ===")
	fmt.Println(separator)

	detector := postprocessor.NewPIIPostprocessor()

	fmt.Println("\nDetecting PII in sample texts:")

	testTexts := []string{
		"Contact john@example.com for more info.",
		"Call us at 555-123-4567 or 1-800-555-0199.",
		"SSN: 123-45-6789, Card: 4111-1111-1111-1111",
		"Server IP: 192.168.1.100",
		"No PII in this text.",
	}

	for _, text := range testTexts {
		matches := detector.DetectPII(text)
		fmt.Printf("\n  Text: %s\n", truncate(text, 50))
		if len(matches) == 0 {
			fmt.Println("  PII found: None")
		} else {
			fmt.Printf("  PII found: %d\n", len(matches))
			for _, match := range matches {
				fmt.Printf("    - %s: %s\n", match.Type, match.Value)
			}
		}
	}

	// 8. Custom PII patterns
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom PII Patterns ===")
	fmt.Println(separator)

	customPatternMasker := postprocessor.NewPIIPostprocessor(
		postprocessor.WithPIIMask(true),
	)

	// Add custom pattern for employee IDs
	err = customPatternMasker.AddPattern("employee_id", `EMP-\d{6}`, "[EMPLOYEE_ID]")
	if err != nil {
		fmt.Printf("Error adding pattern: %v\n", err)
	}

	customNodes := []schema.NodeWithScore{
		{
			Node:  *createNode("Employee EMP-123456 reported the issue."),
			Score: 0.9,
		},
		{
			Node:  *createNode("Contact EMP-789012 or email hr@company.com"),
			Score: 0.85,
		},
	}

	fmt.Println("\nAdded custom pattern for employee IDs (EMP-XXXXXX)")

	customPatternResult, _ := customPatternMasker.PostprocessNodes(ctx, customNodes, nil)

	fmt.Println("\nAfter custom pattern masking:")
	for i, node := range customPatternResult {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Supported PII Types ===")
	fmt.Println(separator)

	fmt.Println("\nBuilt-in PII types:")
	fmt.Println("  - Email addresses")
	fmt.Println("  - Phone numbers (various formats)")
	fmt.Println("  - Social Security Numbers (SSN)")
	fmt.Println("  - Credit card numbers")
	fmt.Println("  - IP addresses")
	fmt.Println()
	fmt.Println("Configuration options:")
	fmt.Println("  - Mask mode: Replace PII with mask strings")
	fmt.Println("  - Filter mode: Remove nodes containing PII")
	fmt.Println("  - Selective types: Only detect specific PII types")
	fmt.Println("  - Custom masks: Use your own mask strings")
	fmt.Println("  - Store original: Preserve original text in metadata")
	fmt.Println("  - Custom patterns: Add domain-specific PII patterns")

	fmt.Println("\n=== PII Masking Demo Complete ===")
}

// createNodesWithPII creates sample nodes containing PII.
func createNodesWithPII() []schema.NodeWithScore {
	texts := []string{
		"Contact John Smith at john.smith@example.com for assistance.",
		"Call our support line at 555-123-4567 or 1-800-HELP-NOW.",
		"Customer SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111",
		"Server located at IP 192.168.1.100 in the data center.",
		"This document contains no personal information.",
	}

	nodes := make([]schema.NodeWithScore, len(texts))
	for i, text := range texts {
		node := schema.NewTextNode(text)
		node.ID = fmt.Sprintf("doc_%d", i+1)
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: 0.9 - float64(i)*0.05,
		}
	}

	return nodes
}

// createNode creates a single node with the given text.
func createNode(text string) *schema.Node {
	node := schema.NewTextNode(text)
	return node
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
