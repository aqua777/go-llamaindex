// Package main demonstrates recency-based postprocessing for time-sensitive retrieval.
// This example corresponds to Python's node_postprocessor/RecencyPostprocessorDemo.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/postprocessor"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Recency Postprocessor Demo ===")
	fmt.Println("\nFilters and reweights nodes based on document timestamps.")

	separator := strings.Repeat("=", 60)

	// Fixed reference time for consistent demo
	referenceTime := time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)

	// 1. Create nodes with timestamps
	nodes := createTimestampedNodes(referenceTime)

	fmt.Println("\nSample nodes with timestamps:")
	for i, node := range nodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. (score: %.2f) [%s] %s\n", i+1, node.Score, dateStr, truncate(node.Node.Text, 35))
	}

	// 2. Basic recency filtering
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Recency Filtering ===")
	fmt.Println(separator)

	recencyFilter := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencyMaxAge(30*24*time.Hour), // 30 days
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nFiltering nodes older than 30 days...")

	filteredNodes, err := recencyFilter.PostprocessNodes(ctx, nodes, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("\nOriginal: %d nodes, After filtering: %d nodes\n", len(nodes), len(filteredNodes))
	fmt.Println("\nFiltered results:")
	for i, node := range filteredNodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. (score: %.2f) [%s] %s\n", i+1, node.Score, dateStr, truncate(node.Node.Text, 35))
	}

	// 3. Linear time decay
	fmt.Println("\n" + separator)
	fmt.Println("=== Linear Time Decay ===")
	fmt.Println(separator)

	linearDecay := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencyMaxAge(60*24*time.Hour), // 60 days
		postprocessor.WithRecencyTimeWeightMode(postprocessor.TimeWeightModeLinear),
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nApplying linear decay over 60 days...")
	fmt.Println("(Score decreases linearly with age)")

	linearNodes, _ := linearDecay.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter linear decay:")
	for i, node := range linearNodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. (adjusted: %.2f) [%s] %s\n", i+1, node.Score, dateStr, truncate(node.Node.Text, 30))
	}

	// 4. Exponential decay
	fmt.Println("\n" + separator)
	fmt.Println("=== Exponential Time Decay ===")
	fmt.Println(separator)

	expDecay := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencyMaxAge(60*24*time.Hour),
		postprocessor.WithRecencyTimeWeightMode(postprocessor.TimeWeightModeExponential),
		postprocessor.WithRecencyDecayRate(0.7),
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nApplying exponential decay (rate: 0.7)...")
	fmt.Println("(Score decays exponentially with age)")

	expNodes, _ := expDecay.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter exponential decay:")
	for i, node := range expNodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. (adjusted: %.2f) [%s] %s\n", i+1, node.Score, dateStr, truncate(node.Node.Text, 30))
	}

	// 5. Step function (recent vs old)
	fmt.Println("\n" + separator)
	fmt.Println("=== Step Function Weighting ===")
	fmt.Println(separator)

	stepWeight := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencyTimeWeightMode(postprocessor.TimeWeightModeStep),
		postprocessor.WithRecencyStepThreshold(14*24*time.Hour, 1.0, 0.3), // 14 days threshold
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nApplying step function:")
	fmt.Println("  - Recent (< 14 days): weight = 1.0")
	fmt.Println("  - Old (>= 14 days): weight = 0.3")

	stepNodes, _ := stepWeight.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nAfter step weighting:")
	for i, node := range stepNodes {
		dateStr := node.Node.Metadata["date"].(string)
		age := referenceTime.Sub(mustParseDate(dateStr))
		category := "OLD"
		if age < 14*24*time.Hour {
			category = "RECENT"
		}
		fmt.Printf("  %d. [%s] (adjusted: %.2f) [%s] %s\n", i+1, category, node.Score, dateStr, truncate(node.Node.Text, 25))
	}

	// 6. Sort by date
	fmt.Println("\n" + separator)
	fmt.Println("=== Sort by Date ===")
	fmt.Println(separator)

	dateSorter := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencySortByDate(true),
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nSorting nodes by date (newest first)...")

	sortedNodes, _ := dateSorter.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nSorted by date:")
	for i, node := range sortedNodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. [%s] %s\n", i+1, dateStr, truncate(node.Node.Text, 40))
	}

	// 7. Top-K with recency
	fmt.Println("\n" + separator)
	fmt.Println("=== Top-K with Recency ===")
	fmt.Println(separator)

	topKRecency := postprocessor.NewNodeRecencyPostprocessor(
		postprocessor.WithRecencyDateKey("date"),
		postprocessor.WithRecencyTimeWeightMode(postprocessor.TimeWeightModeLinear),
		postprocessor.WithRecencyMaxAge(60*24*time.Hour),
		postprocessor.WithRecencyTopK(3),
		postprocessor.WithRecencyNowFunc(func() time.Time { return referenceTime }),
	)

	fmt.Println("\nApplying recency weighting and returning top 3...")

	topKNodes, _ := topKRecency.PostprocessNodes(ctx, nodes, nil)

	fmt.Printf("\nTop %d nodes by recency-adjusted score:\n", len(topKNodes))
	for i, node := range topKNodes {
		dateStr := node.Node.Metadata["date"].(string)
		fmt.Printf("  %d. (adjusted: %.2f) [%s] %s\n", i+1, node.Score, dateStr, truncate(node.Node.Text, 30))
	}

	// 8. Different date formats
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Date Formats ===")
	fmt.Println(separator)

	fmt.Println("\nSupported date formats:")
	fmt.Println("  - RFC3339: 2024-01-15T12:00:00Z")
	fmt.Println("  - ISO date: 2024-01-15")
	fmt.Println("  - US format: 01/15/2024")
	fmt.Println("  - Unix timestamp: 1705320000")
	fmt.Println("  - time.Time objects")

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nRecency Postprocessor Features:")
	fmt.Println("  1. Max Age Filter: Remove documents older than threshold")
	fmt.Println("  2. Linear Decay: Score decreases linearly with age")
	fmt.Println("  3. Exponential Decay: Score decays exponentially")
	fmt.Println("  4. Step Function: Binary recent/old classification")
	fmt.Println("  5. Date Sorting: Order by timestamp")
	fmt.Println("  6. Top-K: Limit results after recency adjustment")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - News and current events retrieval")
	fmt.Println("  - Documentation with version history")
	fmt.Println("  - Time-sensitive knowledge bases")
	fmt.Println("  - Prioritizing recent information")

	fmt.Println("\n=== Recency Postprocessor Demo Complete ===")
}

// createTimestampedNodes creates nodes with various timestamps.
func createTimestampedNodes(referenceTime time.Time) []schema.NodeWithScore {
	data := []struct {
		text    string
		daysAgo int
		score   float64
	}{
		{"Latest update on the project status and milestones.", 2, 0.75},
		{"Weekly report from last week with key metrics.", 7, 0.85},
		{"Monthly summary from two weeks ago.", 14, 0.90},
		{"Quarterly review from last month.", 30, 0.88},
		{"Annual report from two months ago.", 60, 0.92},
		{"Historical document from three months ago.", 90, 0.80},
	}

	nodes := make([]schema.NodeWithScore, len(data))
	for i, d := range data {
		nodeTime := referenceTime.AddDate(0, 0, -d.daysAgo)
		node := schema.NewTextNode(d.text)
		node.ID = fmt.Sprintf("doc_%d", i+1)
		node.Metadata["date"] = nodeTime.Format("2006-01-02")
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: d.score,
		}
	}

	return nodes
}

// mustParseDate parses a date string or panics.
func mustParseDate(dateStr string) time.Time {
	t, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		panic(err)
	}
	return t
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
