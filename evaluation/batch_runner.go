package evaluation

import (
	"context"
	"fmt"
	"sync"
)

// BatchEvalRunner runs evaluations in batch with concurrency control.
type BatchEvalRunner struct {
	evaluators   map[string]Evaluator
	workers      int
	showProgress bool
}

// BatchEvalRunnerOption configures a BatchEvalRunner.
type BatchEvalRunnerOption func(*BatchEvalRunner)

// WithBatchWorkers sets the number of concurrent workers.
func WithBatchWorkers(workers int) BatchEvalRunnerOption {
	return func(r *BatchEvalRunner) {
		r.workers = workers
	}
}

// WithBatchShowProgress enables progress reporting.
func WithBatchShowProgress(show bool) BatchEvalRunnerOption {
	return func(r *BatchEvalRunner) {
		r.showProgress = show
	}
}

// NewBatchEvalRunner creates a new BatchEvalRunner.
func NewBatchEvalRunner(evaluators map[string]Evaluator, opts ...BatchEvalRunnerOption) *BatchEvalRunner {
	r := &BatchEvalRunner{
		evaluators:   evaluators,
		workers:      2,
		showProgress: false,
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// BatchEvalResult contains the results of a batch evaluation.
type BatchEvalResult struct {
	// Results maps evaluator name to list of evaluation results.
	Results map[string][]*EvaluationResult
	// Errors contains any errors that occurred during evaluation.
	Errors []error
}

// NewBatchEvalResult creates a new BatchEvalResult.
func NewBatchEvalResult(evaluatorNames []string) *BatchEvalResult {
	results := make(map[string][]*EvaluationResult)
	for _, name := range evaluatorNames {
		results[name] = []*EvaluationResult{}
	}
	return &BatchEvalResult{
		Results: results,
		Errors:  []error{},
	}
}

// GetAverageScore returns the average score for an evaluator.
func (r *BatchEvalResult) GetAverageScore(evaluatorName string) float64 {
	results, ok := r.Results[evaluatorName]
	if !ok || len(results) == 0 {
		return 0
	}

	var total float64
	var count int
	for _, result := range results {
		if result.Score != nil {
			total += *result.Score
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return total / float64(count)
}

// GetPassingRate returns the passing rate for an evaluator.
func (r *BatchEvalResult) GetPassingRate(evaluatorName string) float64 {
	results, ok := r.Results[evaluatorName]
	if !ok || len(results) == 0 {
		return 0
	}

	var passing int
	for _, result := range results {
		if result.IsPassing() {
			passing++
		}
	}

	return float64(passing) / float64(len(results))
}

// Summary returns a summary of the batch evaluation results.
func (r *BatchEvalResult) Summary() map[string]map[string]float64 {
	summary := make(map[string]map[string]float64)
	for name := range r.Results {
		summary[name] = map[string]float64{
			"average_score": r.GetAverageScore(name),
			"passing_rate":  r.GetPassingRate(name),
			"total_count":   float64(len(r.Results[name])),
		}
	}
	return summary
}

// evalJob represents a single evaluation job.
type evalJob struct {
	evaluatorName string
	evaluator     Evaluator
	input         *EvaluateInput
	index         int
}

// evalResult represents the result of a single evaluation job.
type evalResult struct {
	evaluatorName string
	result        *EvaluationResult
	err           error
	index         int
}

// EvaluateResponseStrs evaluates query, response pairs as strings.
func (r *BatchEvalRunner) EvaluateResponseStrs(
	ctx context.Context,
	queries []string,
	responses []string,
	contextsList [][]string,
	extraKwargs map[string][]interface{},
) (*BatchEvalResult, error) {
	// Validate inputs
	n, err := r.validateInputs(queries, responses, contextsList)
	if err != nil {
		return nil, err
	}

	// Create jobs
	jobs := make([]evalJob, 0, n*len(r.evaluators))
	for i := 0; i < n; i++ {
		input := NewEvaluateInput()

		if queries != nil && i < len(queries) {
			input.WithQuery(queries[i])
		}
		if responses != nil && i < len(responses) {
			input.WithResponse(responses[i])
		}
		if contextsList != nil && i < len(contextsList) {
			input.WithContexts(contextsList[i])
		}

		// Add extra kwargs
		for key, values := range extraKwargs {
			if i < len(values) {
				if key == "reference" {
					if ref, ok := values[i].(string); ok {
						input.WithReference(ref)
					}
				} else {
					input.Extra[key] = values[i]
				}
			}
		}

		for name, evaluator := range r.evaluators {
			jobs = append(jobs, evalJob{
				evaluatorName: name,
				evaluator:     evaluator,
				input:         input,
				index:         i,
			})
		}
	}

	// Run evaluations
	results := r.runJobs(ctx, jobs)

	// Format results
	return r.formatResults(results), nil
}

// EvaluateWithReferences evaluates with reference answers.
func (r *BatchEvalRunner) EvaluateWithReferences(
	ctx context.Context,
	queries []string,
	responses []string,
	references []string,
) (*BatchEvalResult, error) {
	extraKwargs := make(map[string][]interface{})
	if references != nil {
		refInterfaces := make([]interface{}, len(references))
		for i, ref := range references {
			refInterfaces[i] = ref
		}
		extraKwargs["reference"] = refInterfaces
	}

	return r.EvaluateResponseStrs(ctx, queries, responses, nil, extraKwargs)
}

// validateInputs validates and returns the number of items.
func (r *BatchEvalRunner) validateInputs(queries, responses []string, contextsList [][]string) (int, error) {
	var n int

	if queries != nil {
		n = len(queries)
	}
	if responses != nil {
		if n == 0 {
			n = len(responses)
		} else if len(responses) != n {
			return 0, fmt.Errorf("queries and responses must have the same length")
		}
	}
	if contextsList != nil {
		if n == 0 {
			n = len(contextsList)
		} else if len(contextsList) != n {
			return 0, fmt.Errorf("contextsList must have the same length as queries/responses")
		}
	}

	if n == 0 {
		return 0, fmt.Errorf("at least one of queries, responses, or contextsList must be provided")
	}

	return n, nil
}

// runJobs runs evaluation jobs with concurrency control.
func (r *BatchEvalRunner) runJobs(ctx context.Context, jobs []evalJob) []evalResult {
	results := make([]evalResult, len(jobs))
	jobChan := make(chan int, len(jobs))
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < r.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jobIdx := range jobChan {
				job := jobs[jobIdx]

				result, err := job.evaluator.Evaluate(ctx, job.input)
				results[jobIdx] = evalResult{
					evaluatorName: job.evaluatorName,
					result:        result,
					err:           err,
					index:         job.index,
				}
			}
		}()
	}

	// Send jobs
	for i := range jobs {
		jobChan <- i
	}
	close(jobChan)

	// Wait for completion
	wg.Wait()

	return results
}

// formatResults formats evaluation results.
func (r *BatchEvalRunner) formatResults(results []evalResult) *BatchEvalResult {
	evaluatorNames := make([]string, 0, len(r.evaluators))
	for name := range r.evaluators {
		evaluatorNames = append(evaluatorNames, name)
	}

	batchResult := NewBatchEvalResult(evaluatorNames)

	for _, result := range results {
		if result.err != nil {
			batchResult.Errors = append(batchResult.Errors, result.err)
			// Add an invalid result
			invalidResult := NewEvaluationResult().WithInvalid(result.err.Error())
			batchResult.Results[result.evaluatorName] = append(
				batchResult.Results[result.evaluatorName],
				invalidResult,
			)
		} else {
			batchResult.Results[result.evaluatorName] = append(
				batchResult.Results[result.evaluatorName],
				result.result,
			)
		}
	}

	return batchResult
}

// AddEvaluator adds an evaluator to the runner.
func (r *BatchEvalRunner) AddEvaluator(name string, evaluator Evaluator) {
	r.evaluators[name] = evaluator
}

// RemoveEvaluator removes an evaluator from the runner.
func (r *BatchEvalRunner) RemoveEvaluator(name string) {
	delete(r.evaluators, name)
}

// Evaluators returns the evaluators.
func (r *BatchEvalRunner) Evaluators() map[string]Evaluator {
	return r.evaluators
}

// RunSingle runs a single evaluation across all evaluators.
func (r *BatchEvalRunner) RunSingle(ctx context.Context, input *EvaluateInput) (map[string]*EvaluationResult, error) {
	results := make(map[string]*EvaluationResult)

	for name, evaluator := range r.evaluators {
		result, err := evaluator.Evaluate(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("evaluator %s failed: %w", name, err)
		}
		results[name] = result
	}

	return results, nil
}

// CompareResults compares results from two batch evaluations.
func CompareResults(result1, result2 *BatchEvalResult) map[string]map[string]float64 {
	comparison := make(map[string]map[string]float64)

	for name := range result1.Results {
		if _, ok := result2.Results[name]; ok {
			comparison[name] = map[string]float64{
				"score_diff":        result1.GetAverageScore(name) - result2.GetAverageScore(name),
				"passing_rate_diff": result1.GetPassingRate(name) - result2.GetPassingRate(name),
			}
		}
	}

	return comparison
}
