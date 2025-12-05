package reader

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/aqua777/go-llamaindex/schema"
	"github.com/xuri/excelize/v2"
)

// ExcelReader reads Excel files (.xlsx, .xlsm) and converts them to documents.
type ExcelReader struct {
	// InputFiles is a list of Excel file paths to read
	InputFiles []string
	// InputDir is a directory containing Excel files
	InputDir string
	// Recursive determines if subdirectories should be searched
	Recursive bool
	// SheetNames specifies which sheets to read. If empty, all sheets are read.
	SheetNames []string
	// HasHeader indicates if the first row is a header row
	HasHeader bool
	// TextColumns are column names or indices to use as document text.
	// If empty, all columns are concatenated as text.
	TextColumns []string
	// MetadataColumns are column names or indices to extract as metadata.
	// If empty, all non-text columns are used as metadata.
	MetadataColumns []string
	// ConcatRows determines if all rows should be concatenated into a single document.
	// If false (default), each row becomes a separate document.
	ConcatRows bool
	// ConcatSheets determines if all sheets should be concatenated into a single document.
	// If false (default), each sheet is processed separately.
	ConcatSheets bool
	// RowSeparator is used when ConcatRows is true (default: newline)
	RowSeparator string
	// SheetSeparator is used when ConcatSheets is true (default: double newline)
	SheetSeparator string
}

// NewExcelReader creates a new ExcelReader for specific files.
func NewExcelReader(inputFiles ...string) *ExcelReader {
	return &ExcelReader{
		InputFiles:     inputFiles,
		HasHeader:      true,
		RowSeparator:   "\n",
		SheetSeparator: "\n\n",
	}
}

// NewExcelReaderFromDir creates a new ExcelReader for a directory.
func NewExcelReaderFromDir(inputDir string, recursive bool) *ExcelReader {
	return &ExcelReader{
		InputDir:       inputDir,
		Recursive:      recursive,
		HasHeader:      true,
		RowSeparator:   "\n",
		SheetSeparator: "\n\n",
	}
}

// WithSheets sets which sheets to read.
func (r *ExcelReader) WithSheets(sheets ...string) *ExcelReader {
	r.SheetNames = sheets
	return r
}

// WithHeader sets whether the first row is a header.
func (r *ExcelReader) WithHeader(hasHeader bool) *ExcelReader {
	r.HasHeader = hasHeader
	return r
}

// WithTextColumns sets which columns to use as document text.
func (r *ExcelReader) WithTextColumns(columns ...string) *ExcelReader {
	r.TextColumns = columns
	return r
}

// WithMetadataColumns sets which columns to extract as metadata.
func (r *ExcelReader) WithMetadataColumns(columns ...string) *ExcelReader {
	r.MetadataColumns = columns
	return r
}

// WithConcatRows sets whether to concatenate all rows into a single document.
func (r *ExcelReader) WithConcatRows(concat bool) *ExcelReader {
	r.ConcatRows = concat
	return r
}

// WithConcatSheets sets whether to concatenate all sheets into a single document.
func (r *ExcelReader) WithConcatSheets(concat bool) *ExcelReader {
	r.ConcatSheets = concat
	return r
}

// WithRowSeparator sets the separator used when concatenating rows.
func (r *ExcelReader) WithRowSeparator(sep string) *ExcelReader {
	r.RowSeparator = sep
	return r
}

// WithSheetSeparator sets the separator used when concatenating sheets.
func (r *ExcelReader) WithSheetSeparator(sep string) *ExcelReader {
	r.SheetSeparator = sep
	return r
}

// LoadData loads Excel files and returns documents.
func (r *ExcelReader) LoadData() ([]schema.Node, error) {
	files, err := r.getFiles()
	if err != nil {
		return nil, err
	}

	var docs []schema.Node
	for _, file := range files {
		fileDocs, err := r.loadFile(file)
		if err != nil {
			return nil, NewReaderError(file, "failed to load Excel file", err)
		}
		docs = append(docs, fileDocs...)
	}

	return docs, nil
}

// LoadFromFile loads a single Excel file.
func (r *ExcelReader) LoadFromFile(filePath string) ([]schema.Node, error) {
	return r.loadFile(filePath)
}

// Metadata returns reader metadata.
func (r *ExcelReader) Metadata() ReaderMetadata {
	return ReaderMetadata{
		Name:                "ExcelReader",
		SupportedExtensions: []string{".xlsx", ".xlsm", ".xltx", ".xltm"},
		Description:         "Reads Excel files (xlsx, xlsm)",
	}
}

func (r *ExcelReader) getFiles() ([]string, error) {
	if len(r.InputFiles) > 0 {
		return r.InputFiles, nil
	}

	if r.InputDir == "" {
		return nil, fmt.Errorf("no input files or directory specified")
	}

	var files []string
	err := filepath.Walk(r.InputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if !r.Recursive && path != r.InputDir {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".xlsx" || ext == ".xlsm" || ext == ".xltx" || ext == ".xltm" {
			files = append(files, path)
		}
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to walk directory %s: %w", r.InputDir, err)
	}

	return files, nil
}

func (r *ExcelReader) loadFile(filePath string) ([]schema.Node, error) {
	f, err := excelize.OpenFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open Excel file: %w", err)
	}
	defer f.Close()

	// Get sheets to process
	sheetList := f.GetSheetList()
	sheetsToProcess := sheetList
	if len(r.SheetNames) > 0 {
		sheetsToProcess = r.filterSheets(sheetList, r.SheetNames)
	}

	if len(sheetsToProcess) == 0 {
		return nil, nil
	}

	var allDocs []schema.Node

	if r.ConcatSheets {
		// Concatenate all sheets into one document
		var allSheetTexts []string
		for _, sheetName := range sheetsToProcess {
			sheetText, err := r.processSheetAsText(f, sheetName, filePath)
			if err != nil {
				return nil, err
			}
			if sheetText != "" {
				allSheetTexts = append(allSheetTexts, fmt.Sprintf("## %s\n%s", sheetName, sheetText))
			}
		}

		if len(allSheetTexts) > 0 {
			metadata := make(map[string]interface{})
			metadata["source"] = filePath
			metadata["filename"] = filepath.Base(filePath)
			metadata["sheets"] = sheetsToProcess
			metadata["total_sheets"] = len(sheetsToProcess)

			doc := schema.Node{
				ID:       filePath,
				Text:     strings.Join(allSheetTexts, r.SheetSeparator),
				Type:     schema.ObjectTypeDocument,
				Metadata: metadata,
			}
			allDocs = append(allDocs, doc)
		}
	} else {
		// Process each sheet separately
		for _, sheetName := range sheetsToProcess {
			sheetDocs, err := r.processSheet(f, sheetName, filePath)
			if err != nil {
				return nil, err
			}
			allDocs = append(allDocs, sheetDocs...)
		}
	}

	return allDocs, nil
}

func (r *ExcelReader) filterSheets(available, requested []string) []string {
	availableSet := make(map[string]bool)
	for _, s := range available {
		availableSet[strings.ToLower(s)] = true
	}

	var filtered []string
	for _, s := range requested {
		if availableSet[strings.ToLower(s)] {
			// Find the actual case from available
			for _, a := range available {
				if strings.EqualFold(a, s) {
					filtered = append(filtered, a)
					break
				}
			}
		}
	}
	return filtered
}

func (r *ExcelReader) processSheet(f *excelize.File, sheetName, filePath string) ([]schema.Node, error) {
	rows, err := f.GetRows(sheetName)
	if err != nil {
		return nil, fmt.Errorf("failed to read sheet %s: %w", sheetName, err)
	}

	if len(rows) == 0 {
		return nil, nil
	}

	// Determine headers
	var headers []string
	startRow := 0
	if r.HasHeader {
		headers = rows[0]
		startRow = 1
	} else {
		// Generate column indices as headers
		if len(rows) > 0 {
			headers = make([]string, len(rows[0]))
			for i := range headers {
				headers[i] = fmt.Sprintf("col_%d", i)
			}
		}
	}

	// Determine text and metadata column indices
	textIndices := r.getColumnIndices(headers, r.TextColumns)
	metadataIndices := r.getColumnIndices(headers, r.MetadataColumns)

	// If no text columns specified, use all columns
	if len(textIndices) == 0 {
		textIndices = make([]int, len(headers))
		for i := range headers {
			textIndices[i] = i
		}
	}

	// If no metadata columns specified and text columns are specified,
	// use remaining columns as metadata
	if len(metadataIndices) == 0 && len(r.TextColumns) > 0 {
		textSet := make(map[int]bool)
		for _, idx := range textIndices {
			textSet[idx] = true
		}
		for i := range headers {
			if !textSet[i] {
				metadataIndices = append(metadataIndices, i)
			}
		}
	}

	if r.ConcatRows {
		return r.createConcatenatedDocument(rows[startRow:], headers, textIndices, metadataIndices, filePath, sheetName)
	}

	return r.createRowDocuments(rows[startRow:], headers, textIndices, metadataIndices, filePath, sheetName)
}

func (r *ExcelReader) processSheetAsText(f *excelize.File, sheetName, filePath string) (string, error) {
	rows, err := f.GetRows(sheetName)
	if err != nil {
		return "", fmt.Errorf("failed to read sheet %s: %w", sheetName, err)
	}

	if len(rows) == 0 {
		return "", nil
	}

	var headers []string
	startRow := 0
	if r.HasHeader && len(rows) > 0 {
		headers = rows[0]
		startRow = 1
	}

	var allRows []string
	for _, row := range rows[startRow:] {
		var textParts []string
		for i, val := range row {
			val = strings.TrimSpace(val)
			if val != "" {
				if headers != nil && i < len(headers) && headers[i] != "" {
					textParts = append(textParts, fmt.Sprintf("%s: %s", headers[i], val))
				} else {
					textParts = append(textParts, val)
				}
			}
		}
		if len(textParts) > 0 {
			allRows = append(allRows, strings.Join(textParts, " | "))
		}
	}

	return strings.Join(allRows, r.RowSeparator), nil
}

func (r *ExcelReader) getColumnIndices(headers []string, columns []string) []int {
	if len(columns) == 0 {
		return nil
	}

	headerMap := make(map[string]int)
	for i, h := range headers {
		headerMap[strings.ToLower(strings.TrimSpace(h))] = i
	}

	var indices []int
	for _, col := range columns {
		col = strings.TrimSpace(col)
		// Try as column name first
		if idx, ok := headerMap[strings.ToLower(col)]; ok {
			indices = append(indices, idx)
			continue
		}
		// Try as Excel column letter (A, B, C, etc.)
		if colIdx := excelColToIndex(col); colIdx >= 0 && colIdx < len(headers) {
			indices = append(indices, colIdx)
			continue
		}
		// Try as numeric index
		var idx int
		if _, err := fmt.Sscanf(col, "%d", &idx); err == nil && idx >= 0 && idx < len(headers) {
			indices = append(indices, idx)
		}
	}

	return indices
}

// excelColToIndex converts Excel column letters (A, B, ..., Z, AA, AB, ...) to 0-based index.
func excelColToIndex(col string) int {
	col = strings.ToUpper(strings.TrimSpace(col))
	if col == "" {
		return -1
	}

	result := 0
	for _, c := range col {
		if c < 'A' || c > 'Z' {
			return -1
		}
		result = result*26 + int(c-'A') + 1
	}
	return result - 1
}

func (r *ExcelReader) createRowDocuments(rows [][]string, headers []string, textIndices, metadataIndices []int, filePath, sheetName string) ([]schema.Node, error) {
	var docs []schema.Node

	for rowIdx, row := range rows {
		// Build text from text columns
		var textParts []string
		for _, idx := range textIndices {
			if idx < len(row) {
				val := strings.TrimSpace(row[idx])
				if val != "" {
					if len(textIndices) > 1 && idx < len(headers) {
						textParts = append(textParts, fmt.Sprintf("%s: %s", headers[idx], val))
					} else {
						textParts = append(textParts, val)
					}
				}
			}
		}
		text := strings.Join(textParts, "\n")

		// Skip empty rows
		if text == "" {
			continue
		}

		// Build metadata
		metadata := make(map[string]interface{})
		metadata["source"] = filePath
		metadata["filename"] = filepath.Base(filePath)
		metadata["sheet"] = sheetName
		metadata["row"] = rowIdx + 1

		for _, idx := range metadataIndices {
			if idx < len(row) && idx < len(headers) {
				val := strings.TrimSpace(row[idx])
				if val != "" {
					metadata[headers[idx]] = val
				}
			}
		}

		doc := schema.Node{
			ID:       fmt.Sprintf("%s_%s_row_%d", filePath, sheetName, rowIdx),
			Text:     text,
			Type:     schema.ObjectTypeDocument,
			Metadata: metadata,
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

func (r *ExcelReader) createConcatenatedDocument(rows [][]string, headers []string, textIndices, metadataIndices []int, filePath, sheetName string) ([]schema.Node, error) {
	var allRows []string

	for _, row := range rows {
		var textParts []string
		for _, idx := range textIndices {
			if idx < len(row) {
				val := strings.TrimSpace(row[idx])
				if val != "" {
					if len(textIndices) > 1 && idx < len(headers) {
						textParts = append(textParts, fmt.Sprintf("%s: %s", headers[idx], val))
					} else {
						textParts = append(textParts, val)
					}
				}
			}
		}
		if len(textParts) > 0 {
			allRows = append(allRows, strings.Join(textParts, " | "))
		}
	}

	if len(allRows) == 0 {
		return nil, nil
	}

	text := strings.Join(allRows, r.RowSeparator)

	metadata := make(map[string]interface{})
	metadata["source"] = filePath
	metadata["filename"] = filepath.Base(filePath)
	metadata["sheet"] = sheetName
	metadata["total_rows"] = len(rows)
	metadata["headers"] = headers

	doc := schema.Node{
		ID:       fmt.Sprintf("%s_%s", filePath, sheetName),
		Text:     text,
		Type:     schema.ObjectTypeDocument,
		Metadata: metadata,
	}

	return []schema.Node{doc}, nil
}
