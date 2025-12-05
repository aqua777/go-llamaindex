package reader

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/xuri/excelize/v2"
)

func createTestExcelFile(t *testing.T, path string, sheets map[string][][]string) {
	f := excelize.NewFile()
	defer f.Close()

	first := true
	for sheetName, rows := range sheets {
		if first {
			// Rename the default sheet
			f.SetSheetName("Sheet1", sheetName)
			first = false
		} else {
			f.NewSheet(sheetName)
		}

		for rowIdx, row := range rows {
			for colIdx, val := range row {
				cell, _ := excelize.CoordinatesToCellName(colIdx+1, rowIdx+1)
				f.SetCellValue(sheetName, cell, val)
			}
		}
	}

	err := f.SaveAs(path)
	require.NoError(t, err)
}

func TestExcelReader_LoadData(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"name", "age", "city"},
			{"Alice", "30", "New York"},
			{"Bob", "25", "Los Angeles"},
		},
	})

	reader := NewExcelReader(xlsxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	// Check first document
	assert.Contains(t, docs[0].Text, "name: Alice")
	assert.Contains(t, docs[0].Text, "age: 30")
	assert.Contains(t, docs[0].Text, "city: New York")
	assert.Equal(t, xlsxPath, docs[0].Metadata["source"])
	assert.Equal(t, "Sheet1", docs[0].Metadata["sheet"])
	assert.Equal(t, 1, docs[0].Metadata["row"])
}

func TestExcelReader_MultipleSheets(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Users": {
			{"name"},
			{"Alice"},
		},
		"Products": {
			{"product"},
			{"Widget"},
		},
	})

	reader := NewExcelReader(xlsxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	// Should have docs from both sheets
	sheets := make(map[string]bool)
	for _, doc := range docs {
		sheets[doc.Metadata["sheet"].(string)] = true
	}
	assert.True(t, sheets["Users"])
	assert.True(t, sheets["Products"])
}

func TestExcelReader_WithSheets(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Users": {
			{"name"},
			{"Alice"},
		},
		"Products": {
			{"product"},
			{"Widget"},
		},
	})

	reader := NewExcelReader(xlsxPath).WithSheets("Users")
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)
	assert.Equal(t, "Users", docs[0].Metadata["sheet"])
}

func TestExcelReader_WithTextColumns(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"id", "content", "category"},
			{"1", "Hello world", "greeting"},
		},
	})

	reader := NewExcelReader(xlsxPath).
		WithTextColumns("content").
		WithMetadataColumns("id", "category")

	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	assert.Equal(t, "Hello world", docs[0].Text)
	assert.Equal(t, "1", docs[0].Metadata["id"])
	assert.Equal(t, "greeting", docs[0].Metadata["category"])
}

func TestExcelReader_WithConcatRows(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"name", "value"},
			{"item1", "100"},
			{"item2", "200"},
		},
	})

	reader := NewExcelReader(xlsxPath).WithConcatRows(true)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	assert.Contains(t, docs[0].Text, "item1")
	assert.Contains(t, docs[0].Text, "item2")
	assert.Equal(t, 2, docs[0].Metadata["total_rows"])
}

func TestExcelReader_WithConcatSheets(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Users": {
			{"name"},
			{"Alice"},
		},
		"Products": {
			{"product"},
			{"Widget"},
		},
	})

	reader := NewExcelReader(xlsxPath).WithConcatSheets(true)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// Should contain content from both sheets
	assert.Contains(t, docs[0].Text, "Alice")
	assert.Contains(t, docs[0].Text, "Widget")
	assert.Equal(t, 2, docs[0].Metadata["total_sheets"])
}

func TestExcelReader_NoHeader(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"Alice", "30"},
			{"Bob", "25"},
		},
	})

	reader := NewExcelReader(xlsxPath).WithHeader(false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	// Should use col_0, col_1 as headers
	assert.Contains(t, docs[0].Text, "col_0: Alice")
	assert.Contains(t, docs[0].Text, "col_1: 30")
}

func TestExcelReader_FromDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	xlsx1 := filepath.Join(tmpDir, "file1.xlsx")
	xlsx2 := filepath.Join(tmpDir, "file2.xlsx")

	createTestExcelFile(t, xlsx1, map[string][][]string{
		"Sheet1": {{"name"}, {"Alice"}},
	})
	createTestExcelFile(t, xlsx2, map[string][][]string{
		"Sheet1": {{"name"}, {"Bob"}},
	})

	reader := NewExcelReaderFromDir(tmpDir, false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)
}

func TestExcelReader_Metadata(t *testing.T) {
	reader := NewExcelReader("test.xlsx")
	meta := reader.Metadata()

	assert.Equal(t, "ExcelReader", meta.Name)
	assert.Contains(t, meta.SupportedExtensions, ".xlsx")
	assert.Contains(t, meta.SupportedExtensions, ".xlsm")
}

func TestExcelColToIndex(t *testing.T) {
	tests := []struct {
		col      string
		expected int
	}{
		{"A", 0},
		{"B", 1},
		{"Z", 25},
		{"AA", 26},
		{"AB", 27},
		{"AZ", 51},
		{"BA", 52},
		{"", -1},
		{"1", -1},
	}

	for _, tt := range tests {
		t.Run(tt.col, func(t *testing.T) {
			result := excelColToIndex(tt.col)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExcelReader_ColumnByLetter(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"id", "content", "category"},
			{"1", "Hello", "test"},
		},
	})

	// Use Excel column letter B for content
	reader := NewExcelReader(xlsxPath).WithTextColumns("B")
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)
	assert.Equal(t, "Hello", docs[0].Text)
}

func TestExcelReader_EmptySheet(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	f := excelize.NewFile()
	f.SaveAs(xlsxPath)
	f.Close()

	reader := NewExcelReader(xlsxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 0)
}

func TestExcelReader_SkipsEmptyRows(t *testing.T) {
	tmpDir := t.TempDir()
	xlsxPath := filepath.Join(tmpDir, "test.xlsx")

	createTestExcelFile(t, xlsxPath, map[string][][]string{
		"Sheet1": {
			{"name"},
			{"Alice"},
			{""},
			{"Bob"},
		},
	})

	reader := NewExcelReader(xlsxPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	// Should skip the empty row
	assert.Len(t, docs, 2)
}

func TestExcelReader_FileNotFound(t *testing.T) {
	reader := NewExcelReader("/nonexistent/file.xlsx")
	_, err := reader.LoadData()
	assert.Error(t, err)
}

func TestExcelReader_InvalidDirectory(t *testing.T) {
	reader := NewExcelReaderFromDir("/nonexistent/dir", false)
	_, err := reader.LoadData()
	assert.Error(t, err)
}

func TestExcelReader_NoInputSpecified(t *testing.T) {
	reader := &ExcelReader{}
	_, err := reader.LoadData()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no input files or directory specified")
}
