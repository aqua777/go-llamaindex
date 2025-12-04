package reader

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCSVReader_LoadData(t *testing.T) {
	// Create temp CSV file
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `name,age,city
Alice,30,New York
Bob,25,Los Angeles
Charlie,35,Chicago`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(csvPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 3)

	// Check first document
	assert.Contains(t, docs[0].Text, "name: Alice")
	assert.Contains(t, docs[0].Text, "age: 30")
	assert.Contains(t, docs[0].Text, "city: New York")
	assert.Equal(t, csvPath, docs[0].Metadata["source"])
	assert.Equal(t, 1, docs[0].Metadata["row"])

	// Check second document
	assert.Contains(t, docs[1].Text, "name: Bob")
	assert.Equal(t, 2, docs[1].Metadata["row"])
}

func TestCSVReader_WithTextColumns(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `id,content,category
1,Hello world,greeting
2,Goodbye world,farewell`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(csvPath).
		WithTextColumns("content").
		WithMetadataColumns("id", "category")

	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	// Text should only contain content column
	assert.Equal(t, "Hello world", docs[0].Text)
	assert.Equal(t, "1", docs[0].Metadata["id"])
	assert.Equal(t, "greeting", docs[0].Metadata["category"])
}

func TestCSVReader_WithConcatRows(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `name,value
item1,100
item2,200`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(csvPath).WithConcatRows(true)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	// All rows concatenated
	assert.Contains(t, docs[0].Text, "item1")
	assert.Contains(t, docs[0].Text, "item2")
	assert.Equal(t, 2, docs[0].Metadata["total_rows"])
}

func TestCSVReader_TSVFile(t *testing.T) {
	tmpDir := t.TempDir()
	tsvPath := filepath.Join(tmpDir, "test.tsv")
	tsvContent := "name\tage\nAlice\t30\nBob\t25"

	err := os.WriteFile(tsvPath, []byte(tsvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(tsvPath)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	assert.Contains(t, docs[0].Text, "Alice")
	assert.Contains(t, docs[0].Text, "30")
}

func TestCSVReader_NoHeader(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `Alice,30,New York
Bob,25,Los Angeles`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(csvPath).WithHeader(false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)

	// Should use col_0, col_1, etc. as headers
	assert.Contains(t, docs[0].Text, "col_0: Alice")
	assert.Contains(t, docs[0].Text, "col_1: 30")
}

func TestCSVReader_FromDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	// Create multiple CSV files
	csv1 := filepath.Join(tmpDir, "file1.csv")
	csv2 := filepath.Join(tmpDir, "file2.csv")

	err := os.WriteFile(csv1, []byte("name\nAlice"), 0644)
	require.NoError(t, err)
	err = os.WriteFile(csv2, []byte("name\nBob"), 0644)
	require.NoError(t, err)

	reader := NewCSVReaderFromDir(tmpDir, false)
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 2)
}

func TestCSVReader_CustomDelimiter(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := "name;age\nAlice;30"

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader := NewCSVReader(csvPath).WithDelimiter(';')
	docs, err := reader.LoadData()
	require.NoError(t, err)
	assert.Len(t, docs, 1)

	assert.Contains(t, docs[0].Text, "Alice")
	assert.Contains(t, docs[0].Text, "30")
}

func TestCSVReader_Metadata(t *testing.T) {
	reader := NewCSVReader("test.csv")
	meta := reader.Metadata()

	assert.Equal(t, "CSVReader", meta.Name)
	assert.Contains(t, meta.SupportedExtensions, ".csv")
	assert.Contains(t, meta.SupportedExtensions, ".tsv")
}

func TestCSVStreamReader(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `name,age
Alice,30
Bob,25
Charlie,35`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader, err := NewCSVStreamReader(csvPath)
	require.NoError(t, err)
	defer reader.Close()

	// Read headers
	headers, err := reader.ReadHeaders()
	require.NoError(t, err)
	assert.Equal(t, []string{"name", "age"}, headers)

	// Read rows one by one
	doc1, err := reader.ReadNext()
	require.NoError(t, err)
	require.NotNil(t, doc1)
	assert.Contains(t, doc1.Text, "Alice")

	doc2, err := reader.ReadNext()
	require.NoError(t, err)
	require.NotNil(t, doc2)
	assert.Contains(t, doc2.Text, "Bob")

	doc3, err := reader.ReadNext()
	require.NoError(t, err)
	require.NotNil(t, doc3)
	assert.Contains(t, doc3.Text, "Charlie")

	// EOF
	doc4, err := reader.ReadNext()
	require.NoError(t, err)
	assert.Nil(t, doc4)
}

func TestCSVStreamReader_LazyLoad(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	csvContent := `name
Alice
Bob`

	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)

	reader, err := NewCSVStreamReader(csvPath)
	require.NoError(t, err)

	docChan, errChan := reader.LazyLoadData()

	var docs []string
	for doc := range docChan {
		docs = append(docs, doc.Text)
	}

	// Check for errors
	select {
	case err := <-errChan:
		require.NoError(t, err)
	default:
	}

	assert.Len(t, docs, 2)
}
