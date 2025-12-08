package validation

import (
	"testing"
)

func TestValidateChunkParams(t *testing.T) {
	tests := []struct {
		name        string
		chunkSize   int
		chunkOverlap int
		wantErr     bool
	}{
		{
			name:        "valid params",
			chunkSize:   1024,
			chunkOverlap: 200,
			wantErr:     false,
		},
		{
			name:        "zero overlap is valid",
			chunkSize:   1024,
			chunkOverlap: 0,
			wantErr:     false,
		},
		{
			name:        "chunk size zero",
			chunkSize:   0,
			chunkOverlap: 200,
			wantErr:     true,
		},
		{
			name:        "chunk size negative",
			chunkSize:   -1,
			chunkOverlap: 200,
			wantErr:     true,
		},
		{
			name:        "overlap negative",
			chunkSize:   1024,
			chunkOverlap: -1,
			wantErr:     true,
		},
		{
			name:        "overlap equals chunk size",
			chunkSize:   1024,
			chunkOverlap: 1024,
			wantErr:     true,
		},
		{
			name:        "overlap greater than chunk size",
			chunkSize:   1024,
			chunkOverlap: 2000,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateChunkParams(tt.chunkSize, tt.chunkOverlap)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateChunkParams() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidator(t *testing.T) {
	t.Run("no errors", func(t *testing.T) {
		v := NewValidator()
		v.RequirePositive(10, "field")
		v.RequireNotEmpty("value", "field")
		
		if v.HasErrors() {
			t.Error("expected no errors")
		}
		if v.Error() != nil {
			t.Error("expected nil error")
		}
	})

	t.Run("with errors", func(t *testing.T) {
		v := NewValidator()
		v.RequirePositive(-1, "field1")
		v.RequireNotEmpty("", "field2")
		
		if !v.HasErrors() {
			t.Error("expected errors")
		}
		if v.Error() == nil {
			t.Error("expected non-nil error")
		}
		if len(v.Errors()) != 2 {
			t.Errorf("expected 2 errors, got %d", len(v.Errors()))
		}
	})

	t.Run("RequireLessThan", func(t *testing.T) {
		v := NewValidator()
		v.RequireLessThan(5, 10, "a", "b")
		if v.HasErrors() {
			t.Error("5 < 10 should pass")
		}

		v2 := NewValidator()
		v2.RequireLessThan(10, 5, "a", "b")
		if !v2.HasErrors() {
			t.Error("10 < 5 should fail")
		}
	})
}

func TestValidateSentenceSplitterConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     SentenceSplitterConfig
		wantErr bool
	}{
		{
			name: "valid config",
			cfg: SentenceSplitterConfig{
				ChunkSize:    1024,
				ChunkOverlap: 200,
				Separator:    " ",
			},
			wantErr: false,
		},
		{
			name: "invalid chunk size",
			cfg: SentenceSplitterConfig{
				ChunkSize:    0,
				ChunkOverlap: 200,
			},
			wantErr: true,
		},
		{
			name: "overlap too large",
			cfg: SentenceSplitterConfig{
				ChunkSize:    100,
				ChunkOverlap: 200,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSentenceSplitterConfig(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateSentenceSplitterConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateMetadataAwareSplit(t *testing.T) {
	tests := []struct {
		name           string
		chunkSize      int
		metadataLength int
		wantErr        bool
	}{
		{
			name:           "valid - plenty of room",
			chunkSize:      1024,
			metadataLength: 100,
			wantErr:        false,
		},
		{
			name:           "valid - small effective size",
			chunkSize:      100,
			metadataLength: 40,
			wantErr:        false,
		},
		{
			name:           "invalid - metadata equals chunk size",
			chunkSize:      100,
			metadataLength: 100,
			wantErr:        true,
		},
		{
			name:           "invalid - metadata exceeds chunk size",
			chunkSize:      100,
			metadataLength: 200,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateMetadataAwareSplit(MetadataAwareSplitConfig{
				ChunkSize:      tt.chunkSize,
				MetadataLength: tt.metadataLength,
			})
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateMetadataAwareSplit() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetEffectiveChunkSize(t *testing.T) {
	tests := []struct {
		name           string
		chunkSize      int
		metadataLength int
		want           int
		wantErr        bool
	}{
		{
			name:           "normal case",
			chunkSize:      1024,
			metadataLength: 100,
			want:           924,
			wantErr:        false,
		},
		{
			name:           "zero metadata",
			chunkSize:      1024,
			metadataLength: 0,
			want:           1024,
			wantErr:        false,
		},
		{
			name:           "metadata equals chunk size",
			chunkSize:      100,
			metadataLength: 100,
			want:           0,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetEffectiveChunkSize(tt.chunkSize, tt.metadataLength)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetEffectiveChunkSize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("GetEffectiveChunkSize() = %v, want %v", got, tt.want)
			}
		})
	}
}
