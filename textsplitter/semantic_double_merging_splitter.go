package textsplitter

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// LanguageConfig configures the language and model name for SemanticDoubleMergingSplitter
// (aligned with LlamaIndex Python). The Go implementation does not load spaCy; the model
// field is validated for API compatibility.
type LanguageConfig struct {
	Language   string
	SpacyModel string
}

var supportedLanguageModels = map[string][]string{
	"english": {"en_core_web_md", "en_core_web_lg"},
	"german":  {"de_core_news_md", "de_core_news_lg"},
	"spanish": {"es_core_news_md", "es_core_news_lg"},
}

// ValidateLanguageConfig returns an error if the language or spaCy model is not supported.
func ValidateLanguageConfig(c LanguageConfig) error {
	lang := strings.ToLower(strings.TrimSpace(c.Language))
	models, ok := supportedLanguageModels[lang]
	if !ok {
		return fmt.Errorf("language %q is not supported", c.Language)
	}
	for _, m := range models {
		if m == c.SpacyModel {
			return nil
		}
	}
	return fmt.Errorf("spacy model %q does not match language %q", c.SpacyModel, c.Language)
}

// SemanticDoubleMergingSplitter splits text using a double merging semantic algorithm.
type SemanticDoubleMergingSplitter struct {
	LanguageConfig     LanguageConfig
	InitialThreshold   float64
	AppendingThreshold float64
	MergingThreshold   float64
	MaxChunkSize       int
	MergingRange       int
	MergingSeparator   string
	SentenceSplitter   SentenceSplitterStrategy
}

// NewSemanticDoubleMergingSplitter creates a new SemanticDoubleMergingSplitter.
//
// Args:
//
//	languageConfig: Configuration for the language and model.
//	initialThreshold: Sets threshold for initializing new chunk.
//	appendingThreshold: Sets threshold for appending new sentences to chunk.
//	mergingThreshold: Sets threshold for merging whole chunks.
//	maxChunkSize: Maximum size of chunk (in characters).
//	mergingRange: How many chunks ahead beyond the nearest neighbor to merge if similar (1 or 2).
//	mergingSeparator: The separator to use when merging chunks.
//	sentenceSplitter: Strategy to split text into sentences; if nil, RegexSplitterStrategy with DefaultChunkingRegex is used.
//
// Returns:
//
//	A pointer to the newly created SemanticDoubleMergingSplitter. mergingRange is clamped to [1, 2].
func NewSemanticDoubleMergingSplitter(
	languageConfig LanguageConfig,
	initialThreshold float64,
	appendingThreshold float64,
	mergingThreshold float64,
	maxChunkSize int,
	mergingRange int,
	mergingSeparator string,
	sentenceSplitter SentenceSplitterStrategy,
) *SemanticDoubleMergingSplitter {
	if sentenceSplitter == nil {
		sentenceSplitter = NewRegexSplitterStrategy(DefaultChunkingRegex)
	}
	if mergingRange < 1 {
		mergingRange = 1
	}
	if mergingRange > 2 {
		mergingRange = 2
	}
	return &SemanticDoubleMergingSplitter{
		LanguageConfig:     languageConfig,
		InitialThreshold:   initialThreshold,
		AppendingThreshold: appendingThreshold,
		MergingThreshold:   mergingThreshold,
		MaxChunkSize:       maxChunkSize,
		MergingRange:       mergingRange,
		MergingSeparator:   mergingSeparator,
		SentenceSplitter:   sentenceSplitter,
	}
}

// SplitText splits the text into chunks using the double merging algorithm.
//
// Args:
//
//	text: The text string to split.
//
// Returns:
//
//	Semantically grouped chunks. Empty input, no sentences after trimming, or invalid
//	LanguageConfig yields a non-nil empty slice.
func (s *SemanticDoubleMergingSplitter) SplitText(text string) []string {
	if strings.TrimSpace(text) == "" {
		return []string{}
	}
	if err := ValidateLanguageConfig(s.LanguageConfig); err != nil {
		return []string{}
	}
	sentences := filterNonEmptySentences(s.SentenceSplitter.Split(text))
	if len(sentences) == 0 {
		return []string{}
	}
	langKey := strings.ToLower(strings.TrimSpace(s.LanguageConfig.Language))
	h := jaccardSimilarity{stopwords: stopwordsForLanguage(langKey)}
	maxSize := s.MaxChunkSize
	if maxSize < 0 {
		maxSize = 0
	}
	sep := s.MergingSeparator
	initial := createInitialChunks(sentences, s.InitialThreshold, s.AppendingThreshold, maxSize, sep, h.compare)
	merged := mergeInitialChunks(initial, s.MergingThreshold, maxSize, s.MergingRange, sep, h.compare)
	return merged
}

var urlPattern = regexp.MustCompile(`https?://\S+|www\.\S+`)

// cleanTextAdvanced lowercases, strips URLs and punctuation, tokenizes, and removes stopwords
// (aligned with LlamaIndex Python _clean_text_advanced).
func cleanTextAdvanced(text string, stopwords map[string]struct{}) string {
	text = strings.ToLower(text)
	text = urlPattern.ReplaceAllString(text, "")
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			b.WriteRune(r)
		} else {
			b.WriteByte(' ')
		}
	}
	fields := strings.Fields(b.String())
	var out []string
	for _, w := range fields {
		if _, skip := stopwords[w]; !skip {
			out = append(out, w)
		}
	}
	return strings.Join(out, " ")
}

// jaccardSimilarityOnTokens returns |A∩B|/|A∪B| for word sets derived from cleaned strings.
func jaccardSimilarityOnTokens(cleanA, cleanB string) float64 {
	tokA := strings.Fields(cleanA)
	tokB := strings.Fields(cleanB)
	setA := make(map[string]struct{}, len(tokA))
	for _, t := range tokA {
		setA[t] = struct{}{}
	}
	setB := make(map[string]struct{}, len(tokB))
	for _, t := range tokB {
		setB[t] = struct{}{}
	}
	if len(setA) == 0 && len(setB) == 0 {
		return 1
	}
	if len(setA) == 0 || len(setB) == 0 {
		return 0
	}
	inter := 0
	for t := range setA {
		if _, ok := setB[t]; ok {
			inter++
		}
	}
	union := len(setA) + len(setB) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

func similarityBetweenStrings(a, b string, stopwords map[string]struct{}) float64 {
	ca := cleanTextAdvanced(a, stopwords)
	cb := cleanTextAdvanced(b, stopwords)
	return jaccardSimilarityOnTokens(ca, cb)
}

// jaccardSimilarity holds stopwords for Jaccard-based pairwise string similarity in SplitText.
type jaccardSimilarity struct {
	stopwords map[string]struct{}
}

func (h jaccardSimilarity) compare(a, b string) float64 {
	return similarityBetweenStrings(a, b, h.stopwords)
}

// stopwordsForLanguage returns Jaccard similarity cleaning stopwords for LanguageConfig.Language
// (english, german, spanish). Unknown languages fall back to English.
func stopwordsForLanguage(lang string) map[string]struct{} {
	switch lang {
	case "german":
		return germanStopwordSet()
	case "spanish":
		return spanishStopwordSet()
	default:
		return englishStopwordSet()
	}
}

func stringSliceToStopwordSet(words []string) map[string]struct{} {
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

// germanStopwordSet returns a standard German stopword set for Jaccard similarity cleaning.
func germanStopwordSet() map[string]struct{} {
	words := []string{
		"aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "das", "dass", "dein", "deine", "dem", "den", "der", "des", "dich", "die", "dies", "dir", "doch", "dort", "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "einmal", "er", "es", "euch", "euer", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "ihm", "ihn", "ihr", "ihre", "im", "in", "ist", "ja", "jede", "jedem", "jeden", "jeder", "jedes", "jetzt", "kann", "kein", "keine", "können", "könnte", "machen", "man", "mich", "mir", "mit", "muss", "musste", "nach", "nicht", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "sich", "sie", "sind", "so", "solche", "soll", "sollte", "sondern", "über", "um", "und", "uns", "unser", "unter", "viel", "vom", "von", "vor", "war", "waren", "warst", "was", "weg", "weil", "weiter", "welche", "wenn", "werde", "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "würde", "zu", "zum", "zur", "zwischen",
	}
	return stringSliceToStopwordSet(words)
}

// spanishStopwordSet returns a standard Spanish stopword set for Jaccard similarity cleaning.
func spanishStopwordSet() map[string]struct{} {
	words := []string{
		"a", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "ambos", "ante", "antes", "como", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella", "ellas", "ello", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban", "estabas", "estad", "estada", "estadas", "estado", "estados", "estamos", "estando", "estar", "estaremos", "estaré", "estaréis", "estarán", "estarás", "estaría", "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "esto", "estos", "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese", "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis", "estuvo", "está", "estábamos", "estáis", "están", "estás", "esté", "estéis", "estén", "estés", "ha", "haber", "habida", "habidas", "habido", "habidos", "habiendo", "habremos", "habré", "habréis", "habrán", "habrás", "habría", "habríais", "habríamos", "habrían", "habrías", "habéis", "había", "habíais", "habíamos", "habían", "habías", "han", "has", "hasta", "hay", "haya", "hayamos", "hayáis", "hayan", "hayas", "he", "hemos", "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis", "hubo", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "nada", "ni", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que", "qué", "quien", "quienes", "se", "sea", "seamos", "sean", "seas", "ser", "seremos", "será", "serán", "serás", "seré", "seréis", "sería", "seríais", "seríamos", "serían", "serías", "seáis", "si", "sido", "siendo", "siente", "sin", "sintiendo", "sobre", "sois", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "sí", "también", "tanto", "te", "tendremos", "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais", "tendríamos", "tendrían", "tendrías", "tened", "tenemos", "tener", "tenga", "tengamos", "tengan", "tengas", "tengo", "tenida", "tenidas", "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais", "teníamos", "tenían", "tenías", "ti", "tiene", "tienen", "tienes", "todo", "todos", "tu", "tus", "tuve", "tuviera", "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "un", "una", "uno", "unos", "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros", "y", "ya", "yo", "él", "éramos",
	}
	return stringSliceToStopwordSet(words)
}

// englishStopwordSet returns a standard English stopword set for similarity cleaning.
func englishStopwordSet() map[string]struct{} {
	words := []string{
		"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
		"be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
		"could", "did", "do", "does", "doing", "down", "during",
		"each", "few", "for", "from", "further",
		"had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
		"i", "if", "in", "into", "is", "it", "its", "itself",
		"me", "more", "most", "my", "myself",
		"no", "nor", "not",
		"of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
		"same", "she", "should", "so", "some", "such",
		"than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
		"under", "until", "up",
		"very",
		"was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
		"you", "your", "yours", "yourself", "yourselves",
	}
	return stringSliceToStopwordSet(words)
}

func joinLastTwoSentences(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return parts[len(parts)-2] + sep + parts[len(parts)-1]
}

func chunkLenIfJoined(a, b, sep string) int {
	return len(a) + len(sep) + len(b)
}

// createInitialChunks builds the first pass of chunks (LlamaIndex _create_initial_chunks).
func createInitialChunks(
	sentences []string,
	initialThreshold, appendingThreshold float64,
	maxChunkSize int,
	sep string,
	similarity func(a, b string) float64,
) []string {
	if len(sentences) == 0 {
		return nil
	}
	if len(sentences) == 1 {
		return []string{sentences[0]}
	}
	var initial []string
	chunk := sentences[0]
	startNew := true
	var chunkSentences []string
	var lastSentences string

	for _, sentence := range sentences[1:] {
		if startNew {
			if similarity(chunk, sentence) < initialThreshold &&
				chunkLenIfJoined(chunk, sentence, sep) <= maxChunkSize {
				initial = append(initial, chunk)
				chunk = sentence
				continue
			}
			chunkSentences = []string{chunk}
			if chunkLenIfJoined(chunk, sentence, sep) <= maxChunkSize {
				chunkSentences = append(chunkSentences, sentence)
				chunk = strings.Join(chunkSentences, sep)
				startNew = false
				lastSentences = joinLastTwoSentences(chunkSentences, sep)
			} else {
				initial = append(initial, chunk)
				chunk = sentence
				continue
			}
		} else if similarity(lastSentences, sentence) > appendingThreshold &&
			chunkLenIfJoined(chunk, sentence, sep) <= maxChunkSize {
			chunkSentences = append(chunkSentences, sentence)
			lastSentences = joinLastTwoSentences(chunkSentences, sep)
			chunk += sep + sentence
		} else {
			initial = append(initial, chunk)
			chunk = sentence
			startNew = true
		}
	}
	initial = append(initial, chunk)
	return initial
}

// mergeInitialChunks runs the second pass (LlamaIndex _merge_initial_chunks).
func mergeInitialChunks(
	initialChunks []string,
	mergingThreshold float64,
	maxChunkSize int,
	mergingRange int,
	sep string,
	similarity func(a, b string) float64,
) []string {
	if len(initialChunks) == 0 {
		return nil
	}
	if len(initialChunks) == 1 {
		return []string{initialChunks[0]}
	}
	var chunks []string
	skip := 0
	current := initialChunks[0]

	for i := 1; i < len(initialChunks); i++ {
		if skip > 0 {
			skip--
			continue
		}
		if len(current) >= maxChunkSize {
			chunks = append(chunks, current)
			current = initialChunks[i]
			continue
		}

		if similarity(current, initialChunks[i]) > mergingThreshold &&
			chunkLenIfJoined(current, initialChunks[i], sep) <= maxChunkSize {
			current += sep + initialChunks[i]
			continue
		}

		if i <= len(initialChunks)-2 &&
			similarity(current, initialChunks[i+1]) > mergingThreshold &&
			len(current)+len(sep)+len(initialChunks[i])+len(sep)+len(initialChunks[i+1]) <= maxChunkSize {
			current += sep + initialChunks[i] + sep + initialChunks[i+1]
			skip = 1
			continue
		}

		if mergingRange == 2 && i < len(initialChunks)-2 &&
			similarity(current, initialChunks[i+2]) > mergingThreshold &&
			len(current)+len(sep)+len(initialChunks[i])+len(sep)+len(initialChunks[i+1])+len(sep)+len(initialChunks[i+2]) <= maxChunkSize {
			current += sep + initialChunks[i] + sep + initialChunks[i+1] + sep + initialChunks[i+2]
			skip = 2
			continue
		}

		chunks = append(chunks, current)
		current = initialChunks[i]
	}
	chunks = append(chunks, current)
	return chunks
}
