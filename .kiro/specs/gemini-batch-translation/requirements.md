# Requirements Document

## Introduction

The GeminiTranslator currently translates text segments one at a time, issuing a separate Gemini API call for each segment. For videos with many text regions this results in excessive API calls, slower throughput, and higher risk of hitting the free-tier RPM quota. This feature introduces batch translation: grouping up to 10 source texts into a single API request, parsing the ordered response back into individual Translation_Result objects, and gracefully handling partial failures within a batch.

## Glossary

- **Batch_Translator**: The component within GeminiTranslator responsible for assembling, sending, and parsing batch translation requests.
- **Batch**: An ordered collection of up to 10 source texts submitted in a single Gemini API call.
- **Batch_Prompt**: The prompt template used for batch requests, instructing the model to return a numbered list of translations.
- **Batch_Response_Parser**: The logic that splits a numbered-list response back into individual translated strings.
- **Translation_Result**: An immutable dataclass representing the outcome of translating one source string (defined in models.py).
- **RPM_Limiter**: The existing soft rate-per-minute throttle that prevents 429 errors from the Gemini free tier.
- **Cache**: The in-process dictionary keyed on normalized source text that stores previously obtained Translation_Result objects.
- **GeminiTranslator**: The existing class in translator_gemini.py that wraps the Gemini Developer API.
- **Text_Segment**: An immutable dataclass representing a contiguous run of frames where the same text appears.

## Requirements

### Requirement 1: Batch Assembly

**User Story:** As a pipeline operator, I want the translator to group pending texts into batches of up to 10, so that fewer API calls are made and throughput improves.

#### Acceptance Criteria

1. WHEN translate_segments is called with a sequence of Text_Segment objects, THE Batch_Translator SHALL partition the non-cached segments into batches of at most 10 items each and issue one API call per batch.
2. WHEN the number of non-cached segments is not evenly divisible by 10, THE Batch_Translator SHALL create a final batch containing the remaining segments (fewer than 10).
3. WHEN all segments in the input are already present in the Cache, THE Batch_Translator SHALL return cached results without issuing any API call.
4. THE Batch_Translator SHALL preserve the original input ordering of segments both within each batch and across batches, such that batch N contains only segments that appeared before those in batch N+1 in the input sequence.
5. WHEN translate_segments is called with an empty sequence, THE Batch_Translator SHALL return an empty result dictionary without issuing any API call.
6. WHEN translate_segments is called with a mix of cached and non-cached segments, THE Batch_Translator SHALL return a combined result dictionary containing both the cached Translation_Results and the freshly-translated Translation_Results, keyed by segment_id, covering every segment in the input.

### Requirement 2: Batch Prompt Construction

**User Story:** As a pipeline operator, I want the batch prompt to instruct the model to return translations as a numbered list, so that responses can be reliably parsed back into individual results.

#### Acceptance Criteria

1. THE Batch_Prompt SHALL instruct the Gemini model to return exactly one translation per line, prefixed with a sequential number and a period followed by a space matching the input order (e.g., "1. <translation>"), starting at 1 and incrementing by 1 for each item.
2. THE Batch_Prompt SHALL include all source texts in a numbered list within the prompt body, with each source text on its own line prefixed by its sequential number in the same "N. " format used for the expected output.
3. IF max_chars_target is greater than zero, THEN THE Batch_Prompt SHALL include the per-item character length constraint in the instructions specifying that each translated line must not exceed max_chars_target characters.
4. THE Batch_Prompt SHALL instruct the model to return only translations without annotations, quotes, or extra formatting beyond the numbered prefix.
5. THE Batch_Prompt SHALL accept a minimum of 1 and a maximum of 20 source texts per batch invocation.
6. IF the provided list of source texts is empty, THEN THE Batch_Prompt SHALL not be constructed and the system SHALL return an empty result set without calling the model.

### Requirement 3: Batch Response Parsing

**User Story:** As a pipeline operator, I want the batch response to be reliably split into individual translations, so that each segment receives its correct translated text.

#### Acceptance Criteria

1. WHEN the Gemini model returns a numbered-list response, THE Batch_Response_Parser SHALL extract individual translations by splitting on line boundaries and stripping the numeric prefix (matching the pattern "N. " where N is a positive integer).
2. WHEN the number of parsed translations equals the number of source texts in the batch, THE Batch_Response_Parser SHALL map each translation to its corresponding source text by position.
3. IF the number of parsed translations does not equal the number of source texts in the batch, THEN THE Batch_Response_Parser SHALL treat the entire batch as a partial failure and trigger the fallback mechanism.
4. THE Batch_Response_Parser SHALL strip leading and trailing whitespace from each extracted translation.
5. THE Batch_Response_Parser SHALL strip surrounding quotation marks (single or double) from each extracted translation if present.
6. THE Batch_Response_Parser SHALL skip empty lines in the response when parsing, treating them as formatting artifacts rather than empty translations.

### Requirement 4: Partial Failure Handling

**User Story:** As a pipeline operator, I want the system to handle partial batch failures gracefully, so that successfully translated texts are preserved and only failed texts are retried individually.

#### Acceptance Criteria

1. IF a batch API call raises an exception, THEN THE Batch_Translator SHALL fall back to translating each text in that batch individually using the existing single-text translate method.
2. IF the Batch_Response_Parser detects a count mismatch between expected and received translations, THEN THE Batch_Translator SHALL fall back to translating each text in that batch individually.
3. IF an individual translation within the parsed batch response is empty (zero-length string after stripping whitespace and quotes), THEN THE Batch_Translator SHALL retry that specific text using the single-text translate method.
4. WHEN falling back to individual translation, THE Batch_Translator SHALL log a warning indicating the batch index, the number of texts in the batch, and the reason for fallback.
5. WHEN the individual fallback translate method also fails for a text, THE Batch_Translator SHALL record that text with status "untranslated" and the error message from the last failed attempt.

### Requirement 5: Cache Integration

**User Story:** As a pipeline operator, I want batch translation to use and populate the existing cache, so that repeated texts are not re-translated.

#### Acceptance Criteria

1. WHEN assembling a batch, THE Batch_Translator SHALL skip any segment whose normalized source text (leading and trailing whitespace removed) already exists in the Cache with a status of "translated" or "passthrough".
2. IF a cached Translation_Result has status "untranslated", THEN THE Batch_Translator SHALL re-submit that segment for translation instead of returning the cached failure.
3. WHEN a batch translation succeeds, THE Batch_Translator SHALL store each Translation_Result in the Cache keyed by normalized source text.
4. THE Batch_Translator SHALL return cached Translation_Result objects for segments that were skipped during batch assembly, preserving the same Translation_Result field values as the originally cached entry.
5. WHEN multiple segments in the same batch share identical normalized source text, THE Batch_Translator SHALL translate that text at most once and return the same Translation_Result for all matching segments.

### Requirement 6: RPM Limiter Compatibility

**User Story:** As a pipeline operator, I want batch translation to respect the existing RPM limiter, so that the free-tier quota is not exceeded.

#### Acceptance Criteria

1. WHEN a batch API call is about to be made, THE Batch_Translator SHALL invoke the RPM_Limiter exactly once per batch call (not once per text within the batch), counting the entire batch as a single request against the configured RPM quota.
2. WHILE the RPM_Limiter determines that the rate limit would be exceeded, THE Batch_Translator SHALL wait until the limiter permits the call before proceeding.
3. THE Batch_Translator SHALL use a sliding window of 60 seconds to track request timestamps, permitting a new call only when fewer than the configured RPM limit requests have been recorded within that window.

### Requirement 7: Backward Compatibility

**User Story:** As a developer, I want the existing single-text translate method to remain unchanged, so that callers relying on it are not affected.

#### Acceptance Criteria

1. THE GeminiTranslator SHALL continue to expose the translate(text: str) -> Translation_Result method with the same parameter name, type annotation, and return type as the existing implementation.
2. WHEN translate(text) is called, THE GeminiTranslator SHALL produce identical Translation_Result values (same source_text, translated_text, and status fields) as the pre-batch implementation for the same input text and backend responses.
3. WHEN translate_segments is called, THE GeminiTranslator SHALL return a dict[str, Translation_Result] keyed by segment_id, containing one entry per input segment.

### Requirement 8: Batch Size Configuration

**User Story:** As a pipeline operator, I want to configure the batch size, so that I can tune performance based on model limits or quota constraints.

#### Acceptance Criteria

1. THE Gemini_Config SHALL include a batch_size field with a default value of 10.
2. WHEN batch_size is set to 1, THE Batch_Translator SHALL send exactly one text per API call, making one API call for each text segment to be translated.
3. THE Gemini_Config SHALL validate that batch_size is an integer in the range [1, 20].
4. IF batch_size is set to a value less than 1 or greater than 20, THEN THE Gemini_Config SHALL raise an InvalidConfigError indicating the accepted range [1, 20] and the rejected value.
5. IF the number of remaining text segments to translate is less than batch_size, THEN THE Batch_Translator SHALL send a single API call containing only the remaining segments without padding.
