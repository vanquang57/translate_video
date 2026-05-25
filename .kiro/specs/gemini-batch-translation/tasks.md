# Implementation Plan: Gemini Batch Translation

## Overview

Implement batch translation in the existing `GeminiTranslator` class to send up to N texts (configurable, default 10, max 20) in a single Gemini API call. The implementation adds a `batch_size` config field, a batch prompt template, a response parser, and rewrites `translate_segments` to use batch logic with cache deduplication and individual fallback on failure.

## Tasks

- [x] 1. Add batch_size configuration
  - [x] 1.1 Add `batch_size` field to `Gemini_Config` in `src/video_text_translator/models.py`
    - Add `batch_size: int = 10` field after `timeout_seconds`
    - Add validation in `__post_init__`: raise `ValueError` if not in [1, 20]
    - _Requirements: 8.1, 8.3, 8.4_

  - [x] 1.2 Update `configs/default.yaml` to include `batch_size` option
    - Add `batch_size: 10` under the `translator.gemini` section with a comment showing the valid range [1, 20]
    - _Requirements: 8.1_

- [x] 2. Implement batch prompt construction
  - [x] 2.1 Add `_BATCH_PROMPT_TEMPLATE` constant to `src/video_text_translator/translator_gemini.py`
    - Define the template string with placeholders `{length_constraint}` and `{numbered_sources}`
    - Template instructs the model to return numbered translations, one per line, no annotations or quotes
    - _Requirements: 2.1, 2.4_

  - [x] 2.2 Implement `_build_batch_prompt` method on `GeminiTranslator`
    - Signature: `_build_batch_prompt(self, texts: list[str]) -> str`
    - Format source texts as numbered list (`1. text`, `2. text`, ...)
    - Include length constraint line if `self._config.max_chars_target > 0`
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 3. Implement batch response parsing
  - [x] 3.1 Implement `_parse_batch_response` method on `GeminiTranslator`
    - Signature: `_parse_batch_response(self, response: str, expected_count: int) -> list[str] | None`
    - Split response on line boundaries, skip empty lines
    - Strip numeric prefix matching pattern `N. ` (positive integer followed by dot and space)
    - Strip leading/trailing whitespace and surrounding quotes from each translation
    - Return `None` if parsed count does not equal `expected_count`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 4. Implement batch translation orchestration
  - [x] 4.1 Implement `_translate_batch` method on `GeminiTranslator`
    - Signature: `_translate_batch(self, texts: list[str]) -> list[Translation_Result]`
    - Call `_respect_rpm()` once for the batch
    - Build batch prompt via `_build_batch_prompt`
    - Call Gemini API with the batch prompt
    - Parse response via `_parse_batch_response`
    - On success: create `Translation_Result` for each text with status "translated"
    - On parse failure (None returned) or API exception: fall back to individual `translate()` for each text
    - For empty parsed items: retry those specific items via individual `translate()`
    - Log warnings on fallback with batch index, count, and reason
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.1_

- [x] 5. Rewrite translate_segments to use batch logic
  - [x] 5.1 Rewrite `translate_segments` method on `GeminiTranslator`
    - Step 1: Cache lookup — for each segment, check if normalized canonical_text is in cache with status "translated" or "passthrough"; if so, use cached result
    - Step 2: Re-include segments with cached status "untranslated" for re-translation
    - Step 3: Deduplicate — collect unique normalized texts that need translation
    - Step 4: Partition unique texts into batches of `self._config.batch_size`
    - Step 5: Call `_translate_batch` for each partition
    - Step 6: Store results in cache keyed by normalized text
    - Step 7: Merge — assemble output dict keyed by segment_id, mapping each segment to its Translation_Result (from cache or fresh translation)
    - Handle empty input by returning empty dict immediately
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 7.3, 8.2, 8.5_

- [x] 6. Checkpoint - Verify core implementation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Property-based tests
  - [ ]* 7.1 Write property test for partitioning (Property 1)
    - **Property 1: Partitioning preserves order and respects batch_size**
    - Generate lists of segments and batch_size in [1, 20]; verify ceil(N/B) batches, each ≤ B items, no empty batches, concatenation equals original
    - **Validates: Requirements 1.1, 1.2, 1.4, 8.5**

  - [ ]* 7.2 Write property test for all-cached input (Property 2)
    - **Property 2: All-cached input produces zero API calls**
    - Pre-populate cache with all segment texts; verify no API call is made
    - **Validates: Requirements 1.3**

  - [ ]* 7.3 Write property test for output completeness (Property 3)
    - **Property 3: Output completeness**
    - Generate mixed cached/non-cached/duplicate segments; verify output dict has exactly len(segments) entries
    - **Validates: Requirements 1.6, 5.4, 7.3**

  - [ ]* 7.4 Write property test for prompt/parse round-trip (Property 4)
    - **Property 4: Batch prompt/parse round-trip**
    - Generate 1-20 non-empty strings without newlines; format as numbered list and parse back; verify identity
    - **Validates: Requirements 2.1, 2.2, 3.1, 3.2**

  - [ ]* 7.5 Write property test for length constraint inclusion (Property 5)
    - **Property 5: Length constraint conditional inclusion**
    - Generate max_chars_target values; verify constraint text present iff value > 0
    - **Validates: Requirements 2.3**

  - [ ]* 7.6 Write property test for parser robustness (Property 6)
    - **Property 6: Parser robustness against whitespace, quotes, and empty lines**
    - Generate translations with arbitrary whitespace, quotes, empty lines; verify same cleaned output
    - **Validates: Requirements 3.4, 3.5, 3.6**

  - [ ]* 7.7 Write property test for count mismatch detection (Property 7)
    - **Property 7: Count mismatch detection**
    - Generate responses with M ≠ N lines; verify parser returns None
    - **Validates: Requirements 3.3**

  - [ ]* 7.8 Write property test for batch failure fallback (Property 8)
    - **Property 8: Batch failure triggers individual fallback**
    - Mock API to raise exception or return mismatched count; verify individual translate called for each text
    - **Validates: Requirements 4.1, 4.2**

  - [ ]* 7.9 Write property test for empty parsed items retry (Property 9)
    - **Property 9: Empty parsed items trigger per-item retry**
    - Generate responses with K empty items; verify exactly K items retried individually
    - **Validates: Requirements 4.3**

  - [ ]* 7.10 Write property test for cache filtering policy (Property 10)
    - **Property 10: Cache filtering policy**
    - Pre-populate cache with mixed statuses; verify "translated"/"passthrough" excluded, "untranslated" included
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 7.11 Write property test for cache population (Property 11)
    - **Property 11: Successful translation populates cache**
    - Translate a batch; verify all results stored in cache; subsequent call returns cached without API call
    - **Validates: Requirements 5.3**

  - [ ]* 7.12 Write property test for deduplication (Property 12)
    - **Property 12: Deduplication within batches**
    - Generate segments with K duplicates; verify API receives text at most once; all K segments get same result
    - **Validates: Requirements 5.5**

  - [ ]* 7.13 Write property test for RPM limiter per batch (Property 13)
    - **Property 13: RPM limiter called once per batch**
    - Mock _respect_rpm; verify called exactly once per batch regardless of batch size
    - **Validates: Requirements 6.1**

  - [ ]* 7.14 Write property test for batch_size validation (Property 14)
    - **Property 14: batch_size validation**
    - Generate integers; verify Gemini_Config succeeds iff 1 ≤ V ≤ 20, raises ValueError otherwise
    - **Validates: Requirements 8.3, 8.4**

- [ ] 8. Unit tests for edge cases
  - [ ]* 8.1 Write unit tests for batch translation in `tests/unit/test_batch_translation.py`
    - Test empty input returns empty dict
    - Test batch_size=1 sends one text per API call
    - Test batch_size=20 with fewer than 20 segments sends single batch
    - Test config validation boundary values (0, 1, 20, 21)
    - Test fallback logging (captured log assertions)
    - Test integration between cache and batch assembly
    - Test backward compatibility of `translate()` method
    - _Requirements: 1.5, 7.1, 7.2, 8.2, 8.3, 8.4_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All tests mock the Gemini API client — no real API calls in automated tests
- The existing `translate()` method remains unchanged for backward compatibility

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["3.1"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["7.1", "7.4", "7.5", "7.6", "7.7", "7.14", "8.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.8", "7.9", "7.10", "7.11", "7.12", "7.13"] }
  ]
}
```
