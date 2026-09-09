# Canonical JSONL 스키마

JSONL의 각 줄은 하나의 JSON object입니다. 런타임이 사용하는 image reference는 중첩 객체가 아니라
`image_path`/`image` 평면 필드입니다.

```json
{
  "example_id": "stable-hash-or-source-id",
  "source_record_id": "source-native-id",
  "source": "PathVQA",
  "original_split": "train",
  "language": "en",
  "modality": "image_text",
  "private_input": "complete question and, for MCQA, every labeled option",
  "final_answer": "structured gold answer text",
  "target": "answer-first decoder target",
  "options": ["option A", "option B"],
  "gold_option_index": 1,
  "answer_aliases": ["B", "2", "option B"],
  "image_path": "/external/data/image.png",
  "image": "/external/data/image.png",
  "image_sha256": "64-lowercase-hex-digits",
  "input_token_length": 128,
  "target_token_length": 12,
  "metadata": {"native_visual_tokens": 300, "image_grid_thw": [1, 30, 40]}
}
```

## 필드 계약

- `private_input`은 raw private text를 모두 결합한 KURE 입력입니다. MCQA는 공통 instruction, 완전한 질문,
  모든 labeled option을 하나의 sequence에 포함합니다. Qwen decoder의 text token ID 입력으로 재사용하지
  않습니다.
- `final_answer`는 구조화된 gold field에서 얻은 실제 정답 내용입니다. `target`은 answer/EOS loss가 적용될
  decoder target입니다. 객관식은 letter만이 아니라 gold option text를 감독합니다.
- text row에는 image 필드가 없어야 하고, image-text row에는 `image_path`와 실제 파일 바이트의
  `image_sha256`이 필요합니다. 생성된 manifest와 image payload 자체는 Git에 넣지 않습니다.
- `input_token_length`는 고정 KURE tokenizer에서 special token을 포함한 길이, `target_token_length`는 고정
  Qwen tokenizer 길이에 EOS 1개를 더한 값입니다.
- `metadata.native_visual_tokens`는 Qwen native processor의 post-merge token 수입니다. 재현 실행에서는
  processor revision과 image SHA audit을 별도로 고정해야 합니다.
- `repeat_factor`와 `semantic_id`, `content_id`, `input_hash`, `answer_hash`, `group_id`는
  `prepare_data.py`가 다시 계산합니다. stale upstream identity를 신뢰하지 않습니다.

출력 `report.json`은 count와 hash만 담습니다. `private_input`, answer, image bytes, 절대 경로 등 private
payload를 audit report에 복제하지 않습니다.
