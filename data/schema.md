# 입력 JSONL 스키마

한 줄에 예제 하나를 JSON 객체로 저장합니다. 이미지 경로는 최상위 필드 `image_path` 또는 `image`에 넣습니다.

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

## 필드 작성 방법

- `private_input`은 KURE에 전달할 텍스트입니다. 객관식에서는 지시문, 질문, 번호가 붙은 모든 선택지를 한 문자열에 담습니다.
- `final_answer`에는 원본의 정답 내용을, `target`에는 디코더가 생성하도록 학습할 답변을 넣습니다. 객관식에서는 정답 선택지의 텍스트를 사용합니다.
- 이미지-텍스트 예제에는 이미지 경로와 실제 파일의 `image_sha256`이 필요합니다. 텍스트 전용 예제에서는 이미지 필드를 생략합니다.
- `input_token_length`는 고정된 KURE 토크나이저의 특수 토큰을 포함한 입력 길이입니다. `target_token_length`는 고정된 Qwen 토크나이저로 계산한 정답 길이에 EOS 한 토큰을 더한 값입니다.
- `metadata.native_visual_tokens`에는 고정된 Qwen 이미지 프로세서의 머저 출력 토큰 수를 기록합니다.
- `prepare_data.py`는 `repeat_factor`, `semantic_id`, `content_id`, `input_hash`, `answer_hash`, `group_id`를 계산해 전처리 결과에 기록합니다.
