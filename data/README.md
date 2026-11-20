# 데이터 준비

원본 데이터를 [출처 목록](sources.md)에서 확인하고 각 제공처의 이용 조건에 따라 준비합니다. Pri-DDX와 Pri-NLICE에는 원 실험에서 사용한 로컬 자료가 필요합니다.

## 입력과 전처리

데이터셋별 원본을 [공통 JSONL 스키마](schema.md)로 변환한 뒤 `scripts.prepare_data`를 실행합니다. 이미지가 있는 예제는 고정된 Qwen 프로세서로 토큰 수를 측정해 `metadata.native_visual_tokens`에 기록합니다. 스크립트는 다음 순서로 데이터를 준비합니다.

1. 필수 필드, 언어, 객관식 선택지, 이미지 파일과 SHA-256을 확인합니다.
2. 평가 예제를 우선 보존하고 같은 내용이나 같은 이미지·환자 그룹이 학습 데이터와 겹치면 학습 쪽에서 제외합니다.
3. 각 데이터셋 안에서 텍스트 예제는 내용 식별자로, 이미지-텍스트 예제는 내용 식별자와 이미지 SHA-256으로 중복을 제거합니다.
4. KURE 입력 512토큰, Qwen 정답과 EOS 512토큰, 비전 머저 출력 1,024토큰 중 어느 한도를 넘는 예제는 통째로 제외합니다.
5. Stage 2에서는 주요 평가 데이터셋에서 온 학습 예제를 4회, 보조 데이터셋의 예제를 1회 반영합니다. seed 42로 텍스트와 이미지-텍스트 학습 배치를 1:1로 구성합니다.

## 데이터 구성

| 구분 | 예제 수 또는 사용 횟수 | 구성 |
|---|---:|---|
| Stage 1 학습 | 3,488,347 | 텍스트 2,761,963개, 이미지·텍스트 726,384개 |
| Stage 2 학습 | 429,888 | 텍스트 270,125개, 이미지·텍스트 159,763개 |
| Stage 2 학습 예제 사용 횟수 | 889,728 | 전체 배치 크기 64, 13,902회 업데이트 |
| 의료 QA 평가 | 17,734 | 아래 다섯 데이터셋의 검증·테스트 예제 |
| PathVQA 이미지 복원 | 2,350 / 831 / 857 | 학습 / 검증 / 테스트 세트의 고유 이미지 수 |

Stage 2 학습 후보 434,246개 중 언어 필터에서 5개, 정답 길이에서 4,150개, 비전 토큰 수에서 203개를 제외했습니다. 반복 횟수를 적용하면 텍스트 444,848회, 이미지-텍스트 181,639회입니다. 같은 모달리티의 예제 32개(GPU당 8개 × 4)를 한 배치로 묶고, 모달리티별로 444,864회를 사용해 총 889,728회 학습합니다.

논문 의료 QA 평가는 아래 다섯 데이터셋을 사용합니다. 평가 후보 17,929개 중 비전 토큰 한도를 초과한 PathVQA 29개와 VQA-RAD 166개를 제외했습니다.

| 평가 데이터셋 | 예제 수 |
|---|---:|
| PathVQA | 12,949 |
| Pri-DDX | 1,549 |
| Pri-NLICE | 650 |
| SLAKE (English) | 2,114 |
| VQA-RAD | 472 |

## 실행 예시

아래 `/path/to/`를 준비한 JSONL 파일의 경로로 바꿉니다. `--expect-*` 옵션은 전처리 결과가 논문의 행 수 및 학습 예제 사용 횟수와 일치하는지 확인합니다.

```bash
python -m scripts.prepare_data \
  --stage stage2 \
  --train /path/to/canonical_stage2_train_before_budget.jsonl \
  --evaluation /path/to/canonical_paper_five_source_evaluation_before_budget.jsonl \
  --output-dir data/processed/stage2 \
  --seed 42 \
  --homogeneous-block-size 32 \
  --cost-bucket-size 256 \
  --expect-train-rows 429888 \
  --expect-evaluation-rows 17734 \
  --expect-balanced-exposures 889728
```

Stage 2는 새 출력 디렉터리에 `train.jsonl`, `evaluation.jsonl`, `epoch_indices.json`, `report.json`을 생성합니다. `report.json`에서 입출력 SHA-256과 필터별 제외 건수를 확인할 수 있습니다.

Stage 1에서는 원래 모달리티 비율로 1 epoch 학습합니다.

```bash
python -m scripts.prepare_data \
  --stage stage1 \
  --train /path/to/canonical_stage1_train_before_budget.jsonl \
  --evaluation /path/to/canonical_stage1_heldout.jsonl \
  --output-dir data/processed/stage1 \
  --expect-train-rows 3488347
```
