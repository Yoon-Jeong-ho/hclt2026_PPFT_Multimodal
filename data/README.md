# 데이터 준비

이 저장소는 원본 데이터나 생성된 JSONL을 재배포하지 않습니다. 이 디렉터리는 논문에서 사용한 모집단,
표준화 경계, 필터 및 샘플링 규칙을 설명합니다. 각 데이터셋을 적법하게 내려받고 이용 조건을 확인하는 책임은
재현 사용자에게 있습니다.

## 공개 코드의 경계

원본 제공 형식은 데이터셋마다 다르고 일부 Pri 데이터는 이전 연구의 로컬 파일이므로, 공개 코드는 원본
다운로더나 개인 경로를 내장하지 않습니다. 대신 `scripts/prepare_data.py`가 **source adapter를 통과한 canonical
JSONL**을 입력받아 다음 논문 공통 단계를 재현합니다.

1. 필수 필드, 영어 스크립트, MCQA 구조 및 image/SHA 결합을 검사합니다.
2. 평가 split을 우선하여 train의 semantic 및 image/patient group 누수를 제거합니다.
3. source 내부에서 text는 semantic ID, image-text는 semantic ID와 image SHA-256으로 중복을 제거합니다.
4. KURE input 512, Qwen target+EOS 512, native post-merge visual token 1,024를 초과하면 row 전체를
   제외합니다. text 자르기, 임의 resize, visual token pruning은 하지 않습니다.
5. Stage 2 main benchmark train row에는 repeat factor 4, support row에는 1을 부여하고, seed 42의 50/50
   text/image-text 순서를 만듭니다.

Source adapter는 [`schema.md`](schema.md)의 평면 스키마를 만들어야 합니다. 특히 image row의
`metadata.native_visual_tokens`는 **동일한 고정 Qwen processor**로 사전에 측정한 값이어야 합니다. 파일명이나
이미지 크기만으로 visual token 수를 추정해서는 안 됩니다.

## 논문 모집단

| 구분 | 최종 모집단 | 비고 |
|---|---:|---|
| Stage 1 train | 3,488,347 | text 2,761,963 + image-text 726,384, 1 epoch |
| Stage 2 train | 429,888 | text 270,125 + image-text 159,763 |
| Stage 2 balanced exposure | 889,728 | global batch 64에서 13,902 updates |
| 논문 utility evaluation | 17,734 | 아래 다섯 source의 pooled validation/test |
| PathVQA inversion | 2,350 / 831 / 857 | image-deduplicated train/validation/test |

Stage 2의 필터 전 train은 434,246개였습니다. 영어 스크립트 5개, target+EOS 4,150개, native visual
token 203개가 row 단위로 제외되어 429,888개가 남았습니다. repeat 적용 후 balance 전 유효 노출은 text
444,848, image-text 181,639입니다. per-rank batch 8 × 4 ranks인 homogeneous block 32에 맞춰 큰 쪽을
444,864로 padding하고 두 모달리티를 교차해 889,728 exposure를 구성했습니다.

논문 표의 평가는 다음 다섯 source만 사용합니다. 전체 Stage 2 publication에는 MedMCQA validation 4,170개를
포함한 21,904-row sealed derivative도 있었지만, 이는 논문 표의 17,734 denominator에 포함되지 않습니다.
다섯 source의 필터 전 17,929개에서 native visual-token outlier 195개(PathVQA 29, VQA-RAD 166)를
대체/refill 없이 제외한 값이 17,734개입니다.

| source | rows |
|---|---:|
| PathVQA | 12,949 |
| Pri-DDX | 1,549 |
| Pri-NLICE | 650 |
| SLAKE (English) | 2,114 |
| VQA-RAD | 472 |

## 실행 예시

아래 명령은 dataset-specific adapter가 만든 canonical 파일을 검증하고 논문 Stage 2 train/utility subset을
발행합니다. `--expect-*` 값은 다른 모집단을 논문 데이터로 잘못 부르는 것을 막습니다.

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

출력은 `train.jsonl`, `evaluation.jsonl`, `epoch_indices.json`, `report.json`입니다. publication은 새
디렉터리에 원자적으로 생성되며 기존 결과를 덮어쓰지 않습니다. `report.json`에는 row payload나 절대 입력
경로를 넣지 않고 입력/출력 SHA-256, count, exclusion 사유와 보존 규칙만 기록합니다.

Stage 1도 같은 검사/필터를 사용하지만 natural-ratio 1 epoch이므로 balanced order를 만들지 않습니다.

```bash
python -m scripts.prepare_data \
  --stage stage1 \
  --train /path/to/canonical_stage1_train_before_budget.jsonl \
  --evaluation /path/to/canonical_stage1_heldout.jsonl \
  --output-dir data/processed/stage1 \
  --expect-train-rows 3488347
```
