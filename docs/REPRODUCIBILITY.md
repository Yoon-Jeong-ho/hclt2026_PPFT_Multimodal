# 실험 설정과 논문 결과

## 의료 QA 결과

논문 Table 1의 정확도(%)입니다. 세 모델은 같은 Stage 1 체크포인트에서 시작해 서로 다른 노이즈 조건으로 학습했습니다. εt는 텍스트, εv는 이미지에 적용한 값입니다.

| 데이터셋 | 무노이즈 | εt=75, εv=2.0 | εt=75, εv=2.5 |
|---|---:|---:|---:|
| PathVQA | 34.73 | 31.83 | 31.73 |
| Pri-DDX | 42.54 | 35.83 | 36.54 |
| Pri-NLICE | 57.23 | 50.77 | 50.62 |
| SLAKE | 44.80 | 32.26 | 32.78 |
| VQA-RAD | 37.92 | 33.90 | 32.42 |
| 평균 | 43.44 | 36.92 | 36.82 |

평가에는 다섯 데이터셋의 검증·테스트 예제 총 17,734개를 사용했습니다. 객관식은 정답 선택지의 내용을 기준으로 채점하고, 길이 제한에 도달한 답변과 빈 답변도 집계에 포함했습니다. 평균은 다섯 데이터셋 정확도의 산술평균입니다. [데이터셋별 예제 수](../data/README.md)

## 학습 설정

| 구분 | 논문 설정 |
|---|---|
| 언어모델 / 텍스트 인코더 | Qwen3.5-0.8B / KURE-v1 |
| 텍스트 표현 | 패딩을 제외하고 연속된 토큰 두 개의 은닉 상태를 평균 |
| 이미지 표현 | 비전 머저 직전의 토큰을 원래 격자 순서대로 사용 |
| Stage 1 학습 | 예제 3,488,347개, 무노이즈, 1 epoch, 전체 배치 크기 128 |
| Stage 2 학습 | 예제 429,888개, 1 epoch, 전체 배치 크기 64 |
| Stage 2 학습량 | 텍스트와 이미지·텍스트를 같은 비율로 총 889,728회 사용. 모델당 13,902회 업데이트 |
| QA 답변 생성 | 예제별 그리디 디코딩, 최대 512토큰. 노이즈 조건에서는 첫 번째 샘플 사용 |

원 실험에서는 Stage 1의 SQuAD와 CommonsenseQA relaxed accuracy가 각각 0.50 이상인지 확인한 뒤 Stage 2를 시작했습니다. [학습 설정 파일](../configs/)

## PathVQA 이미지 복원

무노이즈로 학습한 Stage 2 모델에서 비전 머저 직전의 표현을 추출합니다. 같은 이미지에 여러 질문이 연결돼 있으므로 이미지 SHA-256을 기준으로 중복을 제거합니다. 여러 split에 걸친 이미지는 test, validation, train 순서로 하나의 split에만 배정합니다.

| 단계 | 이미지 수 | 사용 방법 |
|---|---:|---|
| Train | 2,350 | 무노이즈 표현으로 CNN 학습 |
| Validation | 831 | 매 epoch 복원 MSE 기록 |
| Test | 857 | 10번째 epoch의 CNN으로 노이즈 강도별 MSE 평가 |

복원기는 768차원 입력을 128채널로 투영하고 residual block 3개와 RGB 헤드를 거쳐 이미지를 복원합니다. 원본 비전 프로세서가 만든 패치에서 복원한 RGB 값(0–1)을 정답으로 사용합니다. 학습 전 이미지 4개로 과적합 여부를 점검하고, 모델과 seed를 초기화한 뒤 10 epoch 동안 학습합니다. AdamW의 학습률은 1e-4, weight decay는 0.01이며 gradient accumulation은 4입니다. [전체 복원 실험 설정](../configs/attack/pathvqa.yaml)

저장소 루트에서 다음 명령을 순서대로 실행합니다.

```bash
python -m scripts.extract_vision_representations \
  --stage2-config configs/stage2/no_noise.yaml \
  --checkpoint outputs/checkpoints/stage2/no_noise/final \
  --qwen-path models/qwen35-08b --kure-path models/kure-v1 \
  --manifest data/processed/stage2/train.jsonl \
  --manifest data/processed/stage2/evaluation.jsonl \
  --output-dir data/cache/pathvqa

python -m scripts.train_vision_inversion \
  --config configs/attack/pathvqa.yaml \
  --cache-index data/cache/pathvqa/index.jsonl \
  --output-dir outputs/attacks/pathvqa

python -m scripts.evaluate_vision_inversion \
  --config configs/attack/pathvqa.yaml \
  --cache-index data/cache/pathvqa/index.jsonl \
  --attacker-dir outputs/attacks/pathvqa \
  --output-dir outputs/privacy/pathvqa
```

`outputs/privacy/pathvqa/metrics.csv`와 `summary.json`에는 노이즈 강도별 MSE가, `draws.jsonl`에는 이미지별 결과가 저장됩니다. 각 이미지의 픽셀·채널 평균 오차를 구해 이미지별로 동일하게 가중합니다. 무노이즈는 한 번 평가하고, 각 노이즈 조건은 세 번 샘플링해 평균을 냅니다.

## 원 실험의 코드와 체크포인트

아래는 논문 실험에 사용한 코드 커밋과 체크포인트 해시입니다.

| 원 artifact | 식별자 |
|---|---|
| clean Stage 2 / victim loader source Git | `15e7380cda8962c45b58c22fafae6ea6244ab4a7` |
| split-noise frozen source Git | `db13659f2188d024f8d9e745544c894c0e47d26e` |
| 공통 Stage 1 parent tree SHA-256 | `88863503ac3f39972300c4124f939ff41b39c223aae1f030137f2d9ba00fe5e0` |
| clean Stage 2 victim tree SHA-256 | `e074be6db5e833fc6d866cc507f7b6c26ed9de0cd077cb0bad661635e34801ab` |
| text75/image2 final tree SHA-256 | `384e06f9dd85b79eec13caf74507be194c53ceb2c55b34e2c9b9c0099cfcf8c3` |
| text75/image2.5 final tree SHA-256 | `1a94ddf9769c2edfdde702f61c557216c391ca0e08d1a228268340ca9fa435f0` |
| PathVQA clean CNN checkpoint file SHA-256 | `4ba90c52aa73b0366e2a0b3a2990b20932c990e3a1cffac17ef275624c71a263` |
