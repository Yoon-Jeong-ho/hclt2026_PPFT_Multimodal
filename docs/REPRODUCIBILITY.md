# 논문 설정과 재현 범위

이 저장소는 **원본 이미지/텍스트 전송 없이 민감정보를 처리하는 멀티모달 표현 정렬 및 도메인 학습 프레임워크**의
코드 공개용 정리본입니다. 기존 연구 저장소의 전체 Git 이력이나 운영 환경을 복제하지 않습니다.

## 포함한 실험

| 구분 | 논문 설정 |
|---|---|
| Backbone / text encoder | Qwen3.5-0.8B / KURE-v1, 고정 revision |
| Text representation | token hidden states, padding-aware contiguous mean pooling k=2 |
| Vision representation | native vision tower의 merger 직전 token; native grid/order 유지 |
| Stage 1 | 무노이즈, 1 epoch, 3,488,347 unique rows; global batch 128 |
| Stage 2 | 동일 Stage 1 parent에서 각각 새로 시작, 1 epoch, 429,888 unique rows; global batch 64 |
| Stage 2 exposure | modality-balanced 889,728회, arm당 13,902 optimizer updates |
| Utility arms | clean / text ε=75·vision ε=2 / text ε=75·vision ε=2.5 |
| Utility generation | singleton greedy decoding, 최대 512 new tokens, draw 0 |
| Vision attack | PathVQA clean-trained residual CNN, 10 epochs, final-epoch checkpoint |

Stage 1은 KURE·text projection·native visual merger·언어모델 전체를 학습하고 vision encoder body를 고정합니다.
Stage 2는 KURE·vision encoder·decoder base를 고정하고 projection·native merger·decoder-only LoRA를 학습합니다.
두 noisy utility 열은 **같은 clean 모델의 추론 시 노이즈만 바꾼 결과가 아니라 각각 학습한 Stage 2 arm**입니다.

## 기록된 Table 1

아래는 새로 실행한 결과가 아니라 논문과 완료된 실험 기록에서 확인한 값입니다.

| Dataset | No noise | εt=75, εv=2.0 | εt=75, εv=2.5 |
|---|---:|---:|---:|
| PathVQA | 34.73 | 31.83 | 31.73 |
| Pri-DDX | 42.54 | 35.83 | 36.54 |
| Pri-NLICE | 57.23 | 50.77 | 50.62 |
| SLAKE | 44.80 | 32.26 | 32.78 |
| VQA-RAD | 37.92 | 33.90 | 32.42 |
| Average | 43.44 | 36.92 | 36.82 |

단위는 %입니다. Average는 5개 source의 **동일 가중 macro 평균**으로, 17,734개 전체의 micro accuracy가 아닙니다.
평가는 원본 validation/test를 합친 pooled scope입니다. 원래 실행의 21,904개 평가 row에는 MedMCQA 4,170개도
포함되었지만 논문 표는 이를 제외합니다. 생성 길이 도달·빈 응답 등을 유리하게 제거하지 않습니다.
객관식은 option letter가 아니라 **정답 내용/option text**를 기준으로 기존 relaxed evaluator를 적용합니다.
이는 각 데이터셋의 공식 test-only leaderboard 지표와 같지 않습니다.

## Vision attack의 정확한 의미

- clean Stage 2 victim의 **native merger 이전 표현**을 관찰하는 공격입니다. projection/merger 뒤 표현 공격과
  혼동하지 않습니다. private cache는 반복 실험을 위한 로컬 계산 캐시이며 외부로 공개하지 않습니다.
- PathVQA QA row를 image SHA-256으로 중복 제거합니다. split 중복은 **test > validation > train** 순서로
  보존하여, 2,350 / 831 / 857장의 image-disjoint 집합을 사용합니다.
- 타깃은 원본 파일 크기의 이미지가 아니라 **실제 native processor의 patch tensor를 역변환한 RGB [0,1]**입니다.
  별도 resize로 타깃을 근사하지 않습니다. native flatten order를 되돌리는 처리는 공격 CNN 내부 좌표 해석에만
  적용하며 victim의 vision 경로는 바꾸지 않습니다.
- 768-d input projection → 128-channel residual CNN 3 blocks → RGB head → bilinear interpolation → sigmoid.
  singleton native grids, accumulation 4, AdamW lr=1e-4 / weight decay=0.01, cosine / warmup 0.1,
  gradient clipping 1.0, seed 42. **10 epochs / 23,500 image exposures / 5,880 updates**입니다.
- 4개 train image의 overfit 검사를 먼저 수행한 뒤 seed/model을 초기화하고 본 학습을 시작합니다.
  validation MSE는 기록하지만 **10번째 epoch**를 선택하며 test MSE로 체크포인트를 선택하지 않습니다.
- 무노이즈는 1회, 각 noisy ε는 example ID·ε·draw·seed로 결정되는 3회 평가입니다. 노이즈는 원래 구현과 같이
  CPU fp32에서 샘플링한 뒤 cache dtype으로 되돌립니다. 고정 CNN에는 노이즈 학습을 추가하지 않습니다.
- MSE는 image 내부 pixel/channel 평균 → image 동일 가중 평균 → draw 평균입니다.
  `draw_mean_sd`는 3개 draw 평균의 population SD이며 신뢰구간이나 학습 seed 변동이 아닙니다.

`configs/attack/pathvqa.yaml`은 논문 곡선의 clean + 9개 noisy ε를 보존합니다. 마지막 두 값
1.8019849262262475와 3.603969852452495는 text ε=75/150에 대해 예상 raw noise radius / median feature norm을
맞춘 탐색적 값입니다. 따라서 utility의 εv=2.0/2.5와 공격 곡선의 grid는 별개입니다.
Figure 3의 표시값 1.802는 1.8019849262262475를 반올림한 값입니다. Figure 제작 코드는 포함하지 않습니다.

## 원 실행 provenance

아래 식별자는 원 연구 저장소의 immutable source 및 완료된 체크포인트 기록입니다.
이 정리본의 파일/설정은 경로와 실행 인터페이스가 바뀌었으므로 같은 Git SHA나 checkpoint hash를 갖는다고
주장하지 않습니다. 원 체크포인트/원 데이터/실험 운영 receipt는 재배포하지 않습니다.

| 원 artifact | 식별자 |
|---|---|
| clean Stage 2 / victim loader source Git | `15e7380cda8962c45b58c22fafae6ea6244ab4a7` |
| split-noise frozen source Git | `db13659f2188d024f8d9e745544c894c0e47d26e` |
| 공통 Stage 1 parent tree SHA-256 | `88863503ac3f39972300c4124f939ff41b39c223aae1f030137f2d9ba00fe5e0` |
| clean Stage 2 victim tree SHA-256 | `e074be6db5e833fc6d866cc507f7b6c26ed9de0cd077cb0bad661635e34801ab` |
| text75/image2 final tree SHA-256 | `384e06f9dd85b79eec13caf74507be194c53ceb2c55b34e2c9b9c0099cfcf8c3` |
| text75/image2.5 final tree SHA-256 | `1a94ddf9769c2edfdde702f61c557216c391ca0e08d1a228268340ca9fa435f0` |
| PathVQA clean CNN checkpoint file SHA-256 | `4ba90c52aa73b0366e2a0b3a2990b20932c990e3a1cffac17ef275624c71a263` |

L2-Laplace의 방향/Gamma 반경/원래 norm 보존 의미는 기존 PPFT 구현의 `_inject_noise`에서 계승했습니다.
Qwen의 native vision preprocessing과 transformer 구성, KURE encoder 및 관련 라이브러리는 각각 upstream의
사용 조건을 따릅니다. 원본 데이터와 모델 가중치의 재배포 권한을 이 저장소가 부여하지 않습니다.

## 재현 한계와 해석

- 이 공개용 정리 과정에서 전체 GPU 학습이나 논문 수치를 다시 계산하지 않았습니다. CPU 회귀 테스트와
  static/import/config 검증은 native GPU parity·overfit·full experiment 재실행을 대신하지 않습니다.
- 본 학습 전 새 환경에서 native vision parity, 4-example text/image overfit를 확인해야 합니다. Stage 2의
  원 연구 진입 조건은 Stage 1 SQuAD와 CSQA relaxed accuracy 각각 ≥0.50입니다. 정리본에 성공 receipt를
  꾸며 넣거나 미검증 실행을 원 실험으로 취급하지 않습니다.
- 원 데이터/정확한 split/모델 snapshot 및 GPU kernel 환경이 없으면 동일 수치의 재현을 보장하지 않습니다.
  fresh run의 config/manifest/model/parent hash를 남기고 환경 변경을 기록해야 합니다.
- 코드의 client/server 구분은 표현 전달 인터페이스를 모델링합니다. 완성된 네트워크 서비스나 전송 암호화
  시스템을 제공한다는 뜻이 아닙니다. 답변 출력의 개인정보 노출을 해결한다고 주장하지 않습니다.
- norm-matched grid는 held-out 표현 통계를 참고해 선택한 **탐색적 설정**입니다. 같은 상대적 노이즈 크기가
  같은 프라이버시를 의미하지 않습니다.
- 높은 pixel MSE만으로 질병·객체·민감 속성의 추론이 차단됨을 증명할 수 없습니다. 이 논문 범위의 clean-trained
  CNN 결과를 더 강한/noise-aware 공격 전체에 대한 보장으로 해석하지 않습니다.
- ε는 실험 제어값입니다. formal joint-DP, end-to-end confidentiality 또는 출력 개인정보 보호를 보장하지 않습니다.
