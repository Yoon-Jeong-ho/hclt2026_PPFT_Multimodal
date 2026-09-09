# 데이터 출처와 실행 lineage

이 문서는 논문 실행에서 실제로 사용된 source family와 고정 revision을 구분합니다. 데이터는 이 저장소에
포함되지 않으며, 재배포 권한을 보증하지 않으므로 각 upstream의 최신 이용 조건과 원 저작물 조건을 확인해야 합니다.

## Stage 1

최종 3,488,347-row English Stage 1의 source별 count는 다음과 같습니다.

| source | rows |
|---|---:|
| WebInstructSub | 2,098,886 |
| SmolTalk all | 485,897 |
| LLaVA pretrain 558k | 502,741 |
| VQAv2 train2014 | 200,000 |
| SQuAD | 69,049 |
| UltraChat 200k | 37,500 |
| Alpaca cleaned | 39,627 |
| TextVQA | 20,192 |
| ChatQA NewsQA | 17,235 |
| CommonsenseQA | 9,677 |
| medical meadow flashcards | 4,092 |
| ScienceQA | 3,451 |

기존 English manifest 3,288,372개에 official VQAv2 train question-image pair 200,000개를 추가했습니다.
VQAv2의 443,757 raw pair에서 중복 124개를 제외한 443,633 eligible pair를 만든 뒤, seed 42와 canonical
`{seed, question_id}` SHA-256 순서로 200,000개를 골랐습니다. answer type은 yes/no 75,188, number
25,975, other 98,837입니다. 이후 1,024 visual-token ceiling으로 LLaVA 18개와 TextVQA 7개를 제외했고
VQAv2 row는 모두 유지했습니다.

이 공개 export는 여러 upstream의 다운로드/파싱 코드를 모두 복제하지 않습니다. source-native adapter 결과를
[`schema.md`](schema.md)의 canonical JSONL로 만든 뒤 공통 publication 코드를 실행해야 합니다. 이는 개인
cache 경로와 수 GB 데이터 조작 코드를 논문 핵심 방법처럼 공개하는 것을 피하면서, 실제 필터·누수 제거·샘플링
규칙을 그대로 남기기 위한 경계입니다.

## Stage 2

| source | 고정 revision / provenance | 최종 train rows |
|---|---|---:|
| Pri-DDXPlus | 이전 연구 local `dataset2`, checksum manifest로 고정 | 54,992 |
| Pri-NLICE | 이전 연구 local `dataset2`, checksum manifest로 고정 | 3,249 |
| MedMCQA | `91c6572c454088bf71b679ad90aa8dffcd0d5868` | 178,599 |
| medical meadow flashcards | `7597b32036d67c731cb91bae4f49717fcfe5d5f0` | 33,285 |
| VQA-RAD mirror | `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` | 314 |
| SLAKE mirror | `a9083ce6c34ac3ffb17671a605962924d8a8f9e9` | 339 |
| PathVQA mirror | `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` | 6,639 |
| PMC-VQA v2 | `b56ae594f794867893143b337b4118a835794647` | 152,471 |

Stage 2 main benchmark는 Pri-DDX, Pri-NLICE, VQA-RAD, SLAKE, PathVQA이며 train repeat factor 4를
적용했습니다. MedMCQA, flashcards, PMC-VQA는 support source로 factor 1입니다. SLAKE의 비영어 row는
제외했습니다.

Pri-DDX/Pri-NLICE MCQA는 과거 생성 문장의 reasoning에서 정답을 다시 추출하지 않았습니다. 완전한 질문과
모든 option을 구조화 source와 answer-independent하게 join하고, gold option text를 target으로 사용했습니다.
경계가 모호한 legacy A/B/C/D serialization은 structured MedMCQA option sequence와 정확히 일치할 때만
허용했으며, 일치하지 않으면 fail closed 했습니다.

## PathVQA 복원 공격 split

복원기는 PathVQA의 image SHA-256 기준 중복 제거 split으로 학습/선정/평가했습니다.

| split | unique images | 용도 |
|---|---:|---|
| train | 2,350 | clean representation에서 공격 모델 학습 |
| validation | 831 | 학습 중 validation MSE 기록; 최종 10 epoch checkpoint 사용 |
| test | 857 | clean/noisy representation MSE 평가 |

동일 image SHA-256은 split을 넘지 않으며, 질문 수가 아니라 unique image가 denominator입니다.
