# 데이터 출처와 버전

논문 실험에 사용한 데이터셋과 버전, 전처리 후 예제 수를 정리했습니다. [입력 형식](schema.md) · [전처리 명령](README.md)

## Stage 1

영어 학습 예제 3,488,347개의 데이터셋별 구성입니다.

| 데이터셋 | 학습 예제 수 |
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

기존 영어 예제 3,288,372개에 VQAv2의 질문·이미지 쌍 200,000개를 추가했습니다. VQAv2 원본 443,757쌍에서 중복 124쌍을 제거한 뒤, `{seed, question_id}`의 SHA-256 순서로 샘플링했습니다(seed 42). 선택된 예제는 예·아니요 75,188개, 숫자 25,975개, 기타 98,837개입니다. 이후 비전 토큰 한도(1,024)를 적용해 LLaVA 18개와 TextVQA 7개를 제외했습니다.

## Stage 2

| 데이터셋 | 사용한 버전 또는 출처 | 학습 예제 수 |
|---|---|---:|
| Pri-DDXPlus | 이전 연구 local `dataset2`, checksum manifest로 고정 | 54,992 |
| Pri-NLICE | 이전 연구 local `dataset2`, checksum manifest로 고정 | 3,249 |
| MedMCQA | `91c6572c454088bf71b679ad90aa8dffcd0d5868` | 178,599 |
| medical meadow flashcards | `7597b32036d67c731cb91bae4f49717fcfe5d5f0` | 33,285 |
| VQA-RAD mirror | `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` | 314 |
| SLAKE mirror | `a9083ce6c34ac3ffb17671a605962924d8a8f9e9` | 339 |
| PathVQA mirror | `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` | 6,639 |
| PMC-VQA v2 | `b56ae594f794867893143b337b4118a835794647` | 152,471 |

학습 시 Pri-DDX, Pri-NLICE, VQA-RAD, SLAKE, PathVQA의 예제에는 반복 가중치 4를, MedMCQA, flashcards, PMC-VQA의 예제에는 1을 적용했습니다. SLAKE는 영어 예제만 사용했습니다.

Pri-DDX와 Pri-NLICE의 객관식 예제는 원본 질문과 전체 선택지를 구조화한 뒤 정답 선택지의 텍스트를 학습 대상으로 사용했습니다. 이전 A/B/C/D 형식의 레코드는 구조화된 MedMCQA 선택지 순서와 일치할 때 사용했습니다.

## PathVQA 이미지 복원

이미지 SHA-256을 기준으로 중복을 제거한 뒤 학습 2,350장, 검증 831장, 테스트 857장을 사용했습니다. [분할 기준과 평가 명령](../docs/REPRODUCIBILITY.md#pathvqa-이미지-복원)
