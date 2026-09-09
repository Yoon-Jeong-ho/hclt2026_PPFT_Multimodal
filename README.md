# HCLT 2026 · PPFT Multimodal

**Multimodal Representation Alignment and Privacy-Preserving Domain Adaptation without Raw Input Transmission**

논문 **「원본 이미지/텍스트 전송 없이 민감정보를 처리하는 멀티모달 표현 정렬 및 도메인 학습 프레임워크」**의
학습·평가 코드입니다. 텍스트 중심 PPFT를 이미지-텍스트 입력으로 확장하여, 원본 대신 인코더 표현을 이용한
의료 도메인 학습과 이미지 복원 공격을 평가합니다.

이 저장소에는 논문에 해당하는 **0.8B 모델, 2단계 학습, 의료 QA 평가, PathVQA CNN 복원 공격**만 담았습니다.
데이터·체크포인트·실험 로그·Figure 제작 코드·이후 추가 실험은 포함하지 않습니다.

## 작동 방식

```text
비공개 텍스트 → KURE-v1 → token mean pooling (k=2) → [노이즈] → text projection ─┐
비공개 이미지 → Qwen native vision encoder        → [노이즈] → native merger ──┤
                                                                            ↓
                                                               Qwen3.5-0.8B → 답변
```

- **Stage 1 — 표현 정렬:** 노이즈 없이 KURE, text projection, native visual merger, 언어모델을 학습합니다.
  vision encoder body는 고정합니다.
- **Stage 2 — 도메인 적응:** 같은 Stage 1 체크포인트에서 세 arm을 독립적으로 시작합니다. KURE·vision
  encoder·decoder base를 고정하고 projection·native merger·decoder LoRA만 학습합니다.
- **입력/손실:** 질문과 선택지를 포함한 비공개 텍스트는 KURE로만 들어갑니다. Qwen은 투영된 표현과 답변
  target token을 받으며, 손실은 답변/EOS 위치에만 적용합니다.
- **노이즈:** token별 isotropic L2-Laplace를 더한 뒤 원래 L2 norm으로 재조정합니다. 텍스트와 이미지의 ε는
  별도로 설정합니다. Qwen의 native image preprocessing·공간 병합·token 순서는 변경하지 않습니다.
- **복원 공격:** clean vision 표현으로 residual CNN을 10 epoch 학습하고, 같은 공격기에 평가 시 노이즈를
  적용하여 PathVQA의 이미지별 평균 RGB MSE를 측정합니다.

## 구성

```text
configs/                    # 고정 모델 revision, Stage 1/2, 공격 설정
src/ppft_multimodal/         # 모델·노이즈·학습·평가·복원 구현
scripts/                    # 데이터 준비, 학습, 생성 및 평가 진입점
data/                       # 데이터 출처·구성 절차·입력 schema 설명
tests/                     # 합성 입력 기반 회귀 테스트
docs/REPRODUCIBILITY.md     # 논문 설정, 기록된 결과, provenance와 한계
```

## 설치

Linux, Python 3.11, CUDA GPU 환경을 기준으로 합니다. 학습 설정은 **4 GPU / bf16 / DeepSpeed ZeRO-2**이며,
원 논문 global batch는 Stage 1에서 128, Stage 2에서 64입니다. 의존성 버전은 `pyproject.toml`에 고정했습니다.
CUDA/PyTorch에 맞는 드라이버와 DeepSpeed/kernel 빌드 환경이 필요합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'

hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/qwen35-08b
hf download nlpai-lab/KURE-v1 \
  --revision 4ed4540949c70b7da2c74004a915e1f2d5e46e4f --local-dir models/kure-v1
```

학습/추론은 로컬 model snapshot을 사용합니다. 모델 가중치를 임의의 Base checkpoint로 교체하지 마세요.
Qwen3.5 GPU 실행의 filtered FLA kernel도 고정 revision을 사용하며, 첫 실행 때 upstream kernel 조회/컴파일이
필요할 수 있습니다. 학습 지표는 로컬 JSONL로 기록하며, 외부 로깅 계정이나 인증키는 포함하지 않습니다.

## 1. 데이터 준비

[데이터 구성 설명](data/README.md)과 [입력 schema](data/schema.md)를 먼저 확인하세요.
원 데이터는 각 제공처에서 이용 조건에 맞게 확보하고, private 자료는 접근 권한이 있는 경우에만 사용합니다.
`data/raw/`, `data/processed/`, `data/cache/`는 Git에서 제외됩니다.

논문 설정의 입력 경로는 다음과 같습니다.

| 파일 | 용도 |
|---|---|
| `data/processed/stage1/train.jsonl` | Stage 1 학습, 3,488,347 unique rows |
| `data/processed/stage2/train.jsonl` | Stage 2 학습, 429,888 unique rows |
| `data/processed/stage2/evaluation.jsonl` | 의료 QA pooled 평가; 논문 표는 5개 source, 17,734 rows |

다운로드한 원본을 그대로 학습에 넣지 않고, 문서의 정규화·영어 필터·중복/누수 제거·native visual token budget
절차를 적용합니다. 데이터 버전/권한이 다르면 원 논문과 동일 모집단이 만들어진다고 보장하지 않습니다.

## 2. Stage 1 / Stage 2 학습

아래 명령은 **저장소 루트에서** 실행합니다. config의 data/output 경로를 먼저 확인하세요.
새 환경에서는 본 학습 전에 [재현 조건과 검증 한계](docs/REPRODUCIBILITY.md#재현-한계와-해석)를 확인해야 합니다.

```bash
accelerate launch --config_file configs/accelerate/paper.yaml -m scripts.train_stage1 \
  --config configs/stage1/paper.yaml \
  --qwen-path models/qwen35-08b --kure-path models/kure-v1

# 세 arm은 서로의 checkpoint가 아니라 동일한 Stage 1 final에서 시작합니다.
for arm in no_noise text_75_image_2 text_75_image_2p5; do
  accelerate launch --config_file configs/accelerate/paper.yaml -m scripts.train_stage2 \
    --config "configs/stage2/${arm}.yaml" \
    --qwen-path models/qwen35-08b --kure-path models/kure-v1
done
```

각 Stage 2 arm은 1 epoch / 13,902 updates로 동일합니다. 세 YAML의 공통 데이터·학습 조건을 일치시켜야 하며,
노이즈 값과 출력 디렉터리만 달라집니다. 기존 완료 출력을 덮어쓰지 않고, config·manifest·parent checkpoint
identity를 체크합니다. 작은 `--max-updates` 실행은 진단용이며 논문 전체 학습 결과가 아닙니다.

## 3. 의료 QA 평가

```bash
for arm in no_noise text_75_image_2 text_75_image_2p5; do
  python -m scripts.generate_predictions \
    --config "configs/stage2/${arm}.yaml" \
    --checkpoint "outputs/checkpoints/stage2/${arm}/final" \
    --evaluation data/processed/stage2/evaluation.jsonl \
    --qwen-path models/qwen35-08b --kure-path models/kure-v1 \
    --output "outputs/utility/${arm}.jsonl"
  python -m scripts.evaluate_utility "outputs/utility/${arm}.jsonl" \
    --output "outputs/utility/${arm}.json"
done
```

각 arm은 학습 때와 같은 노이즈 설정으로 평가합니다. singleton greedy 생성, 최대 512 new tokens, draw 0이며,
MCQA는 option label만 맞힌 경우가 아니라 **정답 내용**을 평가합니다. 논문 Average는 5개 source의 macro 평균입니다.
실제 논문 수치와 pooled validation/test 범위는 [재현 문서](docs/REPRODUCIBILITY.md#기록된-table-1)에 정리했습니다.

## 4. PathVQA 이미지 복원 공격

표현 추출은 동일한 victim 계산을 매 epoch 반복하지 않기 위한 **로컬 캐시**입니다. 학습 목표는 캐시의 표현에서
입력 이미지의 native processor-space RGB를 복원하는 것입니다. 원본/복원 이미지는 외부에 업로드하지 않습니다.

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

공격 학습은 **무노이즈만**, 평가는 clean과 설정된 ε grid 전체입니다. 중복 제거 후 train/validation/test는
**2,350 / 831 / 857장**입니다. 최종 출력은 `metrics.csv`, `summary.json`, 로컬 example별 `draws.jsonl`이며,
Figure 생성 코드나 그림 파일은 만들지 않습니다. 노이즈 draw 간 SD는 confidence interval이 아닙니다.

## 검증과 주의사항

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q -m 'not integration'
python -m ruff check src scripts tests
python -m mypy src scripts tests
```

이 정리본에서는 CPU 회귀 테스트와 정적 검증을 수행했으며, 전체 GPU 실험을 재실행한 결과로 표의 수치를
제시하는 것은 아닙니다. 체크포인트·원 데이터는 포함되어 있지 않습니다.

**높은 MSE는 해당 공격기의 pixel 복원이 어렵다는 뜻이지, 모든 민감 정보가 보호된다는 증명은 아닙니다.**
ε를 formal joint-DP 보장으로 해석하거나, 답변 출력/네트워크 전체의 개인정보 보호가 구현됐다고 해석하지 않습니다.
자세한 실험 경계와 원 코드 provenance는 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)를 참고하세요.
