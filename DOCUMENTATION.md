# DermAgent 완전 가이드 문서

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [핵심 개념](#3-핵심-개념)
4. [상세 파일 설명](#4-상세-파일-설명)
5. [사용 방법](#5-사용-방법)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [실행 예제](#7-실행-예제)
8. [아키텍처 및 데이터 흐름](#8-아키텍처-및-데이터-흐름)
9. [확장 및 커스터마이징](#9-확장-및-커스터마이징)
10. [문제 해결](#10-문제-해결)

---

## 1. 프로젝트 개요

### 1.1 DermAgent란?

**DermAgent**는 AI 기반 피부과 진단 시스템으로, 다음을 결합한 고급 진단 프레임워크입니다:

- **계층적 질환 온톨로지**: 369개의 피부 질환을 5단계 계층 구조로 분류
- **Vision-Language Models (VLMs)**: GPT-4o, Qwen-VL, InternVL 등 최신 멀티모달 AI 모델
- **ReAct 패턴 에이전트**: Reasoning + Acting을 결합한 추론 방식
- **계층적 평가 메트릭**: 전통적인 정확도를 넘어선 계층적 평가 지표

### 1.2 주요 기능

| 기능 | 설명 |
|------|------|
| 🔬 **피부 이미지 분석** | VLM을 활용하여 피부 병변의 형태, 색상, 분포 등 임상적 특징 추출 |
| 🌳 **온톨로지 기반 진단** | 계층적 질환 트리를 탐색하며 체계적으로 진단 |
| 🤖 **ReAct 에이전트** | 생각-행동-관찰 사이클을 반복하며 추론 |
| 📊 **계층적 평가** | Exact Match, Hierarchical F1, Partial Credit 등 다양한 메트릭 |
| 🔄 **배치 처리** | CSV 파일로 대량의 이미지 일괄 처리 |

### 1.3 지원 VLM 모델

```
┌─────────────────────────────────────────────────────────┐
│                   지원 VLM 모델                          │
├─────────────────────────────────────────────────────────┤
│  MockVLM     │ 테스트용 더미 모델 (실제 이미지 분석 X)    │
│  GPT-4o      │ OpenAI의 최신 멀티모달 모델 (API 기반)    │
│  Qwen-VL     │ Alibaba의 Qwen2-VL 시리즈 (로컬 실행)   │
│  InternVL    │ OpenGVLab의 InternVL 시리즈 (로컬 실행)  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 프로젝트 구조

### 2.1 디렉토리 트리

```
DermAgent-wonjun/
│
├── 📄 project_path.py              # 전역 경로 설정 모듈
│
├── 📁 dataset/                     # 데이터셋 폴더
│   └── 📁 Derm1M/                  # Derm1M 데이터셋
│       ├── 📄 ontology.json        # ⭐ 핵심: 질환 온톨로지 (369개 질환)
│       ├── 📄 README.md            # 데이터셋 설명
│       └── 📁 random_samples_100/  # 테스트용 샘플 데이터
│           └── 📄 sampled_data.csv # 샘플 CSV
│
└── 📁 derm1m_exp/                  # 실험 코드
    │
    ├── 📁 DermAgent/               # ⭐ 메인 에이전트 프레임워크
    │   ├── 📄 README.md            # 사용 가이드
    │   │
    │   ├── 📁 eval/                # 평가 모듈
    │   │   ├── 📄 ontology_utils.py       # 온톨로지 트리 유틸리티
    │   │   ├── 📄 evaluation_metrics.py   # 계층적 평가 메트릭
    │   │   └── 📄 example_usage.py        # 사용 예제
    │   │
    │   └── 📁 agent/               # 에이전트 모듈
    │       ├── 📄 dermatology_agent.py    # 기본 진단 에이전트
    │       ├── 📄 react_agent.py          # ReAct 패턴 에이전트
    │       ├── 📄 pipeline.py             # 통합 파이프라인
    │       ├── 📄 run_agent.py            # CLI 실행 스크립트
    │       ├── 📄 compare_agents.py       # 에이전트 비교 도구
    │       └── 📄 run_comparison.sh       # 비교 실행 스크립트
    │
    └── 📁 baseline/                # 베이스라인 모델
        ├── 📄 model.py             # VLM 모델 래퍼 (Qwen, GPT, InternVL)
        ├── 📄 baseline.py          # 베이스라인 진단 실행
        ├── 📄 utils.py             # 유틸리티 함수
        ├── 📄 extract_nodes.py     # 온톨로지 노드 추출
        ├── 📄 extracted_node_names.txt  # 추출된 질환 목록
        ├── 📄 requirements.txt     # 의존성 목록
        └── 📁 outputs/             # 결과 저장 폴더
```

### 2.2 핵심 파일 역할

| 파일 | 역할 | 중요도 |
|------|------|--------|
| `ontology.json` | 369개 피부질환의 계층 구조 정의 | ⭐⭐⭐ |
| `ontology_utils.py` | 온톨로지 트리 탐색/조작 클래스 | ⭐⭐⭐ |
| `evaluation_metrics.py` | 계층적 평가 메트릭 구현 | ⭐⭐⭐ |
| `react_agent.py` | ReAct 패턴 에이전트 | ⭐⭐⭐ |
| `pipeline.py` | 통합 실행 파이프라인 | ⭐⭐ |
| `model.py` | VLM 모델 래퍼들 | ⭐⭐ |

---

## 3. 핵심 개념

### 3.1 질환 온톨로지 (Disease Ontology)

피부 질환을 **5단계 계층 구조**로 분류합니다:

```
Level 0: root (루트)
    │
    ├── Level 1: 대분류 (7개)
    │   ├── inflammatory (염증성)
    │   ├── proliferations (증식성)
    │   ├── hereditary (유전성)
    │   ├── exogenous (외인성)
    │   ├── reaction patterns (반응 패턴)
    │   ├── Hair diseases (모발 질환)
    │   └── Nail diseases (손발톱 질환)
    │
    ├── Level 2: 중분류
    │   └── 예: infectious (감염성), non-infectious (비감염성)
    │
    ├── Level 3: 소분류
    │   └── 예: fungal (진균성), bacterial (세균성), viral (바이러스성)
    │
    ├── Level 4: 세부 분류
    │   └── 예: Tinea corporis, Tinea pedis, Candidiasis
    │
    └── Level 5: 최하위 분류 (일부 질환)
```

**예시 경로**:
```
root → inflammatory → infectious → fungal → Tinea corporis
(루트)   (염증성)      (감염성)     (진균성)  (몸백선)
```

### 3.2 ReAct 패턴 (Reasoning + Acting)

에이전트가 **생각(Thought) → 행동(Action) → 관찰(Observation)** 사이클을 반복합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct 추론 사이클                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Step 1: 💭 Thought (생각)                                │
│    "이미지를 먼저 관찰하여 임상적 특징을 파악해야 한다"        │
│                    ↓                                        │
│    Step 2: 🔧 Action (행동)                                 │
│    observe_image(image_path, focus="all")                   │
│                    ↓                                        │
│    Step 3: 📋 Observation (관찰)                            │
│    {"morphology": ["plaque", "scaly"], "color": ["red"]}    │
│                    ↓                                        │
│    Step 4: 💭 Thought (생각)                                │
│    "붉은 비늘 모양 판은 염증성 질환을 시사한다"              │
│                    ↓                                        │
│              ... 반복 ...                                   │
│                    ↓                                        │
│    Final: 🎯 Conclude (결론)                                │
│    primary_diagnosis: "Tinea corporis"                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 계층적 평가 메트릭

전통적인 "맞다/틀리다" 평가를 넘어, **계층적 유사도**를 고려합니다:

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| **Exact Match** | 정확히 일치하는지 | 0 또는 1 |
| **Hierarchical F1** | 조상 집합 기반 Jaccard 유사도 | 0.0 ~ 1.0 |
| **Avg Min Distance** | 트리에서의 평균 최소 거리 | 0 ~ ∞ (낮을수록 좋음) |
| **Partial Credit** | 올바른 조상 예측에 대한 부분 점수 | 0.0 ~ 1.0 |
| **Level-wise Accuracy** | 각 레벨별 정확도 | 0.0 ~ 1.0 |
| **Ancestor Match** | 조상/자손 관계 점수 | 0.0 ~ 1.0 |

**예시**:
```
GT: "Tinea corporis"  (root → inflammatory → infectious → fungal → Tinea corporis)
Pred: "Tinea pedis"   (root → inflammatory → infectious → fungal → Tinea pedis)

결과:
- Exact Match: 0 (정확히 같지 않음)
- Hierarchical F1: 0.8 (대부분의 조상이 같음)
- Distance: 2 (Tinea corporis ↔ fungal ↔ Tinea pedis)
- Partial Credit: 0.75 (4개 레벨 중 3개 일치)
```

---

## 4. 상세 파일 설명

### 4.1 project_path.py

**역할**: 프로젝트 전역 경로 관리

```python
# 주요 경로 변수
PROJECT_PATH      # 프로젝트 루트 경로
DATASET_ROOT      # 데이터셋 폴더
DERM1M_ROOT       # Derm1M 데이터셋 폴더
ONTOLOGY_PATH     # ontology.json 경로
DERMAGENT_ROOT    # DermAgent 폴더
SAMPLED_DATA_CSV  # 샘플 데이터 CSV 경로
```

**사용 예시**:
```python
from project_path import ONTOLOGY_PATH, SAMPLED_DATA_CSV

# 온톨로지 로드
tree = OntologyTree(ONTOLOGY_PATH)

# 샘플 데이터 로드
df = pd.read_csv(SAMPLED_DATA_CSV)
```

### 4.2 ontology.json

**역할**: 369개 피부 질환의 계층적 분류 체계

**구조**:
```json
{
  "root": ["Hair diseases", "Nail diseases", "inflammatory", ...],
  "inflammatory": ["infectious", "non-infectious", "Dermatitis (acute and chronic)", ...],
  "infectious": ["bacterial", "fungal", "viral", "parasitic"],
  "fungal": ["Tinea", "Tinea corporis", "Tinea pedis", "Candidiasis", ...],
  "Tinea corporis": [],  // 리프 노드 (자식 없음)
  ...
}
```

**통계**:
- 총 노드 수: 369개
- 최대 깊이: 5 레벨
- 리프 노드 (실제 질환): 317개
- 루트 카테고리: 7개

### 4.3 ontology_utils.py (354줄)

**역할**: 온톨로지 트리 구조 관리

**핵심 클래스**: `OntologyTree`

```python
class OntologyTree:
    """온톨로지 트리 구조를 관리하는 클래스"""

    def __init__(self, ontology_path: Optional[str] = None):
        """
        Args:
            ontology_path: ontology.json 경로 (None이면 자동 탐색)
        """
```

**주요 메서드**:

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `get_path_to_root(node)` | 노드에서 루트까지 경로 | `List[str]` |
| `get_hierarchical_distance(n1, n2)` | 두 노드 간 거리 | `int` |
| `get_lca(n1, n2)` | 최소 공통 조상 | `str` |
| `get_children(node)` | 자식 노드들 | `List[str]` |
| `get_siblings(node)` | 형제 노드들 | `List[str]` |
| `get_depth(node)` | 노드 깊이 | `int` |
| `get_canonical_name(label)` | 정규화된 이름 | `str` |
| `is_valid_node(node)` | 유효 노드 확인 | `bool` |
| `filter_valid_labels(labels)` | 유효 라벨만 필터링 | `List[str]` |
| `get_all_descendants(node)` | 모든 자손 | `Set[str]` |
| `get_ancestors(node)` | 모든 조상 | `Set[str]` |
| `get_level_labels(node)` | 레벨별 라벨 | `Dict[int, str]` |
| `get_stats()` | 통계 정보 | `Dict` |

**사용 예시**:
```python
from ontology_utils import OntologyTree

# 자동 경로 탐색으로 초기화
tree = OntologyTree()

# 경로 조회
path = tree.get_path_to_root("Tinea corporis")
# ['Tinea corporis', 'fungal', 'infectious', 'inflammatory', 'root']

# 거리 계산
dist = tree.get_hierarchical_distance("Tinea corporis", "Psoriasis")
# 6 (서로 다른 분기)

# 공통 조상
lca = tree.get_lca("Tinea corporis", "Tinea pedis")
# 'fungal'

# 라벨 정규화
canonical = tree.get_canonical_name("TINEA CORPORIS")
# 'Tinea corporis'
```

### 4.4 evaluation_metrics.py (577줄)

**역할**: 계층적 평가 메트릭 구현

**핵심 클래스**: `HierarchicalEvaluator`

**데이터 클래스**:

```python
@dataclass
class PredictionResult:
    """단일 예측 결과"""
    ground_truth: List[str]      # 정답 라벨들
    prediction: List[str]        # 예측 라벨들
    raw_ground_truth: List[str]  # 원본 정답
    raw_prediction: List[str]    # 원본 예측

@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    exact_match: float                    # 정확히 일치 비율
    partial_match: float                  # 부분 일치 비율
    hierarchical_precision: float         # 계층적 정밀도
    hierarchical_recall: float            # 계층적 재현율
    hierarchical_f1: float                # 계층적 F1
    avg_hierarchical_distance: float      # 평균 계층 거리
    level_accuracy: Dict[int, float]      # 레벨별 정확도
    avg_partial_credit: float             # 평균 부분 점수
    total_samples: int                    # 전체 샘플 수
    valid_samples: int                    # 유효 샘플 수
```

**주요 메서드**:

| 메서드 | 설명 |
|--------|------|
| `evaluate_single(gt, pred)` | 단일 샘플 평가 |
| `evaluate_batch(gts, preds)` | 배치 평가 |
| `hierarchical_similarity(l1, l2)` | 계층적 유사도 계산 |
| `hierarchical_precision_recall(gt, pred)` | Precision/Recall/F1 |
| `avg_min_distance(gt, pred)` | 평균 최소 거리 |
| `partial_credit_score(gt, pred)` | 부분 점수 |
| `ancestor_match_score(gt, pred)` | 조상 일치 점수 |
| `level_match(gt, pred, level)` | 특정 레벨 일치 확인 |
| `print_evaluation_report(result)` | 평가 리포트 출력 |

**사용 예시**:
```python
from evaluation_metrics import HierarchicalEvaluator

evaluator = HierarchicalEvaluator()

# 단일 평가
result = evaluator.evaluate_single(
    gt_labels=["Tinea corporis"],
    pred_labels=["Tinea pedis"]
)
print(f"Hierarchical F1: {result['hierarchical_f1']:.4f}")

# 배치 평가
batch_result = evaluator.evaluate_batch(
    ground_truths=[["Tinea corporis"], ["Psoriasis"]],
    predictions=[["Tinea pedis"], ["Psoriasis"]]
)
evaluator.print_evaluation_report(batch_result)
```

### 4.5 dermatology_agent.py (772줄)

**역할**: 기본 피부과 진단 에이전트

**핵심 클래스 및 구조**:

```python
class DiagnosisStep(Enum):
    """진단 단계"""
    INITIAL_ASSESSMENT = "initial_assessment"        # 초기 평가
    CATEGORY_CLASSIFICATION = "category_classification"  # 대분류
    SUBCATEGORY_CLASSIFICATION = "subcategory_classification"  # 중분류
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"  # 감별 진단
    FINAL_DIAGNOSIS = "final_diagnosis"              # 최종 진단

@dataclass
class ObservationResult:
    """관찰 결과"""
    morphology: List[str]    # 형태학적 특징 (papule, plaque 등)
    color: List[str]         # 색상 (red, brown 등)
    distribution: List[str]  # 분포 패턴 (localized, generalized 등)
    location: str            # 신체 위치
    surface: List[str]       # 표면 특징 (scaly, crusted 등)
    symptoms: List[str]      # 증상
    raw_description: str     # 원본 설명

@dataclass
class DiagnosisState:
    """진단 상태 추적"""
    current_step: DiagnosisStep           # 현재 단계
    current_path: List[str]               # 온톨로지 경로
    candidates: List[str]                 # 후보 질환들
    confidence_scores: Dict[str, float]   # 신뢰도 점수
    observations: ObservationResult       # 관찰 결과
    reasoning_history: List[Dict]         # 추론 이력
    final_diagnosis: List[str]            # 최종 진단
```

**도구 클래스**:

```python
class BaseTool(ABC):
    """도구 기본 클래스"""
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def execute(self, *args, **kwargs) -> Any: ...

class OntologyNavigator(BaseTool):
    """온톨로지 탐색 도구"""
    # Actions: get_children, get_path, get_siblings, get_descendants, validate

class DifferentialDiagnosisTool(BaseTool):
    """감별 진단 도구 (VLM 기반)"""
    # VLM을 사용하여 후보 질환들과 관찰 결과 비교
```

**DermatologyAgent 클래스**:

```python
class DermatologyAgent:
    """피부과 진단 에이전트"""

    def __init__(
        self,
        ontology_path: Optional[str] = None,  # 자동 탐색
        vlm_model: Any = None,                # VLM 모델
        verbose: bool = True                  # 상세 출력
    ): ...

    def diagnose(
        self,
        image_path: str,
        max_depth: int = 4
    ) -> Dict[str, Any]:
        """전체 진단 파이프라인 실행"""
        # Step 1: 초기 평가
        # Step 2: 대분류 (Level 1)
        # Step 3: 중분류/소분류 (Level 2-4)
        # Step 4: 감별 진단
        # Step 5: 최종 진단
```

### 4.6 react_agent.py (842줄)

**역할**: ReAct 패턴 기반 고급 진단 에이전트

**핵심 데이터 구조**:

```python
class ActionType(Enum):
    """에이전트 행동 유형"""
    OBSERVE = "observe"         # 이미지 관찰
    NAVIGATE = "navigate"       # 온톨로지 탐색
    COMPARE = "compare"         # 후보 비교
    VERIFY = "verify"           # 진단 검증
    CONCLUDE = "conclude"       # 결론 도출
    ASK_CLARIFICATION = "ask"   # 추가 정보 요청

@dataclass
class Observation:
    """상세 관찰 결과"""
    morphology: List[str]
    color: List[str]
    distribution: List[str]
    surface: List[str]
    border: List[str]
    location: str
    size: str
    symptoms: List[str]
    duration: str
    patient_info: Dict[str, Any]
    confidence: float
    raw_text: str

@dataclass
class ThoughtStep:
    """사고 단계 기록"""
    step_num: int           # 단계 번호
    thought: str            # 현재 생각
    action: ActionType      # 수행할 행동
    action_input: Dict      # 행동 입력
    observation: str        # 행동 결과

@dataclass
class DiagnosisResult:
    """최종 진단 결과"""
    primary_diagnosis: str              # 주요 진단
    differential_diagnoses: List[str]   # 감별 진단 목록
    confidence: float                   # 신뢰도
    ontology_path: List[str]            # 온톨로지 경로
    observations: Observation           # 관찰 결과
    reasoning_chain: List[ThoughtStep]  # 추론 체인
    verification_passed: bool           # 검증 통과 여부
    warnings: List[str]                 # 경고 메시지
```

**도구 클래스**:

| 도구 | 역할 |
|------|------|
| `ObserveTool` | 이미지에서 임상적 특징 추출 |
| `NavigateOntologyTool` | 온톨로지 트리 탐색 |
| `CompareCandidatesTool` | VLM 기반 후보 질환 비교 |
| `VerifyDiagnosisTool` | 진단 결과 검증 |

**ReActDermatologyAgent 클래스**:

```python
class ReActDermatologyAgent:
    """ReAct 패턴 기반 진단 에이전트"""

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model = None,
        max_steps: int = 10,    # 최대 추론 단계
        verbose: bool = True
    ): ...

    def diagnose(self, image_path: str) -> DiagnosisResult:
        """ReAct 루프로 진단 수행"""
        # 1. 초기화
        # 2. 반복: Thought → Action → Observation
        # 3. Conclude 액션 시 종료
        # 4. DiagnosisResult 반환
```

### 4.7 pipeline.py (540줄)

**역할**: 통합 진단 파이프라인

**VLM 팩토리**:

```python
class VLMFactory:
    """VLM 모델 생성 팩토리"""
    @staticmethod
    def create(model_type: str, **kwargs):
        if model_type == "mock": return MockVLM()
        elif model_type == "gpt": return GPT4oVLM(api_key=...)
        elif model_type == "qwen": return QwenVLM(model_path=...)
        elif model_type == "internvl": return InternVLM(model_path=...)
```

**파이프라인 설정**:

```python
@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    ontology_path: Optional[str] = None   # 자동 탐색
    model_type: str = "mock"              # 모델 유형
    model_path: Optional[str] = None      # 모델 경로
    api_key: Optional[str] = None         # API 키
    max_steps: int = 8                    # 최대 단계
    max_depth: int = 4                    # 최대 깊이
    verbose: bool = True                  # 상세 출력
    save_reasoning: bool = True           # 추론 저장
```

**DiagnosisPipeline 클래스**:

```python
class DiagnosisPipeline:
    """통합 진단 파이프라인"""

    def diagnose_single(self, image_path: str) -> Dict
    def diagnose_batch(self, data: List[Dict], image_base_dir: str) -> List[Dict]
    def evaluate(self, results: List[Dict]) -> Dict
    def run(self, input_path: str, output_path: str, ...) -> Dict
```

### 4.8 model.py (623줄)

**역할**: VLM 모델 래퍼

**지원 모델**:

```python
class QwenVL(BaselineModel):
    """Qwen-VL 래퍼"""
    def __init__(self, model_path: str):
        # Qwen2-VL 모델 로드
        # Flash Attention 2 지원 (가능한 경우)

    def chat_img(self, input_text, image_path, max_tokens=512) -> str
    def chat_video(self, input_text, video_path, max_tokens=512) -> str

class InternVL(BaselineModel):
    """InternVL 래퍼"""
    def __init__(self, model_path: str):
        # InternVL 모델 로드

    def chat_img(self, input_text, image_path, max_tokens=512) -> str
    def chat_video(self, input_text, video_path, max_tokens=512) -> str

class GPT4o(BaselineModel):
    """GPT-4o 래퍼 (OpenAI API)"""
    def __init__(self, api_key: str):
        # OpenAI 클라이언트 초기화

    def chat_img(self, input_text, image_path, max_tokens=512) -> str
    def chat_video(self, input_text, video_path, max_tokens=512) -> str
```

**공통 특징**:
- 시스템 명령어에 질환 목록 포함 (`extracted_node_names.txt`)
- 이미지 및 비디오 처리 지원
- 오류 처리 및 이미지 압축

---

## 5. 사용 방법

### 5.1 설치

```bash
# 의존성 설치
pip install -r derm1m_exp/baseline/requirements.txt

# 주요 패키지:
# - torch, transformers
# - openai (GPT-4o용)
# - qwen-vl-utils (Qwen용)
# - numpy, tqdm, pandas, Pillow, opencv-python
```

### 5.2 기본 사용법

#### 방법 1: 평가 시스템만 사용

```python
from derm1m_exp.DermAgent.eval.ontology_utils import OntologyTree
from derm1m_exp.DermAgent.eval.evaluation_metrics import HierarchicalEvaluator

# 1. 온톨로지 트리 로드 (자동 경로)
tree = OntologyTree()

# 2. 트리 탐색
path = tree.get_path_to_root("Tinea corporis")
children = tree.get_children("fungal")
dist = tree.get_hierarchical_distance("Tinea corporis", "Psoriasis")

# 3. 평가 수행
evaluator = HierarchicalEvaluator()
result = evaluator.evaluate_batch(
    ground_truths=[["Tinea corporis"]],
    predictions=[["Tinea pedis"]]
)
evaluator.print_evaluation_report(result)
```

#### 방법 2: 에이전트로 진단

```python
from derm1m_exp.DermAgent.agent.react_agent import ReActDermatologyAgent

# MockVLM으로 테스트
agent = ReActDermatologyAgent(vlm_model=None, verbose=True)
result = agent.diagnose("/path/to/skin_image.jpg")

print(f"진단: {result.primary_diagnosis}")
print(f"신뢰도: {result.confidence}")
print(f"경로: {' → '.join(result.ontology_path)}")
```

#### 방법 3: 파이프라인으로 배치 처리

```python
from derm1m_exp.DermAgent.agent.pipeline import DiagnosisPipeline, PipelineConfig

config = PipelineConfig(
    model_type="mock",
    max_steps=8,
    verbose=True
)

pipeline = DiagnosisPipeline(config)
pipeline.run(
    input_path="data.csv",
    output_path="results.json",
    image_base_dir="/path/to/images"
)
```

### 5.3 CLI 사용법

#### Demo 모드 실행

```bash
# 평가 시스템 데모
cd derm1m_exp/DermAgent/eval
python example_usage.py

# 에이전트 데모
cd derm1m_exp/DermAgent/agent
python run_agent.py --demo --verbose

# 파이프라인 데모
python pipeline.py --demo
```

#### 실제 데이터로 실행

```bash
# Mock VLM으로 테스트
python run_agent.py \
    --input_csv /path/to/data.csv \
    --image_dir /path/to/images \
    --output results.json \
    --model mock \
    --verbose

# GPT-4o로 실행
python run_agent.py \
    --input_csv /path/to/data.csv \
    --image_dir /path/to/images \
    --output gpt_results.json \
    --model gpt \
    --api_key YOUR_API_KEY

# Qwen-VL로 실행
CUDA_VISIBLE_DEVICES=0,1 python run_agent.py \
    --input_csv /path/to/data.csv \
    --image_dir /path/to/images \
    --output qwen_results.json \
    --model qwen \
    --model_path Qwen/Qwen2-VL-7B-Instruct
```

#### Pipeline CLI

```bash
python pipeline.py \
    --input data.csv \
    --output results.json \
    --image_dir /path/to/images \
    --model gpt \
    --api_key YOUR_KEY \
    --max_steps 8 \
    --verbose
```

---

## 6. API 레퍼런스

### 6.1 OntologyTree

```python
class OntologyTree:
    """
    온톨로지 트리 관리 클래스

    Attributes:
        ontology: Dict[str, List[str]] - 온톨로지 딕셔너리
        ontology_path: str - 로드된 파일 경로
        parent_map: Dict[str, str] - 자식→부모 매핑
        valid_nodes: Set[str] - 모든 유효 노드
        normalized_map: Dict[str, str] - 정규화된 이름 매핑
    """

    def __init__(self, ontology_path: Optional[str] = None) -> None:
        """
        Args:
            ontology_path: ontology.json 경로. None이면 자동 탐색.

        Raises:
            FileNotFoundError: 온톨로지 파일을 찾을 수 없을 때
        """

    def get_path_to_root(self, node: str) -> List[str]:
        """
        노드에서 루트까지의 경로 반환

        Args:
            node: 시작 노드 이름

        Returns:
            [node, parent, grandparent, ..., root]
            유효하지 않은 노드면 []
        """

    def get_hierarchical_distance(self, node1: str, node2: str) -> int:
        """
        두 노드 간 계층적 거리

        Args:
            node1, node2: 비교할 노드들

        Returns:
            거리 값. 유효하지 않으면 -1
        """

    def get_lca(self, node1: str, node2: str) -> Optional[str]:
        """최소 공통 조상 (Lowest Common Ancestor)"""

    def get_children(self, node: str) -> List[str]:
        """직계 자식 노드들"""

    def get_siblings(self, node: str) -> List[str]:
        """형제 노드들 (자기 자신 제외)"""

    def get_depth(self, node: str) -> int:
        """노드 깊이 (root=0)"""

    def get_canonical_name(self, node: str) -> Optional[str]:
        """정규화된 이름 반환 (대소문자, 공백 처리)"""

    def is_valid_node(self, node: str) -> bool:
        """유효한 노드인지 확인"""

    def filter_valid_labels(self, labels: List[str]) -> List[str]:
        """유효한 라벨만 필터링"""

    def get_ancestors(self, node: str) -> Set[str]:
        """모든 조상 (자기 자신 제외)"""

    def get_all_descendants(self, node: str) -> Set[str]:
        """모든 자손"""

    def get_level_labels(self, node: str) -> Dict[int, str]:
        """레벨별 라벨 {level: label}"""

    def get_stats(self) -> Dict:
        """온톨로지 통계"""
```

### 6.2 HierarchicalEvaluator

```python
class HierarchicalEvaluator:
    """
    계층적 평가기

    Attributes:
        tree: OntologyTree - 온톨로지 트리
        max_depth: int - 최대 깊이
    """

    def __init__(self, ontology_path: Optional[str] = None) -> None:
        """
        Args:
            ontology_path: 온톨로지 경로 (자동 탐색)
        """

    def evaluate_single(
        self,
        gt_labels: List[str],
        pred_labels: List[str]
    ) -> Dict[str, float]:
        """
        단일 샘플 평가

        Returns:
            {
                'valid': bool,
                'exact_match': float,
                'partial_match': float,
                'hierarchical_precision': float,
                'hierarchical_recall': float,
                'hierarchical_f1': float,
                'avg_min_distance': float,
                'partial_credit': float,
                'ancestor_match': float
            }
        """

    def evaluate_batch(
        self,
        ground_truths: List[List[str]],
        predictions: List[List[str]]
    ) -> EvaluationResult:
        """
        배치 평가

        Args:
            ground_truths: 각 샘플의 GT 라벨 리스트들
            predictions: 각 샘플의 예측 라벨 리스트들

        Returns:
            EvaluationResult 객체
        """

    def hierarchical_similarity(self, label1: str, label2: str) -> float:
        """
        두 라벨의 계층적 유사도 (Jaccard 기반)

        Returns:
            0.0 ~ 1.0
        """

    def print_evaluation_report(self, result: EvaluationResult) -> None:
        """평가 결과 출력"""
```

### 6.3 ReActDermatologyAgent

```python
class ReActDermatologyAgent:
    """
    ReAct 패턴 진단 에이전트

    Attributes:
        tree: OntologyTree
        vlm: VLM 모델 (None이면 Mock)
        max_steps: int - 최대 추론 단계
        verbose: bool
        tools: Dict[str, Tool]
    """

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model = None,
        max_steps: int = 10,
        verbose: bool = True
    ) -> None: ...

    def diagnose(self, image_path: str) -> DiagnosisResult:
        """
        이미지 진단 수행

        Args:
            image_path: 피부 이미지 경로

        Returns:
            DiagnosisResult 객체
        """

    def diagnose_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[DiagnosisResult]:
        """배치 진단"""
```

### 6.4 DiagnosisPipeline

```python
class DiagnosisPipeline:
    """
    통합 진단 파이프라인
    """

    def __init__(self, config: PipelineConfig) -> None: ...

    def diagnose_single(self, image_path: str) -> Dict[str, Any]:
        """단일 이미지 진단"""

    def diagnose_batch(
        self,
        data: List[Dict[str, Any]],
        image_base_dir: str = ""
    ) -> List[Dict[str, Any]]:
        """배치 진단"""

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 평가"""

    def run(
        self,
        input_path: str,
        output_path: str,
        image_base_dir: str = "",
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
```

---

## 7. 실행 예제

### 7.1 온톨로지 탐색

```python
from derm1m_exp.DermAgent.eval.ontology_utils import OntologyTree

tree = OntologyTree()

# 통계 확인
stats = tree.get_stats()
print(f"총 질환 수: {stats['total_nodes']}")
print(f"리프 노드 (실제 질환): {stats['leaf_nodes']}")

# 루트 카테고리
root_cats = tree.get_children("root")
print(f"\n대분류: {root_cats}")

# 특정 질환의 계층 경로
path = tree.get_path_to_root("Tinea corporis")
print(f"\nTinea corporis 경로: {' → '.join(path)}")

# 레벨별 라벨
levels = tree.get_level_labels("Tinea corporis")
for level, label in levels.items():
    print(f"  Level {level}: {label}")

# 형제 질환 (같은 부모)
siblings = tree.get_siblings("Tinea corporis")
print(f"\n형제 질환: {siblings[:5]}")

# 두 질환 간 거리
dist = tree.get_hierarchical_distance("Tinea corporis", "Psoriasis")
lca = tree.get_lca("Tinea corporis", "Psoriasis")
print(f"\nTinea corporis ↔ Psoriasis:")
print(f"  거리: {dist}")
print(f"  공통 조상: {lca}")
```

### 7.2 평가 메트릭 계산

```python
from derm1m_exp.DermAgent.eval.evaluation_metrics import HierarchicalEvaluator

evaluator = HierarchicalEvaluator()

# 케이스 1: 정확히 일치
result1 = evaluator.evaluate_single(["Tinea corporis"], ["Tinea corporis"])
print(f"정확히 일치:")
print(f"  Exact Match: {result1['exact_match']}")
print(f"  Hierarchical F1: {result1['hierarchical_f1']:.4f}")

# 케이스 2: 같은 부모 (fungal)
result2 = evaluator.evaluate_single(["Tinea corporis"], ["Tinea pedis"])
print(f"\n같은 부모 (fungal):")
print(f"  Exact Match: {result2['exact_match']}")
print(f"  Hierarchical F1: {result2['hierarchical_f1']:.4f}")
print(f"  Distance: {result2['avg_min_distance']}")

# 케이스 3: 다른 분기
result3 = evaluator.evaluate_single(["Tinea corporis"], ["Psoriasis"])
print(f"\n다른 분기:")
print(f"  Exact Match: {result3['exact_match']}")
print(f"  Hierarchical F1: {result3['hierarchical_f1']:.4f}")
print(f"  Distance: {result3['avg_min_distance']}")

# 배치 평가
ground_truths = [
    ["Tinea corporis"],
    ["Psoriasis"],
    ["Acne vulgaris"]
]
predictions = [
    ["Tinea pedis"],
    ["Psoriasis"],
    ["Rosacea"]
]

batch_result = evaluator.evaluate_batch(ground_truths, predictions)
evaluator.print_evaluation_report(batch_result)
```

### 7.3 Mock 에이전트 진단

```python
from derm1m_exp.DermAgent.agent.react_agent import ReActDermatologyAgent

# Mock VLM으로 에이전트 생성
agent = ReActDermatologyAgent(
    vlm_model=None,  # None = Mock VLM
    max_steps=6,
    verbose=True
)

# 진단 실행
result = agent.diagnose("/fake/image.jpg")

# 결과 출력
print(f"\n=== 진단 결과 ===")
print(f"주요 진단: {result.primary_diagnosis}")
print(f"감별 진단: {result.differential_diagnoses}")
print(f"신뢰도: {result.confidence}")
print(f"온톨로지 경로: {' → '.join(result.ontology_path)}")

# 추론 체인 출력
print(f"\n=== 추론 과정 ===")
for step in result.reasoning_chain:
    print(f"Step {step.step_num}:")
    print(f"  생각: {step.thought[:100]}...")
    print(f"  행동: {step.action.value}")
```

### 7.4 실제 VLM으로 진단 (GPT-4o)

```python
import os
from derm1m_exp.DermAgent.agent.pipeline import DiagnosisPipeline, PipelineConfig

# 설정
config = PipelineConfig(
    model_type="gpt",
    api_key=os.environ["OPENAI_API_KEY"],
    max_steps=8,
    verbose=True,
    save_reasoning=True
)

# 파이프라인 생성
pipeline = DiagnosisPipeline(config)

# 단일 이미지 진단
result = pipeline.diagnose_single("/path/to/skin_image.jpg")
print(f"진단: {result['primary_diagnosis']}")
print(f"신뢰도: {result['confidence']}")

# 배치 처리
data = [
    {"filename": "img1.jpg", "disease_label": "Tinea corporis"},
    {"filename": "img2.jpg", "disease_label": "Psoriasis"},
]

results = pipeline.diagnose_batch(data, image_base_dir="/path/to/images")
evaluation = pipeline.evaluate(results)

print(f"\n=== 평가 결과 ===")
print(f"Exact Match: {evaluation['exact_match']:.4f}")
print(f"Hierarchical F1: {evaluation['hierarchical_f1']:.4f}")
```

### 7.5 전체 파이프라인 실행

```bash
# CSV 형식 (data.csv):
# filename,disease_label
# image001.jpg,Tinea corporis
# image002.jpg,Psoriasis

python derm1m_exp/DermAgent/agent/pipeline.py \
    --input data.csv \
    --output results.json \
    --image_dir /path/to/images \
    --model gpt \
    --api_key $OPENAI_API_KEY \
    --max_steps 8 \
    --verbose
```

---

## 8. 아키텍처 및 데이터 흐름

### 8.1 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 입력                              │
│                     (이미지 + CSV 데이터)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DiagnosisPipeline                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   VLMFactory    │  │  ReActAgent     │  │  Evaluator      │ │
│  │  (모델 생성)     │  │  (진단 수행)     │  │  (결과 평가)     │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌───────────────────┐  ┌─────────────────────┐  ┌─────────────────┐
│     VLM Models    │  │   OntologyTree      │  │ HierarchicalEval│
│  ┌─────────────┐  │  │                     │  │                 │
│  │   MockVLM   │  │  │  - get_path_to_root │  │ - exact_match   │
│  │   GPT-4o    │  │  │  - get_children     │  │ - hier_f1       │
│  │   Qwen-VL   │  │  │  - get_distance     │  │ - partial_credit│
│  │   InternVL  │  │  │  - get_lca          │  │ - level_acc     │
│  └─────────────┘  │  │                     │  │                 │
└───────────────────┘  └─────────────────────┘  └─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    ontology.json      │
                    │    (369 diseases)     │
                    └───────────────────────┘
```

### 8.2 진단 데이터 흐름

```
입력: 피부 이미지
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: 이미지 관찰                          │
│  VLM이 이미지를 분석하여 임상적 특징 추출                         │
│  - morphology: ["papule", "plaque", "scaly"]                    │
│  - color: ["red", "erythematous"]                               │
│  - distribution: ["localized", "asymmetric"]                    │
│  - location: "trunk"                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 2: 온톨로지 탐색                         │
│  계층적 분류 수행:                                               │
│  root → inflammatory → infectious → fungal                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: 후보 비교                            │
│  VLM이 후보 질환들과 관찰 결과 비교:                             │
│  - Tinea corporis: 0.85 (supporting: annular, scaly)            │
│  - Psoriasis: 0.45 (contradicting: no silvery scales)           │
│  - Nummular eczema: 0.55 (partial match)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 4: 검증 및 결론                          │
│  최종 진단 결정:                                                 │
│  - primary_diagnosis: "Tinea corporis"                          │
│  - confidence: 0.85                                             │
│  - differential_diagnoses: ["Nummular eczema", "Psoriasis"]     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
출력: DiagnosisResult
```

### 8.3 ReAct 추론 사이클

```
┌────────────────────────────────────────────────────────────────┐
│                      ReAct 추론 루프                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1                                                   │  │
│  │  💭 Thought: "이미지를 관찰하여 특징을 파악해야 한다"      │  │
│  │  🔧 Action: observe_image                                 │  │
│  │  📋 Observation: {morphology: ["plaque"], color: ["red"]} │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 2                                                   │  │
│  │  💭 Thought: "붉은 판형 병변은 염증성 질환을 시사한다"     │  │
│  │  🔧 Action: navigate_ontology(get_children, "root")       │  │
│  │  📋 Observation: ["inflammatory", "proliferations", ...]  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 3                                                   │  │
│  │  💭 Thought: "염증성 카테고리의 하위를 탐색해야 한다"      │  │
│  │  🔧 Action: navigate_ontology(get_children, "inflammatory")│  │
│  │  📋 Observation: ["infectious", "non-infectious", ...]    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│                         ...                                    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step N (Final)                                           │  │
│  │  💭 Thought: "충분한 정보를 수집했다. 결론을 내린다"       │  │
│  │  🔧 Action: conclude                                      │  │
│  │  📋 Result: {primary: "Tinea corporis", confidence: 0.85} │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 9. 확장 및 커스터마이징

### 9.1 새로운 VLM 모델 추가

```python
# derm1m_exp/DermAgent/agent/pipeline.py에 추가

class MyCustomVLM:
    """커스텀 VLM 래퍼"""

    def __init__(self, model_path: str):
        # 모델 로드
        pass

    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        """이미지와 함께 대화"""
        # 구현
        pass

# VLMFactory에 등록
class VLMFactory:
    @staticmethod
    def create(model_type: str, **kwargs):
        # ...기존 코드...
        elif model_type == "custom":
            return MyCustomVLM(model_path=kwargs.get("model_path"))
```

### 9.2 새로운 평가 메트릭 추가

```python
# derm1m_exp/DermAgent/eval/evaluation_metrics.py에 추가

class HierarchicalEvaluator:
    # ...기존 코드...

    def my_custom_metric(self, gt_labels: List[str], pred_labels: List[str]) -> float:
        """커스텀 메트릭"""
        # 구현
        pass

    def evaluate_single(self, gt_labels, pred_labels) -> Dict:
        result = {
            # ...기존 메트릭...
            'my_custom_metric': self.my_custom_metric(gt_valid, pred_valid),
        }
        return result
```

### 9.3 새로운 도구 추가 (ReAct 에이전트용)

```python
# derm1m_exp/DermAgent/agent/react_agent.py에 추가

class MyCustomTool(Tool):
    """커스텀 도구"""

    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "설명..."

    @property
    def parameters(self) -> Dict:
        return {"param1": "설명1", "param2": "설명2"}

    def run(self, param1: str, param2: str) -> str:
        # 구현
        return json.dumps({"result": "..."})

# 에이전트에 등록
class ReActDermatologyAgent:
    def _init_tools(self):
        self.tools = {
            # ...기존 도구...
            "my_custom_tool": MyCustomTool(),
        }
```

### 9.4 온톨로지 수정

```python
import json

# ontology.json 로드
with open("dataset/Derm1M/ontology.json", "r") as f:
    ontology = json.load(f)

# 새 질환 추가
ontology["fungal"].append("My New Disease")
ontology["My New Disease"] = []  # 리프 노드

# 저장
with open("dataset/Derm1M/ontology.json", "w") as f:
    json.dump(ontology, f, indent=2, ensure_ascii=False)
```

---

## 10. 문제 해결

### 10.1 자주 발생하는 오류

#### FileNotFoundError: ontology.json

```
Error: ontology.json 파일을 찾을 수 없습니다.
```

**해결방법**:
1. 프로젝트 구조 확인
2. 직접 경로 지정:
```python
tree = OntologyTree("/full/path/to/ontology.json")
```

#### CUDA Out of Memory

```
CUDA out of memory
```

**해결방법**:
1. 더 작은 모델 사용
2. `CUDA_VISIBLE_DEVICES` 설정
3. 배치 크기 줄이기

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py --model qwen --model_path Qwen/Qwen2-VL-2B-Instruct
```

#### API 오류 (GPT-4o)

```
openai.APIError: ...
```

**해결방법**:
1. API 키 확인
2. 요청 한도 확인
3. 이미지 크기 줄이기

```python
# 이미지 압축
from derm1m_exp.baseline.utils import compress_image
compressed = compress_image("large_image.jpg", "small.jpg", quality=50)
```

### 10.2 성능 최적화

#### 메모리 최적화

```python
# Flash Attention 활성화
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

#### 배치 처리

```python
# 대량 이미지 처리 시
pipeline.run(
    input_path="data.csv",
    output_path="results.json",
    limit=100  # 한 번에 처리할 수 제한
)
```

### 10.3 디버깅 팁

#### 상세 로그 활성화

```python
agent = ReActDermatologyAgent(verbose=True)
pipeline = DiagnosisPipeline(PipelineConfig(verbose=True))
```

#### 추론 체인 분석

```python
result = agent.diagnose(image_path)

for step in result.reasoning_chain:
    print(f"\n=== Step {step.step_num} ===")
    print(f"Thought: {step.thought}")
    print(f"Action: {step.action.value}")
    print(f"Input: {step.action_input}")
    print(f"Observation: {step.observation[:200]}...")
```

---

## 부록: 질환 온톨로지 전체 구조

### 대분류 (Level 1)

| 카테고리 | 설명 | 하위 질환 수 |
|----------|------|-------------|
| inflammatory | 염증성 질환 | 180+ |
| proliferations | 증식성 질환 | 100+ |
| hereditary | 유전성 질환 | 15+ |
| exogenous | 외인성 질환 | 25+ |
| reaction patterns | 반응 패턴 | 40+ |
| Hair diseases | 모발 질환 | 5 |
| Nail diseases | 손발톱 질환 | 11 |

### inflammatory (염증성) 하위 구조

```
inflammatory
├── infectious (감염성)
│   ├── bacterial (세균성)
│   │   ├── Impetigo
│   │   ├── Cellulitis
│   │   ├── Folliculitis
│   │   └── ...
│   ├── fungal (진균성)
│   │   ├── Tinea corporis
│   │   ├── Tinea pedis
│   │   ├── Candidiasis
│   │   └── ...
│   ├── viral (바이러스성)
│   │   ├── Herpes simplex
│   │   ├── Herpes zoster
│   │   ├── Molluscum contagiosum
│   │   └── ...
│   └── parasitic (기생충성)
│       ├── Scabies
│       ├── Pediculosis
│       └── ...
└── non-infectious (비감염성)
    ├── Psoriasis
    ├── Eczema
    ├── Lichen planus
    └── ...
```

---

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.
Derm1M 데이터셋은 CC BY-NC 4.0 라이선스를 따릅니다.

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-12-05
**작성자**: AI Assistant
