"""
Derm1M_v2_pretrain.csv의 첫 번째 샘플로 GPT-4o 에이전트 테스트
"""

import os
import sys
import csv
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "agent"))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from dermatology_agent import DermatologyAgent

# 로그 및 결과 파일 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = SCRIPT_DIR / f"gpt4o_pretrain_test_{timestamp}.log"
json_file = SCRIPT_DIR / f"gpt4o_pretrain_result_{timestamp}.json"

# 로깅 설정 (파일 + 콘솔)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# .env 파일에서 API 키 로드
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    logger.error("❌ Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

logger.info(f"✓ API Key loaded: {api_key[:20]}...")
logger.info(f"✓ Log file: {log_file}")
logger.info(f"✓ JSON file: {json_file}\n")

# OpenAI 클라이언트 초기화
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    logger.info("✓ OpenAI client initialized")
except ImportError:
    logger.error("❌ Error: openai package not installed")
    sys.exit(1)


class GPT4oVLM:
    """실제 GPT-4o Vision 모델"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat_img(self, prompt: str, image_paths: list, max_tokens: int = 1024) -> str:
        """GPT-4o Vision API 호출"""
        image_path = image_paths[0] if image_paths else None

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"⚠️  이미지 없음, 텍스트만으로 응답: {image_path}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content

        # 이미지 인코딩
        base64_image = self._encode_image(image_path)

        logger.info(f"\n{'='*80}")
        logger.info(f"📤 GPT-4o에 요청 중...")
        logger.info(f"{'='*80}")
        logger.info(f"프롬프트 (첫 200자):\n{prompt[:200]}...")
        logger.info(f"이미지: {image_path}")
        logger.info(f"{'='*80}\n")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            logger.info(f"\n{'='*80}")
            logger.info(f"📥 GPT-4o 응답:")
            logger.info(f"{'='*80}")
            logger.info(answer)
            logger.info(f"{'='*80}\n")

            return answer

        except Exception as e:
            logger.error(f"\n❌ GPT-4o API 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return "{}"


# CSV 파일 경로
csv_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "Derm1M" / "Derm1M_v2_pretrain.csv"
image_base_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / "Derm1M"

logger.info(f"✓ CSV path: {csv_path}")
logger.info(f"✓ Image base dir: {image_base_dir}\n")

# CSV에서 첫 번째 행 읽기 (BOM 처리)
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    first_row = next(reader)

logger.info("="*80)
logger.info("첫 번째 샘플 정보 (Derm1M_v2_pretrain.csv)")
logger.info("="*80)
logger.info(f"Filename: {first_row['filename']}")
logger.info(f"Disease Label (GT): {first_row.get('disease_label', 'N/A')}")
logger.info(f"Hierarchical Label: {first_row.get('hierarchical_disease_label', 'N/A')}")

# Caption이 있으면 출력
if 'caption' in first_row and first_row['caption']:
    logger.info(f"Caption: {first_row['caption'][:150]}...")
if 'truncated_caption' in first_row and first_row['truncated_caption']:
    logger.info(f"Truncated Caption: {first_row['truncated_caption'][:150]}...")

# 추가 정보
if 'body_location' in first_row:
    logger.info(f"Body Location: {first_row.get('body_location', 'N/A')}")
if 'symptoms' in first_row:
    logger.info(f"Symptoms: {first_row.get('symptoms', 'N/A')}")

logger.info("="*80)

# 이미지 경로 구성
image_path = image_base_dir / first_row['filename']
logger.info(f"\n이미지 경로: {image_path}")

# 이미지 파일 존재 확인
if not image_path.exists():
    logger.warning(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")

    # 다른 가능한 경로들 확인
    possible_paths = [
        image_base_dir / "images" / first_row['filename'],
        Path("/home/work/wonjun/DermAgent/dataset/Derm1M") / first_row['filename'],
    ]

    for p in possible_paths:
        if p.exists():
            image_path = p
            logger.info(f"✓ 찾음: {image_path}")
            break

    if not image_path.exists():
        logger.error("❌ 이미지를 찾을 수 없습니다.")
        sys.exit(1)
else:
    logger.info(f"✓ 이미지 파일 존재 확인\n")

# GPT-4o VLM 초기화
logger.info("="*80)
logger.info("GPT-4o 에이전트 초기화 중...")
logger.info("="*80)

vlm = GPT4oVLM(api_key)

# 에이전트 생성
agent = DermatologyAgent(
    ontology_path=None,
    vlm_model=vlm,
    verbose=True
)

logger.info("\n" + "="*80)
logger.info("진단 시작 (실제 GPT-4o 사용)")
logger.info("="*80)

# 진단 실행
try:
    result = agent.diagnose(str(image_path), max_depth=4)

    logger.info("\n" + "="*80)
    logger.info("진단 결과")
    logger.info("="*80)

    import json
    logger.info(json.dumps(result, indent=2, ensure_ascii=False))

    logger.info("\n" + "="*80)
    logger.info("요약")
    logger.info("="*80)
    logger.info(f"Ground Truth: {first_row.get('disease_label', 'N/A')}")
    logger.info(f"최종 진단: {result['final_diagnosis']}")
    logger.info(f"진단 경로: {' → '.join(result['diagnosis_path'])}")
    logger.info(f"주요 관찰: {result['observations']}")

    # 추론 과정 출력
    logger.info("\n" + "="*80)
    logger.info("추론 과정 (Reasoning History)")
    logger.info("="*80)
    for i, step in enumerate(result['reasoning_history'], 1):
        logger.info(f"\n[Step {i}] {step.get('step', 'unknown')}")
        if 'observations' in step:
            logger.info(f"  관찰: {step['observations']}")
        if 'selected' in step:
            logger.info(f"  선택: {step['selected']} (confidence: {step.get('confidence', 'N/A')})")
        if 'reasoning' in step and step['reasoning']:
            logger.info(f"  추론: {step['reasoning']}")

    # 결과를 파일로 저장
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ 결과 저장: {json_file}")
    logger.info(f"✓ 로그 저장: {log_file}")

except Exception as e:
    logger.error(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "="*80)
logger.info("테스트 완료")
logger.info("="*80)
