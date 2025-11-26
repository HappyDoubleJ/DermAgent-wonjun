"""
첫 번째 샘플로 실제 GPT-4o 에이전트 테스트

CSV 파일의 첫 번째 행을 읽어서 실제 GPT-4o로 진단 흐름을 확인합니다.
"""

import os
import sys
import csv
import base64
from pathlib import Path
from dotenv import load_dotenv

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "agent"))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from dermatology_agent import DermatologyAgent

# .env 파일에서 API 키 로드
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in .env file")
    print(f"   Checked: {env_path}")
    sys.exit(1)

print(f"✓ API Key loaded: {api_key[:20]}...")

# OpenAI 클라이언트 초기화
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")
except ImportError:
    print("❌ Error: openai package not installed")
    print("   Install with: pip install openai")
    sys.exit(1)


class GPT4oVLM:
    """실제 GPT-4o Vision 모델"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # GPT-4o 모델

    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat_img(self, prompt: str, image_paths: list, max_tokens: int = 1024) -> str:
        """
        GPT-4o Vision API 호출

        Args:
            prompt: 텍스트 프롬프트
            image_paths: 이미지 파일 경로 리스트
            max_tokens: 최대 토큰 수

        Returns:
            모델 응답 텍스트
        """
        # 이미지 인코딩
        image_path = image_paths[0] if image_paths else None

        if not image_path or not os.path.exists(image_path):
            print(f"⚠️  이미지 없음, 텍스트만으로 응답: {image_path}")
            # 이미지 없이 텍스트만
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content

        # 이미지 인코딩
        base64_image = self._encode_image(image_path)

        print(f"\n{'='*80}")
        print(f"📤 GPT-4o에 요청 중...")
        print(f"{'='*80}")
        print(f"프롬프트 (첫 200자):\n{prompt[:200]}...")
        print(f"이미지: {image_path}")
        print(f"{'='*80}\n")

        # GPT-4o Vision API 호출
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
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

            print(f"\n{'='*80}")
            print(f"📥 GPT-4o 응답:")
            print(f"{'='*80}")
            print(answer)
            print(f"{'='*80}\n")

            return answer

        except Exception as e:
            print(f"\n❌ GPT-4o API 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return "{}"


# CSV 파일 경로
csv_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "Derm1M" / "Derm1M_v2_pretrain_ontology_sampled_100.csv"
image_base_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / "Derm1M"

print(f"✓ CSV path: {csv_path}")
print(f"✓ Image base dir: {image_base_dir}\n")

# CSV에서 첫 번째 행 읽기
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    first_row = next(reader)

print("="*80)
print("첫 번째 샘플 정보")
print("="*80)
print(f"Filename: {first_row['filename']}")
print(f"Disease Label (GT): {first_row['disease_label']}")
print(f"Hierarchical Label: {first_row.get('hierarchical_disease_label', 'N/A')}")
print(f"Caption: {first_row['caption'][:150]}...")
print(f"Body Location: {first_row.get('body_location', 'N/A')}")
print(f"Symptoms: {first_row.get('symptoms', 'N/A')}")
print("="*80)

# 이미지 경로 구성
image_path = image_base_dir / first_row['filename']
print(f"\n이미지 경로: {image_path}")

# 이미지 파일 존재 확인
if not image_path.exists():
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
    sys.exit(1)
else:
    print(f"✓ 이미지 파일 존재 확인\n")

# GPT-4o VLM 초기화
print("="*80)
print("GPT-4o 에이전트 초기화 중...")
print("="*80)

vlm = GPT4oVLM(api_key)

# 에이전트 생성
agent = DermatologyAgent(
    ontology_path=None,  # 자동 경로
    vlm_model=vlm,
    verbose=True  # 상세 로그 출력
)

print("\n" + "="*80)
print("진단 시작 (실제 GPT-4o 사용)")
print("="*80)

# 진단 실행
try:
    result = agent.diagnose(str(image_path), max_depth=4)

    print("\n" + "="*80)
    print("진단 결과")
    print("="*80)

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "="*80)
    print("요약")
    print("="*80)
    print(f"Ground Truth: {first_row['disease_label']}")
    print(f"최종 진단: {result['final_diagnosis']}")
    print(f"진단 경로: {' → '.join(result['diagnosis_path'])}")
    print(f"주요 관찰: {result['observations']}")

    # 추론 과정 출력
    print("\n" + "="*80)
    print("추론 과정 (Reasoning History)")
    print("="*80)
    for i, step in enumerate(result['reasoning_history'], 1):
        print(f"\n[Step {i}] {step.get('step', 'unknown')}")
        if 'observations' in step:
            print(f"  관찰: {step['observations']}")
        if 'selected' in step:
            print(f"  선택: {step['selected']} (confidence: {step.get('confidence', 'N/A')})")
        if 'reasoning' in step and step['reasoning']:
            print(f"  추론: {step['reasoning']}")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("테스트 완료")
print("="*80)
