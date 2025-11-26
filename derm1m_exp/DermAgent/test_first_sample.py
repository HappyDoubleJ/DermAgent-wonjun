"""
첫 번째 샘플로 GPT-4o 에이전트 테스트

CSV 파일의 첫 번째 행을 읽어서 GPT-4o로 진단 흐름을 확인합니다.
"""

import os
import sys
import csv
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
print(f"Disease Label: {first_row['disease_label']}")
print(f"Hierarchical Label: {first_row.get('hierarchical_disease_label', 'N/A')}")
print(f"Caption: {first_row['caption'][:100]}...")
print(f"Body Location: {first_row.get('body_location', 'N/A')}")
print(f"Symptoms: {first_row.get('symptoms', 'N/A')}")
print("="*80)

# 이미지 경로 구성
image_path = image_base_dir / first_row['filename']
print(f"\n이미지 경로: {image_path}")

# 이미지 파일 존재 확인
if not image_path.exists():
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
    print("\n사용 가능한 경로 확인:")

    # 다른 가능한 경로들 확인
    possible_paths = [
        image_base_dir / "images" / first_row['filename'],
        image_base_dir / first_row['filename'],
    ]

    for p in possible_paths:
        if p.exists():
            image_path = p
            print(f"✓ 찾음: {image_path}")
            break
        else:
            print(f"  ✗ {p}")

    if not image_path.exists():
        print("\n⚠️  이미지를 찾을 수 없어서 Mock 모드로 진행합니다.")
        image_path = "/fake/image.jpg"  # Mock 경로
else:
    print(f"✓ 이미지 파일 존재 확인\n")

# GPT-4o VLM 초기화 (실제 사용하려면 모델 클래스 필요)
# 일단 Mock으로 진행 (GPT-4o 통합은 별도 구현 필요)
print("="*80)
print("에이전트 초기화 중...")
print("="*80)

# Mock VLM으로 테스트 (GPT-4o 통합은 별도 모듈 필요)
class SimpleGPTMock:
    """간단한 GPT Mock (실제 GPT-4o 구현 필요)"""
    def __init__(self, api_key):
        self.api_key = api_key
        # 실제로는 OpenAI 클라이언트 초기화
        # import openai
        # self.client = openai.OpenAI(api_key=api_key)

    def chat_img(self, prompt, image_paths, max_tokens=512):
        """Mock 응답 (실제 GPT-4o 호출로 교체 필요)"""
        import json
        # 실제로는:
        # response = self.client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[{"role": "user", "content": [
        #         {"type": "text", "text": prompt},
        #         {"type": "image_url", "image_url": {"url": image_paths[0]}}
        #     ]}],
        #     max_tokens=max_tokens
        # )
        # return response.choices[0].message.content

        # Mock 응답
        if "morphology" in prompt.lower():
            return json.dumps({
                "morphology": ["atrophy", "muscle wasting"],
                "color": ["normal"],
                "distribution": ["localized"],
                "surface": ["normal"],
                "location": "gluteal region",
                "additional_notes": "muscle atrophy visible"
            })
        elif "major categories" in prompt.lower():
            return json.dumps({
                "selected_category": "no definitive diagnosis",
                "confidence": 0.3,
                "reasoning": "Image shows anatomical changes but no clear dermatological condition"
            })
        return "{}"

# Mock VLM 생성
vlm = SimpleGPTMock(api_key)

# 에이전트 생성
agent = DermatologyAgent(
    ontology_path=None,  # 자동 경로
    vlm_model=vlm,
    verbose=True  # 상세 로그 출력
)

print("\n" + "="*80)
print("진단 시작")
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
    print(f"최종 진단: {result['final_diagnosis']}")
    print(f"진단 경로: {' → '.join(result['diagnosis_path'])}")
    print(f"신뢰도: {result['confidence_scores']}")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("테스트 완료")
print("="*80)
print("\n💡 참고:")
print("  - 현재는 Mock VLM을 사용하고 있습니다")
print("  - 실제 GPT-4o를 사용하려면 OpenAI API 통합이 필요합니다")
print("  - agent/run_agent.py의 GPT 모델 부분을 구현해야 합니다")
