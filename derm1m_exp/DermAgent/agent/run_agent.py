"""
Dermatology Agent Runner

에이전트를 실제 VLM 모델과 함께 실행하는 스크립트
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# 경로 설정 - agent 및 eval 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "agent"))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from dermatology_agent import DermatologyAgent, DiagnosisState
from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator


class MockVLM:
    """테스트용 Mock VLM"""
    
    def __init__(self):
        self.responses = {
            "initial": {
                "morphology": ["papule", "plaque", "scaly"],
                "color": ["red", "erythematous"],
                "distribution": ["localized"],
                "surface": ["scaly"],
                "location": "trunk",
                "additional_notes": "well-demarcated border"
            },
            "category": {
                "selected_category": "inflammatory",
                "confidence": 0.85,
                "reasoning": "Inflammatory features observed"
            },
            "subcategory": {
                "selected_subcategory": "infectious",
                "confidence": 0.7,
                "reasoning": "Pattern suggests infection"
            }
        }
    
    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 512) -> str:
        """Mock 응답 생성"""
        if "initial" in prompt.lower() or "morphology" in prompt.lower():
            return json.dumps(self.responses["initial"])
        elif "major categories" in prompt.lower() or "category" in prompt.lower():
            return json.dumps(self.responses["category"])
        elif "subcategor" in prompt.lower():
            return json.dumps(self.responses["subcategory"])
        else:
            return json.dumps({
                "primary_diagnosis": "Tinea corporis",
                "confidence": 0.7,
                "differential_diagnoses": ["Psoriasis", "Eczema"],
                "reasoning": "Clinical features consistent"
            })


def run_agent_diagnosis(
    agent: DermatologyAgent,
    image_paths: List[str],
    output_path: str = None,
    max_depth: int = 4
) -> List[Dict]:
    """에이전트로 진단 실행"""
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Diagnosing"):
        try:
            result = agent.diagnose(image_path, max_depth=max_depth)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "image_path": image_path,
                "error": str(e),
                "final_diagnosis": []
            })
    
    # 결과 저장
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    return results


def evaluate_results(
    results: List[Dict],
    ground_truths: List[List[str]],
    ontology_path: str
) -> Dict:
    """결과 평가"""
    
    evaluator = HierarchicalEvaluator(ontology_path)
    
    predictions = [r.get("final_diagnosis", []) for r in results]
    
    eval_result = evaluator.evaluate_batch(ground_truths, predictions)
    evaluator.print_evaluation_report(eval_result)
    
    return eval_result


def load_csv_data(csv_path: str) -> tuple:
    """CSV에서 데이터 로드"""
    image_paths = []
    ground_truths = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            disease_label = row.get('disease_label', '')
            
            image_paths.append(filename)
            
            # 쉼표로 구분된 라벨 파싱
            if disease_label:
                labels = [l.strip() for l in disease_label.split(',')]
            else:
                labels = []
            ground_truths.append(labels)
    
    return image_paths, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Run Dermatology Diagnosis Agent")
    parser.add_argument('--ontology', type=str, default=None,
                        help='Path to ontology.json (auto-detect if not specified)')
    parser.add_argument('--input_csv', type=str, required=False,
                        help='Input CSV with image paths and ground truth')
    parser.add_argument('--image_dir', type=str, default='',
                        help='Base directory for images')
    parser.add_argument('--output', type=str, default='agent_results.json',
                        help='Output JSON file path')
    parser.add_argument('--model', type=str, choices=['mock', 'gpt', 'qwen', 'internvl'],
                        default='mock', help='VLM model to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (for qwen/internvl)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (for gpt)')
    parser.add_argument('--max_depth', type=int, default=4,
                        help='Maximum ontology traversal depth')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode')
    
    args = parser.parse_args()
    
    # Demo 모드
    if args.demo:
        print("=== Demo Mode ===\n")

        try:
            agent = DermatologyAgent(
                ontology_path=args.ontology,
                vlm_model=MockVLM(),
                verbose=True
            )
            if args.ontology is None:
                print(f"✓ Ontology auto-detected\n")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\n사용법:")
            print("  python run_agent.py --demo")
            print("  (ontology.json을 자동으로 찾습니다)")
            return

        # 가상의 이미지로 테스트
        result = agent.diagnose("/fake/image.jpg", max_depth=args.max_depth)

        print("\n=== Diagnosis Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # VLM 모델 초기화
    vlm = None
    if args.model == 'mock':
        vlm = MockVLM()
    elif args.model == 'gpt':
        if not args.api_key:
            print("Error: --api_key required for GPT model")
            return
        # GPT4o import (실제 사용 시)
        # from model import GPT4o
        # vlm = GPT4o(api_key=args.api_key)
        print("GPT model selected - implement actual import")
        vlm = MockVLM()  # 임시
    elif args.model == 'qwen':
        if not args.model_path:
            print("Error: --model_path required for Qwen model")
            return
        # from model import QwenVL
        # vlm = QwenVL(model_path=args.model_path)
        print("Qwen model selected - implement actual import")
        vlm = MockVLM()  # 임시
    elif args.model == 'internvl':
        if not args.model_path:
            print("Error: --model_path required for InternVL model")
            return
        # from model import InternVL
        # vlm = InternVL(model_path=args.model_path)
        print("InternVL model selected - implement actual import")
        vlm = MockVLM()  # 임시
    
    # 에이전트 생성
    try:
        agent = DermatologyAgent(
            ontology_path=args.ontology,
            vlm_model=vlm,
            verbose=args.verbose
        )
        if args.ontology is None:
            print(f"✓ Ontology auto-detected\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\n해결 방법:")
        print("  1. 자동 경로: python run_agent.py --demo")
        print("  2. 수동 경로: python run_agent.py --ontology /path/to/ontology.json --demo")
        return
    
    # 데이터 로드
    if args.input_csv:
        image_paths, ground_truths = load_csv_data(args.input_csv)
        
        # 이미지 경로에 base directory 추가
        if args.image_dir:
            image_paths = [os.path.join(args.image_dir, p) for p in image_paths]
        
        print(f"Loaded {len(image_paths)} samples from {args.input_csv}")
        
        # 진단 실행
        results = run_agent_diagnosis(
            agent, 
            image_paths, 
            output_path=args.output,
            max_depth=args.max_depth
        )
        
        # 평가
        if ground_truths:
            print("\n=== Evaluation ===")
            evaluate_results(results, ground_truths, args.ontology)
    else:
        print("No input CSV provided. Use --demo for demo mode or --input_csv for batch processing.")


if __name__ == "__main__":
    main()


"""
Usage Examples:

# Demo mode (Mock VLM으로 구조 테스트, 자동 경로)
python run_agent.py --demo --verbose

# CSV 데이터로 실행 (Mock VLM)
python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output results.json \
    --model mock \
    --verbose

# GPT-4o로 실행
python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output gpt_results.json \
    --model gpt \
    --api_key YOUR_API_KEY \
    --max_depth 4

# Qwen으로 실행
CUDA_VISIBLE_DEVICES=0,1 python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output qwen_results.json \
    --model qwen \
    --model_path Qwen/Qwen2-VL-7B-Instruct
"""
