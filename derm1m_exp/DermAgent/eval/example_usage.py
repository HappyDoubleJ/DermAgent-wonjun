"""
DermAgent 사용 예제 스크립트

이 스크립트는 다양한 사용 시나리오를 보여줍니다.
"""

from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator


def example_1_basic_usage():
    """예제 1: 기본 사용법 (자동 경로)"""
    print("=" * 60)
    print("예제 1: 기본 사용법 - 자동 경로")
    print("=" * 60)

    # ontology.json을 자동으로 찾음
    tree = OntologyTree()
    print(f"✓ Ontology loaded from: {tree.ontology_path}\n")

    # 온톨로지 통계
    stats = tree.get_stats()
    print(f"총 노드 수: {stats['total_nodes']}")
    print(f"최대 깊이: {stats['max_depth']}")
    print(f"리프 노드: {stats['leaf_nodes']}\n")


def example_2_tree_operations():
    """예제 2: 트리 연산"""
    print("=" * 60)
    print("예제 2: 트리 연산")
    print("=" * 60)

    tree = OntologyTree()

    # 경로 확인
    node = "Tinea corporis"
    path = tree.get_path_to_root(node)
    print(f"\n'{node}'의 경로:")
    print(" -> ".join(path))

    # 거리 계산
    node1, node2 = "Tinea corporis", "Tinea pedis"
    distance = tree.get_hierarchical_distance(node1, node2)
    lca = tree.get_lca(node1, node2)
    print(f"\n'{node1}' ↔ '{node2}':")
    print(f"  거리: {distance}")
    print(f"  공통 조상: {lca}")

    # 형제 노드
    siblings = tree.get_siblings(node)
    print(f"\n'{node}'의 형제 노드 ({len(siblings)}개):")
    for s in siblings[:5]:
        print(f"  - {s}")
    if len(siblings) > 5:
        print(f"  ... 외 {len(siblings) - 5}개")


def example_3_evaluation():
    """예제 3: 평가 메트릭"""
    print("\n" + "=" * 60)
    print("예제 3: 평가 메트릭")
    print("=" * 60)

    evaluator = HierarchicalEvaluator()

    # 단일 샘플 평가
    gt = ["Tinea corporis"]
    pred = ["Tinea pedis"]

    result = evaluator.evaluate_single(gt, pred)

    print(f"\nGround Truth: {gt}")
    print(f"Prediction: {pred}")
    print("\n평가 결과:")
    print(f"  Exact Match: {result['exact_match']:.4f}")
    print(f"  Hierarchical F1: {result['hierarchical_f1']:.4f}")
    print(f"  Partial Credit: {result['partial_credit']:.4f}")
    print(f"  Distance: {result['avg_min_distance']:.1f}")


def example_4_batch_evaluation():
    """예제 4: 배치 평가"""
    print("\n" + "=" * 60)
    print("예제 4: 배치 평가")
    print("=" * 60)

    evaluator = HierarchicalEvaluator()

    # 여러 샘플
    ground_truths = [
        ["Tinea corporis"],
        ["Psoriasis"],
        ["Eczema"],
        ["Acne vulgaris"],
        ["Melanoma"],
    ]

    predictions = [
        ["Tinea pedis"],          # 비슷한 질환 (fungal)
        ["Psoriasis"],            # 정확히 일치
        ["Atopic dermatitis"],    # 비슷한 질환
        ["Acne"],                 # 거의 일치
        ["Basal cell carcinoma"], # 다른 질환 (같은 카테고리)
    ]

    result = evaluator.evaluate_batch(ground_truths, predictions)
    evaluator.print_evaluation_report(result)


def example_5_label_normalization():
    """예제 5: 라벨 정규화"""
    print("\n" + "=" * 60)
    print("예제 5: 라벨 정규화 및 유효성 검사")
    print("=" * 60)

    tree = OntologyTree()

    # 다양한 형태의 라벨
    test_labels = [
        "Tinea corporis",      # 정확한 이름
        "tinea corporis",      # 소문자
        "PSORIASIS",          # 대문자
        "  Eczema  ",         # 공백 포함
        "invalid_disease",    # 유효하지 않은 질환
        "Unknown",            # 유효하지 않은 질환
    ]

    print("\n라벨 정규화 테스트:")
    for label in test_labels:
        canonical = tree.get_canonical_name(label)
        status = "✓" if canonical else "✗"
        print(f"  {status} '{label}' -> {canonical}")

    # 유효한 라벨만 필터링
    valid_labels = tree.filter_valid_labels(test_labels)
    print(f"\n유효한 라벨만 필터링: {valid_labels}")


def example_6_different_path_methods():
    """예제 6: 다양한 경로 지정 방법"""
    print("\n" + "=" * 60)
    print("예제 6: 경로 지정 방법")
    print("=" * 60)

    print("\n방법 1: 자동 경로 (권장)")
    print("  tree = OntologyTree()")
    tree1 = OntologyTree()
    print(f"  ✓ Loaded from: {tree1.ontology_path}")

    print("\n방법 2: project_path 사용")
    print("  import project_path")
    print("  tree = OntologyTree(project_path.ONTOLOGY_PATH)")
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        import project_path
        tree2 = OntologyTree(project_path.ONTOLOGY_PATH)
        print(f"  ✓ Loaded from: {tree2.ontology_path}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n방법 3: 직접 경로 지정")
    print("  tree = OntologyTree('/path/to/ontology.json')")
    print("  (필요한 경우에만 사용)")


def main():
    """모든 예제 실행"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "DermAgent 사용 예제" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        example_1_basic_usage()
        example_2_tree_operations()
        example_3_evaluation()
        example_4_batch_evaluation()
        example_5_label_normalization()
        example_6_different_path_methods()

        print("\n" + "=" * 60)
        print("✓ 모든 예제가 성공적으로 실행되었습니다!")
        print("=" * 60)
        print()

    except FileNotFoundError as e:
        print(f"\n✗ 오류: {e}")
        print("\n해결 방법:")
        print("  1. 프로젝트 구조를 확인하세요:")
        print("     DermAgent/dataset/Derm1M/ontology.json")
        print("  2. 또는 직접 경로를 지정하세요:")
        print("     tree = OntologyTree('/path/to/ontology.json')")
        print()
    except Exception as e:
        print(f"\n✗ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
