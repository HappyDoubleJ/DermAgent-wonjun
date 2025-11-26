"""
Advanced Dermatology Diagnosis Agent with ReAct Pattern

ReAct (Reasoning + Acting) 패턴과 Chain-of-Thought 추론을 적용한 
고급 피부과 진단 에이전트
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time

# 경로 설정 - eval 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from ontology_utils import OntologyTree


# ============ 데이터 구조 ============

class ActionType(Enum):
    """에이전트 행동 유형"""
    OBSERVE = "observe"           # 이미지 관찰
    NAVIGATE = "navigate"         # 온톨로지 탐색
    COMPARE = "compare"           # 후보 비교
    VERIFY = "verify"             # 진단 검증
    CONCLUDE = "conclude"         # 결론 도출
    ASK_CLARIFICATION = "ask"     # 추가 정보 요청


@dataclass
class Observation:
    """관찰 결과"""
    morphology: List[str] = field(default_factory=list)
    color: List[str] = field(default_factory=list)
    distribution: List[str] = field(default_factory=list)
    surface: List[str] = field(default_factory=list)
    border: List[str] = field(default_factory=list)
    location: str = ""
    size: str = ""
    symptoms: List[str] = field(default_factory=list)
    duration: str = ""
    patient_info: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "morphology": self.morphology,
            "color": self.color,
            "distribution": self.distribution,
            "surface": self.surface,
            "border": self.border,
            "location": self.location,
            "size": self.size,
            "symptoms": self.symptoms,
            "duration": self.duration,
            "patient_info": self.patient_info,
            "confidence": self.confidence
        }
    
    def to_text(self) -> str:
        parts = []
        if self.morphology:
            parts.append(f"Morphology: {', '.join(self.morphology)}")
        if self.color:
            parts.append(f"Color: {', '.join(self.color)}")
        if self.distribution:
            parts.append(f"Distribution: {', '.join(self.distribution)}")
        if self.surface:
            parts.append(f"Surface: {', '.join(self.surface)}")
        if self.border:
            parts.append(f"Border: {', '.join(self.border)}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        return "; ".join(parts)


@dataclass 
class ThoughtStep:
    """사고 단계"""
    step_num: int
    thought: str           # 현재 생각
    action: ActionType     # 수행할 행동
    action_input: Dict     # 행동 입력
    observation: str       # 행동 결과
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action.value,
            "action_input": self.action_input,
            "observation": self.observation
        }


@dataclass
class DiagnosisResult:
    """진단 결과"""
    primary_diagnosis: str = ""
    differential_diagnoses: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ontology_path: List[str] = field(default_factory=list)
    observations: Optional[Observation] = None
    reasoning_chain: List[ThoughtStep] = field(default_factory=list)
    verification_passed: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "primary_diagnosis": self.primary_diagnosis,
            "differential_diagnoses": self.differential_diagnoses,
            "confidence": self.confidence,
            "ontology_path": self.ontology_path,
            "observations": self.observations.to_dict() if self.observations else {},
            "reasoning_chain": [s.to_dict() for s in self.reasoning_chain],
            "verification_passed": self.verification_passed,
            "warnings": self.warnings
        }


# ============ 도구 정의 ============

class Tool(ABC):
    """도구 기본 클래스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict:
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        pass


class ObserveTool(Tool):
    """이미지 관찰 도구"""
    
    def __init__(self, vlm_model, prompts: Dict[str, str]):
        self.vlm = vlm_model
        self.prompts = prompts
    
    @property
    def name(self) -> str:
        return "observe_image"
    
    @property
    def description(self) -> str:
        return """Observe and analyze the dermatological image to extract clinical features.
Use this tool to identify morphology, color, distribution, surface features, and body location."""
    
    @property
    def parameters(self) -> Dict:
        return {
            "image_path": "Path to the skin image",
            "focus": "Optional: specific aspect to focus on (morphology/color/distribution/all)"
        }
    
    def run(self, image_path: str, focus: str = "all") -> str:
        prompt = self.prompts.get("observation", "")
        
        if self.vlm is None:
            return json.dumps({
                "morphology": ["papule", "plaque"],
                "color": ["red", "erythematous"],
                "distribution": ["localized"],
                "location": "trunk"
            })
        
        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=1024)
            return response
        except Exception as e:
            return json.dumps({"error": str(e)})


class NavigateOntologyTool(Tool):
    """온톨로지 탐색 도구"""
    
    def __init__(self, tree: OntologyTree):
        self.tree = tree
    
    @property
    def name(self) -> str:
        return "navigate_ontology"
    
    @property
    def description(self) -> str:
        return """Navigate the disease ontology tree to explore categories and diseases.
Actions: get_children, get_path, get_siblings, get_parent, search"""
    
    @property
    def parameters(self) -> Dict:
        return {
            "action": "One of: get_children, get_path, get_siblings, get_parent, search",
            "node": "Node name to operate on",
            "query": "Search query (for search action)"
        }
    
    def run(self, action: str, node: str = "root", query: str = "") -> str:
        try:
            if action == "get_children":
                children = self.tree.get_children(node)
                return json.dumps({
                    "node": node,
                    "children": children,
                    "count": len(children)
                })
            
            elif action == "get_path":
                path = self.tree.get_path_to_root(node)
                return json.dumps({
                    "node": node,
                    "path": path,
                    "depth": len(path) - 1
                })
            
            elif action == "get_siblings":
                siblings = self.tree.get_siblings(node)
                return json.dumps({
                    "node": node,
                    "siblings": siblings
                })
            
            elif action == "get_parent":
                parent = self.tree.parent_map.get(
                    self.tree.get_canonical_name(node), None
                )
                return json.dumps({
                    "node": node,
                    "parent": parent
                })
            
            elif action == "search":
                # 간단한 검색: 노드 이름에 쿼리가 포함된 것 찾기
                matches = []
                query_lower = query.lower()
                for node_name in self.tree.ontology.keys():
                    if query_lower in node_name.lower():
                        matches.append(node_name)
                return json.dumps({
                    "query": query,
                    "matches": matches[:20]
                })
            
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
                
        except Exception as e:
            return json.dumps({"error": str(e)})


class CompareCandidatesTool(Tool):
    """후보 질환 비교 도구"""
    
    def __init__(self, tree: OntologyTree, disease_features: Dict):
        self.tree = tree
        self.disease_features = disease_features
    
    @property
    def name(self) -> str:
        return "compare_candidates"
    
    @property
    def description(self) -> str:
        return """Compare clinical observations with candidate diseases to score likelihood.
Returns ranked list of candidates with matching features."""
    
    @property
    def parameters(self) -> Dict:
        return {
            "candidates": "List of candidate disease names",
            "observations": "Dict of observed clinical features"
        }
    
    def run(self, candidates: List[str], observations: Dict) -> str:
        scores = {}
        details = {}
        
        for candidate in candidates:
            canonical = self.tree.get_canonical_name(candidate)
            if not canonical:
                continue
            
            features = self.disease_features.get(canonical, {})
            score, matches = self._calculate_score(features, observations)
            scores[canonical] = score
            details[canonical] = matches
        
        # 정규화 및 정렬
        if scores:
            max_score = max(scores.values()) or 1
            scores = {k: v/max_score for k, v in scores.items()}
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return json.dumps({
            "ranked_candidates": [
                {"disease": d, "score": round(s, 3), "matching_features": details.get(d, [])}
                for d, s in ranked[:10]
            ]
        })
    
    def _calculate_score(self, features: Dict, observations: Dict) -> Tuple[float, List[str]]:
        if not features:
            return 0.5, []
        
        score = 0.0
        matches = []
        
        for key in ["morphology", "color", "distribution", "surface"]:
            feat_list = features.get(key, [])
            obs_list = observations.get(key, [])
            
            if feat_list and obs_list:
                common = set(f.lower() for f in feat_list) & set(o.lower() for o in obs_list)
                if common:
                    score += len(common) / len(feat_list)
                    matches.extend(list(common))
        
        # Keywords in raw text
        keywords = features.get("keywords", [])
        raw_text = observations.get("raw_text", "").lower()
        for kw in keywords:
            if kw.lower() in raw_text:
                score += 0.2
                matches.append(f"keyword:{kw}")
        
        return score, matches


class VerifyDiagnosisTool(Tool):
    """진단 검증 도구"""
    
    def __init__(self, vlm_model, tree: OntologyTree, prompts: Dict[str, str]):
        self.vlm = vlm_model
        self.tree = tree
        self.prompts = prompts
    
    @property
    def name(self) -> str:
        return "verify_diagnosis"
    
    @property
    def description(self) -> str:
        return """Verify if the proposed diagnosis is consistent with the image and observations.
Returns verification result with confidence and any inconsistencies found."""
    
    @property
    def parameters(self) -> Dict:
        return {
            "image_path": "Path to the skin image",
            "diagnosis": "Proposed diagnosis",
            "observations": "Clinical observations"
        }
    
    def run(self, image_path: str, diagnosis: str, observations: Dict) -> str:
        prompt = self.prompts.get("verification", "").format(
            diagnosis=diagnosis,
            observations=json.dumps(observations)
        )
        
        if self.vlm is None:
            return json.dumps({
                "verified": True,
                "confidence": 0.8,
                "consistent_features": observations.get("morphology", [])[:2],
                "inconsistent_features": [],
                "alternative_suggestions": []
            })
        
        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=512)
            return response
        except Exception as e:
            return json.dumps({"error": str(e), "verified": False})


# ============ ReAct 에이전트 ============

class ReActDermatologyAgent:
    """ReAct 패턴 기반 피부과 진단 에이전트"""
    
    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model = None,
        max_steps: int = 10,
        verbose: bool = True
    ):
        self.tree = OntologyTree(ontology_path)  # None이면 자동 탐색
        self.vlm = vlm_model
        self.max_steps = max_steps
        self.verbose = verbose
        
        # 프롬프트 로드
        self._init_prompts()
        
        # 질환 특징 데이터베이스
        self._init_disease_features()
        
        # 도구 초기화
        self._init_tools()
    
    def _init_prompts(self):
        """프롬프트 템플릿 초기화"""
        self.prompts = {
            "observation": """Analyze this dermatological image carefully.

Extract and describe the following clinical features in JSON format:
{
    "morphology": ["primary lesion types: papule/macule/patch/plaque/nodule/vesicle/pustule/bulla/wheal/cyst/erosion/ulcer"],
    "color": ["colors observed: red/pink/brown/black/white/yellow/purple/blue"],
    "distribution": ["pattern: localized/generalized/symmetric/unilateral/linear/dermatomal/follicular"],
    "surface": ["texture: smooth/rough/scaly/crusted/verrucous/ulcerated/erosion"],
    "border": ["border characteristics: well-defined/ill-defined/regular/irregular"],
    "location": "anatomical location",
    "size": "approximate size if visible",
    "confidence": 0.0-1.0
}

Be specific and use standard dermatological terminology. Output ONLY valid JSON.""",

            "react_system": """You are a dermatology diagnostic agent using systematic reasoning.

Available Tools:
{tools}

For each step, output in this EXACT format:
Thought: [Your reasoning about what to do next]
Action: [Tool name]
Action Input: [JSON parameters for the tool]

After receiving observation, continue with next Thought.

When ready to conclude, use:
Thought: [Final reasoning]
Action: conclude
Action Input: {{"primary_diagnosis": "...", "differential_diagnoses": [...], "confidence": 0.0-1.0}}

Important Guidelines:
1. Start by observing the image to gather clinical features
2. Use the ontology to systematically narrow down categories
3. Compare candidates when you have specific observations
4. Verify your diagnosis before concluding
5. Consider differential diagnoses
6. Be confident but acknowledge uncertainty""",

            "category_selection": """Based on these clinical observations:
{observations}

Select the most appropriate category from:
{categories}

Consider the morphology, distribution pattern, and clinical context.
Output JSON: {{"category": "selected_category", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}""",

            "verification": """Verify if '{diagnosis}' is consistent with this skin image.

Clinical observations: {observations}

Evaluate:
1. Are the observed features typical for this diagnosis?
2. Are there any features that contradict this diagnosis?
3. What alternative diagnoses should be considered?

Output JSON:
{{
    "verified": true/false,
    "confidence": 0.0-1.0,
    "consistent_features": ["list of features that support the diagnosis"],
    "inconsistent_features": ["list of features that contradict"],
    "alternative_suggestions": ["other possible diagnoses"]
}}"""
        }
    
    def _init_disease_features(self):
        """질환별 특징 데이터베이스 초기화"""
        self.disease_features = {
            # Fungal
            "Tinea corporis": {
                "morphology": ["annular", "scaly", "patch", "plaque", "ring"],
                "color": ["red", "erythematous", "pink"],
                "distribution": ["localized", "asymmetric"],
                "surface": ["scaly", "slightly elevated border"],
                "keywords": ["ring", "ringworm", "circular", "central clearing"]
            },
            "Tinea pedis": {
                "morphology": ["scaly", "fissured", "macerated", "vesicular"],
                "color": ["red", "white"],
                "distribution": ["interdigital", "moccasin"],
                "location": ["foot", "feet", "toe"],
                "keywords": ["athlete's foot", "interdigital", "itchy feet"]
            },
            "Tinea versicolor": {
                "morphology": ["macule", "patch", "scaly"],
                "color": ["hypopigmented", "hyperpigmented", "pink", "tan"],
                "distribution": ["trunk", "upper arms", "neck"],
                "keywords": ["sun spots", "versicolor", "pityriasis"]
            },
            "Candidiasis": {
                "morphology": ["erythematous", "satellite lesions", "pustules", "macerated"],
                "color": ["red", "white"],
                "distribution": ["intertriginous"],
                "location": ["groin", "axilla", "inframammary", "oral"],
                "keywords": ["thrush", "yeast", "satellite pustules", "diaper"]
            },
            
            # Bacterial
            "Cellulitis": {
                "morphology": ["diffuse", "edematous", "indurated"],
                "color": ["red", "erythematous"],
                "distribution": ["unilateral", "localized"],
                "surface": ["warm", "tender"],
                "keywords": ["spreading", "tender", "fever", "swelling"]
            },
            "Impetigo": {
                "morphology": ["vesicles", "pustules", "crusts", "bullae"],
                "color": ["honey-colored", "yellow", "golden"],
                "distribution": ["localized", "perioral", "perinasal"],
                "surface": ["crusted"],
                "keywords": ["honey crust", "golden crust", "children"]
            },
            "Folliculitis": {
                "morphology": ["papules", "pustules", "follicular"],
                "color": ["red", "yellow"],
                "distribution": ["follicular", "scattered"],
                "keywords": ["hair follicle", "pustule", "shaving"]
            },
            
            # Inflammatory - Eczema
            "Atopic dermatitis": {
                "morphology": ["papules", "vesicles", "lichenification", "excoriation"],
                "color": ["red", "erythematous"],
                "distribution": ["flexural", "symmetric"],
                "surface": ["dry", "scaly", "lichenified"],
                "keywords": ["atopic", "flexural", "itchy", "childhood", "eczema"]
            },
            "Contact dermatitis": {
                "morphology": ["vesicles", "papules", "edema", "erythema"],
                "color": ["red"],
                "distribution": ["geometric", "localized", "linear"],
                "keywords": ["contact", "exposure", "allergic", "irritant"]
            },
            "Seborrheic dermatitis": {
                "morphology": ["scaly", "patch", "greasy"],
                "color": ["red", "yellow"],
                "distribution": ["scalp", "face", "nasolabial"],
                "surface": ["greasy scale", "yellowish"],
                "keywords": ["dandruff", "seborrheic", "nasolabial", "scalp"]
            },
            
            # Inflammatory - Psoriasis
            "Psoriasis": {
                "morphology": ["plaque", "scaly", "papule"],
                "color": ["red", "silvery"],
                "distribution": ["symmetric", "extensor"],
                "surface": ["silvery scale", "thick scale"],
                "border": ["well-defined"],
                "keywords": ["silvery scale", "auspitz", "koebner", "extensor"]
            },
            "Guttate psoriasis": {
                "morphology": ["papules", "small plaques", "drop-like"],
                "color": ["pink", "red"],
                "distribution": ["generalized", "trunk"],
                "keywords": ["guttate", "drop-like", "strep", "children"]
            },
            
            # Viral
            "Herpes simplex virus": {
                "morphology": ["vesicles", "grouped", "erosions", "crusts"],
                "color": ["clear", "yellow"],
                "distribution": ["grouped", "clustered"],
                "location": ["lip", "genital", "perioral"],
                "keywords": ["cold sore", "fever blister", "grouped vesicles"]
            },
            "Herpes zoster": {
                "morphology": ["vesicles", "papules", "crusts"],
                "color": ["clear", "hemorrhagic"],
                "distribution": ["dermatomal", "unilateral"],
                "keywords": ["shingles", "dermatomal", "painful", "unilateral"]
            },
            "Molluscum contagiosum": {
                "morphology": ["papules", "umbilicated", "dome-shaped"],
                "color": ["skin-colored", "pink", "pearly"],
                "distribution": ["grouped", "localized"],
                "keywords": ["umbilicated", "central dimple", "children", "pearly"]
            },
            
            # Benign tumors
            "Seborrheic keratosis": {
                "morphology": ["papule", "plaque", "stuck-on"],
                "color": ["brown", "tan", "black", "waxy"],
                "surface": ["verrucous", "waxy", "rough"],
                "border": ["well-defined"],
                "keywords": ["stuck-on", "waxy", "elderly", "horn cysts"]
            },
            "Dermatofibroma": {
                "morphology": ["nodule", "papule", "firm"],
                "color": ["brown", "pink", "tan"],
                "distribution": ["solitary"],
                "location": ["extremities", "leg"],
                "keywords": ["dimple sign", "firm", "button-like"]
            },
            
            # Malignant
            "Basal cell carcinoma": {
                "morphology": ["nodule", "papule", "ulcer", "pearly"],
                "color": ["pink", "pearly", "translucent"],
                "border": ["rolled border"],
                "surface": ["telangiectasia", "ulcerated"],
                "keywords": ["pearly", "rolled border", "telangiectasia", "rodent ulcer"]
            },
            "Squamous cell carcinoma": {
                "morphology": ["nodule", "plaque", "ulcer", "keratotic"],
                "color": ["red", "pink", "flesh-colored"],
                "surface": ["scaly", "crusted", "ulcerated"],
                "keywords": ["sun-exposed", "keratotic", "indurated", "ulcerated"]
            },
            "Melanoma": {
                "morphology": ["macule", "papule", "nodule", "asymmetric"],
                "color": ["brown", "black", "variegated", "blue", "red"],
                "border": ["irregular", "notched"],
                "keywords": ["ABCDE", "asymmetry", "border irregular", "color variegation", "diameter", "evolving"]
            }
        }
    
    def _init_tools(self):
        """도구 초기화"""
        self.tools = {
            "observe_image": ObserveTool(self.vlm, self.prompts),
            "navigate_ontology": NavigateOntologyTool(self.tree),
            "compare_candidates": CompareCandidatesTool(self.tree, self.disease_features),
            "verify_diagnosis": VerifyDiagnosisTool(self.vlm, self.tree, self.prompts),
        }
    
    def _log(self, message: str, level: str = "info"):
        """로깅"""
        if self.verbose:
            prefix = {"info": "ℹ️", "thought": "💭", "action": "🔧", "result": "📋", "warning": "⚠️", "success": "✅"}
            print(f"{prefix.get(level, '•')} {message}")
    
    def _parse_action(self, response: str) -> Tuple[str, Dict]:
        """응답에서 액션 파싱"""
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)
        
        action = action_match.group(1) if action_match else "conclude"
        
        try:
            action_input = json.loads(input_match.group(1)) if input_match else {}
        except json.JSONDecodeError:
            action_input = {}
        
        return action, action_input
    
    def _execute_tool(self, tool_name: str, params: Dict) -> str:
        """도구 실행"""
        if tool_name == "conclude":
            return json.dumps(params)
        
        tool = self.tools.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        try:
            return tool.run(**params)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _generate_react_step(
        self, 
        image_path: str,
        step_num: int,
        history: List[ThoughtStep],
        current_observations: Optional[Observation]
    ) -> Tuple[str, str, Dict, str]:
        """ReAct 한 단계 생성"""
        
        # 히스토리 텍스트 구성
        history_text = ""
        for step in history:
            history_text += f"\nStep {step.step_num}:\n"
            history_text += f"Thought: {step.thought}\n"
            history_text += f"Action: {step.action.value}\n"
            history_text += f"Action Input: {json.dumps(step.action_input)}\n"
            history_text += f"Observation: {step.observation[:500]}...\n" if len(step.observation) > 500 else f"Observation: {step.observation}\n"
        
        # 도구 설명
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        tools_desc += "\n- conclude: Finalize diagnosis with primary_diagnosis, differential_diagnoses, and confidence"
        
        # 프롬프트 구성
        prompt = f"""{self.prompts['react_system'].format(tools=tools_desc)}

{history_text}

Current observations: {current_observations.to_text() if current_observations else 'None yet'}

Now continue with Step {step_num}:
Thought:"""
        
        # VLM 호출
        if self.vlm is None:
            # Mock 응답 생성
            if step_num == 1:
                thought = "I need to first observe the image to identify clinical features."
                action = "observe_image"
                action_input = {"image_path": image_path, "focus": "all"}
            elif step_num == 2:
                thought = "Now I should explore the ontology to find matching categories."
                action = "navigate_ontology"
                action_input = {"action": "get_children", "node": "root"}
            elif step_num == 3:
                thought = "Based on observations, this looks like an inflammatory condition. Let me explore further."
                action = "navigate_ontology"
                action_input = {"action": "get_children", "node": "inflammatory"}
            elif step_num == 4:
                thought = "The features suggest an infectious etiology. Let me check fungal infections."
                action = "navigate_ontology"
                action_input = {"action": "get_children", "node": "fungal"}
            elif step_num == 5:
                thought = "Let me compare the candidates with my observations."
                action = "compare_candidates"
                obs_dict = current_observations.to_dict() if current_observations else {}
                action_input = {
                    "candidates": ["Tinea corporis", "Tinea pedis", "Candidiasis"],
                    "observations": obs_dict
                }
            else:
                thought = "Based on my analysis, I'm ready to conclude."
                action = "conclude"
                action_input = {
                    "primary_diagnosis": "Tinea corporis",
                    "differential_diagnoses": ["Psoriasis", "Nummular eczema"],
                    "confidence": 0.75
                }
            
            response = f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}"
        else:
            try:
                response = self.vlm.chat_img(prompt, [image_path], max_tokens=512)
            except Exception as e:
                self._log(f"VLM error: {e}", "warning")
                response = 'Thought: Error occurred, concluding.\nAction: conclude\nAction Input: {"primary_diagnosis": "", "confidence": 0.0}'
        
        # 파싱
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        action, action_input = self._parse_action(response)
        
        # 도구 실행
        if action == "observe_image":
            action_input["image_path"] = image_path
        
        observation = self._execute_tool(action, action_input)
        
        return thought, action, action_input, observation
    
    def diagnose(self, image_path: str) -> DiagnosisResult:
        """이미지 진단 수행"""
        self._log(f"\n{'='*60}")
        self._log(f"Starting diagnosis for: {image_path}")
        self._log(f"{'='*60}\n")
        
        result = DiagnosisResult()
        history: List[ThoughtStep] = []
        current_observations: Optional[Observation] = None
        
        for step_num in range(1, self.max_steps + 1):
            self._log(f"\n--- Step {step_num} ---", "info")
            
            # ReAct 단계 실행
            thought, action, action_input, observation = self._generate_react_step(
                image_path, step_num, history, current_observations
            )
            
            self._log(f"Thought: {thought}", "thought")
            self._log(f"Action: {action}", "action")
            self._log(f"Observation: {observation[:200]}...", "result")
            
            # 관찰 결과 업데이트
            if action == "observe_image":
                try:
                    obs_data = json.loads(observation)
                    current_observations = Observation(
                        morphology=obs_data.get("morphology", []),
                        color=obs_data.get("color", []),
                        distribution=obs_data.get("distribution", []),
                        surface=obs_data.get("surface", []),
                        border=obs_data.get("border", []),
                        location=obs_data.get("location", ""),
                        confidence=obs_data.get("confidence", 0.5),
                        raw_text=observation
                    )
                    result.observations = current_observations
                except json.JSONDecodeError:
                    pass
            
            # 히스토리 기록
            try:
                action_type = ActionType(action) if action in [a.value for a in ActionType] else ActionType.OBSERVE
            except ValueError:
                action_type = ActionType.OBSERVE
            
            step = ThoughtStep(
                step_num=step_num,
                thought=thought,
                action=action_type,
                action_input=action_input,
                observation=observation
            )
            history.append(step)
            result.reasoning_chain.append(step)
            
            # 종료 조건
            if action == "conclude":
                try:
                    conclusion = json.loads(observation)
                    result.primary_diagnosis = conclusion.get("primary_diagnosis", "")
                    result.differential_diagnoses = conclusion.get("differential_diagnoses", [])
                    result.confidence = conclusion.get("confidence", 0.5)
                    
                    # 유효성 검증
                    canonical = self.tree.get_canonical_name(result.primary_diagnosis)
                    if canonical:
                        result.primary_diagnosis = canonical
                        result.ontology_path = self.tree.get_path_to_root(canonical)
                        result.verification_passed = True
                    else:
                        result.warnings.append(f"Primary diagnosis '{result.primary_diagnosis}' not found in ontology")
                        result.verification_passed = False
                    
                except json.JSONDecodeError:
                    result.warnings.append("Failed to parse conclusion")
                
                break
        
        self._log(f"\n{'='*60}")
        self._log(f"Diagnosis complete!", "success")
        self._log(f"Primary: {result.primary_diagnosis}")
        self._log(f"Confidence: {result.confidence:.2f}")
        self._log(f"Path: {' → '.join(result.ontology_path)}")
        self._log(f"{'='*60}\n")
        
        return result
    
    def diagnose_batch(
        self, 
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[DiagnosisResult]:
        """배치 진단"""
        results = []
        
        iterator = image_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Diagnosing")
            except ImportError:
                pass
        
        for path in iterator:
            try:
                result = self.diagnose(path)
            except Exception as e:
                result = DiagnosisResult(
                    warnings=[f"Error: {str(e)}"]
                )
            results.append(result)
        
        return results


# ============ 데모 ============

def demo():
    """데모 실행"""
    print("="*60)
    print("ReAct Dermatology Agent Demo")
    print("="*60)

    try:
        # 에이전트 생성 (VLM 없이 Mock 모드, 자동 경로)
        agent = ReActDermatologyAgent(
            ontology_path=None,
            vlm_model=None,
            max_steps=6,
            verbose=True
        )

        # 진단 실행
        result = agent.diagnose("/demo/skin_image.jpg")

        print("\n" + "="*60)
        print("DIAGNOSIS RESULT")
        print("="*60)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure ontology.json is in the correct location.")


if __name__ == "__main__":
    demo()
