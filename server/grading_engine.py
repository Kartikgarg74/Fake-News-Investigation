"""Multi-signal reward computation for the Fake News Investigator environment."""

from typing import Dict, List, Optional

# Label adjacency for partial credit
LABEL_ORDER = [
    "PANTS_ON_FIRE",
    "FALSE",
    "MOSTLY_FALSE",
    "HALF_TRUE",
    "MOSTLY_TRUE",
    "TRUE",
]


LABEL_ALIASES = {
    "PANTS_FIRE": "PANTS_ON_FIRE",
    "BARELY_TRUE": "MOSTLY_FALSE",
    "MOSTLY_TRUE": "MOSTLY_TRUE",
}


def _normalize_label(label: str) -> str:
    """Normalize a verdict label to the canonical LABEL_ORDER form."""
    normalized = label.upper().replace("-", "_").replace(" ", "_")
    return LABEL_ALIASES.get(normalized, normalized)


def score_verdict(predicted: str, ground_truth: str) -> float:
    """Score the agent's verdict against ground truth.

    Exact match = 1.0, adjacent label = 0.5, wrong direction = 0.0.
    """
    predicted = _normalize_label(predicted)
    ground_truth = _normalize_label(ground_truth)

    if predicted == ground_truth:
        return 1.0

    try:
        pred_idx = LABEL_ORDER.index(predicted)
        truth_idx = LABEL_ORDER.index(ground_truth)
    except ValueError:
        return 0.0

    distance = abs(pred_idx - truth_idx)
    if distance == 1:
        return 0.5
    if distance == 2:
        return 0.25
    return 0.0


def score_evidence(cited: List[str], gold: List[str]) -> float:
    """Score evidence quality using F1 of cited vs gold-standard sources."""
    if not gold:
        return 1.0 if not cited else 0.5
    if not cited:
        return 0.0

    cited_set = set(cited)
    gold_set = set(gold)
    overlap = cited_set & gold_set

    if not overlap:
        return 0.0

    precision = len(overlap) / len(cited_set)
    recall = len(overlap) / len(gold_set)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def score_efficiency(steps_used: int, max_budget: int) -> float:
    """Score investigation efficiency (fewer steps = higher score)."""
    if max_budget <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (steps_used / max_budget)), 4)


def score_confidence(confidence: float, verdict_correct: bool) -> float:
    """Score confidence calibration.

    High confidence + correct = good.
    High confidence + wrong = very bad.
    Low confidence + wrong = acceptable.
    """
    actual = 1.0 if verdict_correct else 0.0
    return round(1.0 - abs(confidence - actual), 4)


def score_reasoning(agent_reasoning: Optional[str], gold_reasoning: str) -> float:
    """Score reasoning quality using keyword overlap.

    Uses simple keyword overlap instead of BERTScore to avoid
    heavy model dependency. Checks if key factual terms from
    the gold reasoning appear in the agent's reasoning.
    """
    if not agent_reasoning:
        return 0.0
    if not gold_reasoning:
        return 0.5

    # Simple keyword overlap scoring
    gold_words = set(gold_reasoning.lower().split())
    agent_words = set(agent_reasoning.lower().split())

    # Remove stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "that",
        "this", "these", "those", "it", "its", "they", "them", "their",
    }
    gold_keywords = gold_words - stopwords
    agent_keywords = agent_words - stopwords

    if not gold_keywords:
        return 0.5

    overlap = gold_keywords & agent_keywords
    recall = len(overlap) / len(gold_keywords)
    return round(min(1.0, recall * 1.2), 4)  # slight boost, cap at 1.0


def compute_reward(
    predicted_verdict: str,
    ground_truth_verdict: str,
    cited_evidence: List[str],
    gold_evidence: List[str],
    steps_used: int,
    max_budget: int,
    confidence: float,
    agent_reasoning: Optional[str],
    gold_reasoning: str,
    penalties: float = 0.0,
) -> Dict[str, float]:
    """Compute the full multi-signal reward.

    Returns a dict with breakdown and total score.
    """
    verdict_sc = score_verdict(predicted_verdict, ground_truth_verdict)
    evidence_sc = score_evidence(cited_evidence, gold_evidence)
    efficiency_sc = score_efficiency(steps_used, max_budget)
    confidence_sc = score_confidence(confidence, verdict_sc >= 0.5)
    reasoning_sc = score_reasoning(agent_reasoning, gold_reasoning)

    total = (
        0.30 * verdict_sc
        + 0.25 * evidence_sc
        + 0.15 * efficiency_sc
        + 0.15 * confidence_sc
        + 0.15 * reasoning_sc
        - penalties
    )
    total = round(max(0.0, min(1.0, total)), 4)

    return {
        "total": total,
        "verdict_accuracy": verdict_sc,
        "evidence_quality": evidence_sc,
        "efficiency": efficiency_sc,
        "confidence_calibration": confidence_sc,
        "reasoning_quality": reasoning_sc,
        "penalties": penalties,
    }
