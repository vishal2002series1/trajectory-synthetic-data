"""
Simplified Quality Analysis for Training Data
Analyzes quality metrics without requiring project dependencies
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class QualityMetrics:
    """Quality metrics for a training example."""
    example_id: int
    cot_length: int
    has_valid_decision: bool
    has_meaningful_cot: bool
    has_mock_data: bool
    decision_type: str
    tool_count: int
    answer_quality_score: float
    overall_score: float
    issues: list
    
    def is_high_quality(self, threshold: float = 0.6) -> bool:
        return self.overall_score >= threshold


class QualityAnalyzer:
    """Analyzes training example quality."""
    
    MIN_COT_LENGTH = 20
    MIN_ANSWER_LENGTH = 50
    
    MOCK_PATTERNS = [
        r"mock\s+data",
        r"placeholder",
        r"simulated\s+(calculation|data|result)",
        r"example\s+response",
        r"test\s+data",
    ]
    
    def __init__(self):
        self.stats = defaultdict(int)
    
    def analyze_dataset(self, examples):
        """Analyze all examples."""
        print(f"\nAnalyzing {len(examples)} training examples...")
        
        all_metrics = []
        
        for idx, example in enumerate(examples):
            metrics = self.evaluate_example(example, idx)
            all_metrics.append(metrics)
            
            if metrics.is_high_quality(0.6):
                self.stats["high_quality"] += 1
            else:
                self.stats["low_quality"] += 1
        
        return all_metrics
    
    def evaluate_example(self, example, example_id):
        """Evaluate single example."""
        issues = []
        
        # Extract fields
        query = example.get("Q", "")
        cot = example.get("COT", "")
        decision = example.get("Decision", "")
        tool_set = example.get("Tool Set", [])
        context = example.get("Context", [])
        metadata = example.get("metadata", {})
        
        decision_type = metadata.get("decision_type", "UNKNOWN")
        
        # COT quality
        cot_length = len(cot)
        if cot_length < self.MIN_COT_LENGTH:
            issues.append(f"COT too short ({cot_length} chars)")
        
        # COT meaningfulness
        has_meaningful_cot = self._is_meaningful_cot(cot)
        if not has_meaningful_cot:
            issues.append("COT lacks substance")
        
        # Decision validity
        has_valid_decision = self._is_valid_decision(decision, decision_type)
        if not has_valid_decision:
            issues.append("Invalid decision format")
        
        # Mock data check
        has_mock_data = self._contains_mock_data(example)
        if has_mock_data:
            issues.append("Contains mock/placeholder data")
        
        # Answer quality
        answer_quality_score = self._evaluate_answer_quality(decision, decision_type)
        
        # Tool count
        tool_count = len(tool_set) if isinstance(tool_set, list) else 0
        
        # Overall score
        overall_score = self._calculate_score(
            cot_length, has_meaningful_cot, has_valid_decision,
            has_mock_data, answer_quality_score, decision_type
        )
        
        return QualityMetrics(
            example_id=example_id,
            cot_length=cot_length,
            has_valid_decision=has_valid_decision,
            has_meaningful_cot=has_meaningful_cot,
            has_mock_data=has_mock_data,
            decision_type=decision_type,
            tool_count=tool_count,
            answer_quality_score=answer_quality_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _is_meaningful_cot(self, cot):
        """Check COT meaningfulness."""
        if len(cot) < self.MIN_COT_LENGTH:
            return False
        
        reasoning_indicators = [
            "because", "therefore", "however", "based on",
            "given that", "since", "which", "indicates", "suggests"
        ]
        
        cot_lower = cot.lower()
        has_reasoning = any(ind in cot_lower for ind in reasoning_indicators)
        
        return has_reasoning or len(cot) > 100
    
    def _is_valid_decision(self, decision, decision_type):
        """Check decision validity."""
        if not decision:
            return False
        
        decision_upper = decision.upper()
        
        if decision_type == "CALL":
            return "CALL" in decision_upper or len(decision) < 50
        elif decision_type == "ASK":
            return "ASK" in decision_upper or "?" in decision
        elif decision_type == "ANSWER":
            return "ANSWER" in decision_upper or len(decision) > self.MIN_ANSWER_LENGTH
        
        return True
    
    def _contains_mock_data(self, example):
        """Check for mock data."""
        example_str = json.dumps(example).lower()
        
        for pattern in self.MOCK_PATTERNS:
            if re.search(pattern, example_str, re.IGNORECASE):
                return True
        
        return False
    
    def _evaluate_answer_quality(self, decision, decision_type):
        """Evaluate answer quality."""
        if decision_type != "ANSWER":
            return 1.0
        
        if not decision or len(decision) < self.MIN_ANSWER_LENGTH:
            return 0.0
        
        score = 0.0
        
        if len(decision) > self.MIN_ANSWER_LENGTH:
            score += 0.3
        
        has_structure = any(m in decision for m in ["**", "##", "1.", "2.", "-"])
        if has_structure:
            score += 0.3
        
        has_specifics = any(w in decision.lower() for w in [
            "allocation", "portfolio", "performance", "return", "risk"
        ])
        if has_specifics:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_score(self, cot_length, meaningful, valid_decision,
                        mock_data, answer_quality, decision_type):
        """Calculate overall score."""
        score = 0.0
        
        # COT (0-0.4)
        if meaningful:
            score += 0.2
        if cot_length >= 50:
            score += 0.1
        if cot_length >= 100:
            score += 0.1
        
        # Decision (0-0.3)
        if valid_decision:
            score += 0.3
        
        # Answer (0-0.2)
        if decision_type == "ANSWER":
            score += answer_quality * 0.2
        else:
            score += 0.2
        
        # Mock data (0-0.1)
        if not mock_data:
            score += 0.1
        
        return min(score, 1.0)


def main():
    """Main analysis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_analysis_simple.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    print("="*80)
    print("TRAINING DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Load examples
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(examples)} examples")
    
    # Analyze
    analyzer = QualityAnalyzer()
    metrics = analyzer.analyze_dataset(examples)
    
    # Report
    print("\n" + "-"*80)
    print("QUALITY ANALYSIS RESULTS")
    print("-"*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  High quality (≥0.6): {analyzer.stats['high_quality']}")
    print(f"  Low quality (<0.6): {analyzer.stats['low_quality']}")
    print(f"  Pass rate: {analyzer.stats['high_quality']/len(examples)*100:.1f}%")
    
    # Distribution
    print(f"\nQuality Score Distribution:")
    score_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in score_ranges:
        count = sum(1 for m in metrics if low <= m.overall_score < high)
        print(f"  {low:.1f}-{high:.1f}: {count} examples")
    
    # Common issues
    print(f"\nCommon Issues:")
    issue_counts = defaultdict(int)
    for m in metrics:
        for issue in m.issues:
            issue_counts[issue] += 1
    
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  - {issue}: {count} examples")
    
    # Low quality examples
    print(f"\nLow Quality Examples (score <0.5):")
    low_quality = [m for m in metrics if m.overall_score < 0.5]
    for m in low_quality[:5]:
        print(f"\n  Example {m.example_id}:")
        print(f"    Score: {m.overall_score:.2f}")
        print(f"    Issues: {', '.join(m.issues)}")
    
    # Save detailed report
    output_file = input_file.replace('.jsonl', '_quality_report.json')
    report = {
        "total_examples": len(examples),
        "high_quality": analyzer.stats['high_quality'],
        "low_quality": analyzer.stats['low_quality'],
        "pass_rate": analyzer.stats['high_quality']/len(examples)*100,
        "metrics": [asdict(m) for m in metrics]
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()