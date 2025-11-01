"""
Quality Filtering System for Training Examples

Filters out low-quality training examples based on multiple criteria:
1. Content completeness
2. Reasoning quality
3. Mock data presence
4. Tool usage appropriateness
5. Answer quality
6. Length constraints

Usage:
    from src.processing.quality_filter import QualityFilter
    
    filter = QualityFilter()
    results = filter.filter_examples("input.jsonl", "output.jsonl")
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Metrics for quality assessment."""
    score: float
    passed: bool
    reasons: List[str]
    flags: List[str]


class QualityFilter:
    """
    Filters training examples based on quality criteria.
    
    Quality Checks:
    - Mock data detection (rejects examples with mock/placeholder data)
    - Answer completeness (ensures ANSWER decisions have substantive content)
    - Tool usage appropriateness (validates tool calls make sense)
    - COT quality (checks reasoning depth and relevance)
    - Query complexity (filters out extremely simple/complex queries)
    - Consistency (validates field consistency across iterations)
    """
    
    # Patterns that indicate mock/placeholder data
    MOCK_DATA_PATTERNS = [
        r"mock\s+data",
        r"placeholder",
        r"simulated\s+(calculation|result|data)",
        r"example\s+(data|result)",
        r"test\s+(data|result)",
        r"dummy\s+data",
        r"\[sample\s+data\]",
        r"status.*success.*data.*mock",
    ]
    
    # Minimum requirements
    MIN_COT_LENGTH = 50  # Characters
    MIN_ANSWER_LENGTH = 100  # Characters for ANSWER decisions
    MAX_QUERY_LENGTH = 1000  # Extremely long queries are suspicious
    MIN_QUERY_LENGTH = 10  # Too short queries lack substance
    
    def __init__(
        self,
        min_quality_score: float = 0.6,
        reject_mock_data: bool = True,
        verbose: bool = True
    ):
        """
        Initialize quality filter.
        
        Args:
            min_quality_score: Minimum score (0-1) to pass
            reject_mock_data: If True, reject examples with mock data
            verbose: If True, log detailed filtering information
        """
        self.min_quality_score = min_quality_score
        self.reject_mock_data = reject_mock_data
        self.verbose = verbose
        
        # Compile patterns
        self.mock_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.MOCK_DATA_PATTERNS
        ]
        
        logger.info(
            f"QualityFilter initialized "
            f"(min_score={min_quality_score}, reject_mock={reject_mock_data})"
        )
    
    def assess_quality(self, example: Dict[str, Any]) -> QualityMetrics:
        """
        Assess quality of a single training example.
        
        Returns QualityMetrics with score (0-1) and reasons.
        """
        score_components = []
        reasons = []
        flags = []
        
        # Extract fields
        query = example.get("Q", "")
        cot = example.get("COT", "")
        tool_set = example.get("Tool Set", [])
        decision = example.get("Decision", "")
        context = example.get("Context", [])
        
        # Check 1: Query quality
        query_score, query_reasons = self._check_query_quality(query)
        score_components.append(query_score)
        reasons.extend(query_reasons)
        
        # Check 2: COT quality
        cot_score, cot_reasons = self._check_cot_quality(cot, query)
        score_components.append(cot_score)
        reasons.extend(cot_reasons)
        
        # Check 3: Mock data detection
        if self.reject_mock_data:
            has_mock, mock_reasons = self._detect_mock_data(example)
            if has_mock:
                score_components.append(0.0)
                reasons.extend(mock_reasons)
                flags.append("MOCK_DATA")
            else:
                score_components.append(1.0)
        
        # Check 4: Tool usage appropriateness
        tool_score, tool_reasons = self._check_tool_usage(
            decision, tool_set, context
        )
        score_components.append(tool_score)
        reasons.extend(tool_reasons)
        
        # Check 5: Answer quality (for ANSWER decisions)
        if decision.startswith("ANSWER"):
            answer_score, answer_reasons = self._check_answer_quality(decision)
            score_components.append(answer_score)
            reasons.extend(answer_reasons)
        else:
            score_components.append(1.0)  # CALL/ASK don't need answer checks
        
        # Check 6: Consistency
        consistency_score, consistency_reasons = self._check_consistency(example)
        score_components.append(consistency_score)
        reasons.extend(consistency_reasons)
        
        # Calculate overall score
        overall_score = sum(score_components) / len(score_components)
        passed = overall_score >= self.min_quality_score
        
        return QualityMetrics(
            score=overall_score,
            passed=passed,
            reasons=reasons,
            flags=flags
        )
    
    def _check_query_quality(self, query: str) -> Tuple[float, List[str]]:
        """Check query quality."""
        reasons = []
        score = 1.0
        
        if not query:
            return 0.0, ["Empty query"]
        
        query_len = len(query)
        
        # Too short
        if query_len < self.MIN_QUERY_LENGTH:
            score -= 0.5
            reasons.append(f"Query too short ({query_len} chars)")
        
        # Too long (might be an error)
        if query_len > self.MAX_QUERY_LENGTH:
            score -= 0.3
            reasons.append(f"Query extremely long ({query_len} chars)")
        
        # Check for actual content
        if query.strip() and len(query.strip().split()) < 3:
            score -= 0.3
            reasons.append("Query has very few words")
        
        return max(0.0, score), reasons
    
    def _check_cot_quality(self, cot: str, query: str) -> Tuple[float, List[str]]:
        """Check chain of thought quality."""
        reasons = []
        score = 1.0
        
        if not cot:
            return 0.0, ["Empty COT"]
        
        # Length check
        if len(cot) < self.MIN_COT_LENGTH:
            score -= 0.4
            reasons.append(f"COT too short ({len(cot)} chars)")
        
        # Generic/template responses
        generic_phrases = [
            "need to retrieve",
            "requires accessing",
            "cannot provide without",
            "the user is asking",
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in cot.lower())
        if generic_count >= 2:
            score -= 0.2
            reasons.append("COT appears generic/templated")
        
        # Check if COT relates to query
        query_words = set(query.lower().split())
        cot_words = set(cot.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "in", "on"}
        query_words -= common_words
        cot_words -= common_words
        
        if query_words and cot_words:
            overlap = len(query_words.intersection(cot_words)) / len(query_words)
            if overlap < 0.1:  # Less than 10% overlap
                score -= 0.3
                reasons.append("COT doesn't relate well to query")
        
        return max(0.0, score), reasons
    
    def _detect_mock_data(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Detect if example contains mock/placeholder data."""
        reasons = []
        
        # Convert entire example to string for searching
        example_str = json.dumps(example, indent=2).lower()
        
        # Check against patterns
        for pattern in self.mock_patterns:
            matches = pattern.findall(example_str)
            if matches:
                reasons.append(f"Found mock data: {matches[0][:50]}")
        
        has_mock = len(reasons) > 0
        return has_mock, reasons
    
    def _check_tool_usage(
        self,
        decision: str,
        tool_set: List[Dict],
        context: List[Dict]
    ) -> Tuple[float, List[str]]:
        """Check if tool usage is appropriate."""
        reasons = []
        score = 1.0
        
        # CALL decision should have tools
        if decision == "CALL":
            if not tool_set:
                score -= 0.5
                reasons.append("CALL decision but no tools specified")
            elif len(tool_set) > 5:
                score -= 0.2
                reasons.append(f"Too many tools ({len(tool_set)})")
        
        # ANSWER decision should not have tools (tools in context are ok)
        if decision.startswith("ANSWER"):
            if tool_set and len(tool_set) > 0:
                score -= 0.3
                reasons.append("ANSWER decision should have empty tool set")
        
        # Check context validity
        if context:
            for ctx in context:
                if not isinstance(ctx, dict):
                    score -= 0.2
                    reasons.append("Invalid context format")
                    break
        
        return max(0.0, score), reasons
    
    def _check_answer_quality(self, decision: str) -> Tuple[float, List[str]]:
        """Check quality of ANSWER decisions."""
        reasons = []
        score = 1.0
        
        # Extract answer text
        if not decision.startswith("ANSWER:"):
            return 1.0, []  # Not an answer, skip
        
        answer = decision.replace("ANSWER:", "").strip()
        
        # Length check
        if len(answer) < self.MIN_ANSWER_LENGTH:
            score -= 0.4
            reasons.append(f"Answer too short ({len(answer)} chars)")
        
        # Check for actual content vs. disclaimers
        disclaimer_phrases = [
            "cannot provide",
            "don't have access",
            "unable to",
            "I apologize",
            "error occurred",
        ]
        
        disclaimer_count = sum(
            1 for phrase in disclaimer_phrases 
            if phrase.lower() in answer.lower()
        )
        
        if disclaimer_count >= 2:
            score -= 0.3
            reasons.append("Answer contains multiple disclaimers")
        
        # Check for structure (paragraphs, sections)
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if len(paragraphs) < 2 and len(answer) > 300:
            score -= 0.1
            reasons.append("Long answer lacks structure")
        
        return max(0.0, score), reasons
    
    def _check_consistency(self, example: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check internal consistency."""
        reasons = []
        score = 1.0
        
        decision = example.get("Decision", "")
        context = example.get("Context", [])
        metadata = example.get("metadata", {})
        
        # Check iteration consistency
        iteration = metadata.get("iteration", 0)
        
        # Iteration 0 should not have context
        if iteration == 0 and context:
            score -= 0.2
            reasons.append("Iteration 0 has context (should be empty)")
        
        # Later iterations should have context
        if iteration > 0 and not context:
            score -= 0.3
            reasons.append(f"Iteration {iteration} has no context")
        
        # ANSWER should typically be at final iteration
        if decision.startswith("ANSWER") and context:
            # Check if we have tool results
            has_tool_results = len(context) > 0
            if not has_tool_results and iteration > 0:
                score -= 0.2
                reasons.append("ANSWER without tool results at later iteration")
        
        return max(0.0, score), reasons
    
    def filter_examples(
        self,
        input_file: str,
        output_file: str,
        stats_file: str = None
    ) -> Dict[str, Any]:
        """
        Filter training examples from input file to output file.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            stats_file: Optional path to save filtering statistics
            
        Returns:
            Dictionary with filtering statistics
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Filtering examples from {input_file}")
        logger.info(f"Output will be saved to {output_file}")
        
        # Statistics
        stats = {
            "total_examples": 0,
            "passed": 0,
            "failed": 0,
            "mock_data_rejected": 0,
            "quality_scores": [],
            "failure_reasons": Counter(),
            "flags": Counter()
        }
        
        # Read, filter, and write
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    example = json.loads(line.strip())
                    stats["total_examples"] += 1
                    
                    # Assess quality
                    metrics = self.assess_quality(example)
                    stats["quality_scores"].append(metrics.score)
                    
                    # Log verbose details
                    if self.verbose and not metrics.passed:
                        logger.info(
                            f"Line {line_num}: REJECTED "
                            f"(score={metrics.score:.2f}) - {metrics.reasons[:2]}"
                        )
                    
                    # Track statistics
                    for reason in metrics.reasons:
                        stats["failure_reasons"][reason] += 1
                    
                    for flag in metrics.flags:
                        stats["flags"][flag] += 1
                        if flag == "MOCK_DATA":
                            stats["mock_data_rejected"] += 1
                    
                    # Write if passed
                    if metrics.passed:
                        stats["passed"] += 1
                        outfile.write(json.dumps(example) + "\n")
                    else:
                        stats["failed"] += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    logger.error(f"Line {line_num}: Error - {e}")
        
        # Calculate summary statistics
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
            stats["min_quality_score"] = min(stats["quality_scores"])
            stats["max_quality_score"] = max(stats["quality_scores"])
        
        # Save statistics
        if stats_file:
            stats_path = Path(stats_file)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert Counter to dict for JSON serialization
            stats_export = stats.copy()
            stats_export["failure_reasons"] = dict(stats["failure_reasons"])
            stats_export["flags"] = dict(stats["flags"])
            
            with open(stats_path, 'w') as f:
                json.dump(stats_export, f, indent=2)
            
            logger.info(f"Statistics saved to {stats_file}")
        
        # Log summary
        logger.info("=" * 80)
        logger.info("QUALITY FILTERING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total examples processed: {stats['total_examples']}")
        logger.info(f"Passed: {stats['passed']} ({stats['passed']/stats['total_examples']*100:.1f}%)")
        logger.info(f"Failed: {stats['failed']} ({stats['failed']/stats['total_examples']*100:.1f}%)")
        if stats["quality_scores"]:
            logger.info(f"Avg quality score: {stats['avg_quality_score']:.3f}")
        logger.info(f"Mock data rejected: {stats['mock_data_rejected']}")
        
        if stats["failure_reasons"]:
            logger.info("\nTop failure reasons:")
            for reason, count in stats["failure_reasons"].most_common(5):
                logger.info(f"  - {reason}: {count}")
        
        return stats


if __name__ == "__main__":
    # Example usage
    from ..utils import setup_logger
    
    setup_logger("INFO")
    
    filter = QualityFilter(
        min_quality_score=0.6,
        reject_mock_data=True,
        verbose=True
    )
    
    stats = filter.filter_examples(
        input_file="data/output/training_examples.jsonl",
        output_file="data/output/filtered_examples.jsonl",
        stats_file="data/output/quality_stats.json"
    )