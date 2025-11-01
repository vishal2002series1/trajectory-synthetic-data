"""
Training Data Quality Filtering and Deduplication

This script:
1. Filters low-quality training examples
2. Deduplicates using semantic similarity (VectorDB embeddings)
3. Generates clean dataset for fine-tuning

Usage:
    python filter_and_deduplicate.py \
        --input training_examples.jsonl \
        --output cleaned_training_data.jsonl \
        --stats filter_stats.json

Quality Filters:
- COT length (min 20 chars)
- Decision format validation
- Tool usage patterns
- Answer quality (for ANSWER decisions)
- Mock data detection

Deduplication:
- Uses embeddings to detect semantic similarity
- Removes duplicates with >90% similarity
- Keeps highest quality example from each cluster
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, asdict

# Add project root to path if running standalone
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core import BedrockProvider
from src.utils import load_config, get_logger, read_jsonl, write_jsonl, write_json

logger = get_logger(__name__)


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
    answer_quality_score: float  # 0-1
    overall_score: float  # 0-1
    issues: List[str]
    
    def is_high_quality(self, threshold: float = 0.6) -> bool:
        """Check if example meets quality threshold."""
        return self.overall_score >= threshold


class QualityFilter:
    """Filters training examples based on quality criteria."""
    
    # Minimum thresholds
    MIN_COT_LENGTH = 20
    MIN_ANSWER_LENGTH = 50
    
    # Mock data patterns
    MOCK_PATTERNS = [
        r"mock\s+data",
        r"placeholder",
        r"simulated\s+(calculation|data|result)",
        r"example\s+response",
        r"test\s+data",
    ]
    
    def __init__(self):
        """Initialize quality filter."""
        self.stats = defaultdict(int)
        logger.info("QualityFilter initialized")
    
    def filter_dataset(
        self,
        examples: List[Dict[str, Any]],
        quality_threshold: float = 0.6
    ) -> Tuple[List[Dict[str, Any]], List[QualityMetrics]]:
        """
        Filter dataset for quality.
        
        Args:
            examples: List of training examples
            quality_threshold: Minimum quality score (0-1)
            
        Returns:
            Tuple of (filtered_examples, all_metrics)
        """
        logger.info(f"Filtering {len(examples)} examples (threshold={quality_threshold})")
        
        all_metrics = []
        filtered_examples = []
        
        for idx, example in enumerate(examples):
            metrics = self.evaluate_example(example, idx)
            all_metrics.append(metrics)
            
            if metrics.is_high_quality(quality_threshold):
                filtered_examples.append(example)
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1
                logger.debug(f"Example {idx} failed: {metrics.issues}")
        
        logger.info(f"Filtered: {len(filtered_examples)}/{len(examples)} passed")
        return filtered_examples, all_metrics
    
    def evaluate_example(
        self,
        example: Dict[str, Any],
        example_id: int
    ) -> QualityMetrics:
        """
        Evaluate quality of a single example.
        
        Returns:
            QualityMetrics object with scores and issues
        """
        issues = []
        
        # Extract fields
        query = example.get("Q", "")
        cot = example.get("COT", "")
        decision = example.get("Decision", "")
        tool_set = example.get("Tool Set", [])
        context = example.get("Context", [])
        metadata = example.get("metadata", {})
        
        decision_type = metadata.get("decision_type", "UNKNOWN")
        
        # 1. Check COT length
        cot_length = len(cot)
        if cot_length < self.MIN_COT_LENGTH:
            issues.append(f"COT too short ({cot_length} chars)")
        
        # 2. Check COT meaningfulness
        has_meaningful_cot = self._is_meaningful_cot(cot)
        if not has_meaningful_cot:
            issues.append("COT lacks substance")
        
        # 3. Check decision format
        has_valid_decision = self._is_valid_decision(decision, decision_type)
        if not has_valid_decision:
            issues.append("Invalid decision format")
        
        # 4. Check for mock data
        has_mock_data = self._contains_mock_data(example)
        if has_mock_data:
            issues.append("Contains mock/placeholder data")
        
        # 5. Check answer quality (for ANSWER decisions)
        answer_quality_score = self._evaluate_answer_quality(decision, decision_type)
        
        # 6. Check tool usage patterns
        tool_count = len(tool_set) if isinstance(tool_set, list) else 0
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            cot_length=cot_length,
            has_meaningful_cot=has_meaningful_cot,
            has_valid_decision=has_valid_decision,
            has_mock_data=has_mock_data,
            answer_quality_score=answer_quality_score,
            decision_type=decision_type
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
    
    def _is_meaningful_cot(self, cot: str) -> bool:
        """Check if COT has meaningful content."""
        if not cot or len(cot) < self.MIN_COT_LENGTH:
            return False
        
        # Check for generic/template phrases
        generic_phrases = [
            "the user is asking",
            "i need to",
            "this query requires",
            "error occurred",
        ]
        
        cot_lower = cot.lower()
        
        # If COT is mostly generic phrases, it's not meaningful
        generic_count = sum(1 for phrase in generic_phrases if phrase in cot_lower)
        if generic_count >= 2:
            return False
        
        # Check for actual reasoning
        reasoning_indicators = [
            "because",
            "therefore",
            "however",
            "based on",
            "given that",
            "since",
            "which",
            "indicates",
            "suggests",
        ]
        
        has_reasoning = any(indicator in cot_lower for indicator in reasoning_indicators)
        
        return has_reasoning or len(cot) > 100
    
    def _is_valid_decision(self, decision: str, decision_type: str) -> bool:
        """Check if decision format is valid."""
        if not decision:
            return False
        
        decision_upper = decision.upper()
        
        # Check if decision type matches content
        if decision_type == "CALL":
            return "CALL" in decision_upper or len(decision) < 50
        elif decision_type == "ASK":
            return "ASK" in decision_upper or "?" in decision
        elif decision_type == "ANSWER":
            return "ANSWER" in decision_upper or len(decision) > self.MIN_ANSWER_LENGTH
        
        return True
    
    def _contains_mock_data(self, example: Dict[str, Any]) -> bool:
        """Check if example contains mock/placeholder data."""
        # Convert example to string for pattern matching
        example_str = json.dumps(example).lower()
        
        # Check for mock patterns
        for pattern in self.MOCK_PATTERNS:
            if re.search(pattern, example_str, re.IGNORECASE):
                return True
        
        # Check context for mock data
        context = example.get("Context", [])
        for item in context:
            if isinstance(item, dict):
                data = item.get("data", {})
                if isinstance(data, dict):
                    # Check if data values are generic
                    data_str = str(data).lower()
                    if "mock" in data_str or "placeholder" in data_str:
                        return True
        
        return False
    
    def _evaluate_answer_quality(self, decision: str, decision_type: str) -> float:
        """Evaluate quality of ANSWER decision (0-1 score)."""
        if decision_type != "ANSWER":
            return 1.0  # Not applicable
        
        if not decision or len(decision) < self.MIN_ANSWER_LENGTH:
            return 0.0
        
        score = 0.0
        
        # Check length (0-0.3)
        if len(decision) > self.MIN_ANSWER_LENGTH:
            score += 0.3
        
        # Check for structured content (0-0.3)
        has_structure = any(marker in decision for marker in ["**", "##", "1.", "2.", "-", "•"])
        if has_structure:
            score += 0.3
        
        # Check for specific information (0-0.4)
        has_specifics = any(word in decision.lower() for word in [
            "allocation", "portfolio", "performance", "return", "risk",
            "stocks", "bonds", "asset", "investment", "data", "information"
        ])
        if has_specifics:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_overall_score(
        self,
        cot_length: int,
        has_meaningful_cot: bool,
        has_valid_decision: bool,
        has_mock_data: bool,
        answer_quality_score: float,
        decision_type: str
    ) -> float:
        """Calculate overall quality score (0-1)."""
        score = 0.0
        
        # COT quality (0-0.4)
        if has_meaningful_cot:
            score += 0.2
        if cot_length >= 50:
            score += 0.1
        if cot_length >= 100:
            score += 0.1
        
        # Decision quality (0-0.3)
        if has_valid_decision:
            score += 0.3
        
        # Answer quality (0-0.2 for ANSWER type)
        if decision_type == "ANSWER":
            score += answer_quality_score * 0.2
        else:
            score += 0.2  # Full score for non-ANSWER
        
        # Mock data penalty (0-0.1)
        if not has_mock_data:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics."""
        return dict(self.stats)


class SemanticDeduplicator:
    """Deduplicates training examples using semantic similarity."""
    
    def __init__(self, bedrock_provider: BedrockProvider, similarity_threshold: float = 0.90):
        """
        Initialize deduplicator.
        
        Args:
            bedrock_provider: BedrockProvider for embeddings
            similarity_threshold: Similarity threshold for duplicates (0-1)
        """
        self.provider = bedrock_provider
        self.similarity_threshold = similarity_threshold
        self.stats = defaultdict(int)
        logger.info(f"SemanticDeduplicator initialized (threshold={similarity_threshold})")
    
    def deduplicate(
        self,
        examples: List[Dict[str, Any]],
        metrics: List[QualityMetrics]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Deduplicate examples using semantic similarity.
        
        Args:
            examples: List of training examples
            metrics: Quality metrics for each example
            
        Returns:
            Tuple of (deduplicated_examples, dedup_stats)
        """
        logger.info(f"Deduplicating {len(examples)} examples")
        
        if len(examples) == 0:
            return [], {}
        
        # Generate embeddings for all examples
        embeddings = self._generate_embeddings(examples)
        
        # Find duplicate clusters
        duplicate_clusters = self._find_duplicate_clusters(embeddings)
        
        # Keep best example from each cluster
        keep_indices = self._select_best_from_clusters(
            examples, metrics, duplicate_clusters
        )
        
        # Filter examples
        deduplicated = [examples[i] for i in sorted(keep_indices)]
        
        # Generate stats
        dedup_stats = {
            "total_examples": len(examples),
            "unique_examples": len(deduplicated),
            "duplicates_removed": len(examples) - len(deduplicated),
            "duplicate_clusters": len(duplicate_clusters),
            "similarity_threshold": self.similarity_threshold
        }
        
        logger.info(
            f"Deduplication complete: {len(deduplicated)}/{len(examples)} unique "
            f"({dedup_stats['duplicates_removed']} duplicates removed)"
        )
        
        return deduplicated, dedup_stats
    
    def _generate_embeddings(self, examples: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for all examples."""
        logger.info("Generating embeddings...")
        
        embeddings = []
        for idx, example in enumerate(examples):
            # Create text representation for embedding
            text = self._example_to_text(example)
            
            # Generate embedding
            embedding = self.provider.generate_embedding(text)
            embeddings.append(embedding)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Generated {idx + 1}/{len(examples)} embeddings")
        
        return embeddings
    
    def _example_to_text(self, example: Dict[str, Any]) -> str:
        """Convert example to text for embedding."""
        query = example.get("Q", "")
        cot = example.get("COT", "")
        decision = example.get("Decision", "")
        
        # Extract just the decision type if it's a long answer
        if len(decision) > 200:
            decision = decision[:200] + "..."
        
        return f"Query: {query}\nReasoning: {cot}\nDecision: {decision}"
    
    def _find_duplicate_clusters(
        self,
        embeddings: List[List[float]]
    ) -> List[Set[int]]:
        """
        Find clusters of duplicate examples.
        
        Returns:
            List of sets, each set contains indices of duplicate examples
        """
        n = len(embeddings)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find all examples similar to example i
            cluster = {i}
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity >= self.similarity_threshold:
                    cluster.add(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
                logger.debug(f"Found duplicate cluster of size {len(cluster)}: {cluster}")
            
            visited.add(i)
        
        return clusters
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _select_best_from_clusters(
        self,
        examples: List[Dict[str, Any]],
        metrics: List[QualityMetrics],
        clusters: List[Set[int]]
    ) -> Set[int]:
        """
        Select best example from each duplicate cluster.
        
        Returns:
            Set of indices to keep
        """
        # Start with all indices
        keep_indices = set(range(len(examples)))
        
        # For each cluster, keep only the best
        for cluster in clusters:
            # Find best example in cluster
            best_idx = max(cluster, key=lambda i: metrics[i].overall_score)
            
            # Remove others from keep set
            for idx in cluster:
                if idx != best_idx:
                    keep_indices.discard(idx)
                    self.stats["removed_as_duplicate"] += 1
        
        return keep_indices


def main():
    """Main filtering and deduplication pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter and deduplicate training data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file (cleaned)")
    parser.add_argument("--stats", default="filter_stats.json", help="Output stats file")
    parser.add_argument("--quality-threshold", type=float, default=0.6, 
                       help="Quality threshold (0-1)")
    parser.add_argument("--similarity-threshold", type=float, default=0.90,
                       help="Similarity threshold for deduplication (0-1)")
    parser.add_argument("--skip-dedup", action="store_true",
                       help="Skip deduplication (only filter)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRAINING DATA QUALITY FILTERING & DEDUPLICATION")
    print("="*80)
    
    # Load config and initialize provider
    config = load_config()
    provider = BedrockProvider(
        model_id=config.bedrock.model_id,
        embedding_model_id=config.bedrock.embedding_model_id,
        region=config.bedrock.region
    )
    
    # Load examples
    print(f"\nLoading examples from: {args.input}")
    examples = read_jsonl(args.input)
    print(f"Loaded {len(examples)} examples")
    
    # Step 1: Quality Filtering
    print("\n" + "-"*80)
    print("STEP 1: QUALITY FILTERING")
    print("-"*80)
    
    quality_filter = QualityFilter()
    filtered_examples, all_metrics = quality_filter.filter_dataset(
        examples, quality_threshold=args.quality_threshold
    )
    
    filter_stats = quality_filter.get_stats()
    print(f"\n✅ Quality filtering complete:")
    print(f"   Passed: {filter_stats['passed']}")
    print(f"   Failed: {filter_stats['failed']}")
    print(f"   Pass rate: {filter_stats['passed']/len(examples)*100:.1f}%")
    
    # Step 2: Deduplication
    if args.skip_dedup:
        print("\n⏭️  Skipping deduplication")
        final_examples = filtered_examples
        dedup_stats = {"skipped": True}
    else:
        print("\n" + "-"*80)
        print("STEP 2: SEMANTIC DEDUPLICATION")
        print("-"*80)
        
        deduplicator = SemanticDeduplicator(
            provider, similarity_threshold=args.similarity_threshold
        )
        
        # Only deduplicate examples that passed quality filter
        filtered_metrics = [m for m in all_metrics if m.is_high_quality(args.quality_threshold)]
        
        final_examples, dedup_stats = deduplicator.deduplicate(
            filtered_examples, filtered_metrics
        )
        
        print(f"\n✅ Deduplication complete:")
        print(f"   Unique: {dedup_stats['unique_examples']}")
        print(f"   Duplicates removed: {dedup_stats['duplicates_removed']}")
        print(f"   Duplicate clusters: {dedup_stats['duplicate_clusters']}")
    
    # Save results
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    # Save cleaned examples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_examples, output_path)
    print(f"✅ Saved {len(final_examples)} cleaned examples to: {output_path}")
    
    # Save stats
    stats_output = {
        "input_file": args.input,
        "output_file": args.output,
        "total_input_examples": len(examples),
        "quality_filter": filter_stats,
        "deduplication": dedup_stats,
        "final_examples": len(final_examples),
        "reduction_percentage": (1 - len(final_examples)/len(examples)) * 100,
        "quality_threshold": args.quality_threshold,
        "similarity_threshold": args.similarity_threshold,
        "quality_metrics_summary": {
            "avg_cot_length": sum(m.cot_length for m in all_metrics) / len(all_metrics),
            "avg_quality_score": sum(m.overall_score for m in all_metrics) / len(all_metrics),
            "examples_with_mock_data": sum(1 for m in all_metrics if m.has_mock_data),
        }
    }
    
    stats_path = Path(args.stats)
    write_json(stats_output, stats_path)
    print(f"✅ Saved statistics to: {stats_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input:  {len(examples)} examples")
    print(f"Output: {len(final_examples)} examples")
    print(f"Reduction: {stats_output['reduction_percentage']:.1f}%")
    print("\nPipeline complete! ✅")
    print("="*80)


if __name__ == "__main__":
    main()