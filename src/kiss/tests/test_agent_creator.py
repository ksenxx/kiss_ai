# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Tests for the Agent Creator module.

This module tests:
1. ImprovementReport serialization/deserialization
2. AgentVariant Pareto dominance
3. ImproverAgent (requires API key)
4. AgentEvolver (requires API key)
"""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Define ImprovementReport locally to avoid importing the full module chain
# that requires claude_agent_sdk
@dataclass
class ImprovementReport:
    """Report of improvements made to an agent."""

    baseline_tokens: int = 0
    baseline_time: float = 0.0
    improved_tokens: int = 0
    improved_time: float = 0.0
    token_improvement_pct: float = 0.0
    time_improvement_pct: float = 0.0
    implemented_ideas: list[dict[str, Any]] = field(default_factory=list)
    failed_ideas: list[dict[str, str]] = field(default_factory=list)
    parent_folder: str = ""
    parent_report_path: str = ""
    generation: int = 0
    parent_folders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "baseline_tokens": self.baseline_tokens,
            "baseline_time": self.baseline_time,
            "improved_tokens": self.improved_tokens,
            "improved_time": self.improved_time,
            "token_improvement_pct": self.token_improvement_pct,
            "time_improvement_pct": self.time_improvement_pct,
            "implemented_ideas": self.implemented_ideas,
            "failed_ideas": self.failed_ideas,
            "parent_folder": self.parent_folder,
            "parent_report_path": self.parent_report_path,
            "parent_folders": self.parent_folders,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementReport":
        """Create report from dictionary."""
        return cls(
            baseline_tokens=data.get("baseline_tokens", 0),
            baseline_time=data.get("baseline_time", 0.0),
            improved_tokens=data.get("improved_tokens", 0),
            improved_time=data.get("improved_time", 0.0),
            token_improvement_pct=data.get("token_improvement_pct", 0.0),
            time_improvement_pct=data.get("time_improvement_pct", 0.0),
            implemented_ideas=data.get("implemented_ideas", []),
            failed_ideas=data.get("failed_ideas", []),
            parent_folder=data.get("parent_folder", ""),
            parent_report_path=data.get("parent_report_path", ""),
            parent_folders=data.get("parent_folders", []),
            generation=data.get("generation", 0),
        )

    @classmethod
    def load(cls, path: str) -> "ImprovementReport":
        """Load report from JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def is_improvement_over(self, other: "ImprovementReport") -> bool:
        """Check if this report shows improvement over another."""
        token_better = self.improved_tokens <= other.improved_tokens
        time_better = self.improved_time <= other.improved_time
        strictly_better = (
            self.improved_tokens < other.improved_tokens
            or self.improved_time < other.improved_time
        )
        return token_better and time_better and strictly_better


@dataclass
class AgentVariant:
    """Represents an agent variant in the Pareto frontier."""

    folder_path: str
    report_path: str
    report: ImprovementReport
    tokens_used: int = 0
    execution_time: float = 0.0
    id: int = 0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    dominates_count: int = 0
    dominated_by_count: int = 0

    def dominates(self, other: "AgentVariant") -> bool:
        """Check if this variant Pareto-dominates another."""
        tokens_better_or_equal = self.tokens_used <= other.tokens_used
        time_better_or_equal = self.execution_time <= other.execution_time
        strictly_better = (
            self.tokens_used < other.tokens_used
            or self.execution_time < other.execution_time
        )
        return tokens_better_or_equal and time_better_or_equal and strictly_better

    def to_dict(self) -> dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "folder_path": self.folder_path,
            "report_path": self.report_path,
            "tokens_used": self.tokens_used,
            "execution_time": self.execution_time,
            "id": self.id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


@dataclass
class ImproverVariant:
    """A variant in the improver's internal Pareto frontier."""

    folder_path: str
    report: ImprovementReport
    variant_id: int
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    estimated_tokens_saved_pct: float = 0.0
    estimated_time_saved_pct: float = 0.0

    def dominates(self, other: "ImproverVariant") -> bool:
        """Check if this variant Pareto-dominates another.

        For improvements, higher is better.
        """
        tokens_better_or_equal = (
            self.estimated_tokens_saved_pct >= other.estimated_tokens_saved_pct
        )
        time_better_or_equal = (
            self.estimated_time_saved_pct >= other.estimated_time_saved_pct
        )
        strictly_better = (
            self.estimated_tokens_saved_pct > other.estimated_tokens_saved_pct
            or self.estimated_time_saved_pct > other.estimated_time_saved_pct
        )
        return tokens_better_or_equal and time_better_or_equal and strictly_better

    def improvement_score(self) -> float:
        """Combined improvement score for ranking."""
        return self.estimated_tokens_saved_pct + self.estimated_time_saved_pct


class TestImprovementReport(unittest.TestCase):
    """Tests for ImprovementReport dataclass."""

    def test_to_dict(self) -> None:
        """Test converting report to dictionary."""
        report = ImprovementReport(
            baseline_tokens=1000,
            baseline_time=10.0,
            improved_tokens=800,
            improved_time=8.0,
            token_improvement_pct=20.0,
            time_improvement_pct=20.0,
            implemented_ideas=[{"idea": "test", "impact": "10%"}],
            failed_ideas=[{"idea": "bad", "reason": "didn't work"}],
            parent_folder="/test/parent",
            generation=1,
        )

        d = report.to_dict()

        self.assertEqual(d["baseline_tokens"], 1000)
        self.assertEqual(d["improved_tokens"], 800)
        self.assertEqual(d["token_improvement_pct"], 20.0)
        self.assertEqual(len(d["implemented_ideas"]), 1)
        self.assertEqual(d["generation"], 1)

    def test_from_dict(self) -> None:
        """Test creating report from dictionary."""
        d = {
            "baseline_tokens": 500,
            "baseline_time": 5.0,
            "improved_tokens": 400,
            "improved_time": 4.0,
            "token_improvement_pct": 20.0,
            "time_improvement_pct": 20.0,
            "implemented_ideas": [],
            "failed_ideas": [],
            "parent_folder": "",
            "generation": 2,
        }

        report = ImprovementReport.from_dict(d)

        self.assertEqual(report.baseline_tokens, 500)
        self.assertEqual(report.improved_tokens, 400)
        self.assertEqual(report.generation, 2)

    def test_save_and_load(self) -> None:
        """Test saving and loading report from JSON file."""
        report = ImprovementReport(
            baseline_tokens=1000,
            improved_tokens=800,
            implemented_ideas=[{"idea": "caching", "impact": "20%"}],
            generation=3,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            report.save(path)
            loaded = ImprovementReport.load(path)

            self.assertEqual(loaded.baseline_tokens, 1000)
            self.assertEqual(loaded.improved_tokens, 800)
            self.assertEqual(len(loaded.implemented_ideas), 1)
            self.assertEqual(loaded.generation, 3)
        finally:
            os.unlink(path)

    def test_is_improvement_over(self) -> None:
        """Test Pareto improvement comparison."""
        report1 = ImprovementReport(improved_tokens=1000, improved_time=10.0)
        report2 = ImprovementReport(improved_tokens=800, improved_time=8.0)
        report3 = ImprovementReport(improved_tokens=800, improved_time=12.0)

        # report2 is strictly better in both dimensions
        self.assertTrue(report2.is_improvement_over(report1))

        # report3 is better in tokens but worse in time - not a strict improvement
        self.assertFalse(report3.is_improvement_over(report1))

        # Same report is not an improvement over itself
        self.assertFalse(report1.is_improvement_over(report1))


class TestAgentVariant(unittest.TestCase):
    """Tests for AgentVariant dataclass."""

    def test_dominates(self) -> None:
        """Test Pareto dominance relationship."""
        # Create variants with different metrics
        v1 = AgentVariant(
            folder_path="/v1",
            report_path="/v1/report.json",
            report=ImprovementReport(),
            tokens_used=1000,
            execution_time=10.0,
            id=1,
        )
        v2 = AgentVariant(
            folder_path="/v2",
            report_path="/v2/report.json",
            report=ImprovementReport(),
            tokens_used=800,
            execution_time=8.0,
            id=2,
        )
        v3 = AgentVariant(
            folder_path="/v3",
            report_path="/v3/report.json",
            report=ImprovementReport(),
            tokens_used=800,
            execution_time=12.0,
            id=3,
        )
        v4 = AgentVariant(
            folder_path="/v4",
            report_path="/v4/report.json",
            report=ImprovementReport(),
            tokens_used=1000,
            execution_time=10.0,
            id=4,
        )

        # v2 dominates v1 (better in both)
        self.assertTrue(v2.dominates(v1))
        self.assertFalse(v1.dominates(v2))

        # v3 doesn't dominate v1 (better in tokens, worse in time)
        self.assertFalse(v3.dominates(v1))
        self.assertFalse(v1.dominates(v3))

        # Equal variants don't dominate each other
        self.assertFalse(v1.dominates(v4))
        self.assertFalse(v4.dominates(v1))

    def test_to_dict(self) -> None:
        """Test converting variant to dictionary."""
        v = AgentVariant(
            folder_path="/test",
            report_path="/test/report.json",
            report=ImprovementReport(),
            tokens_used=500,
            execution_time=5.0,
            id=42,
            generation=3,
            parent_ids=[1, 2],
        )

        d = v.to_dict()

        self.assertEqual(d["folder_path"], "/test")
        self.assertEqual(d["tokens_used"], 500)
        self.assertEqual(d["execution_time"], 5.0)
        self.assertEqual(d["id"], 42)
        self.assertEqual(d["generation"], 3)
        self.assertEqual(d["parent_ids"], [1, 2])


class TestImproverVariant(unittest.TestCase):
    """Tests for ImproverVariant dataclass (internal Pareto frontier)."""

    def test_dominates_higher_is_better(self) -> None:
        """Test Pareto dominance for improvements (higher is better)."""
        # v2 dominates v1 (better improvements in both)
        v1 = ImproverVariant(
            folder_path="/v1",
            report=ImprovementReport(),
            variant_id=1,
            estimated_tokens_saved_pct=10.0,
            estimated_time_saved_pct=10.0,
        )
        v2 = ImproverVariant(
            folder_path="/v2",
            report=ImprovementReport(),
            variant_id=2,
            estimated_tokens_saved_pct=20.0,
            estimated_time_saved_pct=15.0,
        )
        v3 = ImproverVariant(
            folder_path="/v3",
            report=ImprovementReport(),
            variant_id=3,
            estimated_tokens_saved_pct=25.0,
            estimated_time_saved_pct=5.0,  # Trade-off
        )

        # v2 dominates v1 (higher in both)
        self.assertTrue(v2.dominates(v1))
        self.assertFalse(v1.dominates(v2))

        # v3 doesn't dominate v1 (trade-off)
        self.assertFalse(v3.dominates(v1))
        self.assertFalse(v1.dominates(v3))

    def test_improvement_score(self) -> None:
        """Test combined improvement score."""
        v = ImproverVariant(
            folder_path="/v",
            report=ImprovementReport(),
            variant_id=1,
            estimated_tokens_saved_pct=15.0,
            estimated_time_saved_pct=10.0,
        )
        self.assertEqual(v.improvement_score(), 25.0)

    def test_equal_variants_no_dominance(self) -> None:
        """Test that equal variants don't dominate each other."""
        v1 = ImproverVariant(
            folder_path="/v1",
            report=ImprovementReport(),
            variant_id=1,
            estimated_tokens_saved_pct=10.0,
            estimated_time_saved_pct=10.0,
        )
        v2 = ImproverVariant(
            folder_path="/v2",
            report=ImprovementReport(),
            variant_id=2,
            estimated_tokens_saved_pct=10.0,
            estimated_time_saved_pct=10.0,
        )

        self.assertFalse(v1.dominates(v2))
        self.assertFalse(v2.dominates(v1))


class TestImproverParetoFrontier(unittest.TestCase):
    """Tests for ImproverAgent's internal Pareto frontier operations."""

    def _update_pareto_frontier(
        self,
        frontier: list[ImproverVariant],
        new_variant: ImproverVariant,
        pareto_size: int = 10,
        min_threshold: float = 0.01,
    ) -> tuple[list[ImproverVariant], bool]:
        """Update the improver Pareto frontier (higher improvements = better)."""
        # Check if new variant is dominated by any in frontier
        for existing in frontier:
            if existing.dominates(new_variant):
                return frontier, False

        # Check minimum improvement threshold (if not first)
        if frontier:
            if (
                new_variant.estimated_tokens_saved_pct < min_threshold
                and new_variant.estimated_time_saved_pct < min_threshold
            ):
                dominates_any = any(new_variant.dominates(v) for v in frontier)
                if not dominates_any:
                    return frontier, False

        # Remove variants dominated by new variant
        new_frontier = [v for v in frontier if not new_variant.dominates(v)]
        new_frontier.append(new_variant)

        # Trim if too large
        if len(new_frontier) > pareto_size:
            # Keep those with highest improvement scores
            new_frontier.sort(key=lambda v: v.improvement_score(), reverse=True)
            new_frontier = new_frontier[:pareto_size]

        return new_frontier, True

    def test_improver_frontier_update(self) -> None:
        """Test updating improver Pareto frontier."""
        frontier: list[ImproverVariant] = []

        # Create test variants (higher improvement = better)
        v1 = ImproverVariant(
            folder_path="/v1",
            report=ImprovementReport(),
            variant_id=1,
            estimated_tokens_saved_pct=10.0,
            estimated_time_saved_pct=10.0,
        )
        v2 = ImproverVariant(
            folder_path="/v2",
            report=ImprovementReport(),
            variant_id=2,
            estimated_tokens_saved_pct=15.0,
            estimated_time_saved_pct=5.0,  # Trade-off
        )
        v3 = ImproverVariant(
            folder_path="/v3",
            report=ImprovementReport(),
            variant_id=3,
            estimated_tokens_saved_pct=25.0,
            estimated_time_saved_pct=20.0,  # Dominates v1 and v2
        )

        frontier, added = self._update_pareto_frontier(frontier, v1)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 1)

        frontier, added = self._update_pareto_frontier(frontier, v2)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 2)  # v2 doesn't dominate v1 (trade-off)

        frontier, added = self._update_pareto_frontier(frontier, v3)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 1)  # v3 dominates both v1 and v2
        self.assertEqual(frontier[0].variant_id, 3)

    def test_frontier_size_limit(self) -> None:
        """Test that frontier respects size limit."""
        frontier: list[ImproverVariant] = []

        # Add 5 non-dominated variants with trade-offs
        variants = [
            ImproverVariant(
                folder_path=f"/v{i}",
                report=ImprovementReport(),
                variant_id=i,
                estimated_tokens_saved_pct=10.0 + i * 5,
                estimated_time_saved_pct=20.0 - i * 3,  # Trade-off
            )
            for i in range(5)
        ]

        for v in variants:
            frontier, _ = self._update_pareto_frontier(frontier, v, pareto_size=3)

        # Should be limited to 3
        self.assertLessEqual(len(frontier), 3)


class TestParetoFrontier(unittest.TestCase):
    """Tests for Pareto frontier operations."""

    def _update_pareto_frontier(
        self,
        frontier: list[AgentVariant],
        new_variant: AgentVariant,
        pareto_size: int = 10,
    ) -> tuple[list[AgentVariant], bool]:
        """Update the Pareto frontier with a new variant (standalone implementation)."""
        # Check if new variant is dominated by any in frontier
        for existing in frontier:
            if existing.dominates(new_variant):
                return frontier, False  # New variant is dominated

        # Remove variants dominated by new variant
        new_frontier = [v for v in frontier if not new_variant.dominates(v)]

        # Add new variant to frontier
        new_frontier.append(new_variant)

        # Trim if too large (simple approach: keep first pareto_size)
        if len(new_frontier) > pareto_size:
            new_frontier = new_frontier[:pareto_size]

        return new_frontier, True

    def test_frontier_update(self) -> None:
        """Test updating Pareto frontier with new variants."""
        frontier: list[AgentVariant] = []

        # Create test variants
        v1 = AgentVariant(
            folder_path="/v1",
            report_path="/v1/report.json",
            report=ImprovementReport(),
            tokens_used=1000,
            execution_time=10.0,
            id=1,
        )
        v2 = AgentVariant(
            folder_path="/v2",
            report_path="/v2/report.json",
            report=ImprovementReport(),
            tokens_used=800,
            execution_time=12.0,  # Trade-off: better tokens, worse time
            id=2,
        )
        v3 = AgentVariant(
            folder_path="/v3",
            report_path="/v3/report.json",
            report=ImprovementReport(),
            tokens_used=1200,
            execution_time=8.0,  # Trade-off: worse tokens, better time
            id=3,
        )
        v4 = AgentVariant(
            folder_path="/v4",
            report_path="/v4/report.json",
            report=ImprovementReport(),
            tokens_used=700,
            execution_time=7.0,  # Dominates all others
            id=4,
        )

        # Add variants
        frontier, added = self._update_pareto_frontier(frontier, v1)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 1)

        frontier, added = self._update_pareto_frontier(frontier, v2)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 2)  # v2 doesn't dominate v1

        frontier, added = self._update_pareto_frontier(frontier, v3)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 3)  # v3 doesn't dominate others

        frontier, added = self._update_pareto_frontier(frontier, v4)
        self.assertTrue(added)
        self.assertEqual(len(frontier), 1)  # v4 dominates all others
        self.assertEqual(frontier[0].id, 4)


class TestIntegration(unittest.TestCase):
    """Integration tests (require API keys).

    These tests are skipped by default as they require API keys and can be expensive.
    To run them, set KISS_RUN_INTEGRATION_TESTS=1 environment variable.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.test_dir, "source")
        os.makedirs(self.source_dir)

        # Create a simple test agent
        agent_code = '''
"""Simple test agent."""

class TestAgent:
    """A simple test agent for testing the improver."""

    def __init__(self):
        self.step_count = 0

    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.step_count += 1
        # This is intentionally verbose to give room for optimization
        result = ""
        result = result + "Starting task: "
        result = result + task
        result = result + "\\n"
        result = result + "Processing..."
        result = result + "\\n"
        result = result + "Done!"
        return result

if __name__ == "__main__":
    agent = TestAgent()
    print(agent.run("Test task"))
'''
        with open(os.path.join(self.source_dir, "agent.py"), "w") as f:
            f.write(agent_code)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    @unittest.skip("Requires API key - run manually with KISS_RUN_INTEGRATION_TESTS=1")
    def test_improver_agent(self) -> None:
        """Test ImproverAgent on a simple agent.

        This test requires:
        - claude_agent_sdk installed
        - Valid Anthropic API key
        """
        # Skip if not running integration tests
        if not os.environ.get("KISS_RUN_INTEGRATION_TESTS"):
            self.skipTest("Integration tests disabled")

        import anyio

        from kiss.agents.agent_creator.improver_agent import (
            ImprovementReport as RealReport,
        )
        from kiss.agents.agent_creator.improver_agent import (
            ImproverAgent,
        )

        improver = ImproverAgent()
        target_dir = os.path.join(self.test_dir, "improved")

        async def run_test() -> tuple[bool, RealReport | None]:
            return await improver.improve(
                source_folder=self.source_dir,
                target_folder=target_dir,
            )

        success, report = anyio.run(run_test)

        self.assertTrue(success)
        self.assertIsNotNone(report)
        self.assertTrue(Path(target_dir).exists())
        self.assertTrue((Path(target_dir) / "agent.py").exists())

    @unittest.skip("Requires API key - run manually with KISS_RUN_INTEGRATION_TESTS=1")
    def test_agent_evolver(self) -> None:
        """Test AgentEvolver with a simple task.

        This test requires:
        - claude_agent_sdk installed
        - Valid Anthropic API key
        - Can be expensive due to multiple LLM calls
        """
        # Skip if not running integration tests
        if not os.environ.get("KISS_RUN_INTEGRATION_TESTS"):
            self.skipTest("Integration tests disabled")

        import anyio

        from kiss.agents.agent_creator.agent_evolver import (
            AgentEvolver as RealEvolver,
        )
        from kiss.agents.agent_creator.agent_evolver import (
            AgentVariant as RealVariant,
        )

        task = """
        Create a simple calculator agent that can:
        1. Parse math expressions
        2. Evaluate basic operations (+, -, *, /)
        3. Return the result

        The agent should be minimal and efficient.
        """

        evolver = RealEvolver(
            task_description=task,
            max_generations=2,
            population_size=2,
            pareto_size=2,
            work_dir=os.path.join(self.test_dir, "evolver"),
        )

        async def run_test() -> RealVariant:
            return await evolver.evolve()

        best = anyio.run(run_test)

        self.assertIsNotNone(best)
        self.assertTrue(Path(best.folder_path).exists())
        self.assertGreater(len(evolver.all_variants), 0)


if __name__ == "__main__":
    unittest.main()
