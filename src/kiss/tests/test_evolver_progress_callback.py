# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Integration tests for AgentEvolver progress_callback functionality.

Tests the EvolverPhase enum, EvolverProgress dataclass, create_progress_callback
helper, and the full callback integration in AgentEvolver.evolve() without
any mocks or patches. Uses a TestableAgentEvolver subclass that creates real
agent files and evaluates them in-process.
"""

import os
import unittest

from kiss.agents.create_and_optimize_agent.agent_evolver import (
    AgentEvolver,
    AgentVariant,
    EvolverPhase,
    EvolverProgress,
    create_progress_callback,
)
from kiss.agents.create_and_optimize_agent.improver_agent import (
    ImprovementReport,
)

SIMPLE_AGENT_CODE = '''
def agent_run(task: str) -> dict:
    return {
        "metrics": {
            "success": 0,
            "tokens_used": 100,
            "execution_time": 1.5,
        }
    }
'''

VARIANT_AGENT_TEMPLATE = '''
def agent_run(task: str) -> dict:
    return {{
        "metrics": {{
            "success": {success},
            "tokens_used": {tokens},
            "execution_time": {time},
        }}
    }}
'''


def _create_agent_dir(base_dir: str, variant_id: int,
                      success: int = 0, tokens: int = 100,
                      time: float = 1.5) -> tuple[str, str]:
    """Create a real agent directory with agent.py and report files."""
    folder = os.path.join(base_dir, f"variant_{variant_id}")
    os.makedirs(folder, exist_ok=True)

    agent_code = VARIANT_AGENT_TEMPLATE.format(
        success=success, tokens=tokens, time=time,
    )
    with open(os.path.join(folder, "agent.py"), "w") as f:
        f.write(agent_code)

    with open(os.path.join(folder, "__init__.py"), "w") as f:
        f.write("")

    report = ImprovementReport(
        metrics={"tokens_used": tokens, "execution_time": time},
        implemented_ideas=[{"idea": "test", "source": "test"}],
        failed_ideas=[],
        generation=0,
    )
    report_path = os.path.join(folder, "improvement_report.json")
    report.save(report_path)
    return folder, report_path


class TestableAgentEvolver(AgentEvolver):
    """A real AgentEvolver subclass that uses fast local agent files.

    Overrides _create_initial_agent and _mutate/_crossover to produce real
    agent directories without invoking LLM calls, while preserving all
    progress callback behavior.
    """

    def __init__(self, agent_metrics: list[dict[str, float]] | None = None):
        super().__init__()
        self._test_metrics_iter = iter(agent_metrics or [])
        self._default_metrics = {"success": 0, "tokens_used": 100, "execution_time": 1.5}

    def _next_test_metrics(self) -> dict[str, float]:
        try:
            return next(self._test_metrics_iter)
        except StopIteration:
            return self._default_metrics.copy()

    def _create_initial_agent(self, variant_id: int) -> AgentVariant:
        metrics = self._next_test_metrics()
        folder, report_path = _create_agent_dir(
            str(self.work_dir), variant_id,
            success=int(metrics.get("success", 0)),
            tokens=int(metrics.get("tokens_used", 100)),
            time=metrics.get("execution_time", 1.5),
        )
        report = ImprovementReport(
            metrics=metrics,
            implemented_ideas=[{"idea": "Initial implementation", "source": "initial"}],
            failed_ideas=[],
            generation=0,
        )
        report.save(report_path)
        return AgentVariant(
            folder_path=folder,
            report_path=report_path,
            report=report,
            metrics={},
            id=variant_id,
            generation=0,
            parent_ids=[],
        )

    def _mutate(self, variant: AgentVariant) -> AgentVariant | None:
        new_id = self._next_variant_id()
        metrics = self._next_test_metrics()
        folder, report_path = _create_agent_dir(
            str(self.work_dir), new_id,
            success=int(metrics.get("success", 0)),
            tokens=int(metrics.get("tokens_used", 100)),
            time=metrics.get("execution_time", 1.5),
        )
        report = ImprovementReport(
            metrics=metrics,
            implemented_ideas=[{"idea": "Mutation improvement", "source": "mutation"}],
            failed_ideas=[],
            generation=self._generation,
        )
        report.save(report_path)
        return AgentVariant(
            folder_path=folder,
            report_path=report_path,
            report=report,
            metrics={},
            id=new_id,
            generation=self._generation,
            parent_ids=[variant.id],
        )

    def _crossover(self, primary: AgentVariant, secondary: AgentVariant) -> AgentVariant | None:
        new_id = self._next_variant_id()
        metrics = self._next_test_metrics()
        folder, report_path = _create_agent_dir(
            str(self.work_dir), new_id,
            success=int(metrics.get("success", 0)),
            tokens=int(metrics.get("tokens_used", 100)),
            time=metrics.get("execution_time", 1.5),
        )
        report = ImprovementReport(
            metrics=metrics,
            implemented_ideas=[{"idea": "Crossover", "source": "crossover"}],
            failed_ideas=[],
            generation=self._generation,
        )
        report.save(report_path)
        return AgentVariant(
            folder_path=folder,
            report_path=report_path,
            report=report,
            metrics={},
            id=new_id,
            generation=self._generation,
            parent_ids=[primary.id, secondary.id],
        )


class TestEvolverProgressDataclass(unittest.TestCase):
    """Test EvolverProgress dataclass structure."""

    def test_progress_has_required_fields(self):
        progress = EvolverProgress(
            generation=1,
            max_generations=10,
            phase=EvolverPhase.EVALUATING,
            variant_id=5,
            parent_ids=[1, 2],
            frontier_size=3,
            best_score=1500.0,
            current_metrics={"success": 0, "tokens_used": 100},
            added_to_frontier=True,
            message="Testing progress",
        )

        self.assertEqual(progress.generation, 1)
        self.assertEqual(progress.max_generations, 10)
        self.assertEqual(progress.phase, EvolverPhase.EVALUATING)
        self.assertEqual(progress.variant_id, 5)
        self.assertEqual(progress.parent_ids, [1, 2])
        self.assertEqual(progress.frontier_size, 3)
        self.assertEqual(progress.best_score, 1500.0)
        self.assertEqual(progress.current_metrics, {"success": 0, "tokens_used": 100})
        self.assertTrue(progress.added_to_frontier)
        self.assertEqual(progress.message, "Testing progress")

    def test_progress_default_values(self):
        progress = EvolverProgress(
            generation=0,
            max_generations=5,
            phase=EvolverPhase.INITIALIZING,
        )

        self.assertIsNone(progress.variant_id)
        self.assertEqual(progress.parent_ids, [])
        self.assertEqual(progress.frontier_size, 0)
        self.assertIsNone(progress.best_score)
        self.assertEqual(progress.current_metrics, {})
        self.assertIsNone(progress.added_to_frontier)
        self.assertEqual(progress.message, "")

    def test_progress_default_list_not_shared(self):
        p1 = EvolverProgress(generation=0, max_generations=1, phase=EvolverPhase.EVALUATING)
        p2 = EvolverProgress(generation=0, max_generations=1, phase=EvolverPhase.EVALUATING)
        p1.parent_ids.append(99)
        self.assertEqual(p2.parent_ids, [])

    def test_progress_default_dict_not_shared(self):
        p1 = EvolverProgress(generation=0, max_generations=1, phase=EvolverPhase.EVALUATING)
        p2 = EvolverProgress(generation=0, max_generations=1, phase=EvolverPhase.EVALUATING)
        p1.current_metrics["x"] = 1.0
        self.assertEqual(p2.current_metrics, {})


class TestEvolverPhaseEnum(unittest.TestCase):
    """Test EvolverPhase enum values."""

    def test_phase_enum_values(self):
        expected = {
            "initializing", "evaluating", "mutation",
            "crossover", "pareto_update", "complete",
        }
        actual = {phase.value for phase in EvolverPhase}
        self.assertEqual(actual, expected)

    def test_phase_enum_names(self):
        self.assertEqual(EvolverPhase.INITIALIZING.value, "initializing")
        self.assertEqual(EvolverPhase.EVALUATING.value, "evaluating")
        self.assertEqual(EvolverPhase.MUTATION.value, "mutation")
        self.assertEqual(EvolverPhase.CROSSOVER.value, "crossover")
        self.assertEqual(EvolverPhase.PARETO_UPDATE.value, "pareto_update")
        self.assertEqual(EvolverPhase.COMPLETE.value, "complete")

    def test_all_phases_accessible_by_value(self):
        for val in ["initializing", "evaluating", "mutation",
                     "crossover", "pareto_update", "complete"]:
            phase = EvolverPhase(val)
            self.assertIsNotNone(phase)


class TestCreateProgressCallback(unittest.TestCase):
    """Test the create_progress_callback helper function."""

    def test_returns_callable(self):
        cb = create_progress_callback()
        self.assertTrue(callable(cb))

    def test_verbose_false_prints_eval_complete(self, ):
        import io
        import sys
        cb = create_progress_callback(verbose=False)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            # Eval with metrics (complete) should print
            cb(EvolverProgress(
                generation=1, max_generations=5, phase=EvolverPhase.EVALUATING,
                current_metrics={"success": 0, "tokens_used": 100},
                message="Variant 1: success=0",
            ))
            output = buf.getvalue()
            self.assertIn("evaluating", output)
            self.assertIn("Variant 1", output)
        finally:
            sys.stdout = old_stdout

    def test_verbose_false_suppresses_init_phase(self):
        import io
        import sys
        cb = create_progress_callback(verbose=False)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=0, max_generations=5, phase=EvolverPhase.INITIALIZING,
                message="Creating initial variant",
            ))
            output = buf.getvalue()
            self.assertEqual(output, "")
        finally:
            sys.stdout = old_stdout

    def test_verbose_false_prints_pareto_update(self):
        import io
        import sys
        cb = create_progress_callback(verbose=False)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=1, max_generations=5, phase=EvolverPhase.PARETO_UPDATE,
                message="Added variant 2",
            ))
            output = buf.getvalue()
            self.assertIn("pareto_update", output)
        finally:
            sys.stdout = old_stdout

    def test_verbose_false_prints_complete(self):
        import io
        import sys
        cb = create_progress_callback(verbose=False)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=5, max_generations=5, phase=EvolverPhase.COMPLETE,
                message="Done",
            ))
            output = buf.getvalue()
            self.assertIn("complete", output)
        finally:
            sys.stdout = old_stdout

    def test_verbose_true_prints_all_phases(self):
        import io
        import sys
        cb = create_progress_callback(verbose=True)

        for phase in EvolverPhase:
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                cb(EvolverProgress(
                    generation=1, max_generations=5, phase=phase,
                    message=f"Phase: {phase.value}",
                ))
                output = buf.getvalue()
                self.assertIn(phase.value, output)
            finally:
                sys.stdout = old_stdout

    def test_best_score_formatting_none(self):
        import io
        import sys
        cb = create_progress_callback(verbose=True)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=0, max_generations=5, phase=EvolverPhase.INITIALIZING,
                best_score=None, message="test",
            ))
            output = buf.getvalue()
            self.assertIn("N/A", output)
        finally:
            sys.stdout = old_stdout

    def test_best_score_formatting_value(self):
        import io
        import sys
        cb = create_progress_callback(verbose=True)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=1, max_generations=5, phase=EvolverPhase.EVALUATING,
                best_score=1234.56, message="test",
            ))
            output = buf.getvalue()
            self.assertIn("1234.56", output)
        finally:
            sys.stdout = old_stdout

    def test_verbose_false_suppresses_eval_without_metrics(self):
        import io
        import sys
        cb = create_progress_callback(verbose=False)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            cb(EvolverProgress(
                generation=1, max_generations=5, phase=EvolverPhase.EVALUATING,
                current_metrics={},
                message="Evaluating variant 3",
            ))
            output = buf.getvalue()
            self.assertEqual(output, "")
        finally:
            sys.stdout = old_stdout


class TestEvolverCallbackIntegration(unittest.TestCase):
    """Integration tests for progress_callback in AgentEvolver.evolve()."""

    def setUp(self):
        self.original_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_callback_called_during_evolve(self):
        progress_updates: list[EvolverProgress] = []

        def callback(p: EvolverProgress) -> None:
            progress_updates.append(p)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(progress_updates), 0)

    def test_callback_not_called_when_none(self):
        evolver = TestableAgentEvolver()
        best = evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=None,
        )
        self.assertIsNotNone(best)

    def test_all_phases_seen_across_full_evolution(self):
        phases_seen: set[EvolverPhase] = set()

        def callback(p: EvolverProgress) -> None:
            phases_seen.add(p.phase)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        # Need 2+ in frontier for crossover; mutation_probability=0 forces crossover
        metrics = [
            {"success": 0, "tokens_used": 100, "execution_time": 1.0},
            {"success": 0, "tokens_used": 200, "execution_time": 0.5},
            {"success": 0, "tokens_used": 150, "execution_time": 0.8},
            {"success": 0, "tokens_used": 120, "execution_time": 0.9},
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=2,
            initial_frontier_size=2,
            max_frontier_size=4,
            mutation_probability=0.0,
            progress_callback=callback,
        )

        expected = {
            EvolverPhase.INITIALIZING,
            EvolverPhase.EVALUATING,
            EvolverPhase.PARETO_UPDATE,
            EvolverPhase.MUTATION,
            EvolverPhase.CROSSOVER,
            EvolverPhase.COMPLETE,
        }
        self.assertTrue(expected.issubset(phases_seen))

    def test_callback_generation_tracking_init_is_zero(self):
        generations: list[int] = []

        def callback(p: EvolverProgress) -> None:
            generations.append(p.generation)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        # First callbacks are during initialization (generation=0)
        self.assertEqual(generations[0], 0)

    def test_callback_generation_tracking_evolve_one_indexed(self):
        gen_phase_pairs: list[tuple[int, EvolverPhase]] = []

        def callback(p: EvolverProgress) -> None:
            gen_phase_pairs.append((p.generation, p.phase))

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=2,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        evolution_gens = {g for g, phase in gen_phase_pairs
                         if phase in (EvolverPhase.MUTATION, EvolverPhase.CROSSOVER)}
        self.assertTrue(evolution_gens.issubset({1, 2}))

    def test_callback_max_generations_consistent(self):
        max_gens_seen: set[int] = set()

        def callback(p: EvolverProgress) -> None:
            max_gens_seen.add(p.max_generations)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=3,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertEqual(max_gens_seen, {3})

    def test_callback_frontier_size_increases_during_init(self):
        frontier_sizes: list[int] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.PARETO_UPDATE:
                frontier_sizes.append(p.frontier_size)

        metrics = [
            {"success": 0, "tokens_used": 100, "execution_time": 1.0},
            {"success": 0, "tokens_used": 200, "execution_time": 0.5},
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=2,
            max_frontier_size=4,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        # Frontier size should increase as init variants are added
        self.assertGreater(len(frontier_sizes), 0)
        self.assertGreaterEqual(frontier_sizes[-1], 1)

    def test_callback_best_score_progression(self):
        best_scores: list[float | None] = []

        def callback(p: EvolverProgress) -> None:
            best_scores.append(p.best_score)

        # Provide improving metrics
        metrics = [
            {"success": 0, "tokens_used": 500, "execution_time": 5.0},
            {"success": 0, "tokens_used": 100, "execution_time": 1.0},
            {"success": 0, "tokens_used": 50, "execution_time": 0.5},
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=2,
            initial_frontier_size=1,
            max_frontier_size=4,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertIsNone(best_scores[0])
        non_none = [s for s in best_scores if s is not None]
        self.assertGreater(len(non_none), 0)
        for i in range(1, len(non_none)):
            self.assertLessEqual(non_none[i], non_none[i - 1])

    def test_callback_variant_id_set_on_eval(self):
        variant_ids: list[int | None] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.EVALUATING and p.variant_id is not None:
                variant_ids.append(p.variant_id)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(variant_ids), 0)
        for vid in variant_ids:
            self.assertIsInstance(vid, int)

    def test_callback_parent_ids_on_mutation(self):
        parent_ids_collected: list[list[int]] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.MUTATION:
                parent_ids_collected.append(p.parent_ids)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        # Mutation reports the parent variant (which has no parents itself)
        self.assertGreater(len(parent_ids_collected), 0)

    def test_callback_parent_ids_on_crossover(self):
        parent_ids_collected: list[list[int]] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.CROSSOVER:
                parent_ids_collected.append(p.parent_ids)

        metrics = [
            {"success": 0, "tokens_used": 100, "execution_time": 1.0},
            {"success": 0, "tokens_used": 200, "execution_time": 0.5},
            {"success": 0, "tokens_used": 150, "execution_time": 0.8},
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=2,
            max_frontier_size=4,
            mutation_probability=0.0,
            progress_callback=callback,
        )

        # Crossover reports the primary variant's parent_ids (empty for init)
        self.assertGreater(len(parent_ids_collected), 0)

    def test_callback_metrics_populated_after_eval(self):
        metrics_collected: list[dict[str, float]] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.EVALUATING and p.current_metrics:
                metrics_collected.append(p.current_metrics)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(metrics_collected), 0)
        for m in metrics_collected:
            self.assertIn("success", m)
            self.assertIn("tokens_used", m)

    def test_callback_added_to_frontier_bool(self):
        added_values: list[bool | None] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.PARETO_UPDATE:
                added_values.append(p.added_to_frontier)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(added_values), 0)
        for v in added_values:
            self.assertIsInstance(v, bool)

    def test_callback_messages_are_descriptive(self):
        messages: list[str] = []

        def callback(p: EvolverProgress) -> None:
            messages.append(p.message)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        non_empty = [m for m in messages if m]
        self.assertGreater(len(non_empty), 0)
        found_variant = any("variant" in m.lower() for m in messages)
        self.assertTrue(found_variant)

    def test_callback_complete_message_contains_best(self):
        complete_msgs: list[str] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.COMPLETE:
                complete_msgs.append(p.message)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertEqual(len(complete_msgs), 1)
        self.assertIn("complete", complete_msgs[0].lower())
        self.assertIn("variant", complete_msgs[0].lower())

    def test_callback_with_multiple_generations(self):
        generation_counts: dict[int, int] = {}

        def callback(p: EvolverProgress) -> None:
            generation_counts[p.generation] = generation_counts.get(p.generation, 0) + 1

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=3,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        # Should see generation 0 (init), 1, 2, 3 (evolution + complete)
        self.assertIn(0, generation_counts)
        self.assertIn(1, generation_counts)
        self.assertIn(2, generation_counts)
        self.assertIn(3, generation_counts)

    def test_callback_with_larger_initial_frontier(self):
        init_variant_ids: list[int | None] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.INITIALIZING:
                init_variant_ids.append(p.variant_id)

        metrics = [
            {"success": 0, "tokens_used": 100, "execution_time": 1.0},
            {"success": 0, "tokens_used": 200, "execution_time": 0.5},
            {"success": 0, "tokens_used": 150, "execution_time": 0.8},
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=3,
            max_frontier_size=4,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertEqual(len(init_variant_ids), 3)

    def test_callback_phase_ordering(self):
        def run_and_collect(evolver, generation, **kwargs):
            phases: list[EvolverPhase] = []

            def callback(p: EvolverProgress) -> None:
                if p.generation == generation:
                    phases.append(p.phase)

            evolver.evolve(
                task_description="test task",
                max_generations=1,
                progress_callback=callback,
                **kwargs,
            )
            return phases

        with self.subTest("init"):
            evolver = TestableAgentEvolver()
            phase_order = run_and_collect(
                evolver,
                generation=0,
                initial_frontier_size=1,
                max_frontier_size=2,
                mutation_probability=1.0,
            )
            self.assertGreaterEqual(len(phase_order), 3)
            first_init = phase_order.index(EvolverPhase.INITIALIZING)
            first_eval = phase_order.index(EvolverPhase.EVALUATING)
            first_pareto = phase_order.index(EvolverPhase.PARETO_UPDATE)
            self.assertLess(first_init, first_eval)
            self.assertLess(first_eval, first_pareto)

        with self.subTest("mutation"):
            evolver = TestableAgentEvolver()
            gen1_phases = run_and_collect(
                evolver,
                generation=1,
                initial_frontier_size=1,
                max_frontier_size=2,
                mutation_probability=1.0,
            )
            self.assertIn(EvolverPhase.MUTATION, gen1_phases)
            self.assertIn(EvolverPhase.EVALUATING, gen1_phases)
            self.assertIn(EvolverPhase.PARETO_UPDATE, gen1_phases)
            first_mut = gen1_phases.index(EvolverPhase.MUTATION)
            first_eval = gen1_phases.index(EvolverPhase.EVALUATING)
            first_pareto = gen1_phases.index(EvolverPhase.PARETO_UPDATE)
            self.assertLess(first_mut, first_eval)
            self.assertLess(first_eval, first_pareto)

        with self.subTest("crossover"):
            metrics = [
                {"success": 0, "tokens_used": 100, "execution_time": 1.0},
                {"success": 0, "tokens_used": 200, "execution_time": 0.5},
                {"success": 0, "tokens_used": 150, "execution_time": 0.8},
            ]
            evolver = TestableAgentEvolver(agent_metrics=metrics)
            gen1_phases = run_and_collect(
                evolver,
                generation=1,
                initial_frontier_size=2,
                max_frontier_size=4,
                mutation_probability=0.0,
            )
            self.assertIn(EvolverPhase.CROSSOVER, gen1_phases)
            self.assertIn(EvolverPhase.EVALUATING, gen1_phases)
            self.assertIn(EvolverPhase.PARETO_UPDATE, gen1_phases)
            first_cross = gen1_phases.index(EvolverPhase.CROSSOVER)
            first_eval = gen1_phases.index(EvolverPhase.EVALUATING)
            first_pareto = gen1_phases.index(EvolverPhase.PARETO_UPDATE)
            self.assertLess(first_cross, first_eval)
            self.assertLess(first_eval, first_pareto)

    def test_evolve_returns_valid_variant_with_callback(self):
        progress_updates: list[EvolverProgress] = []

        def callback(p: EvolverProgress) -> None:
            progress_updates.append(p)

        evolver = TestableAgentEvolver()
        best = evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertIsInstance(best, AgentVariant)
        self.assertIn("success", best.metrics)
        self.assertIn("tokens_used", best.metrics)

    def test_callback_with_create_progress_callback_helper(self):
        import io
        import sys

        cb = create_progress_callback(verbose=True)
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()

        try:
            evolver = TestableAgentEvolver()
            evolver.evolve(
                task_description="test task",
                max_generations=1,
                initial_frontier_size=1,
                max_frontier_size=2,
                mutation_probability=1.0,
                progress_callback=cb,
            )
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        self.assertIn("Gen", output)

    def test_callback_frontier_size_never_exceeds_max(self):
        frontier_sizes: list[int] = []

        def callback(p: EvolverProgress) -> None:
            frontier_sizes.append(p.frontier_size)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=3,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        for size in frontier_sizes:
            self.assertLessEqual(size, 2)

    def test_callback_added_true_for_first_variant(self):
        first_pareto: list[EvolverProgress] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.PARETO_UPDATE and p.generation == 0:
                first_pareto.append(p)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(first_pareto), 0)
        self.assertTrue(first_pareto[0].added_to_frontier)

    def test_callback_with_dominated_variant(self):
        added_flags: list[bool | None] = []

        def callback(p: EvolverProgress) -> None:
            if p.phase == EvolverPhase.PARETO_UPDATE and p.generation > 0:
                added_flags.append(p.added_to_frontier)

        # Second variant dominates, third is dominated
        metrics = [
            {"success": 1, "tokens_used": 500, "execution_time": 10.0},  # initial (bad)
            {"success": 0, "tokens_used": 50, "execution_time": 0.1},    # gen1 (dominates)
            {"success": 1, "tokens_used": 1000, "execution_time": 20.0}, # gen2 (dominated)
        ]
        evolver = TestableAgentEvolver(agent_metrics=metrics)
        evolver.evolve(
            task_description="test task",
            max_generations=2,
            initial_frontier_size=1,
            max_frontier_size=4,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertGreater(len(added_flags), 0)
        # Should have at least one True (gen1 added) and possibly a False (gen2 dominated)
        self.assertIn(True, added_flags)

    def test_callback_complete_phase_is_last(self):
        all_phases: list[EvolverPhase] = []

        def callback(p: EvolverProgress) -> None:
            all_phases.append(p.phase)

        evolver = TestableAgentEvolver()
        evolver.evolve(
            task_description="test task",
            max_generations=2,
            initial_frontier_size=1,
            max_frontier_size=2,
            mutation_probability=1.0,
            progress_callback=callback,
        )

        self.assertEqual(all_phases[-1], EvolverPhase.COMPLETE)
        # COMPLETE should appear exactly once
        self.assertEqual(all_phases.count(EvolverPhase.COMPLETE), 1)


class TestReportProgressDirectly(unittest.TestCase):
    """Test _report_progress method directly on a configured evolver."""

    def setUp(self):
        self.original_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_report_progress_does_nothing_without_callback(self):
        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
            progress_callback=None,
        )
        # Should not raise
        evolver._report_progress(
            generation=0,
            phase=EvolverPhase.INITIALIZING,
            message="test",
        )

    def test_report_progress_calls_callback(self):
        received: list[EvolverProgress] = []

        def cb(p: EvolverProgress) -> None:
            received.append(p)

        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=5,
            initial_frontier_size=1,
            max_frontier_size=2,
            progress_callback=cb,
        )
        evolver._report_progress(
            generation=2,
            phase=EvolverPhase.MUTATION,
            message="hello",
        )

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].generation, 2)
        self.assertEqual(received[0].max_generations, 5)
        self.assertEqual(received[0].phase, EvolverPhase.MUTATION)
        self.assertEqual(received[0].message, "hello")

    def test_report_progress_with_variant(self):
        received: list[EvolverProgress] = []

        def cb(p: EvolverProgress) -> None:
            received.append(p)

        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=5,
            initial_frontier_size=1,
            max_frontier_size=2,
            progress_callback=cb,
        )

        variant = AgentVariant(
            folder_path="/tmp/test",
            report_path="/tmp/test/report.json",
            report=ImprovementReport(metrics={}, implemented_ideas=[], failed_ideas=[]),
            metrics={"success": 0, "tokens_used": 200},
            id=42,
            generation=1,
            parent_ids=[10, 20],
        )

        evolver._report_progress(
            generation=1,
            phase=EvolverPhase.EVALUATING,
            variant=variant,
            added_to_frontier=True,
            message="variant done",
        )

        self.assertEqual(len(received), 1)
        p = received[0]
        self.assertEqual(p.variant_id, 42)
        self.assertEqual(p.parent_ids, [10, 20])
        self.assertEqual(p.current_metrics, {"success": 0, "tokens_used": 200})
        self.assertTrue(p.added_to_frontier)

    def test_report_progress_without_variant(self):
        received: list[EvolverProgress] = []

        def cb(p: EvolverProgress) -> None:
            received.append(p)

        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=5,
            initial_frontier_size=1,
            max_frontier_size=2,
            progress_callback=cb,
        )
        evolver._report_progress(
            generation=0,
            phase=EvolverPhase.INITIALIZING,
            message="starting",
        )

        p = received[0]
        self.assertIsNone(p.variant_id)
        self.assertEqual(p.parent_ids, [])
        self.assertEqual(p.current_metrics, {})
        self.assertIsNone(p.added_to_frontier)

    def test_update_best_score_tracks_minimum(self):
        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
        )

        self.assertIsNone(evolver._best_score)

        # Add variant with high score
        v1 = AgentVariant(
            folder_path="/tmp/a",
            report_path="/tmp/a/r.json",
            report=ImprovementReport(metrics={}, implemented_ideas=[], failed_ideas=[]),
            metrics={"success": 1, "tokens_used": 500, "execution_time": 5.0},
            id=1, generation=0, parent_ids=[],
        )
        evolver.pareto_frontier.append(v1)
        evolver._update_best_score()
        first_best = evolver._best_score
        self.assertIsNotNone(first_best)

        # Add better variant
        v2 = AgentVariant(
            folder_path="/tmp/b",
            report_path="/tmp/b/r.json",
            report=ImprovementReport(metrics={}, implemented_ideas=[], failed_ideas=[]),
            metrics={"success": 0, "tokens_used": 50, "execution_time": 0.5},
            id=2, generation=1, parent_ids=[1],
        )
        evolver.pareto_frontier.append(v2)
        evolver._update_best_score()
        self.assertIsNotNone(evolver._best_score)
        assert evolver._best_score is not None  # for type narrowing
        assert first_best is not None
        self.assertLess(evolver._best_score, first_best)

    def test_update_best_score_noop_on_empty_frontier(self):
        evolver = TestableAgentEvolver()
        evolver._reset(
            task_description="test",
            max_generations=1,
            initial_frontier_size=1,
            max_frontier_size=2,
        )
        evolver._update_best_score()
        self.assertIsNone(evolver._best_score)


class TestEvolverProgressImport(unittest.TestCase):
    """Test that new types are properly exported from the package."""

    def test_import_from_package(self):
        from kiss.agents.create_and_optimize_agent import (
            EvolverPhase,
            EvolverProgress,
            create_progress_callback,
        )
        self.assertIsNotNone(EvolverPhase)
        self.assertIsNotNone(EvolverProgress)
        self.assertIsNotNone(create_progress_callback)

    def test_import_from_module(self):
        from kiss.agents.create_and_optimize_agent.agent_evolver import (
            EvolverPhase,
            EvolverProgress,
            create_progress_callback,
        )
        self.assertIsNotNone(EvolverPhase)
        self.assertIsNotNone(EvolverProgress)
        self.assertIsNotNone(create_progress_callback)


if __name__ == "__main__":
    unittest.main()
