
import os
import shutil
import unittest
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from kiss.agents.obsolete.create_and_optimize_agent.agent_evolver import (
        AgentEvolver,
        AgentVariant,
    )
    from kiss.agents.obsolete.create_and_optimize_agent.improver_agent import (
        ImprovementReport,
    )

VARIANT_AGENT_TEMPLATE = """
def agent_run(task: str) -> dict:
    return {{
        "metrics": {{
            "success": {success},
            "tokens_used": {tokens},
            "execution_time": {time},
        }}
    }}
"""


def _create_agent_dir(
    base_dir: str, variant_id: int, success: int = 0, tokens: int = 100, time: float = 1.5
) -> tuple[str, str]:
    folder = os.path.join(base_dir, f"variant_{variant_id}")
    os.makedirs(folder, exist_ok=True)

    agent_code = VARIANT_AGENT_TEMPLATE.format(
        success=success,
        tokens=tokens,
        time=time,
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


class _TestableAgentEvolver(AgentEvolver):
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
            str(self.work_dir),
            variant_id,
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
            str(self.work_dir),
            new_id,
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
            str(self.work_dir),
            new_id,
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


class TestReportProgressDirectly(unittest.TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self._work_dirs: list[str] = []

    def tearDown(self):
        os.chdir(self.original_cwd)
        for d in self._work_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _make_evolver(
        self, agent_metrics: list[dict[str, float]] | None = None, **reset_kwargs: object
    ) -> _TestableAgentEvolver:
        evolver = _TestableAgentEvolver(agent_metrics)
        defaults: dict[str, object] = {
            "task_description": "test",
            "max_generations": 1,
            "initial_frontier_size": 1,
            "max_frontier_size": 2,
        }
        defaults.update(reset_kwargs)
        evolver._reset(**defaults)  # type: ignore[arg-type]
        self._work_dirs.append(str(evolver.work_dir))
        return evolver

    def test_update_best_score_noop_on_empty_frontier(self):
        evolver = self._make_evolver()
        evolver._update_best_score()
        self.assertIsNone(evolver._best_score)


if __name__ == "__main__":
    unittest.main()
