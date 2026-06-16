"""Aider Polyglot rollout harness — drives the aider Coder API directly.

The installed ``aider-chat`` (0.86.2) does NOT ship ``aider.benchmark``, so this harness drives
the public Coder API itself: ``Coder.create(...).run(with_message=...)`` performs one edit round
against an isolated copy of an Exercism-style exercise, then we run the exercise's pytest file as
the grader-of-record. Aider + pytest IS the official grade for Polyglot, so it is computed here,
during rollout; :class:`~agent_evals.benchmarks.aider_polyglot.AiderPolyglotGrader` is a
pass-through of the verdict this harness writes into the prediction JSON.

``litellm`` and ``aider`` are imported only inside this module — both live behind the ``[aider]``
extra and must never be pulled in by core modules.

Savings capture
---------------
aider issues its model calls through litellm. We register a litellm ``CustomLogger`` (once,
idempotently) whose ``log_success_event`` reads the per-response Headroom headers off
``response_obj._hidden_params["additional_headers"]`` (litellm prefixes provider headers with
``llm_provider-``), strips that prefix, and feeds the result to
:func:`~agent_evals.metrics.savings.parse_savings_headers`. The current task is identified via a
module-level :class:`contextvars.ContextVar` that the worker thread sets (``_active_task``) around
the Coder loop.

Concurrency caveat
------------------
litellm invokes the success callback synchronously, on the same thread that made the request, so
inside one worker thread the contextvar correctly attributes every response to the task that
thread is running. This holds for orchestrator concurrency == 1 within an arm. At concurrency > 1
in one arm, multiple worker threads share the single process-wide callback; the contextvar is
per-thread so attribution is still correct PROVIDED litellm keeps calling the callback inline on
the requesting thread (it does today). Until per-call-id correlation is added we recommend running
the aider arm at concurrency == 1.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_evals.config import Settings
from agent_evals.logging import get_logger
from agent_evals.metrics.savings import SavingsStore, parse_savings_headers
from agent_evals.models import ArmName, BenchTask, Pricing, Provider, RolloutResult

logger = get_logger("harnesses.aider")

# litellm prefixes upstream provider response headers with this when surfacing them on
# ``response_obj._hidden_params["additional_headers"]``. We strip it before parsing so the keys
# match what ``parse_savings_headers`` expects (the raw ``x-headroom-*`` names).
_LITELLM_PROVIDER_HEADER_PREFIX = "llm_provider-"
_HIDDEN_PARAMS_ATTR = "_hidden_params"
_ADDITIONAL_HEADERS_KEY = "additional_headers"

# Env var the arm hands us (its proxy base_url) vs the var litellm actually reads. We translate
# one to the other per provider before constructing the Model.
_ARM_ENV_BY_PROVIDER: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_BASE_URL",
    Provider.OPENAI: "OPENAI_BASE_URL",
}
_LITELLM_ENV_BY_PROVIDER: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_BASE",
    Provider.OPENAI: "OPENAI_API_BASE",
}

# Identifies the task a litellm response belongs to. Set per worker thread around the Coder loop.
current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_evals_aider_current_task_id", default=None
)


@contextmanager
def _active_task(task_id: str) -> Iterator[None]:
    """Bind ``current_task_id`` to ``task_id`` for the duration of the block (then reset)."""

    token = current_task_id.set(task_id)
    try:
        yield
    finally:
        current_task_id.reset(token)


def _strip_provider_prefix(headers: Mapping[str, Any]) -> dict[str, str]:
    """Drop litellm's ``llm_provider-`` prefix and stringify values for header parsing.

    Keys without the prefix are passed through unchanged so the parser can still see plain
    ``x-headroom-*`` headers if litellm ever surfaces them un-prefixed.
    """

    out: dict[str, str] = {}
    for key, value in headers.items():
        name = str(key)
        if name.startswith(_LITELLM_PROVIDER_HEADER_PREFIX):
            name = name[len(_LITELLM_PROVIDER_HEADER_PREFIX) :]
        out[name] = str(value)
    return out


def _extract_additional_headers(response_obj: Any) -> dict[str, Any] | None:
    """Pull ``_hidden_params["additional_headers"]`` off a litellm response, or ``None``."""

    hidden = getattr(response_obj, _HIDDEN_PARAMS_ATTR, None)
    if not isinstance(hidden, Mapping):
        return None
    headers = hidden.get(_ADDITIONAL_HEADERS_KEY)
    if not isinstance(headers, Mapping):
        return None
    return dict(headers)


def make_savings_logger(
    store: SavingsStore,
    task_id_getter: Callable[[], str | None],
    pricing: Pricing,
) -> Any:
    """Build a litellm ``CustomLogger`` that records per-response Headroom savings.

    Closes over ``store``, ``task_id_getter`` and ``pricing`` so a single registered instance
    serves every task: on each successful litellm call it resolves the active task id, reads the
    Headroom headers off the response, and (when both resolve) stores the parsed savings. Nothing
    is recorded when no task is active or the response carries no Headroom headers — savings are
    never fabricated. ``litellm`` is imported lazily here to keep it out of core modules.
    """

    from litellm.integrations.custom_logger import CustomLogger

    class _HeadroomSavingsLogger(CustomLogger):  # type: ignore[misc]
        def log_success_event(
            self,
            kwargs: dict[str, Any],
            response_obj: Any,
            start_time: Any,
            end_time: Any,
        ) -> None:
            task_id = task_id_getter()
            if task_id is None:
                return
            headers = _extract_additional_headers(response_obj)
            if not headers:
                return
            savings = parse_savings_headers(_strip_provider_prefix(headers), pricing)
            if savings is None:
                return
            store.add(task_id, savings)
            logger.debug(
                "captured task savings via litellm callback",
                extra={
                    "fields": {
                        "task_id": task_id,
                        "tokens_before": savings.tokens_before,
                        "tokens_after": savings.tokens_after,
                        "tokens_saved": savings.tokens_saved,
                    }
                },
            )

    return _HeadroomSavingsLogger()


def _aider_version() -> str:
    """Installed aider version string (imported lazily — aider is an ``[aider]``-extra dep)."""
    try:
        import aider
    except ModuleNotFoundError:
        return "not-installed"

    return str(getattr(aider, "__version__", "unknown"))


class AiderHarness:
    """Harness that rolls out Aider Polyglot exercises and grades them with pytest in-process.

    One instance is reused across tasks. Construction registers the litellm savings callback
    exactly once (idempotent across instances sharing the module-level guard), so concurrent
    rollouts all feed the same :class:`SavingsStore`.
    """

    # Module-level guard so the callback is registered once per process even with many harnesses.
    _callback_registered: bool = False

    def __init__(self, settings: Settings, store: SavingsStore | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else SavingsStore()
        self.name = "aider"
        self.version = _aider_version()
        self.supported_providers: set[Provider] = {Provider.ANTHROPIC, Provider.OPENAI}
        self._register_callback_once()

    def _register_callback_once(self) -> None:
        """Register the litellm savings callback exactly once per process."""

        if AiderHarness._callback_registered:
            return
        import litellm

        savings_logger = make_savings_logger(self.store, current_task_id.get, self.settings.pricing)
        callbacks = list(getattr(litellm, "callbacks", []) or [])
        callbacks.append(savings_logger)
        litellm.callbacks = callbacks
        AiderHarness._callback_registered = True
        logger.info(
            "registered litellm savings callback",
            extra={"fields": {"harness": self.name, "version": self.version}},
        )

    async def run_task(
        self, task: BenchTask, env: dict[str, str], workdir: Path, task_tag: str
    ) -> RolloutResult:
        """Roll out one exercise in a worker thread (blocking aider/pytest work off the loop)."""

        return await asyncio.to_thread(self._run_sync, task, env, workdir, task_tag)

    # --- sync worker -----------------------------------------------------------------------

    def _run_sync(
        self, task: BenchTask, env: dict[str, str], workdir: Path, task_tag: str
    ) -> RolloutResult:
        start = time.monotonic()
        try:
            return self._rollout(task, env, workdir, task_tag, start)
        except Exception as exc:  # noqa: BLE001 — surface any failure as a loud RolloutResult.
            wall_ms = (time.monotonic() - start) * 1000.0
            logger.error(
                "aider rollout failed",
                exc_info=True,
                extra={"fields": {"task_id": task.task_id, "task_tag": task_tag}},
            )
            return RolloutResult(
                task_id=task.task_id,
                arm=ArmName.A0_DIRECT,
                run_index=0,
                prediction="",
                trajectory_path=workdir,
                savings=None,
                wall_ms=wall_ms,
                error=repr(exc),
            )

    def _rollout(
        self,
        task: BenchTask,
        env: dict[str, str],
        workdir: Path,
        task_tag: str,
        start: float,
    ) -> RolloutResult:
        cfg = self.settings.aider
        src_exercise = Path(task.payload["exercise_dir"])

        # 1. Isolate: copy the exercise into the workdir so edits/tests never touch the checkout.
        # Resolve to an ABSOLUTE path: pytest runs as a subprocess with cwd set to this dir, so a
        # relative path would resolve against the wrong base and pytest would not find the tests
        # (the root cause of spurious all-fail + the model then "helpfully" rewriting the tests).
        workdir.mkdir(parents=True, exist_ok=True)
        dest_exercise = (workdir / src_exercise.name).resolve()
        if dest_exercise.exists():
            shutil.rmtree(dest_exercise)
        shutil.copytree(src_exercise, dest_exercise)

        # 2. Translate the arm's base_url env into the var litellm actually reads (per provider).
        self._apply_base_url(env)

        # 3. Resolve instructions + solution/test paths (relative to the copied exercise).
        instructions = self._read_instructions(dest_exercise, task)
        solution_paths = self._resolve_paths(dest_exercise, task.payload["solution_files"])
        test_paths = self._resolve_paths(dest_exercise, task.payload["test_files"])
        if not solution_paths:
            raise ValueError(f"no solution files for exercise {task.task_id!r}")
        if not test_paths:
            raise ValueError(f"no test files for exercise {task.task_id!r}")

        logger.info(
            "starting aider rollout",
            extra={
                "fields": {
                    "task_id": task.task_id,
                    "task_tag": task_tag,
                    "exercise": dest_exercise.name,
                    "solution_files": [str(p) for p in solution_paths],
                    "test_files": [str(p) for p in test_paths],
                    "tries": cfg.tries,
                    "edit_format": cfg.edit_format,
                }
            },
        )

        coder = self._build_coder(solution_paths)

        # 4/5. Edit -> test loop, up to cfg.tries; feed pytest output back as the next prompt.
        tests_outcomes: list[bool] = []
        resolved = False
        message = instructions
        # Key savings by the unique per-cell tag (arm-run-task), NOT task_id: the same exercise
        # runs under every arm against one shared store, so task_id alone would mix B's savings
        # with A1's for the same exercise. task_tag isolates each (arm, run, task) cell.
        test_rel_paths = list(task.payload["test_files"])
        with _active_task(task_tag):
            for attempt in range(cfg.tries):
                coder.run(with_message=message)
                # Grader integrity: the model edits only the solution, but aider applies whatever
                # diffs the model emits — including (observed) rewriting the test file with bogus
                # expectations. Restore the PRISTINE test file(s) from the source checkout before
                # grading so the verdict always reflects the real tests, never model-tampered ones.
                self._restore_tests(src_exercise, dest_exercise, test_rel_paths)
                # Run pytest FROM the exercise dir so `import <solution_module>` resolves (pytest's
                # prepend import mode adds the test's dir to sys.path) and relative test discovery
                # works. dest_exercise is absolute, so the subprocess cwd is unambiguous.
                passed, feedback = self._run_pytest(test_paths, dest_exercise, cfg.pytest_timeout_s)
                tests_outcomes.append(passed)
                logger.info(
                    "aider attempt complete",
                    extra={
                        "fields": {
                            "task_id": task.task_id,
                            "attempt": attempt + 1,
                            "passed": passed,
                        }
                    },
                )
                if passed:
                    resolved = True
                    break
                # Next try sees the failing-test output as feedback.
                message = feedback

        # 6. Aggregate savings captured by the litellm callback for this cell (keyed by task_tag).
        savings = self.store.aggregate(task_tag, self.settings.pricing)

        # 7. The grader reads this verdict verbatim.
        prediction = json.dumps(
            {
                "resolved": resolved,
                "tests_outcomes": tests_outcomes,
                "exercise": dest_exercise.name,
            }
        )

        wall_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "aider rollout complete",
            extra={
                "fields": {
                    "task_id": task.task_id,
                    "resolved": resolved,
                    "tests_outcomes": tests_outcomes,
                    "wall_ms": wall_ms,
                    "tokens_saved": (savings.tokens_saved if savings else None),
                }
            },
        )
        return RolloutResult(
            task_id=task.task_id,
            arm=ArmName.A0_DIRECT,  # orchestrator overwrites arm/run_index.
            run_index=0,
            prediction=prediction,
            trajectory_path=workdir,
            savings=savings,
            wall_ms=wall_ms,
            error=None,
        )

    # --- helpers ---------------------------------------------------------------------------

    def _apply_base_url(self, env: Mapping[str, str]) -> None:
        """Set the litellm base-url env var from the arm-provided base_url for our provider."""

        provider = self.settings.provider
        arm_var = _ARM_ENV_BY_PROVIDER[provider]
        litellm_var = _LITELLM_ENV_BY_PROVIDER[provider]
        base_url = env.get(arm_var)
        if base_url:
            os.environ[litellm_var] = base_url
            logger.debug(
                "applied base_url for litellm",
                extra={"fields": {"provider": provider.value, litellm_var: base_url}},
            )

    @staticmethod
    def _resolve_paths(exercise_dir: Path, rel_paths: list[str]) -> list[Path]:
        """Resolve config-relative file paths against the copied exercise dir."""

        return [exercise_dir / rel for rel in rel_paths]

    @staticmethod
    def _read_instructions(exercise_dir: Path, task: BenchTask) -> str:
        """Concatenate the exercise instruction docs (main + optional append) into one prompt."""

        rel_paths = task.payload.get("instructions_paths") or []
        parts: list[str] = []
        for rel in rel_paths:
            path = exercise_dir / rel
            if path.exists():
                parts.append(path.read_text(encoding="utf-8"))
        if not parts:
            raise ValueError(f"no instruction docs found for exercise {exercise_dir.name!r}")
        return "\n\n".join(parts)

    def _build_coder(self, solution_paths: list[Path]) -> Any:
        """Construct the aider Coder bound to the solution file(s)."""

        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model

        io = InputOutput(yes=True, pretty=False, fancy_input=False)
        model = Model(self.settings.litellm_model_name())
        return Coder.create(
            main_model=model,
            edit_format=self.settings.aider.edit_format,
            io=io,
            fnames=[str(p) for p in solution_paths],
            use_git=False,
            stream=False,
            suggest_shell_commands=False,
            auto_test=False,
            # Critical: do NOT scrape URLs found in the instructions. Without this, aider fetches
            # every linked page (wikipedia/python-docs) into context — observed 313k tokens for one
            # tiny exercise (~$1/call) and an attempted playwright auto-install. We grade the model
            # on the instructions alone, as the official Polyglot benchmark does.
            detect_urls=False,
        )

    @staticmethod
    def _restore_tests(src_exercise: Path, dest_exercise: Path, test_rel_paths: list[str]) -> None:
        """Copy the pristine test file(s) from the source checkout over the working copy."""

        for rel in test_rel_paths:
            src = src_exercise / rel
            dest = dest_exercise / rel
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    @staticmethod
    def _run_pytest(test_paths: list[Path], cwd_dir: Path, timeout_s: float) -> tuple[bool, str]:
        """Run the exercise's pytest file(s) from ``cwd_dir``; return (passed, output-for-feedback)."""

        cmd = [sys.executable, "-m", "pytest", *[str(p) for p in test_paths], "-q"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            feedback = f"pytest timed out after {timeout_s}s\n{stdout}\n{stderr}"
            logger.warning(
                "pytest timed out",
                extra={"fields": {"timeout_s": timeout_s, "cmd": cmd}},
            )
            return False, feedback
        passed = proc.returncode == 0
        feedback = f"{proc.stdout}\n{proc.stderr}".strip()
        return passed, feedback
