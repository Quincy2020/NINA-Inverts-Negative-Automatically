from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from threading import Thread
from time import perf_counter, sleep

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from qnegative.core.models import DustRemovalParams


class DustMaskPreviewSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, object, object, object)
    failed = Signal(int, object, str)


class DustMaskPreviewTask(QRunnable):
    def __init__(
        self,
        *,
        job_id: int,
        path: Path | None,
        linear_rgb: np.ndarray,
        params: DustRemovalParams,
        cancel_event: Event | None = None,
    ) -> None:
        super().__init__()
        self.job_id = int(job_id)
        self.path = path
        self.linear_rgb = np.ascontiguousarray(linear_rgb.astype(np.float32, copy=True))
        self.params = deepcopy(params)
        self.cancel_event = cancel_event
        self._last_progress_emit = 0.0
        self.signals = DustMaskPreviewSignals()

    def run(self) -> None:
        try:
            mask, stats = self._run_subprocess()
        except Exception as exc:
            self.signals.failed.emit(self.job_id, self.path, str(exc))
            return

        self.signals.finished.emit(self.job_id, self.path, mask.astype(bool), stats)

    def _run_subprocess(self) -> tuple[np.ndarray, dict]:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("Dust mask generation cancelled.")

        with TemporaryDirectory(prefix="nina_dust_mask_") as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.npz"
            params_path = tmp_dir / "params.json"
            output_path = tmp_dir / "output.npz"
            np.savez(input_path, linear_rgb=self.linear_rgb)
            with params_path.open("w", encoding="utf-8") as handle:
                json.dump(asdict(self.params), handle)

            command = [
                sys.executable,
                "-m",
                "qnegative.tools.dust_mask_worker",
                "--input",
                str(input_path),
                "--params",
                str(params_path),
                "--output",
                str(output_path),
            ]
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stdout_queue: Queue[str] = Queue()
            stderr_queue: Queue[str] = Queue()
            stdout_thread = Thread(target=_read_pipe, args=(process.stdout, stdout_queue), daemon=True)
            stderr_thread = Thread(target=_read_pipe, args=(process.stderr, stderr_queue), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            while process.poll() is None:
                self._drain_stdout(stdout_queue)
                if self.cancel_event is not None and self.cancel_event.is_set():
                    _terminate_process(process)
                    raise RuntimeError("Dust mask generation cancelled.")
                sleep(0.05)

            stdout_thread.join(timeout=0.5)
            stderr_thread.join(timeout=0.5)
            self._drain_stdout(stdout_queue)
            stderr_text = _drain_text(stderr_queue)
            if process.returncode != 0:
                message = stderr_text.strip() or f"Dust mask worker exited with code {process.returncode}"
                raise RuntimeError(message)
            if not output_path.exists():
                raise RuntimeError("Dust mask worker did not write an output mask.")

            with np.load(output_path, allow_pickle=True) as output:
                mask = output["mask"].astype(bool)
                stats_value = output["stats"].item()
            stats = json.loads(str(stats_value)) if stats_value is not None else {}
            return mask, stats

    def _drain_stdout(self, stdout_queue: Queue[str]) -> None:
        while True:
            try:
                line = stdout_queue.get_nowait()
            except Empty:
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "progress":
                self._progress(
                    int(payload.get("value", 0)),
                    str(payload.get("text") or "Auto mask"),
                )

    def _run_in_process_fallback(self) -> tuple[np.ndarray, dict]:
        from qnegative.core.dust_removal import linear_to_srgb_float, predict_dust_mask

        try:
            self.signals.progress.emit(self.job_id, 3, "Preparing image")
            srgb = linear_to_srgb_float(self.linear_rgb)
            mask, stats = predict_dust_mask(
                srgb,
                self.params,
                model_root=Path.cwd(),
                progress_callback=self._progress,
                cancel_event=self.cancel_event,
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        return mask.astype(bool), stats

    def _progress(self, value: int, text: str) -> None:
        now = perf_counter()
        value = max(0, min(100, int(value)))
        if value < 100 and now - self._last_progress_emit < 0.12:
            return
        self._last_progress_emit = now
        self.signals.progress.emit(self.job_id, value, text)


def _read_pipe(pipe, queue: Queue[str]) -> None:
    if pipe is None:
        return
    try:
        for line in pipe:
            if line:
                queue.put(line.strip())
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _drain_text(queue: Queue[str]) -> str:
    lines: list[str] = []
    while True:
        try:
            lines.append(queue.get_nowait())
        except Empty:
            return "\n".join(lines)


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
