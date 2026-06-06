from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderStart:
    job_id: int
    render_token: int


@dataclass(frozen=True)
class PendingRender:
    show_errors: bool


class PreviewRenderController:
    def __init__(self) -> None:
        self.job_id = 0
        self.render_tokens: dict[Path, int] = {}
        self.in_progress = False
        self._pending: PendingRender | None = None

    def start(self, path: Path) -> RenderStart:
        self.job_id += 1
        token = self.bump_token(path)
        self.in_progress = True
        return RenderStart(job_id=self.job_id, render_token=token)

    def cancel(self, path: Path | None) -> None:
        if path is not None:
            self.bump_token(path)
        self.job_id += 1
        self.in_progress = False
        self._pending = None

    def bump_token(self, path: Path) -> int:
        token = self.render_tokens.get(path, 0) + 1
        self.render_tokens[path] = token
        return token

    def defer(self, path: Path | None, *, show_errors: bool) -> None:
        if path is not None:
            self.bump_token(path)
        self._pending = PendingRender(
            show_errors=bool(show_errors or (self._pending.show_errors if self._pending else False))
        )

    def mark_idle(self) -> None:
        self.in_progress = False

    def is_latest_job(self, job_id: int) -> bool:
        return int(job_id) == self.job_id

    def output_is_current(self, output) -> bool:
        return output.render_token == self.render_tokens.get(output.path, 0)

    def has_pending(self) -> bool:
        return self._pending is not None

    def consume_pending(self) -> PendingRender | None:
        pending = self._pending
        self._pending = None
        return pending
