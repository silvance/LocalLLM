"""Lightweight extensions to fastapi.staticfiles.

Pulled into its own module so app.web.app and app.agent.routes can both
import it without a circular dependency (web → agent → web).
"""
from __future__ import annotations

from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles with ``Cache-Control: no-cache`` on every response.

    Default StaticFiles relies on browser heuristic caching (may keep
    a CSS file for hours), which means an updated stylesheet won't
    show up until the user hard-refreshes — pretty hostile during
    iterative UI work. ``no-cache`` tells the browser to revalidate on
    every request; combined with the parent class's ETag/Last-Modified
    handling, unchanged files still get a cheap 304.
    """
    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response
