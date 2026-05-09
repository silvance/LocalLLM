"""Ollama model-management facade for the /hardware page's pull flow.

We talk to the local Ollama daemon over HTTP rather than shelling out
to ``ollama pull`` because:

- The CLI is a thin wrapper around the same HTTP API and shelling adds
  a fork + a tty quirks layer.
- Ollama is already running (the chat path connects to it), so its
  configured model directory / GPU offload / etc. all apply.
- Model names get into a subprocess's argv if we shell out — easy
  command-injection foot-gun even with shlex, and pointless when the
  daemon's HTTP endpoint accepts the same name.

The pull endpoint streams NDJSON progress events. We wrap that in a
generator so the route can pump each line as a JobManager checkpoint
event, reusing the existing SSE machinery.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Iterator


logger = logging.getLogger("localllm")


# Permissive but bounded — Ollama tags can include slashes (registry
# paths) and colons (tag separator). Keep alpha/digit/underscore/dash/
# dot plus the structural separators. Reject anything else, especially
# spaces, semicolons, ampersands, and quote characters.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_./\-]+(?::[A-Za-z0-9_.\-]+)?$")


def validate_model_name(name: str) -> str:
    """Normalize + bounds-check a model name. Raises ValueError on
    anything that doesn't look like a real Ollama tag."""
    name = (name or "").strip()
    if not name:
        raise ValueError("empty model name")
    if len(name) > 200:
        raise ValueError("model name too long")
    if not _MODEL_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid model name: {name!r}")
    # `.` is allowed in registry hostnames and version tags, but `..`
    # or a leading `.` looks like path-traversal — reject explicitly.
    if name.startswith(".") or ".." in name:
        raise ValueError(f"invalid model name: {name!r}")
    return name


def _http_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def list_installed(base_url: str) -> list[dict]:
    """GET /api/tags. Returns the daemon's view of installed models —
    full entries (name, size, digest) so callers can show metadata.

    Returns ``[]`` on failure so legacy callers don't have to wrap.
    Use ``list_installed_or_raise`` instead if you need to distinguish
    "daemon reachable but empty" from "couldn't reach daemon."
    """
    try:
        return list_installed_or_raise(base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("ollama list_installed failed: %s", exc)
        return []


def list_installed_or_raise(base_url: str) -> list[dict]:
    """Same as ``list_installed`` but propagates errors so the caller
    can distinguish unreachable from empty. Used by the model-list
    discovery path that needs to populate the UI's reachability
    banner with a real error message."""
    url = _http_url(base_url, "/api/tags")
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("models") or [])


def installed_names(base_url: str) -> list[str]:
    """Just the model name strings — what the hardware page displays.

    Returns ``[]`` on failure (delegates to swallowing
    ``list_installed``). For UI status that needs to know WHY the
    list was empty, call ``installed_names_or_raise``.
    """
    out: list[str] = []
    for entry in list_installed(base_url):
        n = entry.get("name") or entry.get("model")
        if n:
            out.append(n)
    return out


def installed_names_or_raise(base_url: str) -> list[str]:
    """``installed_names`` that propagates network errors. Use this
    when the caller needs to surface the actual error to the user."""
    out: list[str] = []
    for entry in list_installed_or_raise(base_url):
        n = entry.get("name") or entry.get("model")
        if n:
            out.append(n)
    return out


def pull_model(model: str, base_url: str) -> Iterator[dict]:
    """POST /api/pull and yield each NDJSON progress event.

    Ollama's pull stream emits one JSON object per line, each with a
    ``status`` key plus optional ``digest`` / ``total`` / ``completed``
    when downloading layers. We just yield them; the route turns each
    into an SSE event.

    The generator is cancellation-aware via GeneratorExit — calling
    ``gen.close()`` on the consumer side closes the underlying urlopen
    response so an in-flight read can't keep the thread wedged when
    the operator hits Stop.
    """
    name = validate_model_name(model)
    body = json.dumps({"name": name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        _http_url(base_url, "/api/pull"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = None
    try:
        try:
            # Long-running download — no read timeout. Connection timeout
            # still applies via the default. The caller is responsible
            # for cancellation via gen.close().
            resp = urllib.request.urlopen(req, timeout=None)
        except urllib.error.HTTPError as exc:
            # Ollama returns the error body as JSON; surface it so the
            # UI can show e.g. "model not found in registry".
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = {"error": exc.reason or str(exc)}
            yield {"error": str(payload.get("error") or payload), "http_status": exc.code}
            return
        except (urllib.error.URLError, OSError) as exc:
            yield {"error": f"transport error: {exc}"}
            return

        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"status": line}
        except (OSError, ConnectionError) as exc:
            # http.client.IncompleteRead is a HTTPException, not an
            # OSError on every Python — match by name as a safety net.
            yield {"error": f"stream interrupted: {exc}"}
        except Exception as exc:  # noqa: BLE001 — IncompleteRead etc.
            cls = type(exc).__name__
            if cls == "IncompleteRead" or "Read" in cls:
                yield {"error": f"stream interrupted: {exc}"}
            else:
                raise
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


def delete_model(model: str, base_url: str) -> None:
    """DELETE /api/delete through the Ollama daemon.

    Do not write into Ollama's model directory directly; the daemon owns
    model layout and reference counting.
    """
    name = validate_model_name(model)
    body = json.dumps({"name": name}).encode("utf-8")
    req = urllib.request.Request(
        _http_url(base_url, "/api/delete"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
