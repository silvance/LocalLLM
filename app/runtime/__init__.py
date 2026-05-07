"""Runtime helpers used when LocalLLM is launched from the single-file
desktop binary (PyInstaller onefile). Modules here own the lifecycle of
external sidecars (currently just the embedded Ollama process) and the
filesystem layout we expose to the user.

Importable from dev mode too — they all degrade gracefully when nothing
has been "frozen" by PyInstaller.
"""
