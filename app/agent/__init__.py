"""Internet-connected agent variant.

Excluded from the airgap bundle by build_bundle.py — see APP_IGNORE.
The agent runs in the SAME FastAPI app as the chat / review pages, but
behind try-import so the airgap deploy (which doesn't ship app/agent/)
silently skips mounting the routes.

Tools (current):
    - web_search: DuckDuckGo via ddgs (no API key)
    - http_fetch: httpx + trafilatura clean-text extraction
"""
