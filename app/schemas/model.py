"""Type aliases for model selection.

`ModelKey` stays a closed set — these are the three preset slots the
auto-router knows how to choose between. They map to user-configured
Ollama model names via Settings.model_map.

`ModelSelection` is wider: "auto" picks via the router, "granite" /
"gemma" / "qwen" pick a preset slot, and any other string is a raw
Ollama model name (e.g. "qwen2.5-coder:32b", "deepseek-coder-v2:latest")
which ChatService.get_adapter() handles by instantiating an adapter
on the fly. Lets the user select any installed model without
re-declaring slots in config.
"""
from typing import Literal


ModelKey = Literal["granite", "gemma", "qwen"]
# str rather than Literal["auto", ...] — the dropdowns now include
# every installed Ollama model, not just the three preset slots.
ModelSelection = str
