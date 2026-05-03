import json
import re
import os
import hashlib
import shutil
import time
import uuid
from pathlib import Path
from collections import OrderedDict
from urllib.parse import urlparse
import sys
import threading

from .CloudflareBypasser import CloudflareBypasser
from DrissionPage import ChromiumPage, ChromiumOptions
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal, Union, Any
import argparse
from dotenv import load_dotenv

try:
    from pyvirtualdisplay import Display
except ImportError:
    Display = None
import uvicorn
import atexit
load_dotenv()

# Module-level default so endpoints work when not launched via __main__
log = True

# Import DeepSeekAPI
from .api import DeepSeekAPI, AuthenticationError, RateLimitError, NetworkError, APIError, ModelType

# Check if running in Docker mode
DOCKER_MODE = os.getenv("DOCKERMODE", "false").lower() == "true"

SERVER_PORT = int(os.getenv("SERVER_PORT", 5005))

# Chromium options arguments
arguments = [
    # "--remote-debugging-port=9222",  # Add this line for remote debugging
    "-no-first-run",
    "-force-color-profile=srgb",
    "-metrics-recording-only",
    "-password-store=basic",
    "-use-mock-keychain",
    "-export-tagged-pdf",
    "-no-default-browser-check",
    "-disable-background-mode",
    "-enable-features=NetworkService,NetworkServiceInProcess,LoadCryptoTokenExtension,PermuteTLSExtensions",
    "-disable-features=FlashDeprecationWarning,EnablePasswordsAccountStorage",
    "-deny-permission-prompts",
    "-disable-gpu",
    "-accept-lang=en-US",
    #"-incognito" # You can add this line to open the browser in incognito mode by default
]

_chrome_candidates = ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]
browser_path = next((p for p in _chrome_candidates if os.path.isfile(p)), shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser") or "/usr/bin/google-chrome")
app = FastAPI()


# --- Session manager (LRU cache + disk) ---
SESSIONS_FILE = Path(__file__).parent / 'chat_sessions.json'
MAX_CACHE_SIZE = 20


def _build_key(messages: list) -> str:
    """Build a conversation key from the first user message only.

    Using only the first user message is far more stable than hashing all
    user messages because:
      - System messages are excluded (they contain dynamic IDE context)
      - Only the first message is used (it rarely changes between turns,
        whereas later messages may be rewritten or reordered by the client)
      - A separate conversation_id (UUID) is the primary session identifier;
        this key serves as a fallback when no conversation_id is available.
    """
    for msg in messages:
        role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
        if role == "user":
            raw = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            content = _extract_content(raw)
            return hashlib.sha256(f"u0:{content}".encode()).hexdigest()[:16]
    return hashlib.sha256(b"empty").hexdigest()[:16]


class SessionManager:
    """Disk-persistent session manager for conversation→DeepSeek session mapping.
    
    All data is written to disk immediately on every mutation so sessions
    survive server crashes (e.g. curl_cffi heap corruption).
    """

    def __init__(self, max_sessions: int = MAX_CACHE_SIZE, filepath: Path = SESSIONS_FILE):
        self._max = max_sessions
        self._file = filepath
        self._lock = threading.Lock()
        self._data: OrderedDict[str, Dict[str, Optional[str]]] = OrderedDict()
        self._load()

    def _load(self):
        try:
            if self._file.exists():
                with open(self._file, 'r') as f:
                    raw = json.load(f)
                    for k, v in raw.items():
                        if isinstance(v, str):
                            self._data[k] = {'session_id': v, 'parent_message_id': None}
                        else:
                            self._data[k] = v
        except (json.JSONDecodeError, IOError):
            self._data = OrderedDict()

    def _save(self):
        try:
            with open(self._file, 'w') as f:
                json.dump(dict(self._data), f, indent=2)
        except IOError as e:
            print(f"[session] WARNING: Failed to save sessions: {e}")

    def get(self, key: str) -> Optional[Dict[str, Optional[str]]]:
        """Look up session info by key."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return None

    def put(self, key: str, info: Dict[str, Optional[str]]):
        """Store session info and persist to disk immediately."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = info
            while len(self._data) > self._max:
                evicted_key, _ = self._data.popitem(last=False)
                print(f"[session] Evicted oldest session {evicted_key[:8]}…")
            self._save()

    def find_by_conversation_id(self, conversation_id: str) -> Optional[tuple]:
        """Find a session entry by its conversation_id. Returns (key, info) or None."""
        with self._lock:
            for key, info in self._data.items():
                if info.get('conversation_id') == conversation_id:
                    self._data.move_to_end(key)
                    return (key, info)
            return None

    def update_parent_message_id(self, key: str, parent_message_id: str):
        """Update parent_message_id and persist to disk immediately."""
        with self._lock:
            if key in self._data:
                self._data[key]['parent_message_id'] = parent_message_id
                self._save()


session_mgr = SessionManager()


# Pydantic model for the response
class CookieResponse(BaseModel):
    cookies: Dict[str, str]
    user_agent: str


# OpenAI-compatible models
# --- Tool calling models (ported from ds-free-api) ---
class FunctionDefinition(BaseModel):
    model_config = {"extra": "ignore"}
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    model_config = {"extra": "ignore"}
    type: str = "function"
    function: Optional[FunctionDefinition] = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatCompletionChunkDeltaToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[Dict[str, str]] = None


class ChatMessage(BaseModel):
    model_config = {"extra": "ignore"}
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Union[str, List[Any], None] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "deepseek-v4-flash"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    thinking_enabled: Optional[bool] = None
    search_enabled: Optional[bool] = None
    model_type: Optional[ModelType] = None  # 'default'=Instant(V4 Flash), 'expert'=V4 Pro; auto-resolved from model field if not set
    conversation_id: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Any] = None  # 'none', 'auto', 'required', or {'type':'function','function':{'name':'...'}}
    parallel_tool_calls: Optional[bool] = None
    # ds-free-api compatible fields
    reasoning_effort: Optional[str] = None  # "none" disables thinking; any other value (default "high") enables it
    web_search_options: Optional[Any] = None  # dict with search_context_size; present=enabled, "none"=disabled


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[ChatCompletionResponseUsage] = None
    conversation_id: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    role: Optional[Literal["assistant"]] = None
    tool_calls: Optional[List[ChatCompletionChunkDeltaToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    conversation_id: Optional[str] = None


# Function to check if the URL is safe
def is_safe_url(url: str) -> bool:
    parsed_url = urlparse(url)
    ip_pattern = re.compile(
        r"^(127\.0\.0\.1|localhost|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|172\.1[6-9]\.\d+\.\d+|172\.2[0-9]\.\d+\.\d+|172\.3[0-1]\.\d+\.\d+|192\.168\.\d+\.\d+)$"
    )
    hostname = parsed_url.hostname
    return parsed_url.scheme != "file" and not (hostname and ip_pattern.match(hostname))


# Function to verify if the page has loaded properly
def verify_page_loaded(driver: ChromiumPage) -> bool:
    """Verify if the page has loaded properly"""
    try:
        # Wait for body element to be present
        body = driver.ele('tag:body', timeout=10)
        # Check if page has actual content
        return len(body.html) > 100
    except Exception:
        return False


# Function to bypass Cloudflare protection
def _build_chrome_options(proxy: str = None) -> ChromiumOptions:
    options = ChromiumOptions().auto_port().set_paths(browser_path=browser_path).headless(False)
    if DOCKER_MODE:
        options.set_argument("--auto-open-devtools-for-tabs", "true")
        options.set_argument("--remote-debugging-port=9222")
        options.set_argument("--no-sandbox")  # Necessary for Docker
        options.set_argument("--disable-gpu")  # Optional, helps in some cases
    if proxy:
        options.set_proxy(proxy)
    return options


def bypass_cloudflare(url: str, retries: int, log: bool, proxy: str = None) -> ChromiumPage:
    max_load_retries = 3

    for load_attempt in range(max_load_retries):
        driver = ChromiumPage(addr_or_opts=_build_chrome_options(proxy))
        try:
            driver.get(url)
            # Wait for initial page load
            time.sleep(5)

            if not verify_page_loaded(driver):
                driver.quit()
                if load_attempt < max_load_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    raise Exception("Failed to load page properly after multiple attempts")

            cf_bypasser = CloudflareBypasser(driver, retries, log)
            cf_bypasser.bypass()
            return driver
        except Exception as e:
            driver.quit()
            if load_attempt < max_load_retries - 1:
                time.sleep(3)
                continue
            raise e


def _extract_cookies(driver: ChromiumPage) -> Dict[str, str]:
    return {cookie.get("name", ""): cookie.get("value", " ") for cookie in driver.cookies()}


# Endpoint to get cookies
@app.get("/cookies", response_model=CookieResponse)
async def get_cookies(url: str, retries: int = 5, proxy: str = None):
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        driver = bypass_cloudflare(url, retries, log, proxy)
        cookies = _extract_cookies(driver)
        user_agent = driver.user_agent
        driver.quit()
        return CookieResponse(cookies=cookies, user_agent=user_agent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to get HTML content and cookies
@app.get("/html")
async def get_html(url: str, retries: int = 5, proxy: str = None):
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        driver = bypass_cloudflare(url, retries, log, proxy)
        html = driver.html
        cookies_json = _extract_cookies(driver)
        response = Response(content=html, media_type="text/html")
        response.headers["cookies"] = json.dumps(cookies_json)
        response.headers["user_agent"] = driver.user_agent
        driver.quit()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model name resolution helper (matching ds-free-api model_types + model_aliases)
# Only base model IDs and aliases; thinking/search controlled via reasoning_effort & web_search_options
MODEL_MAP = {
    'deepseek-default':   {'model_type': 'default'},
    'deepseek-expert':    {'model_type': 'expert'},
    'deepseek-v4-flash':  {'model_type': 'default'},
    'deepseek-v4-pro':    {'model_type': 'expert'},
}

def _resolve_model(model: str, model_type=None, thinking_enabled=None, search_enabled=None,
                   reasoning_effort: Optional[str] = None, web_search_options: Optional[Any] = None):
    """Resolve model name to (model_type, thinking_enabled, search_enabled).

    Resolution order (matching ds-free-api resolver.rs):
      1. model_type from MODEL_MAP or explicit param
      2. thinking_enabled: explicit param > reasoning_effort ("none"=off, else on) > default True
      3. search_enabled: explicit param > web_search_options (present=on, "none"=off) > default True
    """
    resolved = MODEL_MAP.get(model.lower(), None)

    # Determine model_type
    resolved_model_type = resolved['model_type'] if resolved else 'default'
    final_model_type = model_type if model_type is not None else resolved_model_type

    # Determine thinking_enabled (ds-free-api: reasoning_effort defaults to "high", thinking on unless "none")
    if thinking_enabled is not None:
        final_thinking = thinking_enabled
    elif reasoning_effort is not None:
        final_thinking = reasoning_effort != 'none'
    else:
        final_thinking = True  # ds-free-api default: reasoning ON

    # Determine search_enabled (ds-free-api: search ON by default for stronger system prompts)
    if search_enabled is not None:
        final_search = search_enabled
    elif web_search_options is not None:
        # web_search_options present → check for explicit disable
        if isinstance(web_search_options, dict):
            ctx_size = web_search_options.get('search_context_size', '')
            final_search = ctx_size != 'none'
        else:
            final_search = True
    else:
        final_search = True  # ds-free-api default: search ON

    return (final_model_type, final_thinking, final_search)


# OpenAI-compatible models endpoint
@app.get("/v1/models")
async def list_models():
    _MODEL_LABELS = {'default': 'V4 Flash', 'expert': 'V4 Pro'}
    models = []
    for mid, cfg in MODEL_MAP.items():
        label = _MODEL_LABELS.get(cfg['model_type'], cfg['model_type'])
        desc = f"DeepSeek {label} (thinking & search on by default; disable via reasoning_effort/web_search_options)"
        models.append({
            "id": mid,
            "object": "model",
            "created": 1740000000,
            "owned_by": "deepseek",
            "description": desc,
            "model_type": cfg['model_type'],
        })
    return {"object": "list", "data": models}


def _extract_content(content):
    """Extract text from message content, handling both string and list formats."""
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


# ---------------------------------------------------------------------------
# Tool calling — prompt injection (ported from ds-free-api)
# ---------------------------------------------------------------------------
# DeepSeek uses these internal role tags; we use the Unicode variant forms
# that the model actually produces (｜ = U+FF5C, ▁ = U+2581).
TOOL_CALL_START = "<|tool_calls_begin|>"
TOOL_CALL_END = "<|tool_calls_end|>"

# Tag constants for DeepSeek native prompt format
_TAG_SYSTEM_START = "<｜System｜>"
_TAG_SYSTEM_END = "<｜System｜>"
_TAG_USER_START = "<｜end▁of▁sentence｜><｜User｜>"
_TAG_ASSISTANT_START = "<｜Assistant｜>"
_TAG_TOOL_OUTPUTS_BEGIN = "<｜tool_outputs_begin｜>"
_TAG_TOOL_OUTPUT_END = "<｜tool_output_end｜>"
_TAG_TOOL_OUTPUTS_END = "<｜tool_outputs_end｜>"
_TAG_TOOL_OUTPUT_BEGIN = "<｜tool_output_begin｜>"
_THINK_BLOCK = "Hmm, I was just reminded by the system that I need to follow these rules:\n\n"


def _example_args(name: str) -> str:
    """Return realistic example arguments for common tool names (from ds-free-api)."""
    _MAP = {
        "Read": '"file_path": "/path/to/file"',
        "read_file": '"file_path": "/path/to/file"',
        "Bash": '"command": "ls -la"',
        "execute_command": '"command": "ls -la"',
        "exec_command": '"command": "ls -la"',
        "Write": '"file_path": "/path/to/file", "content": "hello"',
        "write_to_file": '"file_path": "/path/to/file", "content": "hello"',
        "Edit": '"file_path": "/path/to/file", "old_string": "foo", "new_string": "bar"',
        "Glob": '"pattern": "**/*.py", "path": "."',
        "search_files": '"query": "TODO", "path": "."',
        "get_weather": '"city": "Beijing"',
        "get_time": '"timezone": "Asia/Shanghai"',
        "list_files": '"path": "."',
    }
    return "{" + _MAP.get(name, '"key": "value"') + "}"


def _example_nested_args(name: str) -> str:
    if name == "Edit":
        return '{"file_path": "/path/to/file", "edits": [{"old_string": "foo", "new_string": "bar"}, {"old_string": "x", "new_string": "y"}]}'
    return '{"config": {"enabled": true, "items": ["a", "b"]}}'


def _build_tool_instruction_block(tools: List[ToolDefinition]) -> str:
    """Build the format specification + rules + examples block (from ds-free-api tools.rs)."""
    lines: list[str] = []
    lines.append("**Tool Call Format — Please strictly follow:**")
    lines.append("")
    lines.append("Wrap the JSON array in tool call markers:")
    lines.append("")
    lines.append(f'{TOOL_CALL_START}[{{"name": "tool_name", "arguments": {{param_json}}}}]{TOOL_CALL_END}')
    lines.append("")

    # Rules
    lines.append("**Rules:**")
    lines.append("")
    lines.append(
        "**Core: When you decide to call a tool, your response must ONLY contain the tool call text itself. "
        "No explanations, prefixes, summaries, greetings, or any other extra content.**"
    )
    lines.append("")
    lines.append(f"1. The JSON array MUST start with `{TOOL_CALL_START}` and end with `{TOOL_CALL_END}`, "
                 "wrapping the array **completely** inside the markers.")
    lines.append("2. All tool calls must be placed in **one** JSON array, multiple calls separated by commas.")
    lines.append(f"3. After outputting `{TOOL_CALL_END}`, **stop immediately** — do not add any subsequent text, XML tags, or explanations.")
    lines.append("4. Do NOT wrap tool calls in markdown code blocks.")
    lines.append("5. String argument values must be wrapped in **double quotes** (JSON standard).")
    lines.append(f"6. The **first non-whitespace character** of your response must be `{TOOL_CALL_START}`.")
    lines.append(f"7. Only **one** `{TOOL_CALL_START}` block may appear in the entire response — do not output multiple blocks.")
    lines.append(f"8. **Repeat:** Only one `{TOOL_CALL_START}` block in the entire response. Do not repeat.")
    lines.append(f"9. **Repeat:** No text before `{TOOL_CALL_START}` — no explanations, confirmations, summaries, or greetings.")
    lines.append("")

    # Correct examples using actual tool names
    tool_names = [t.function.name for t in tools if t.function]
    a = tool_names[0] if tool_names else "tool_a"

    lines.append("**Correct examples:**")
    lines.append("")
    lines.append("**Example A** — Call one tool:")
    lines.append(f'{TOOL_CALL_START}[{{"name": "{a}", "arguments": {_example_args(a)}}}]{TOOL_CALL_END}')
    lines.append("")

    if len(tool_names) >= 2:
        items = [f'{{"name": "{n}", "arguments": {_example_args(n)}}}' for n in tool_names[:2]]
        lines.append("**Example B** — Call two tools in parallel (one array contains all calls):")
        lines.append("")
        lines.append(f"{TOOL_CALL_START}[{', '.join(items)}]{TOOL_CALL_END}")
        lines.append("")

    if len(tool_names) >= 3:
        items = [f'{{"name": "{n}", "arguments": {_example_args(n)}}}' for n in tool_names[:3]]
        lines.append("**Example C** — Call three tools in parallel (all calls in one array):")
        lines.append("")
        lines.append(f"{TOOL_CALL_START}[{', '.join(items)}]{TOOL_CALL_END}")
        lines.append("")

    # Nested args example
    d_name = tool_names[0] if tool_names else "tool_a"
    lines.append("**Example D** — Nested object/array arguments (still standard JSON):")
    lines.append("")
    lines.append(f'{TOOL_CALL_START}[{{"name": "{d_name}", "arguments": {_example_nested_args(d_name)}}}]{TOOL_CALL_END}')
    lines.append("")

    return "\n".join(lines)


def _format_tool_def(tool: ToolDefinition, idx: int) -> str:
    """Format a single tool definition as natural language description."""
    if tool.type == "function" and tool.function:
        func = tool.function
        params = json.dumps(func.parameters or {})
        call_example = f'{TOOL_CALL_START}[{{"name": "{func.name}", "arguments": {params}}}]{TOOL_CALL_END}'
        desc = (func.description or "").strip()
        desc_block = f"~~~markdown\n  {desc}\n~~~\n" if desc else "  No description"
        return f'- **{func.name}** (function):\n  - Call: `{call_example}`\n  - Description:\n{desc_block}'
    return f"- tools[{idx}]: unsupported type '{tool.type}'"


def _build_tool_defs_text(tools: List[ToolDefinition]) -> str:
    """Build the tool definitions section."""
    lines = ["You can use the following tools:"]
    for i, tool in enumerate(tools):
        lines.append(_format_tool_def(tool, i))
    return "\n".join(lines)


def _build_instruction_text(tool_choice: Any, parallel_tool_calls: Optional[bool]) -> str:
    """Build behavioral instructions based on tool_choice and parallel_tool_calls."""
    lines: list[str] = []

    if isinstance(tool_choice, str):
        if tool_choice == "required":
            lines.append("**Note: You MUST call one or more tools.**")
    elif isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "xxx"}}
        func_name = tool_choice.get("function", {}).get("name", "")
        if func_name:
            lines.append(f"**Note: You MUST call the '{func_name}' tool.**")

    if parallel_tool_calls is False:
        lines.append("**Note: You may only call one tool at a time.**")

    return "\n".join(lines)


class ToolContext:
    """Extracted tool context for prompt injection (mirrors ds-free-api ToolContext)."""
    def __init__(self, format_block: Optional[str], defs_text: Optional[str],
                 instruction_text: Optional[str]):
        self.format_block = format_block
        self.defs_text = defs_text
        self.instruction_text = instruction_text


def _extract_tool_context(req: ChatCompletionRequest) -> ToolContext:
    """Extract and validate tool info from the request (mirrors ds-free-api tools.rs extract())."""
    has_tools = bool(req.tools)

    tool_choice = req.tool_choice if req.tool_choice is not None else ("auto" if has_tools else "none")

    # If tool_choice is 'none', no injection
    if tool_choice == "none":
        return ToolContext(None, None, None)

    format_block = _build_tool_instruction_block(req.tools) if has_tools else None
    defs_text = _build_tool_defs_text(req.tools) if has_tools else None
    instruction_text = _build_instruction_text(tool_choice, req.parallel_tool_calls) or None

    return ToolContext(format_block, defs_text, instruction_text)


def _sse(cid: str, created: int, model: str, delta: ChatCompletionChunkDelta, finish_reason=None) -> str:
    """Build an SSE data frame for a streaming chat completion chunk."""
    chunk = ChatCompletionChunk(
        id=cid, created=created, model=model,
        choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish_reason)]
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _build_tool_sections(tool_ctx: ToolContext, include_defs: bool = True) -> list[str]:
    """Build ordered list of tool instruction sections from a ToolContext."""
    sections: list[str] = []
    if tool_ctx.format_block:
        sections.append(f"### Format Specification\n{tool_ctx.format_block}")
    if include_defs and tool_ctx.defs_text:
        sections.append(f"### Tool Definitions\n{tool_ctx.defs_text}")
    if tool_ctx.instruction_text:
        sections.append(f"### Call Instructions\n{tool_ctx.instruction_text}")
    return sections


def _build_prompt(req: ChatCompletionRequest, tool_ctx: ToolContext) -> str:
    """Build the full prompt using DeepSeek native tag format (mirrors ds-free-api prompt.rs build()).

    Layout:
      <｜System｜>{system + reminder}<｜System｜>
      <｜end▁of▁sentence｜><｜User｜>{user msg}
      <｜Assistant｜>{assistant msg}
      <｜tool_outputs_begin｜><｜tool_output_begin｜>{tool result}<｜tool_output_end｜><｜tool_outputs_end｜>
      <｜Assistant｜>Hmm, I was just reminded...{format + instruction only}
    """
    messages = req.messages
    parts: list[str] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg.role == "tool":
            # Collect consecutive tool messages
            tool_contents = []
            while i < len(messages) and messages[i].role == "tool":
                c = _extract_content(messages[i].content)
                if c:
                    tool_contents.append(c)
                i += 1
            inner = "".join(
                f"{_TAG_TOOL_OUTPUT_BEGIN}{c}{_TAG_TOOL_OUTPUT_END}" for c in tool_contents
            )
            parts.append(f"{_TAG_TOOL_OUTPUTS_BEGIN}{inner}{_TAG_TOOL_OUTPUTS_END}")
        else:
            parts.append(_format_message(msg))
            i += 1

    # Build reminder sections
    tool_sections = _build_tool_sections(tool_ctx)

    if not tool_sections:
        return "".join(parts)

    reminder_body = "## Tool Calling\n" + "\n\n".join(tool_sections)

    # Inject full reminder into System block tail
    sys_content = f"\n\n{reminder_body}"
    sys_idx = next((j for j, p in enumerate(parts) if p.startswith(_TAG_SYSTEM_START)), None)
    if sys_idx is not None:
        # Insert before closing tag
        close = parts[sys_idx].rfind(_TAG_SYSTEM_END)
        if close != -1:
            parts[sys_idx] = parts[sys_idx][:close] + sys_content + parts[sys_idx][close:]
    else:
        parts.insert(0, f"{_TAG_SYSTEM_START}{sys_content}{_TAG_SYSTEM_END}\n")

    # Think block: only format spec + instructions (no tool defs — keeps it short)
    think_sections = _build_tool_sections(tool_ctx, include_defs=False)

    if think_sections:
        think_reminder = _THINK_BLOCK + "## Tool Calling\n" + "\n\n".join(think_sections)
        parts.append(f"{_TAG_ASSISTANT_START}{think_reminder}\n")

    return "".join(parts)


def _format_message(msg: ChatMessage) -> str:
    """Format a single message using DeepSeek native tags."""
    if msg.role == "system":
        body = _extract_content(msg.content)
        return f"{_TAG_SYSTEM_START}{body}{_TAG_SYSTEM_END}"
    elif msg.role == "user":
        body = _extract_content(msg.content)
        return f"{_TAG_USER_START}{body}"
    elif msg.role == "assistant":
        body_parts: list[str] = []
        if msg.content:
            body_parts.append(_extract_content(msg.content))
        if msg.tool_calls:
            items = []
            for tc in msg.tool_calls:
                if tc.function:
                    try:
                        args_val = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, ValueError):
                        args_val = tc.function.arguments
                    items.append(
                        f'{{"name": {json.dumps(tc.function.name)}, "arguments": {json.dumps(args_val)}}}'
                    )
            body_parts.append(f"{TOOL_CALL_START}\n[{', '.join(items)}]\n{TOOL_CALL_END}")
        body = "\n".join(body_parts)
        return f"{_TAG_ASSISTANT_START}{body}"
    else:
        body = _extract_content(msg.content)
        return f"<｜{msg.role.capitalize()}｜>{body}"


def _build_tool_system_prompt(base_system: str, tool_ctx: ToolContext) -> str:
    """Build system prompt with tool instructions injected.

    Instead of embedding everything into a giant tagged prompt (which duplicates
    the conversation history), we inject tool definitions into the system_prompt
    parameter so DeepSeek's chat API handles history via session + parent_message_id.
    """
    tool_sections = _build_tool_sections(tool_ctx)

    if not tool_sections:
        return base_system

    tool_block = "## Tool Calling\n" + "\n\n".join(tool_sections)
    return f"{base_system}\n\n{tool_block}" if base_system else tool_block


def _extract_tool_aware_prompt(messages: list) -> str:
    """Extract prompt for tool-aware requests, following session continuation logic.

    Handles two cases:
      1. Last message is a user message → extract it (same as non-tool path)
      2. Last messages are tool results → format as tool output block
    """
    if not messages:
        return ""

    last_msg = messages[-1]

    if last_msg.role == "user":
        return _extract_content(last_msg.content)

    if last_msg.role == "tool":
        # Collect consecutive trailing tool messages
        tool_results: list[str] = []
        i = len(messages) - 1
        while i >= 0 and messages[i].role == "tool":
            tool_results.insert(0, _extract_content(messages[i].content))
            i -= 1

        # Format as tool output block that DeepSeek understands
        inner = "".join(
            f"{_TAG_TOOL_OUTPUT_BEGIN}{c}{_TAG_TOOL_OUTPUT_END}" for c in tool_results
        )
        return f"{_TAG_TOOL_OUTPUTS_BEGIN}{inner}{_TAG_TOOL_OUTPUTS_END}"

    # Fallback: last user message
    user_messages = [msg for msg in messages if msg.role == "user"]
    if user_messages:
        return _extract_content(user_messages[-1].content)
    return ""


# ---------------------------------------------------------------------------
# Tool calling — response parsing (ported from ds-free-api tool_parser.rs)
# ---------------------------------------------------------------------------

# Unicode normalization: ｜(U+FF5C) → |, ▁(U+2581) → _
def _norm_tag_char(c: str) -> str:
    if c == '\uff5c':
        return '|'
    if c == '\u2581':
        return '_'
    return c


def _eq_tag_char(a: str, b: str) -> bool:
    return a == b or _norm_tag_char(a) == _norm_tag_char(b)


def _fuzzy_match_tag(haystack: str, partial: str) -> Optional[tuple]:
    """Fuzzy match a tag in haystack, supporting ｜↔| and ▁↔_ equivalence."""
    n_chars = list(partial)
    h_chars = list(haystack)
    if not n_chars or len(h_chars) < len(n_chars):
        return None
    for start in range(len(h_chars) - len(n_chars) + 1):
        matched = True
        for j in range(len(n_chars)):
            if not _eq_tag_char(n_chars[j], h_chars[start + j]):
                matched = False
                break
        if matched:
            byte_pos = sum(len(c.encode('utf-8')) for c in h_chars[:start])
            tag_len = sum(len(c.encode('utf-8')) for c in h_chars[start:start + len(n_chars)])
            return (byte_pos, haystack[byte_pos:byte_pos + tag_len])
    return None


def _match_start_tag(s: str, tag: str) -> Optional[tuple]:
    """Match a start tag in string, with fuzzy fallback."""
    partial = tag.rstrip('>')
    pos = s.find(partial)
    if pos != -1:
        return (pos, s[pos:pos + len(partial)])
    return _fuzzy_match_tag(s, partial)


def _find_start_tag(s: str) -> Optional[tuple]:
    """Find the first tool call start tag in s."""
    return _match_start_tag(s, TOOL_CALL_START)


def _find_end_tag(s: str, from_pos: int = 0, start_tag: Optional[str] = None) -> Optional[tuple]:
    """Find the tool call end tag in s starting from from_pos."""
    search = s[from_pos:]

    # Try to derive close tag from start_tag
    if start_tag:
        open_tag = start_tag.rstrip('>')
        close_tag = f"</{open_tag[1:]}>"
        pos = search.find(close_tag)
        if pos != -1:
            abs_pos = from_pos + pos
            return (abs_pos, s[abs_pos:abs_pos + len(close_tag)])
        # Fuzzy fallback
        close_partial = close_tag.rstrip('>')
        result = _fuzzy_match_tag(search, close_partial)
        if result:
            rel_pos, matched = result
            abs_pos = from_pos + rel_pos
            return (abs_pos, s[abs_pos:abs_pos + len(matched)])

    # Try known end tags
    for end in (TOOL_CALL_END,):
        pos = search.find(end)
        if pos != -1:
            abs_pos = from_pos + pos
            return (abs_pos, s[abs_pos:abs_pos + len(end)])
        end_partial = end.rstrip('>')
        result = _fuzzy_match_tag(search, end_partial)
        if result:
            rel_pos, matched = result
            abs_pos = from_pos + rel_pos
            return (abs_pos, s[abs_pos:abs_pos + len(matched)])

    return None


def _is_inside_code_fence(text: str, tag_pos: int) -> bool:
    """Check if tag_pos is inside a markdown code fence."""
    return text[:tag_pos].count('```') % 2 == 1


def _repair_invalid_backslashes(s: str) -> str:
    """Fix invalid backslash escapes in JSON strings."""
    out: list[str] = []
    chars = list(s)
    i = 0
    while i < len(chars):
        if chars[i] == '\\' and i + 1 < len(chars):
            next_c = chars[i + 1]
            if next_c in '"\\/bfnrtu':
                out.append('\\')
                out.append(next_c)
                i += 2
            else:
                out.append('\\\\')
                out.append(next_c)
                i += 2
        else:
            out.append(chars[i])
            i += 1
    return ''.join(out)


def _repair_unquoted_keys(s: str) -> str:
    """Fix unquoted JSON keys like {name: "value"} → {"name": "value"}."""
    out: list[str] = []
    chars = list(s)
    length = len(chars)
    i = 0
    while i < length:
        if chars[i] in '{,' and i + 1 < length:
            out.append(chars[i])
            i += 1
            while i < length and chars[i] in ' \t\n\r':
                out.append(chars[i])
                i += 1
            if i < length and (chars[i].isalpha() or chars[i] == '_'):
                key_start = i
                while i < length and (chars[i].isalnum() or chars[i] == '_'):
                    i += 1
                if i < length and chars[i] == ':':
                    out.append('"')
                    out.extend(chars[key_start:i])
                    out.append('"')
                else:
                    out.extend(chars[key_start:i])
                    continue
            continue
        out.append(chars[i])
        i += 1
    return ''.join(out)


def _repair_json(s: str) -> Optional[str]:
    """Three-layer JSON repair: direct parse → backslash repair → unquoted key repair."""
    try:
        json.loads(s)
        return s
    except (json.JSONDecodeError, ValueError):
        pass

    step1 = _repair_invalid_backslashes(s)
    try:
        json.loads(step1)
        return step1
    except (json.JSONDecodeError, ValueError):
        pass

    step2 = _repair_unquoted_keys(step1)
    try:
        json.loads(step2)
        return step2
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _next_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _parse_tool_calls_from_text(text: str) -> Optional[tuple]:
    """Parse tool calls from model output text.

    Returns (list_of_ToolCall, remaining_text) or None if no tool calls found.
    Mirrors ds-free-api parse_tool_calls_with().
    """
    # Find start tag
    start_result = _find_start_tag(text)
    if start_result is None:
        return None
    start, start_tag = start_result
    after_start = start + len(start_tag)

    # Skip if inside code fence
    if _is_inside_code_fence(text, start):
        return None

    # Find end tag
    end_result = _find_end_tag(text, after_start, start_tag)
    if end_result:
        end_pos, matched_end = end_result
        inner_end = end_pos
        after_end = end_pos + len(matched_end)
    else:
        inner_end = len(text)
        after_end = len(text)

    inner = text[after_start:inner_end]

    # Extract JSON array or object
    arr = None
    arr_start = inner.find('[')
    if arr_start != -1:
        arr_end = inner.rfind(']')
        if arr_end == -1:
            arr_end = len(inner)
        json_str = inner[arr_start:arr_end + 1]
        if json_str.strip() == '[]':
            return None
        try:
            arr = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            repaired = _repair_json(json_str)
            if repaired:
                try:
                    arr = json.loads(repaired)
                except (json.JSONDecodeError, ValueError):
                    arr = None
            if arr is None:
                # Try to parse as single object
                obj_start = json_str.find('{')
                obj_end = json_str.rfind('}')
                if obj_start != -1 and obj_end != -1:
                    obj_str = json_str[obj_start:obj_end + 1]
                    repaired_obj = _repair_json(obj_str)
                    if repaired_obj:
                        try:
                            obj = json.loads(repaired_obj)
                            if isinstance(obj, dict):
                                arr = [obj]
                        except (json.JSONDecodeError, ValueError):
                            pass

    if arr is None:
        # Try single object
        obj_start = inner.find('{')
        obj_end = inner.rfind('}')
        if obj_start != -1 and obj_end != -1:
            json_str = inner[obj_start:obj_end + 1]
            repaired = _repair_json(json_str)
            if repaired:
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict):
                        arr = [obj]
                except (json.JSONDecodeError, ValueError):
                    pass

    if not arr:
        return None

    # Convert to ToolCall objects
    calls: list[ToolCall] = []
    for idx, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if not name or not isinstance(name, str):
            continue
        arguments = item.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        else:
            arguments = str(arguments) if arguments else '{}'

        calls.append(ToolCall(
            id=_next_call_id(),
            type="function",
            function=ToolCallFunction(name=name, arguments=arguments)
        ))

    if not calls:
        return None

    remaining = text[:start] + text[after_end:]
    return (calls, remaining)


# ---------------------------------------------------------------------------
# Sliding-window tool call detector for streaming (ported from ds-free-api)
# ---------------------------------------------------------------------------
_SCAN_WINDOW = 71  # Same as ds-free-api's W

class _ToolCallDetector:
    """Stateful sliding-window detector that finds tool call tags in a text stream.

    Usage:
        detector = _ToolCallDetector()
        for text_chunk in stream:
            result = detector.push(text_chunk)
            if result is not None:
                # result = (tool_calls_list, remaining_text_after_tags)
                break
        # After stream ends, call detector.finish() for any remaining buffer
    """
    def __init__(self):
        self._buffer = ""
        self._found = False

    def push(self, chunk: str) -> Optional[tuple]:
        """Push a new text chunk. Returns (calls, remaining) if tool calls detected, else None."""
        self._buffer += chunk

        if not self._found:
            # Detecting state: scan for start tag
            result = _find_start_tag(self._buffer)
            if result is None:
                # No tag found — release safe portion of buffer
                safe_len = max(0, len(self._buffer) - _SCAN_WINDOW)
                if safe_len > 0:
                    released = self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    return None  # caller should emit released as content
                return None
            else:
                self._found = True
                # Fall through to collecting

        if self._found:
            # Collecting state: check if we have both start and end
            parse_result = _parse_tool_calls_from_text(self._buffer)
            if parse_result is not None:
                return parse_result
            # End tag not yet seen — keep buffering
            return None

    @property
    def buffer(self) -> str:
        return self._buffer

    def finish(self) -> Optional[tuple]:
        """Call after stream ends. Attempts final parse of remaining buffer."""
        if not self._buffer:
            return None
        return _parse_tool_calls_from_text(self._buffer)


# --- Singleton DeepSeekAPI instance (avoids re-loading WASM per request) ---
_api_lock = threading.Lock()
_api_instance = None

def _get_api() -> DeepSeekAPI:
    """Get or create the singleton DeepSeekAPI instance."""
    global _api_instance
    auth_token = os.getenv("DEEPSEEK_AUTH_TOKEN")
    if not auth_token:
        raise HTTPException(status_code=500, detail="DEEPSEEK_AUTH_TOKEN not set")
    with _api_lock:
        if _api_instance is None:
            _api_instance = DeepSeekAPI(auth_token)
    return _api_instance


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-Id"),
):
    try:
        api = _get_api()

        # --- Tool context extraction ---
        has_tools = bool(request.tools)
        tool_ctx = _extract_tool_context(request) if has_tools else ToolContext(None, None, None)

        # --- Build prompt ---
        # System prompt stays as the original client system message (both paths).
        # Tool instructions are embedded directly in the prompt so the model sees them.
        # Only the NEW user message (or tool results) is sent — not the entire history.
        system_messages = [msg for msg in request.messages if msg.role == "system"]
        base_system = _extract_content(system_messages[-1].content) if system_messages else ""

        if has_tools and (tool_ctx.format_block or tool_ctx.defs_text or tool_ctx.instruction_text):
            # Build tool instruction block
            tool_sections: list[str] = []
            if tool_ctx.defs_text:
                tool_sections.append(tool_ctx.defs_text)
            if tool_ctx.format_block:
                tool_sections.append(tool_ctx.format_block)
            if tool_ctx.instruction_text:
                tool_sections.append(tool_ctx.instruction_text)
            tool_block = "\n\n".join(tool_sections)

            # Extract new user message or tool results
            raw_prompt = _extract_tool_aware_prompt(request.messages)

            # Embed tool instructions in the prompt itself
            prompt = f"{tool_block}\n\n---\n\n{raw_prompt}"
            system_prompt = base_system or None
            print(f"[tools] Injected {len(request.tools)} tool definitions into prompt")
        else:
            # No tools — simple extraction (backward compatible)
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message provided")
            prompt = _extract_content(user_messages[-1].content)
            system_prompt = base_system or None

        # --- Resolve session ---
        # Priority: 1) conversation_id from request body or header  2) first-user-message hash fallback
        conv_id = request.conversation_id or x_conversation_id
        fallback_key = _build_key(request.messages)

        if conv_id:
            # Direct lookup by conversation_id (most reliable)
            found = session_mgr.find_by_conversation_id(conv_id)
            if found:
                lookup_key, session_info = found
                session_id = session_info['session_id']
                parent_message_id = session_info.get('parent_message_id')
                print(f"[session] Reusing session {session_id} (conv_id={conv_id[:8]}…, parent_msg={parent_message_id})")
            else:
                # conversation_id provided but not found — could be from a different server restart
                # Fall back to first-message hash, but keep the conversation_id
                session_info = session_mgr.get(fallback_key)
                if session_info:
                    lookup_key = fallback_key
                    session_id = session_info['session_id']
                    parent_message_id = session_info.get('parent_message_id')
                    print(f"[session] Reusing session {session_id} (fallback_key={fallback_key[:8]}…, conv_id={conv_id[:8]}…)")
                else:
                    lookup_key = fallback_key
                    session_id = api.create_chat_session()
                    parent_message_id = None
                    print(f"[session] Created new session {session_id} (conv_id={conv_id[:8]}…, fallback_key={fallback_key[:8]}…)")
        else:
            # No conversation_id — use first-user-message hash
            session_info = session_mgr.get(fallback_key)
            if session_info:
                lookup_key = fallback_key
                session_id = session_info['session_id']
                parent_message_id = session_info.get('parent_message_id')
                conv_id = session_info.get('conversation_id')
                print(f"[session] Reusing session {session_id} (key={fallback_key[:8]}…, parent_msg={parent_message_id})")
            else:
                lookup_key = fallback_key
                session_id = api.create_chat_session()
                parent_message_id = None
                conv_id = str(uuid.uuid4())
                print(f"[session] Created new session {session_id} (key={fallback_key[:8]}…, conv_id={conv_id[:8]}…)")

        session_mgr.put(lookup_key, {
            'session_id': session_id,
            'parent_message_id': parent_message_id,
            'conversation_id': conv_id,
        })

        # Generate a unique ID for this completion
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # --- Resolve model params ---
        # ds-free-api behavior: thinking ON by default, search ON by default
        # (DeepSeek injects stronger system prompts in search mode, improving tool compliance)
        model_type, thinking_enabled, search_enabled = _resolve_model(
            request.model, request.model_type, request.thinking_enabled, request.search_enabled,
            reasoning_effort=request.reasoning_effort, web_search_options=request.web_search_options
        )
        if has_tools and not search_enabled:
            search_enabled = True
            print("[tools] Auto-enabled search mode for better tool call compliance")

        if request.stream:
            async def stream_generator():
                captured_message_id = None
                try:
                    chunks = api.chat_completion(
                        session_id,
                        prompt,
                        parent_message_id=parent_message_id,
                        thinking_enabled=thinking_enabled,
                        search_enabled=search_enabled,
                        model_type=model_type,
                        system_prompt=system_prompt
                    )

                    # Send first chunk with role + conversation_id
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None
                        )],
                        conversation_id=conv_id
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"

                    finish_reason = "stop"
                    text_parts: list[str] = []

                    for chunk in chunks:
                        if chunk.get('type') == 'message_id' and chunk.get('message_id'):
                            captured_message_id = chunk['message_id']
                            print(f"[session] Captured message_id: {captured_message_id}")
                            continue
                        if chunk['type'] == 'thinking' and chunk['content']:
                            yield _sse(completion_id, created, request.model,
                                       ChatCompletionChunkDelta(reasoning_content=chunk['content']))
                        elif chunk['type'] == 'text' and chunk['content']:
                            if has_tools:
                                text_parts.append(chunk['content'])
                            else:
                                yield _sse(completion_id, created, request.model,
                                           ChatCompletionChunkDelta(content=chunk['content']))

                    # Tool call parsing (buffered text mode)
                    if has_tools:
                        buffered_text = "".join(text_parts)
                        parse_result = _parse_tool_calls_from_text(buffered_text)
                        if parse_result:
                            tool_calls, remaining_text = parse_result
                            print(f"[tools] Parsed {len(tool_calls)} tool call(s) from response")
                            if remaining_text.strip():
                                yield _sse(completion_id, created, request.model,
                                           ChatCompletionChunkDelta(content=remaining_text.strip()))
                            for idx, tc in enumerate(tool_calls):
                                yield _sse(completion_id, created, request.model,
                                           ChatCompletionChunkDelta(tool_calls=[
                                               ChatCompletionChunkDeltaToolCall(
                                                   index=idx, id=tc.id, type="function",
                                                   function={"name": tc.function.name, "arguments": tc.function.arguments}
                                               )]))
                            finish_reason = "tool_calls"
                        elif buffered_text:
                            yield _sse(completion_id, created, request.model,
                                       ChatCompletionChunkDelta(content=buffered_text))

                    # Update parent_message_id for the next turn
                    if captured_message_id:
                        session_mgr.update_parent_message_id(lookup_key, captured_message_id)
                        print(f"[session] Updated parent_message_id for key={lookup_key[:8]}… → {captured_message_id}")

                    # Send final chunk
                    yield _sse(completion_id, created, request.model,
                               ChatCompletionChunkDelta(), finish_reason=finish_reason)
                    yield "data: [DONE]\n\n"

                except (AuthenticationError, RateLimitError, NetworkError, APIError) as e:
                    error_data = {"error": {"message": str(e), "type": type(e).__name__}}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Conversation-Id": conv_id,
                }
            )
        else:
            # Non-streaming response
            full_content = ""
            reasoning_content = ""
            captured_message_id = None

            chunks = api.chat_completion(
                session_id,
                prompt,
                parent_message_id=parent_message_id,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
                model_type=model_type,
                system_prompt=system_prompt
            )

            for chunk in chunks:
                if chunk.get('type') == 'message_id' and chunk.get('message_id'):
                    captured_message_id = chunk['message_id']
                    print(f"[session] Captured message_id: {captured_message_id}")
                elif chunk['type'] == 'thinking':
                    reasoning_content += chunk['content']
                elif chunk['type'] == 'text':
                    full_content += chunk['content']

            # Update parent_message_id for the next turn
            if captured_message_id:
                session_mgr.update_parent_message_id(lookup_key, captured_message_id)
                print(f"[session] Updated parent_message_id for key={lookup_key[:8]}… → {captured_message_id}")

            # Parse for tool calls (non-streaming)
            finish_reason = "stop"
            tool_calls_result = None

            if has_tools:
                parse_result = _parse_tool_calls_from_text(full_content)
                if parse_result:
                    tool_calls, remaining_text = parse_result
                    print(f"[tools] Parsed {len(tool_calls)} tool call(s) from response")
                    tool_calls_result = tool_calls
                    full_content = remaining_text.strip() or None
                    finish_reason = "tool_calls"

            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=full_content,
                        reasoning_content=reasoning_content or None,
                        tool_calls=tool_calls_result,
                    ),
                    finish_reason=finish_reason
                )],
                usage=ChatCompletionResponseUsage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len((full_content or "").split()) + len(reasoning_content.split()),
                    total_tokens=len(prompt.split()) + len((full_content or "").split()) + len(reasoning_content.split())
                ),
                conversation_id=conv_id
            )

            return Response(
                content=response.model_dump_json(),
                media_type="application/json",
                headers={"X-Conversation-Id": conv_id}
            )

    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except NetworkError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloudflare bypass api")

    parser.add_argument("--nolog", action="store_true", help="Disable logging")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()
    display = None

    if (args.headless or DOCKER_MODE) and Display is not None:
        display = Display(visible=0, size=(1920, 1080))
        display.start()

        def cleanup_display():
            if display:
                display.stop()
        atexit.register(cleanup_display)
    elif args.headless or DOCKER_MODE:
        print("Warning: pyvirtualdisplay not installed. Headless mode may not work properly.")
        print("Install it with: pip install pyvirtualdisplay")

    log = not args.nolog

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)