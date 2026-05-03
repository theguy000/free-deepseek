import json
import re
import os
import hashlib
import shutil
from pathlib import Path
from collections import OrderedDict
from urllib.parse import urlparse
import sys

from .CloudflareBypasser import CloudflareBypasser
from DrissionPage import ChromiumPage, ChromiumOptions
from fastapi import FastAPI, HTTPException, Response
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

# Import DeepSeekAPI
from .api import DeepSeekAPI, AuthenticationError, RateLimitError, NetworkError, APIError

# Check if running in Docker mode
DOCKER_MODE = os.getenv("DOCKERMODE", "false").lower() == "true"

SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))

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


def _build_key(messages: list, include_last_user: bool = False) -> str:
    """Build a conversation key from system + user messages only."""
    parts = []
    user_idx = 0
    for msg in messages:
        role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
        if role == "system":
            parts.append(f"s:{content}")
        elif role == "user":
            parts.append(f"u{user_idx}:{content}")
            user_idx += 1
    if not include_last_user and parts and parts[-1].startswith("u"):
        parts = parts[:-1]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class SessionManager:
    """Disk-persistent session manager for conversation→DeepSeek session mapping.
    
    All data is written to disk immediately on every mutation so sessions
    survive server crashes (e.g. curl_cffi heap corruption).
    """

    def __init__(self, max_sessions: int = MAX_CACHE_SIZE, filepath: Path = SESSIONS_FILE):
        self._max = max_sessions
        self._file = filepath
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
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, info: Dict[str, Optional[str]]):
        """Store session info and persist to disk immediately."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = info
        while len(self._data) > self._max:
            evicted_key, _ = self._data.popitem(last=False)
            print(f"[session] Evicted oldest session {evicted_key[:8]}…")
        self._save()

    def update_parent_message_id(self, key: str, parent_message_id: str):
        """Update parent_message_id and persist to disk immediately."""
        if key in self._data:
            self._data[key]['parent_message_id'] = parent_message_id
            self._save()


session_mgr = SessionManager()


# Pydantic model for the response
class CookieResponse(BaseModel):
    cookies: Dict[str, str]
    user_agent: str


# OpenAI-compatible models
class ChatMessage(BaseModel):
    model_config = {"extra": "ignore"}
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Union[str, List[Any]]


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "deepseek-chat"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    thinking_enabled: Optional[bool] = True
    search_enabled: Optional[bool] = False
    conversation_id: Optional[str] = None


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


class ChatCompletionChunkDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["assistant"]] = None


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


# Function to check if the URL is safe
def is_safe_url(url: str) -> bool:
    parsed_url = urlparse(url)
    ip_pattern = re.compile(
        r"^(127\.0\.0\.1|localhost|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|172\.1[6-9]\.\d+\.\d+|172\.2[0-9]\.\d+\.\d+|172\.3[0-1]\.\d+\.\d+|192\.168\.\d+\.\d+)$"
    )
    hostname = parsed_url.hostname
    if (hostname and ip_pattern.match(hostname)) or parsed_url.scheme == "file":
        return False
    return True


# Function to verify if the page has loaded properly
def verify_page_loaded(driver: ChromiumPage) -> bool:
    """Verify if the page has loaded properly"""
    try:
        # Wait for body element to be present
        body = driver.ele('tag:body', timeout=10)
        # Check if page has actual content
        return len(body.html) > 100
    except:
        return False


# Function to bypass Cloudflare protection
def bypass_cloudflare(url: str, retries: int, log: bool, proxy: str = None) -> ChromiumPage:
    max_load_retries = 3

    for load_attempt in range(max_load_retries):
        options = ChromiumOptions().auto_port()
        if DOCKER_MODE:
            options.set_argument("--auto-open-devtools-for-tabs", "true")
            options.set_argument("--remote-debugging-port=9222")
            options.set_argument("--no-sandbox")  # Necessary for Docker
            options.set_argument("--disable-gpu")  # Optional, helps in some cases
            options.set_paths(browser_path=browser_path).headless(False)
        else:
            options.set_paths(browser_path=browser_path).headless(False)

        if proxy:
            options.set_proxy(proxy)

        driver = ChromiumPage(addr_or_opts=options)
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


# Endpoint to get cookies
@app.get("/cookies", response_model=CookieResponse)
async def get_cookies(url: str, retries: int = 5, proxy: str = None):
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        driver = bypass_cloudflare(url, retries, log, proxy)
        cookies = {cookie.get("name", ""): cookie.get("value", " ") for cookie in driver.cookies()}
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
        cookies_json = {cookie.get("name", ""): cookie.get("value", " ") for cookie in driver.cookies()}
        response = Response(content=html, media_type="text/html")
        response.headers["cookies"] = json.dumps(cookies_json)
        response.headers["user_agent"] = driver.user_agent
        driver.quit()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    auth_token = os.getenv("DEEPSEEK_AUTH_TOKEN")
    if not auth_token:
        raise HTTPException(status_code=500, detail="DEEPSEEK_AUTH_TOKEN not set")

    try:
        api = DeepSeekAPI(auth_token)
        
        # Extract the last user message as the prompt
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message provided")
        
        raw_content = user_messages[-1].content
        if isinstance(raw_content, list):
            prompt = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            prompt = raw_content
        
        # --- Resolve session ---
        if request.conversation_id:
            lookup_key = request.conversation_id
            store_key = request.conversation_id
        else:
            lookup_key = _build_key(request.messages, include_last_user=False)
            store_key = _build_key(request.messages, include_last_user=True)

        session_info = session_mgr.get(lookup_key)

        if session_info:
            session_id = session_info['session_id']
            parent_message_id = session_info.get('parent_message_id')
            print(f"[session] Reusing session {session_id} (key={lookup_key[:8]}…, parent_msg={parent_message_id})")
        else:
            session_id = api.create_chat_session()
            parent_message_id = None
            print(f"[session] Created new session {session_id} (key={lookup_key[:8]}…)")

        session_mgr.put(store_key, {'session_id': session_id, 'parent_message_id': parent_message_id})
        
        # Generate a unique ID for this completion
        import time
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        
        if request.stream:
            async def stream_generator():
                captured_message_id = None
                try:
                    chunks = api.chat_completion(
                        session_id,
                        prompt,
                        parent_message_id=parent_message_id,
                        thinking_enabled=request.thinking_enabled,
                        search_enabled=request.search_enabled
                    )
                    
                    # Send first chunk with role
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"
                    
                    # Stream content chunks
                    for chunk in chunks:
                        # Capture message_id for threading
                        if chunk.get('type') == 'message_id' and chunk.get('message_id'):
                            captured_message_id = chunk['message_id']
                            print(f"[session] Captured message_id: {captured_message_id}")
                            continue
                        if chunk['type'] == 'text' and chunk['content']:
                            content_chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=request.model,
                                choices=[ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=chunk['content']),
                                    finish_reason=None
                                )]
                            )
                            yield f"data: {content_chunk.model_dump_json()}\n\n"
                    
                    # Update parent_message_id for the next turn
                    if captured_message_id:
                        session_mgr.update_parent_message_id(store_key, captured_message_id)
                        print(f"[session] Updated parent_message_id for key={store_key[:8]}… → {captured_message_id}")
                    
                    # Send final chunk
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop"
                        )]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except AuthenticationError:
                    raise HTTPException(status_code=401, detail="Invalid authentication token")
                except RateLimitError:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                except NetworkError as e:
                    raise HTTPException(status_code=503, detail=str(e))
                except APIError as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            full_content = ""
            captured_message_id = None
            chunks = api.chat_completion(
                session_id,
                prompt,
                parent_message_id=parent_message_id,
                thinking_enabled=request.thinking_enabled,
                search_enabled=request.search_enabled
            )
            
            for chunk in chunks:
                if chunk.get('type') == 'message_id' and chunk.get('message_id'):
                    captured_message_id = chunk['message_id']
                    print(f"[session] Captured message_id: {captured_message_id}")
                elif chunk['type'] == 'text':
                    full_content += chunk['content']
            
            # Update parent_message_id for the next turn
            if captured_message_id:
                session_mgr.update_parent_message_id(store_key, captured_message_id)
                print(f"[session] Updated parent_message_id for key={store_key[:8]}… → {captured_message_id}")
            
            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_content),
                    finish_reason="stop"
                )],
                usage=ChatCompletionResponseUsage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(full_content.split()),
                    total_tokens=len(prompt.split()) + len(full_content.split())
                )
            )
            
            return response
            
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

    if args.nolog:
        log = False
    else:
        log = True

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)