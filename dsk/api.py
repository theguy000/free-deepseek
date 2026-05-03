from curl_cffi import requests
from typing import Optional, Dict, Any, Generator, Literal
import json
import queue
import threading
from .pow import DeepSeekPOW
from importlib.metadata import version, PackageNotFoundError
import sys
from pathlib import Path
import subprocess
import time

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']
ModelType = Literal['default', 'expert']  # default=Instant(V4 Flash), expert=V4 Pro

class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors"""
    pass

class AuthenticationError(DeepSeekError):
    """Raised when authentication fails"""
    pass

class RateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded"""
    pass

class NetworkError(DeepSeekError):
    """Raised when network communication fails"""
    pass

class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request"""
    pass

class APIError(DeepSeekError):
    """Raised when API returns an error response"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        try:
            if version('curl-cffi') != '0.8.1b9':
                print("\033[93mWarning: curl-cffi version 0.8.1b9 is required. Install with: pip install curl-cffi==0.8.1b9\033[0m", file=sys.stderr)
        except PackageNotFoundError:
            print("\033[93mWarning: curl-cffi not found. Install version 0.8.1b9 with: pip install curl-cffi==0.8.1b9\033[0m", file=sys.stderr)

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()
        self._session_parent_ids: Dict[str, Optional[str]] = {}

        self.cookies = self._load_cookies()

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '2.0.0',
            'x-client-timezone-offset': '-21600',
        }

        if pow_response:
            headers['x-ds-pow-response'] = pow_response

        return headers

    def _load_cookies(self) -> dict:
        cookies_path = Path(__file__).parent / 'cookies.json'
        try:
            with open(cookies_path, 'r') as f:
                return json.load(f).get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"\033[93mWarning: Could not load cookies from {cookies_path}: {e}\033[0m", file=sys.stderr)
            return {}

    def _refresh_cookies(self) -> None:
        """Run the cookie refresh script and reload cookies"""
        try:
            script_path = Path(__file__).parent / 'bypass.py'
            subprocess.run([sys.executable, script_path], check=True)
            time.sleep(2)
            self.cookies = self._load_cookies()
        except Exception as e:
            print(f"\033[93mWarning: Failed to refresh cookies: {e}\033[0m", file=sys.stderr)

    def _make_request(self, method: str, endpoint: str, json_data: Dict[str, Any], pow_required: bool = False) -> Any:
        url = f"{self.BASE_URL}{endpoint}"

        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                pow_response = None
                if pow_required:
                    pow_response = self.pow_solver.solve_challenge(self._get_pow_challenge())
                headers = self._get_headers(pow_response)

                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies,
                    impersonate='chrome120',
                    timeout=None
                )

                # Check if we hit Cloudflare protection
                if "<!DOCTYPE html>" in response.text and "Just a moment" in response.text:
                    print("\033[93mWarning: Cloudflare protection detected. Bypassing...\033[0m", file=sys.stderr)
                    if retry_count < max_retries - 1:
                        self._refresh_cookies()  # Refresh cookies
                        retry_count += 1
                        continue

                # Handle other response codes
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(f"Server error occurred: {response.text}", response.status_code)
                elif response.status_code != 200:
                    raise APIError(f"API request failed: {response.text}", response.status_code)

                return response.json()

            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred: {str(e)}")
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    def _get_pow_challenge(self) -> Dict[str, Any]:
        try:
            response = self._make_request(
                'POST',
                '/chat/create_pow_challenge',
                {'target_path': '/api/v0/chat/completion'}
            )
            return response['data']['biz_data']['challenge']
        except KeyError:
            raise APIError("Invalid challenge response format from server")

    def create_chat_session(self) -> str:
        """Creates a new chat session and returns the session ID"""
        try:
            response = self._make_request(
                'POST',
                '/chat_session/create',
                {}
            )
            biz_data = response['data']['biz_data']
            # Handle both old format (biz_data.id) and new format (biz_data.chat_session.id)
            if 'id' in biz_data:
                return biz_data['id']
            elif 'chat_session' in biz_data:
                return biz_data['chat_session']['id']
            raise KeyError('id')
        except KeyError:
            raise APIError("Invalid session creation response format from server")

    def chat_completion(self,
                    chat_session_id: str,
                    prompt: str,
                    parent_message_id: Optional[Any] = None,
                    thinking_enabled: bool = True,
                    search_enabled: bool = False,
                    model_type: ModelType = 'default',
                    system_prompt: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and get streaming response

        Args:
            chat_session_id (str): The ID of the chat session
            prompt (str): The message to send
            parent_message_id (Optional[str]): ID of the parent message for threading
            thinking_enabled (bool): Whether to show the thinking process
            search_enabled (bool): Whether to enable web search for up-to-date information
            model_type (str): Model to use - 'default' for Instant (V4 Flash), 'expert' for V4 Pro
            system_prompt (Optional[str]): Optional system prompt to prepend to the conversation

        The 7 available model variations are:
            1. V4 Flash: model_type='default', thinking_enabled=False, search_enabled=False
            2. V4 Flash + Search: model_type='default', thinking_enabled=False, search_enabled=True
            3. V4 Flash + DeepThink: model_type='default', thinking_enabled=True, search_enabled=False
            4. V4 Flash + DeepThink + Search: model_type='default', thinking_enabled=True, search_enabled=True
            5. V4 Pro: model_type='expert', thinking_enabled=False, search_enabled=False
            6. V4 Pro + DeepThink: model_type='expert', thinking_enabled=True, search_enabled=False
            7. V4 Pro + DeepThink + Search: model_type='expert', thinking_enabled=True, search_enabled=True

        Returns:
            Generator[Dict[str, Any], None, None]: Yields message chunks with content and type

        Raises:
            AuthenticationError: If the authentication token is invalid
            RateLimitError: If the API rate limit is exceeded
            NetworkError: If a network error occurs
            APIError: If any other API error occurs
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        # Per-call parsing context (thread-safe — not stored on self)
        _parse_ctx = {'fragment_type': 'RESPONSE', 'last_content_path': ''}

        # Use stored parent_message_id for the session if not explicitly provided
        if parent_message_id is None:
            parent_message_id = self._session_parent_ids.get(chat_session_id)

        json_data = {
            'chat_session_id': chat_session_id,
            'parent_message_id': parent_message_id,
            'model_type': model_type,
            'prompt': prompt,
            'ref_file_ids': [],
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
            'system_prompt': system_prompt or '',
            'preempt': False,
        }

        # Use content_callback + queue to avoid curl_cffi heap corruption
        # (the stream=True / iter_lines() path triggers a double-free in libcurl cleanup)
        chunk_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()
        _line_buffer = bytearray()

        def _on_data(data: bytes):
            """Callback invoked by libcurl for each chunk of response data."""
            nonlocal _line_buffer
            _line_buffer.extend(data)
            # Split on newlines; SSE lines are \n-delimited
            while b'\n' in _line_buffer:
                line, _, _line_buffer = _line_buffer.partition(b'\n')
                line = line.strip()
                if line:
                    chunk_queue.put(line)

        _request_error = [None]  # mutable container to pass error from thread
        _response_holder = [None]  # keep response alive to prevent premature GC

        def _do_request():
            try:
                resp = requests.post(
                    f"{self.BASE_URL}/chat/completion",
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies,
                    impersonate='chrome120',
                    content_callback=_on_data,
                    timeout=None,
                )
                _response_holder[0] = resp
                # Push any remaining data in the line buffer
                remaining = _line_buffer.strip()
                if remaining:
                    chunk_queue.put(bytes(remaining))
                # Check status code
                if resp.status_code != 200:
                    error_text = resp.text if hasattr(resp, 'text') else ''
                    if resp.status_code == 401:
                        _request_error[0] = AuthenticationError("Invalid or expired authentication token")
                    elif resp.status_code == 429:
                        _request_error[0] = RateLimitError("API rate limit exceeded")
                    else:
                        _request_error[0] = APIError(f"API request failed: {error_text}", resp.status_code)
            except requests.exceptions.RequestException as e:
                _request_error[0] = NetworkError(f"Network error occurred during streaming: {str(e)}")
            except Exception as e:
                _request_error[0] = APIError(f"Request error: {str(e)}")
            finally:
                chunk_queue.put(_SENTINEL)

        try:
            headers = self._get_headers(
                pow_response=self.pow_solver.solve_challenge(
                    self._get_pow_challenge()
                )
            )

            # Run the blocking request in a background thread
            request_thread = threading.Thread(target=_do_request, daemon=True)
            request_thread.start()

            while True:
                try:
                    item = chunk_queue.get(timeout=120)
                except queue.Empty:
                    raise APIError("Streaming response timed out")

                if item is _SENTINEL:
                    # Request finished — check for errors
                    if _request_error[0]:
                        raise _request_error[0]
                    break

                try:
                    parsed_events = self._parse_chunk(item, _parse_ctx)
                    for parsed in parsed_events:
                        # Track message_id for conversation threading
                        if parsed.get('type') == 'message_id':
                            self._session_parent_ids[chat_session_id] = parsed['message_id']
                        yield parsed
                        if parsed.get('finish_reason') == 'stop':
                            return
                except Exception as e:
                    raise APIError(f"Error parsing response chunk: {str(e)}")

            # Wait for the thread to fully finish to prevent premature cleanup
            request_thread.join(timeout=5)

        except (AuthenticationError, RateLimitError, NetworkError, APIError):
            raise
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")

    @staticmethod
    def _frag_content(frag: dict) -> list[Dict[str, Any]]:
        """Extract content events from a single fragment dict."""
        content = frag.get('content', '')
        if not content:
            return []
        ct = 'thinking' if frag.get('type', 'RESPONSE') == 'THINK' else 'text'
        return [{'content': content, 'type': ct, 'finish_reason': None}]

    @staticmethod
    def _msg_id_event(message_id: str) -> Dict[str, Any]:
        return {'content': '', 'type': 'message_id', 'message_id': message_id, 'finish_reason': None}

    def _parse_chunk(self, chunk: bytes, ctx: Dict[str, str]) -> list[Dict[str, Any]]:
        """Parse a SSE chunk from the API response.

        Returns a list of event dicts (may be empty, one, or multiple).

        The API uses a fragments-based incremental patch format:
        - Initial dict chunk contains fragments list with first fragment (THINK or RESPONSE)
        - Content chunks use path 'response/fragments/-1/content' (-1 = last fragment)
        - When p is empty, the chunk inherits the previous path (incremental append)
        - When thinking finishes, a new RESPONSE fragment is appended to 'response/fragments'
        - We track _current_fragment_type and _last_content_path for proper routing
        """
        if not chunk:
            return []

        try:
            if not chunk.startswith(b'data: '):
                return []

            data = json.loads(chunk[6:])

            # Check for top-level response_message_id (first chunk of response)
            if 'response_message_id' in data:
                return [self._msg_id_event(data['response_message_id'])]

            if 'v' in data:
                path = data.get('p', '')
                operation = data.get('o', '')

                # Handle dict values (initial response object with fragments)
                if isinstance(data['v'], dict):
                    response_obj = data['v'].get('response', {})
                    if 'message_id' not in response_obj:
                        return []
                    results = [self._msg_id_event(response_obj['message_id'])]
                    fragments = response_obj.get('fragments', [])
                    if fragments:
                        ctx['fragment_type'] = fragments[-1].get('type', 'RESPONSE')
                    for frag in fragments:
                        results.extend(self._frag_content(frag))
                    return results

                # Handle list values (new fragment appended - signals THINK→RESPONSE transition)
                if isinstance(data['v'], list):
                    if path != 'response/fragments' or operation != 'APPEND':
                        return []
                    new_fragments = data['v']
                    if new_fragments and isinstance(new_fragments[-1], dict):
                        ctx['fragment_type'] = new_fragments[-1].get('type', 'RESPONSE')
                    results = []
                    for frag in new_fragments:
                        if isinstance(frag, dict):
                            results.extend(self._frag_content(frag))
                    return results

                # Handle string values as content
                if isinstance(data['v'], str):
                    # Track content path: when p is set, update _last_content_path;
                    # when p is empty, inherit the last seen content path
                    if path:
                        ctx['last_content_path'] = path
                    else:
                        path = ctx['last_content_path']

                    # Content via fragments path
                    if path == 'response/fragments/-1/content':
                        content_type = 'thinking' if ctx['fragment_type'] == 'THINK' else 'text'
                        return [{'content': data['v'], 'type': content_type, 'finish_reason': data.get('finish_reason')}]

                    # Legacy paths
                    if path == 'response/content':
                        return [{'content': data['v'], 'type': 'text', 'finish_reason': data.get('finish_reason')}]
                    if path == 'response/thinking_content':
                        return [{'content': data['v'], 'type': 'thinking', 'finish_reason': None}]

                    # Filter out metadata chunks (token usage, status, elapsed_secs, etc.)
                    return []

            # Check for message_id at top level
            if 'message_id' in data:
                return [self._msg_id_event(data['message_id'])]

            if 'choices' in data and data['choices']:
                choice = data['choices'][0]
                if 'delta' in choice:
                    delta = choice['delta']
                    return [{'content': delta.get('content', ''), 'type': delta.get('type', ''), 'finish_reason': choice.get('finish_reason')}]
        except json.JSONDecodeError:
            raise APIError("Invalid JSON in response chunk")
        except Exception as e:
            raise APIError(f"Error parsing chunk: {str(e)}")

        return []
