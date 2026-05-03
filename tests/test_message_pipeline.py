"""Tests that user messages are properly sent through the server pipeline.

Covers:
  - _extract_content: string / list / None content handling
  - _extract_tool_aware_prompt: user msg, tool results, fallback extraction
  - _build_key: session key generation (first user message only)
  - _format_message: DeepSeek native tag formatting
  - /v1/chat/completions endpoint: prompt actually reaches api.chat_completion()
"""

import json
import os
from unittest.mock import patch, MagicMock
import pytest

from fastapi.testclient import TestClient

# ── Import functions under test ──────────────────────────────────────────
from dsk.server import (
    app,
    _extract_content,
    _extract_tool_aware_prompt,
    _build_key,
    _format_message,
    _build_tool_system_prompt,
    ChatMessage,
    ChatCompletionRequest,
    ToolContext,
    ToolDefinition,
    FunctionDefinition,
    session_mgr,
    _TAG_SYSTEM_START,
    _TAG_SYSTEM_END,
    _TAG_USER_START,
    _TAG_ASSISTANT_START,
)


# ── _extract_content ────────────────────────────────────────────────────

class TestExtractContent:
    def test_string_content(self):
        assert _extract_content("hello world") == "hello world"

    def test_none_content(self):
        assert _extract_content(None) == ""

    def test_list_content_with_text_parts(self):
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert _extract_content(content) == "hello  world"

    def test_list_content_with_non_dict_parts(self):
        assert _extract_content(["abc", "def"]) == "abc def"

    def test_empty_string(self):
        assert _extract_content("") == ""


# ── _build_key ──────────────────────────────────────────────────────────

class TestBuildKey:
    def _make_msgs(self, *pairs):
        """Helper: accepts (role, content) tuples → list of ChatMessage."""
        return [ChatMessage(role=r, content=c) for r, c in pairs]

    def test_system_only(self):
        msgs = self._make_msgs(("system", "you are helpful"))
        key = _build_key(msgs)
        assert key  # returns a hash even with no user messages

    def test_single_user(self):
        msgs = self._make_msgs(("system", "sys"), ("user", "hi"))
        key = _build_key(msgs)
        assert key
        # Same first user message → same key
        msgs2 = self._make_msgs(("system", "different sys"), ("user", "hi"))
        assert _build_key(msgs2) == key  # system message ignored

    def test_only_first_user_matters(self):
        msgs1 = self._make_msgs(
            ("system", "sys"), ("user", "first"), ("user", "second")
        )
        msgs2 = self._make_msgs(
            ("system", "sys"), ("user", "first"), ("user", "different")
        )
        # Only the first user message is hashed
        assert _build_key(msgs1) == _build_key(msgs2)

    def test_different_first_user(self):
        msgs1 = self._make_msgs(("user", "hello"))
        msgs2 = self._make_msgs(("user", "world"))
        assert _build_key(msgs1) != _build_key(msgs2)

    def test_deterministic(self):
        msgs = self._make_msgs(("system", "sys"), ("user", "hello"))
        assert _build_key(msgs) == _build_key(msgs)

    def test_list_content(self):
        msgs = self._make_msgs(("user", [{"type": "text", "text": "hello"}]))
        key1 = _build_key(msgs)
        msgs2 = self._make_msgs(("user", "hello"))
        key2 = _build_key(msgs2)
        assert key1 == key2  # _extract_content normalizes list to string


# ── _format_message ─────────────────────────────────────────────────────

class TestFormatMessage:
    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful")
        result = _format_message(msg)
        assert result.startswith(_TAG_SYSTEM_START)
        assert "You are helpful" in result
        assert _TAG_SYSTEM_END in result

    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello there")
        result = _format_message(msg)
        assert result.startswith(_TAG_USER_START)
        assert result.endswith("Hello there")

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi!")
        result = _format_message(msg)
        assert result.startswith(_TAG_ASSISTANT_START)
        assert "Hi!" in result

    def test_user_message_with_list_content(self):
        msg = ChatMessage(
            role="user",
            content=[{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
        )
        result = _format_message(msg)
        assert "part1" in result
        assert "part2" in result


# ── _extract_tool_aware_prompt ──────────────────────────────────────────

class TestExtractToolAwarePrompt:
    def test_empty_messages(self):
        assert _extract_tool_aware_prompt([]) == ""

    def test_last_user_message(self):
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hello"),
        ]
        assert _extract_tool_aware_prompt(msgs) == "hello"

    def test_trailing_tool_results(self):
        msgs = [
            ChatMessage(role="user", content="read file"),
            ChatMessage(role="assistant", content=None),
            ChatMessage(role="tool", content="file contents here"),
        ]
        result = _extract_tool_aware_prompt(msgs)
        assert "file contents here" in result
        assert "<｜tool_output_begin｜>" in result

    def test_multiple_trailing_tool_results(self):
        msgs = [
            ChatMessage(role="user", content="read two files"),
            ChatMessage(role="assistant", content=None),
            ChatMessage(role="tool", content="file1"),
            ChatMessage(role="tool", content="file2"),
        ]
        result = _extract_tool_aware_prompt(msgs)
        assert "file1" in result
        assert "file2" in result

    def test_fallback_to_last_user(self):
        msgs = [
            ChatMessage(role="user", content="first question"),
            ChatMessage(role="assistant", content="answer"),
        ]
        # Last msg is assistant → fallback to last user
        result = _extract_tool_aware_prompt(msgs)
        assert result == "first question"


# ── _build_tool_system_prompt ───────────────────────────────────────────

class TestBuildToolSystemPrompt:
    def test_no_tools(self):
        ctx = ToolContext(None, None, None)
        assert _build_tool_system_prompt("base system", ctx) == "base system"

    def test_empty_base_with_tools(self):
        ctx = ToolContext("format", "defs", "instructions")
        result = _build_tool_system_prompt("", ctx)
        assert "Tool Calling" in result
        assert "format" in result

    def test_base_with_tools(self):
        ctx = ToolContext("format", "defs", "instructions")
        result = _build_tool_system_prompt("base system", ctx)
        assert result.startswith("base system")
        assert "Tool Calling" in result


# ── Full endpoint test (mocked API) ────────────────────────────────────

class TestChatCompletionsEndpoint:
    """Verify that the user message actually reaches api.chat_completion()."""

    @pytest.fixture(autouse=True)
    def _clear_sessions(self, tmp_path):
        """Reset session manager between tests."""
        session_mgr._data.clear()

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _mock_api(self, response_chunks):
        """Return a patch that replaces _get_api with a mock.

        response_chunks: list of dicts to be yielded by chat_completion.
        """
        mock_api = MagicMock()
        mock_api.create_chat_session.return_value = "fake-session-id"
        mock_api.chat_completion.return_value = iter(response_chunks)
        return patch("dsk.server._get_api", return_value=mock_api)

    # ── Non-streaming ───────────────────────────────────────────────

    def test_simple_user_message_non_streaming(self, client):
        """A simple string user message should be passed as the prompt."""
        chunks = [
            {"type": "message_id", "message_id": "msg-123"},
            {"type": "text", "content": "Hello!", "finish_reason": None},
            {"type": "text", "content": "", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Hello!"

        # Verify the prompt passed to chat_completion
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        assert call_args is not None
        # prompt is the 2nd positional arg to chat_completion(session_id, prompt, ...)
        prompt_arg = call_args[0][1]
        assert prompt_arg == "What is 2+2?"

    def test_list_content_user_message_non_streaming(self, client):
        """User message with list content should be concatenated into prompt."""
        chunks = [
            {"type": "message_id", "message_id": "msg-456"},
            {"type": "text", "content": "Response", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": "Hello "},
                            {"type": "text", "text": "World"},
                        ]},
                    ],
                },
            )
        assert resp.status_code == 200
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        prompt_arg = call_args[0][1]
        assert prompt_arg == "Hello  World"

    def test_system_prompt_passed_non_streaming(self, client):
        """System message should be passed as system_prompt to chat_completion."""
        chunks = [
            {"type": "message_id", "message_id": "msg-789"},
            {"type": "text", "content": "ok", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "Be concise"},
                        {"role": "user", "content": "hi"},
                    ],
                },
            )
        assert resp.status_code == 200
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        assert call_args[1]["system_prompt"] == "Be concise"

    def test_no_user_message_returns_400(self, client):
        """Request with no user message should return 400."""
        with self._mock_api([]):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "Be helpful"},
                    ],
                },
            )
        assert resp.status_code == 400

    # ── Streaming ───────────────────────────────────────────────────

    def test_simple_user_message_streaming(self, client):
        """Streaming: user message should still be passed as prompt."""
        chunks = [
            {"type": "message_id", "message_id": "msg-stream"},
            {"type": "text", "content": "Hi!", "finish_reason": None},
            {"type": "text", "content": "", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Say hello"},
                    ],
                },
            )
        assert resp.status_code == 200
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        prompt_arg = call_args[0][1]
        assert prompt_arg == "Say hello"

        # Verify SSE stream contains content
        full_text = resp.text
        assert "Hi!" in full_text
        assert "[DONE]" in full_text

    # ── Tool-aware path ─────────────────────────────────────────────

    def test_tool_aware_user_message(self, client):
        """When tools are present, user message is still extracted properly."""
        chunks = [
            {"type": "message_id", "message_id": "msg-tool"},
            {"type": "text", "content": "I used a tool", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "You have tools"},
                        {"role": "user", "content": "Read the file"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                    },
                                    "required": ["path"],
                                },
                            },
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        prompt_arg = call_args[0][1]
        # The prompt should contain the user message
        assert "Read the file" in prompt_arg
        # And tool definitions should be injected
        assert "read_file" in prompt_arg

    def test_tool_aware_tool_results(self, client):
        """When last messages are tool results, they should be formatted properly."""
        chunks = [
            {"type": "message_id", "message_id": "msg-toolres"},
            {"type": "text", "content": "Got it", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks) as mock_get:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "You have tools"},
                        {"role": "user", "content": "Read file"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [{"id": "call_1", "type": "function",
                                         "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}}]},
                        {"role": "tool", "content": "file contents here"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                            },
                        }
                    ],
                },
            )
        assert resp.status_code == 200
        mock_api = mock_get.return_value
        call_args = mock_api.chat_completion.call_args
        prompt_arg = call_args[0][1]
        # Tool results should be in the prompt
        assert "file contents here" in prompt_arg
        assert "<｜tool_output_begin｜>" in prompt_arg

    # ── Conversation continuity ─────────────────────────────────────

    def test_session_reuse_on_followup(self, client):
        """Second request in same conversation should reuse session."""
        chunks1 = [
            {"type": "message_id", "message_id": "msg-first"},
            {"type": "text", "content": "First answer", "finish_reason": "stop"},
        ]
        chunks2 = [
            {"type": "message_id", "message_id": "msg-second"},
            {"type": "text", "content": "Second answer", "finish_reason": "stop"},
        ]

        with self._mock_api(chunks1) as mock_get1:
            resp1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "first question"},
                    ],
                },
            )
        assert resp1.status_code == 200

        # Second request — same system + first user, plus new user message
        with self._mock_api(chunks2) as mock_get2:
            resp2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "follow up question"},
                    ],
                },
            )
        assert resp2.status_code == 200

        # Verify the second call got the follow-up message as prompt
        mock_api2 = mock_get2.return_value
        call_args = mock_api2.chat_completion.call_args
        prompt_arg = call_args[0][1]
        assert prompt_arg == "follow up question"

        # Verify session was reused (create_chat_session called only once)
        mock_api2.create_chat_session.assert_not_called()

    def test_conversation_id_returned_in_response(self, client):
        """First request should return a conversation_id in body and header."""
        chunks = [
            {"type": "message_id", "message_id": "msg-1"},
            {"type": "text", "content": "Hi", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "conversation_id" in body
        assert body["conversation_id"] is not None
        # Header should also be set
        assert "x-conversation-id" in resp.headers

    def test_conversation_id_reuses_session(self, client):
        """Sending conversation_id back should reuse the session even with different messages."""
        chunks1 = [
            {"type": "message_id", "message_id": "msg-1"},
            {"type": "text", "content": "First", "finish_reason": "stop"},
        ]
        chunks2 = [
            {"type": "message_id", "message_id": "msg-2"},
            {"type": "text", "content": "Second", "finish_reason": "stop"},
        ]

        with self._mock_api(chunks1) as mock_get1:
            resp1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        conv_id = resp1.json()["conversation_id"]

        # Second request with same conversation_id but completely different messages
        with self._mock_api(chunks2) as mock_get2:
            resp2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "conversation_id": conv_id,
                    "messages": [{"role": "user", "content": "totally different"}],
                },
            )
        assert resp2.status_code == 200
        # Session should be reused — no new chat session created
        mock_api2 = mock_get2.return_value
        mock_api2.create_chat_session.assert_not_called()

    def test_conversation_id_via_header(self, client):
        """X-Conversation-Id header should also work for session reuse."""
        chunks1 = [
            {"type": "message_id", "message_id": "msg-1"},
            {"type": "text", "content": "First", "finish_reason": "stop"},
        ]
        chunks2 = [
            {"type": "message_id", "message_id": "msg-2"},
            {"type": "text", "content": "Second", "finish_reason": "stop"},
        ]

        with self._mock_api(chunks1) as mock_get1:
            resp1 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        conv_id = resp1.headers["x-conversation-id"]

        # Second request using header instead of body field
        with self._mock_api(chunks2) as mock_get2:
            resp2 = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "stream": False,
                    "messages": [{"role": "user", "content": "different"}],
                },
                headers={"X-Conversation-Id": conv_id},
            )
        assert resp2.status_code == 200
        mock_api2 = mock_get2.return_value
        mock_api2.create_chat_session.assert_not_called()

    # ── Thinking content ─────────────────────────────────────────────

    def test_thinking_content_non_streaming(self, client):
        """Thinking content should be captured in reasoning_content."""
        chunks = [
            {"type": "message_id", "message_id": "msg-think"},
            {"type": "thinking", "content": "Let me think..."},
            {"type": "text", "content": "The answer is 4", "finish_reason": "stop"},
        ]
        with self._mock_api(chunks):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash-deepthink",
                    "stream": False,
                    "messages": [
                        {"role": "user", "content": "2+2?"},
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["reasoning_content"] == "Let me think..."
        assert body["choices"][0]["message"]["content"] == "The answer is 4"
