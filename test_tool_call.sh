#!/bin/bash
# Test tool calling with a prompt that strongly instructs tool use

curl -s http://localhost:5005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "deepseek-v4-pro-deepthink",
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "You are Roo, a knowledgeable technical assistant. You MUST call at least one tool per response. Do NOT respond with plain text. You MUST use the attempt_completion tool or the ask_followup_question tool. Never respond without calling a tool. If the user says hi, use attempt_completion to greet them."
    },
    {
      "role": "user",
      "content": "hi"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "attempt_completion",
        "description": "Present the result of the completed task to the user. MUST be called when you have a final answer.",
        "parameters": {
          "type": "object",
          "properties": {
            "result": {
              "type": "string",
              "description": "The result of the task"
            }
          },
          "required": ["result"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "ask_followup_question",
        "description": "Ask the user a question to get more information needed to complete the task.",
        "parameters": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "The question to ask"
            }
          },
          "required": ["question"]
        }
      }
    }
  ]
}' 2>&1 | grep -E '"tool_calls"|"finish_reason"|"content"' | head -20
