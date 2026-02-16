# Wrangle AI Python Library

The official Python library for the WrangleAI.

This library provides a drop-in replacement for the OpenAI SDK, adding **Smart Routing**, **Cost Tracking**, and **Enterprise Governance** capabilities. It allows you to automatically route prompts to the most cost-effective and capable model (GPT-5, Gemini 2.5 Mini, Mistral, etc.) without changing your code logic.

[![PyPI version](https://img.shields.io/pypi/v/wrangleai.svg)](https://pypi.org/project/wrangleai/)
[![License](https://img.shields.io/pypi/l/wrangleai.svg)](https://pypi.org/project/wrangleai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wrangleai.svg)](https://pypi.org/project/wrangleai/)

---

## What's New in v0.3.0

 **Native Async Support** - Full asyncio support with `AsyncWrangleAI` client  
 **Type Safety** - Pydantic models for all API responses with IDE autocomplete  
 **Smart Error Handling** - Specific exception types for different error scenarios  
 **Python 3.13 Compatible** - Improved streaming stability

See [Migration Guide](#migration-guide) for upgrading from v0.2.x.

---

## Installation

```bash
pip install wrangleai
```

## Authentication

The library needs your API key to communicate with the server. You can pass it explicitly or define it in your environment variables.

**Option 1: Environment Variable (Recommended)**
```bash
export WRANGLE_API_KEY="sk-..."
```

**Option 2: Explicit Initialization**
```python
from wrangleai import WrangleAI

client = WrangleAI(api_key="sk-...")
```

---

## Async Support (New in v0.3.0)

For asyncio applications (FastAPI, WebSocket servers, LiveKit agents), use the `AsyncWrangleAI` client:

```python
from wrangleai import AsyncWrangleAI

async def main():
    async with AsyncWrangleAI() as client:
        # Non-streaming
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
        
        # Streaming
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Count to 5"}],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")

# Run with asyncio
import asyncio
asyncio.run(main())
```

**Key Benefits:**
- Native asyncio support (no `asyncio.to_thread` wrappers needed)
- Async context manager support (`async with`)
- Async generators for streaming (`async for`)
- Same API surface as sync client

---

## Chat Completions

### 1. Smart Routing (`model="auto"`)

The unique feature of Wrangle AI is the **Auto Router**. Instead of hardcoding a model, set `model="auto"`. WrangleAI analyzes your prompt's complexity and routes it to the optimal model (e.g., routing simple queries to `gpt-4o-mini` and complex coding tasks to `gpt-5` or `gemini-2.5-pro`).

```python
from wrangleai import WrangleAI

client = WrangleAI()

completion = client.chat.completions.create(
    model="auto",  # <--- Let WrangleAI decide
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
)

# Standard OpenAI-compatible response structure
print(completion.choices[0].message.content)
```

### 2. Standard Models

You can still request specific models if you require deterministic provider behavior.

```python
completion = client.chat.completions.create(
    model="gpt-4o", # or 'gemini-2.5-pro', 'gpt-5-mini'
    messages=[{"role": "user", "content": "Hello world!"}]
)
```

### 3. Streaming Responses

Full support for Server-Sent Events (SSE) via standard Python generators.

```python
stream = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about servers."}],
    stream=True
)

print("Streaming: ", end="")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()
```

---

### 4. Web Search (Grounding)

Wrangle AI supports live web access. When using the `web_search` tool, the server returns a specialized response format.

```
completion = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Compare Apple and Google stock prices."}],
    tools=[{
        "type": "web_search",
        "web_search": {"external_web_access": True}
    }]
)

# 1. Check for Standard Chat Response
if completion.choices:
    print(completion.choices[0].message.content)

# 2. Check for Grounded Response (Web Search Results)
elif completion.output:
    # Iterate through output items to find the message
    for item in completion.output:
        if item.type == 'message':
            for content in item.content:
                if content.type == 'output_text':
                    print(f"Response: {content.text}\n")
                    
                    # Access Citations safely (check if they exist)
                    if content.annotations:
                        print("--- Sources ---")
                        for cite in content.annotations:
                            print(f"• {cite.title} ({cite.url})")
```


### 5. Function Calling (Tools)

You can define custom functions for the model to call. This works seamlessly with Smart Routing.

```python
completion = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }]
)

choice = completion.choices[0]

# Check if the model wants to call a tool
if choice.finish_reason == "tool_calls":
    tool_call = choice.message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

---


### 6. Efficiency-First Routing (SLM) [BETA]

Use the **Efficiency Tier** to route tasks to specialized Small Language Models (SLMs) for maximum speed and cost savings. This tier is ideal for high-volume tasks like coding snippets, summarization, and data extraction where premium reasoning is not required.

### Basic SLM Request

By setting `useSlm: True`, WrangleAI will automatically select the best cost-effective model (e.g., `mistral-nemo`, `llama-3.1-8b`) based on prompt complexity.

```python
completion = client.chat.completions.create(
    model="auto",
    slm={
        "useSlm": True,
        "useCase": "coding" # Optional: 'chat', 'tool_use', 'reasoning', 'summarization', etc.
    },
    messages=[{"role": "user", "content": "Write a Python function to parse CSV files."}]
)
```

### Auto-Scaling for Tools

The Efficiency Tier is tool-aware. If you set `tool_use` within `useCase` in your request, the WrangleAI router will pivot to high-capability SLMs (such as **Qwen 2.5 72B** or **DeepSeek R1**) to ensure strict JSON schema adherence and reliable function calling.

This gives you the best of both worlds: low cost for standard text generation, and high reliability for agentic workflows.

```python
completion = client.chat.completions.create(
    model="auto",
    slm={"useSlm": True},
    messages=[{"role": "user", "content": "Get the stock price for NVDA."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "parameters": { "type": "object", "properties": { "symbol": {"type": "string"} } }
        }
    }]
)
```

### Supported Use Cases

Providing a `useCase` helps the router select a specialist model.

| Use Case | Description |
| :--- | :--- |
| `coding` | Optimized for Python, JS, SQL, and debugging. |
| `reasoning` | Tuned for math, logic puzzles, and multi-step deduction. |
| `chat` | Optimized for natural conversation flow and "human" vibes. |
| `summarization` | High-context window models for condensing text. |
| `classification` | Fast, low-latency models for tagging and labeling. |
| `creative_writing` | Tuned for storytelling and reduced refusal rates. |
| `tool_use` | Setup for high-reliability function calling capabilities |
| `other` | (Default) General-purpose instruction following. |

> **Pro Tip:** For complex tool-use scenarios with SLMs, we recommend adding a system prompt instructing the model to "Always use the provided tool if applicable" to overcome potential passivity in smaller models.


## Management API

Programmatically monitor your token usage, costs, and key status.

### Usage Statistics
Get aggregated usage data. You can optionally filter by date range.

```python
# Get all-time stats
usage = client.usage.retrieve()

# Get stats for a specific date range
# usage = client.usage.retrieve(start_date="2023-12-01", end_date="2023-12-31")

print(f"Total Requests: {usage.total_requests}")
print(f"Total Tokens:   {usage.total_tokens}")
print(f"Optimized:      {usage.optimized}") # True if you are using 'auto' models

print("\n--- Breakdown by Model ---")
for model_stat in usage.usage_by_model:
    print(f"{model_stat.model}: {model_stat.requests} requests (${model_stat.total_cost})")
```

### Cost Tracking
Get the total accrued cost for the API Key.

```python
cost = client.cost.retrieve()
print(f"Total Spend: ${cost.total_cost}")
```

### API Key Verification
The SDK provides specific exception types for different error scenarios:

```python
from wrangleai import WrangleAI
from wrangleai.exceptions import (
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    APIError,
    APIConnectionError
)

client = WrangleAI()

try:
    response = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
    # Handle auth errors (check API key)
except RateLimitError as e:
    print(f"Rate limit hit: {e}")
    # Implement retry with backoff
except BadRequestError as e:
    print(f"Invalid request: {e}")
    # Fix request parameters
except APIError as e:
    print(f"Server error: {e}")
    # Retry or alert monitoring
except APIConnectionError as e:
    print(f"Network error: {e}")
    # Check internet connection
```

**Exception Hierarchy:**
- `WrangleError` (base exception)
  - `AuthenticationError` (401, 403)
  - `RateLimitError` (429)
  - `BadRequestError` (400)
  - `APIError` (500+)
  - `APIConnectionError` (network/timeout issues)

---

## Type Safety (New in v0.3.0)

All API responses use Pydantic models for type safety and IDE autocomplete:

```python
from wrangleai import WrangleAI
from wrangleai.types import ChatCompletion, ChatCompletionChunk

client = WrangleAI()

# Non-streaming returns ChatCompletion
response: ChatCompletion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi"}]
)

# Access with full type hints
content: str | None = response.choices[0].message.content
tokens: int = response.usage.total_tokens

# Streaming returns Generator[ChatCompletionChunk]
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi"}],
    stream=True
)

for chunk in stream:
    # IDE provides autocomplete for .delta, .choices, etc.
    delta_content: str | None = chunk.choices[0].delta.content
```

**Available Models:**
- `ChatCompletion` - Non-streaming response
- `ChatCompletionChunk` - Streaming chunk
- `Choice` - Individual completion choice
- `Message` - Complete message with role/content
- `Delta` - Incremental content update
- `ToolCall` - Function call details
- `Usage` - Token usage statistics

### Legacy Response Mode

If you need the old `WrangleObject` behavior (v0.2.x), use `legacy_response=True`:

```python
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hi"}],
    legacy_response=True  # Returns WrangleObject instead of ChatCompletion
)
```

---

## Migration Guide

### Upgrading from v0.2.x to v0.3.0

**Breaking Changes: None** (fully backward compatible)

**Recommended Updates:**

1. **Install new dependencies:**
   ```bash
   pip install --upgrade wrangleai
   ```

2. **Use async client for asyncio apps:**
   ```python
   # Old (v0.2.x) - required manual threading
   import asyncio
   from wrangleai import WrangleAI
   
   client = WrangleAI()
   stream = await asyncio.to_thread(
       client.chat.completions.create,
       model="auto",
       messages=[...],
       stream=True
   )
   
   # New (v0.3.0) - native async
   from wrangleai import AsyncWrangleAI
   
   async with AsyncWrangleAI() as client:
       stream = await client.chat.completions.create(
           model="auto",
           messages=[...],
           stream=True
       )
       async for chunk in stream:
           print(chunk.choices[0].delta.content)
   ```

3. **Add type hints (optional but recommended):**
   ```python
   from wrangleai.types import ChatCompletion
   
   response: ChatCompletion = client.chat.completions.create(...)
   ```

4. **Update error handling (optional):**
   ```python
   # Old
   try:
       response = client.chat.completions.create(...)
   except Exception as e:
       print(f"Error: {e}")
   
   # New - specific exceptions
   from wrangleai.exceptions import RateLimitError
   
   try:
       response = client.chat.completions.create(...)
   except RateLimitError:
       # Implement retry logic
       pass
   ```

---

## Requirements

*   `Python 3.8+`
*   `httpx>=0.27.0`
*   `pydantic>=2.0.0`

---

## Error Handling

Check if your current key is valid and active.

```python
key_info = client.keys.verify()

if key_info.valid:
    print(f"Status: {key_info.keyStatus}") # e.g., 'ACTIVE'
    print(f"Key ID: {key_info.apiKeyId}")
else:
    print("Invalid Key")
```

---

## Configuration

### Timeouts
The default timeout is 60 seconds. You can adjust this globally.

```python
client = WrangleAI(timeout=120.0) # 2 minutes
```

## Check Version

To verify which version of the library you are installed:

**Python:**
```python
import wrangleai
print(wrangleai.__version__)
```

**Command Line:**
```bash
pip show wrangleai
```

## Error Handling

Errors are raised as standard exceptions. The client attempts to parse the Server's error message for clarity.

```python
try:
    client.chat.completions.create(model="auto", messages=[...])
except Exception as e:
    print(f"An error occurred: {e}")
```

## Requirements

*   `Python 3.8+`
*   `httpx`

## License

`MIT`
