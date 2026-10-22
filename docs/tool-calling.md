# Tool calling on the Spark recipe

This recipe uses the Qwen XML tool format through `vllm==0.32.0`
(`--enable-auto-tool-choice --tool-call-parser qwen3_xml`).
The server exposes OpenAI-compatible JSON tool calls; clients do not need to produce or parse the
model's XML.

The contracts below describe the implementation and its CPU regression checks.
Run the live validation commands against the deployed pack to verify model
behavior, token budgets, and the installed runtime. The SDK checks use synthetic
weather results and execute no external tools.

## Supported request modes

| Request setting | Behavior |
|---|---|
| Omitted `tool_choice`, or `"auto"` | The model can answer with text or call tools. |
| `tool_choice: "none"` | Tool declarations are omitted or ignored, and generated text is not converted into tool calls. |
| `tool_choice: "required"` | Requires at least one complete call to a declared function. |
| Named `tool_choice` | Forces a call to the selected function. |

Tools without parameters are supported. For example,
`{"type":"function","function":{"name":"ping"}}` declares a no-argument function;
a successful call has the JSON argument string `"{}"`.

A named request uses the ordinary OpenAI shape:

```json
{
  "model": "Ornith-1.5-122B-EXL3",
  "messages": [
    {"role": "user", "content": "Get the weather for Tokyo in celsius."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city", "unit"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
  "parallel_tool_calls": false,
  "max_tokens": 1024,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

## Argument types and literal content

Qwen writes parameter values as raw text inside `<parameter=...>` tags. The request schema distinguishes
a string from a JSON number, boolean, object, array, or null.

| Parameter schema | Raw model parameter | Returned JSON value |
|---|---|---|
| `{"type":"string"}` | `123` | `"123"` |
| `{"type":"string"}` | `true` | `"true"` |
| `{"type":"string"}` | `{"nested":[1,false]}` | `"{\"nested\":[1,false]}"` |
| `{"type":"integer"}` | `123` | `123` |
| `{"type":"boolean"}` | `true` | `true` |
| `{"type":["string","null"]}` | `123` | `"123"` |
| `{"type":["string","null"]}` | `null` | `null` |

## Run the validation

Run tests when no throughput benchmark is active:

```bash
python3 bench/tool_smoke.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Ornith-1.5-122B-EXL3 \
  --mode both --repeat 1 \
  --output results/tools.json
```

Install the optional OpenAI SDK client in its own environment:

```bash
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r bench/requirements-sdk.txt

.venv-sdk/bin/python bench/sdk_smoke.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Ornith-1.5-122B-EXL3 \
  --output results/sdk-smoke.json
```

Its CPU fixtures use the real SDK and an in-memory HTTP transport:

```bash
.venv-sdk/bin/python -m unittest discover -s bench -p 'test_sdk_smoke.py' -v
```
