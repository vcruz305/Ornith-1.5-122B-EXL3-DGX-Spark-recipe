"""CPU tests for Ornith-1.5-122B / Qwen3.5 tool-call rendering and parsing."""
from __future__ import annotations

import json
import unittest


def format_qwen3_5_tool_call(name: str, arguments: dict) -> str:
    parts = [f"<tool_call>\n<function={name}>"]
    for key, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"<parameter={key}>\n{rendered}\n</parameter>")
    parts.append("</function>\n</tool_call>")
    return "\n".join(parts)


class OrnithTemplateRoundtripTests(unittest.TestCase):
    def test_xml_tool_call_envelope_preserves_parameter_blocks(self):
        rendered = format_qwen3_5_tool_call("get_weather", {"city": "Tokyo", "unit": "celsius"})
        self.assertIn("<tool_call>", rendered)
        self.assertIn("<function=get_weather>", rendered)
        self.assertIn("<parameter=city>\nTokyo\n</parameter>", rendered)
        self.assertIn("</function>\n</tool_call>", rendered)


if __name__ == "__main__":
    unittest.main()
