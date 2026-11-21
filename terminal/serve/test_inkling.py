"""serve/test_inkling.py - the server's side of Inkling 2 (thinkingmachines/Inkling-2-Mini, engine/src/inkling): its chat template
rendered as transformers renders it, its effort words, its reply parsed into reasoning, answer and tool calls. No model, no engine.

    python -m unittest serve.test_inkling

The template is the model's own (serve/fixtures/inkling2_chat_template.jinja, from the GGUF's tokenizer.chat_template;
Apache-2.0, Thinking Machines Lab). The replies are the token texts the model writes (seen from UD-IQ1_S through
llama.cpp and strata-inkling).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from serve import frontend
from serve.frontend import ChatTemplate, InklingOutputParser, effort_kwargs, parser_for_template, use_family

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "inkling2_chat_template.jinja"
TOOLS = [{"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


def run(parser, deltas):
    events = []
    for d in deltas:
        events += parser.feed(d)
    events += parser.finish("stop")
    text = {"reasoning": "", "content": ""}
    calls = []
    for e in events:
        if e.kind in text:
            text[e.kind] += e.text
        elif e.kind == "tool_call" and e.call is not None:
            calls.append(e.call)
    return text, calls


def tokens(s: str) -> list[str]:
    """The reply split as an engine streams it: every marker alone, the text a few characters at a time."""
    out, i = [], 0
    while i < len(s):
        if s.startswith("<|", i):
            j = s.index("|>", i) + 2
            out.append(s[i:j])
            i = j
        else:
            j = s.find("<|", i)
            j = len(s) if j < 0 else j
            out += [s[k:min(j, k + 3)] for k in range(i, j, 3)]
            i = j
    return out


class Template(unittest.TestCase):
    def setUp(self):
        self.t = ChatTemplate(FIXTURE)

    def test_it_is_known_as_inkling_and_renders_the_generation_tag(self):
        self.assertTrue(self.t.inkling)
        self.assertIs(parser_for_template(self.t.source), InklingOutputParser)
        out = self.t.render([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                             {"role": "user", "content": "How are you?"}])
        self.assertIn("<|message_model|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>", out)
        self.assertNotIn("generation", out)

    def test_the_prompt_ends_after_the_last_turn_and_the_effort(self):
        out = self.t.render([{"role": "user", "content": "What is 17*3?"}])
        self.assertTrue(out.startswith("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"))
        self.assertTrue(out.endswith("<|message_user|><|content_text|>What is 17*3?<|end_message|>"))

    def test_thinking_off_is_effort_none(self):
        out = self.t.render([{"role": "user", "content": "x"}], enable_thinking=False)
        self.assertIn("Thinking effort level: 0<|end_message|>", out)
        out = self.t.render([{"role": "user", "content": "x"}], reasoning_effort="low")
        self.assertIn("Thinking effort level: 0.2<|end_message|>", out)

    def test_tool_calls_and_results_render(self):
        msgs = [{"role": "user", "content": "Weather in Oslo?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "get_weather",
                                                                                            "arguments": {"city": "Oslo"}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "-3 C, light snow"}]
        out = self.t.render(msgs, tools=TOOLS)
        self.assertIn('<|message_system|>tool_declare<|content_xml|>', out)
        self.assertIn('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Oslo"}}'
                      '<|end_message|>', out)
        self.assertIn("<|message_tool|>get_weather<|content_text|>-3 C, light snow<|end_message|>", out)

    def test_caps(self):
        caps = self.t.caps
        self.assertTrue(caps["supports_tools"])
        self.assertTrue(caps["supports_tool_calls"])
        self.assertTrue(caps["supports_parallel_tool_calls"])
        self.assertTrue(caps["supports_system_role"])
        self.assertTrue(caps["supports_preserve_reasoning"])


class Effort(unittest.TestCase):
    def tearDown(self):
        use_family(None)

    def test_words_go_straight_to_the_template(self):
        use_family("inkling")
        self.assertEqual(effort_kwargs("none"), {"reasoning_effort": "none"})
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(effort_kwargs("max"), {"reasoning_effort": "max"})
        self.assertEqual(frontend.budget_effort(1000), {"reasoning_effort": "low"})
        use_family(None)
        self.assertEqual(effort_kwargs("high"), {"reasoning_effort": "xhigh"})   # the Qwen table is untouched


class Replies(unittest.TestCase):
    def test_thinking_then_the_answer(self):
        reply = ("<|message_model|><|content_thinking|>Compute: 17*3 = 51.<|end_message|>"
                 "<|message_model|><|content_text|>17 × 3 = 51<|end_message|><|content_model_end_sampling|>")
        text, calls = run(InklingOutputParser(), tokens(reply))
        self.assertEqual(text["reasoning"].strip(), "Compute: 17*3 = 51.")
        self.assertEqual(text["content"].strip(), "17 × 3 = 51")
        self.assertEqual(calls, [])

    def test_an_answer_without_thinking(self):
        reply = "<|message_model|><|content_text|>Hello there.<|end_message|><|content_model_end_sampling|>"
        text, _ = run(InklingOutputParser(), tokens(reply))
        self.assertEqual(text["reasoning"].strip(), "")
        self.assertEqual(text["content"].strip(), "Hello there.")

    def test_a_tool_call(self):
        reply = ("<|message_model|><|content_thinking|>Need the tool.<|end_message|>"
                 '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Oslo"}}'
                 "<|end_message|><|content_model_end_sampling|>")
        text, calls = run(InklingOutputParser(tools=TOOLS), tokens(reply))
        self.assertEqual(text["reasoning"].strip(), "Need the tool.")
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Oslo"})])
        self.assertEqual(text["content"].strip(), "")
        self.assertNotIn("get_weather", text["content"])     # the call's name header is not text

    def test_two_calls(self):
        reply = ('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Oslo"}}<|end_message|>'
                 '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Bergen"}}<|end_message|>'
                 "<|content_model_end_sampling|>")
        _, calls = run(InklingOutputParser(tools=TOOLS), tokens(reply))
        self.assertEqual([c.arguments["city"] for c in calls], ["Oslo", "Bergen"])

    def test_markers_split_across_deltas(self):
        reply = ("<|message_model|><|content_thinking|>abc<|end_message|><|message_model|><|content_text|>"
                 "a <| b<|end_message|><|content_model_end_sampling|>")
        deltas = [reply[i:i + 4] for i in range(0, len(reply), 4)]
        text, _ = run(InklingOutputParser(), deltas)
        self.assertEqual(text["reasoning"].strip(), "abc")
        self.assertEqual(text["content"].strip(), "a <| b")   # a "<|" that is no marker stays text

    def test_a_reply_cut_off_while_thinking(self):
        text, _ = run(InklingOutputParser(), tokens("<|message_model|><|content_thinking|>still think"))
        self.assertEqual(text["reasoning"].strip(), "still think")
        self.assertEqual(text["content"], "")

    def test_the_call_json_is_converted(self):
        p = InklingOutputParser(tools=TOOLS)
        p._ink_call = '{"name":"get_weather","args":{"city":"Oslo"}}'
        body = p._call_text()
        self.assertEqual(json.loads(body.split("\n")[1]), {"name": "get_weather", "arguments": {"city": "Oslo"}})


if __name__ == "__main__":
    unittest.main()
