from types import SimpleNamespace

from google.genai import errors

from app import ask as ask_module


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    class FakeChat:
        def send_message(self, question):
            return SimpleNamespace(
                text="answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
            )

    answer = ask_module.ask("question", FakeChat(), ask_module.RequestThrottle(0), 0)

    assert answer == "answer"
    output = capsys.readouterr().out
    assert "latency=" in output
    assert "input_tokens=12 output_tokens=7" in output


def test_main_reuses_chat_for_followup_questions(monkeypatch, capsys):
    questions = []
    creations = []

    class FakeChat:
        def __init__(self, history):
            self.history = list(history)

        def send_message(self, question):
            questions.append(question)
            self.history.extend(
                [SimpleNamespace(role="user"), SimpleNamespace(role="model")]
            )
            return SimpleNamespace(
                text=f"answer to {question}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
            )

        def get_history(self):
            return self.history

    class FakeChats:
        def create(self, *, model, history=None):
            creations.append((model, list(history or [])))
            return FakeChat(history or [])

    class FakeClient:
        def __init__(self, *, api_key):
            self.chats = FakeChats()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "sys.argv",
        ["app.ask", "First question", "--request-interval", "0", "--max-history-turns", "1"],
    )
    inputs = iter(["Follow-up question", "/model gemini-2.5-flash", "One more", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask_module.main()

    assert questions == ["First question", "Follow-up question", "One more"]
    assert [model for model, _ in creations] == [
        ask_module.MODEL,
        ask_module.MODEL,
        "gemini-2.5-flash",
    ]
    assert [len(history) for _, history in creations] == [0, 2, 2]
    output = capsys.readouterr().out
    assert output.count("[metrics]") == 3
    assert "[model] gemini-2.5-flash" in output


def test_send_with_retry_uses_exponential_backoff(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.calls = 0

        def send_message(self, question):
            self.calls += 1
            if self.calls == 1:
                raise errors.APIError(
                    code=429,
                    response_json={"error": {"message": "rate limited"}},
                )
            return SimpleNamespace(text="answer")

    sleeps = []
    monkeypatch.setattr(ask_module.time, "sleep", sleeps.append)
    chat = FakeChat()

    response, latency = ask_module.send_with_retry(
        chat, "question", ask_module.RequestThrottle(0), max_retries=1
    )

    assert response.text == "answer"
    assert chat.calls == 2
    assert sleeps == [1]
    assert latency >= 0
    assert "[retry] API error 429" in capsys.readouterr().out


def test_request_throttle_waits_between_requests(monkeypatch):
    times = iter([0.0, 0.25, 1.0])
    sleeps = []
    monkeypatch.setattr(ask_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(ask_module.time, "sleep", sleeps.append)
    throttle = ask_module.RequestThrottle(1)

    throttle.wait()
    throttle.wait()

    assert sleeps == [0.75]


def test_trim_history_keeps_only_requested_turns():
    history = list(range(8))

    assert ask_module.trim_history(history, 2) == [4, 5, 6, 7]