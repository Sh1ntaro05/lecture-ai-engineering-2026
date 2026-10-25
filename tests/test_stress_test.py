from types import SimpleNamespace

from google.genai import errors

from stress_test import MAX_REQUESTS, run_stress_test


def test_stops_on_429_and_reports_attempt(monkeypatch, capsys):
    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise errors.APIError(
                    code=429,
                    response_json={"error": {"message": "rate limited"}},
                )
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=20,
                )
            )

    models = FakeModels()
    sleeps = []
    monkeypatch.setattr("stress_test.time.sleep", sleeps.append)

    run_stress_test(SimpleNamespace(models=models), ["model-a", "model-b"])

    output = capsys.readouterr().out
    assert models.calls == 3
    assert sleeps == [1.0, 1.0]
    assert "model_attempt=3" in output
    assert "total_attempt=3" in output
    assert "[model] model-b" not in output


def test_never_exceeds_global_request_cap(monkeypatch, capsys):
    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=20,
                )
            )

    models = FakeModels()
    monkeypatch.setattr("stress_test.time.sleep", lambda _: None)

    run_stress_test(SimpleNamespace(models=models), ["model-a", "model-b"])

    output = capsys.readouterr().out
    assert models.calls == MAX_REQUESTS
    assert "[done] 429なし。送信リクエスト数: 15/15" in output
    assert "[model] model-b" not in output
