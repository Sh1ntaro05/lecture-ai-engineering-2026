"""Gemini APIを使う、会話履歴付きの対話型チャットボット。"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
DEFAULT_REQUEST_INTERVAL = 4.0
DEFAULT_MAX_HISTORY_TURNS = 10
MAX_BACKOFF_SECONDS = 30


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0以上の整数を指定してください")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0以上の秒数を指定してください")
    return number


def trim_history(history, max_turns: int):
    """履歴の末尾から指定ターン分だけを保持する。"""
    return history[-2 * max_turns :] if history else []


class RequestThrottle:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_request_at = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            remaining = self.min_interval - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


def is_retryable(error: errors.APIError) -> bool:
    return error.code in (408, 429) or 500 <= error.code < 600


def send_with_retry(chat, question: str, throttle: RequestThrottle, max_retries: int):
    request_started_at = None
    for attempt in range(max_retries + 1):
        throttle.wait()
        if request_started_at is None:
            request_started_at = time.perf_counter()
        try:
            response = chat.send_message(question)
        except errors.APIError as error:
            if not is_retryable(error) or attempt == max_retries:
                raise
            delay = min(2**attempt, MAX_BACKOFF_SECONDS)
            print(f"[retry] API error {error.code}; {delay}秒後に再試行します")
            time.sleep(delay)
        else:
            return response, time.perf_counter() - request_started_at

    raise RuntimeError("再試行ループが予期せず終了しました")


def ask(question: str, chat, throttle: RequestThrottle, max_retries: int) -> str:
    response, latency = send_with_retry(chat, question, throttle, max_retries)
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    input_tokens = input_tokens if input_tokens is not None else "unknown"
    output_tokens = output_tokens if output_tokens is not None else "unknown"
    print(
        f"[metrics] latency={latency:.3f}s "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini APIに質問する対話型チャットボット")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="最初の質問")
    parser.add_argument("--model", default=MODEL, help="最初に使うGeminiモデル（既定: %(default)s）")
    parser.add_argument(
        "--request-interval",
        type=non_negative_float,
        default=DEFAULT_REQUEST_INTERVAL,
        metavar="SECONDS",
        help="APIリクエスト間の最小秒数（既定: %(default)s）",
    )
    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=5,
        help="一時的なAPIエラーの最大再試行回数（既定: %(default)s）",
    )
    parser.add_argument(
        "--max-history-turns",
        type=positive_int,
        default=DEFAULT_MAX_HISTORY_TURNS,
        help="モデルに渡す会話履歴の最大ターン数（既定: %(default)s）",
    )
    args = parser.parse_args()

    print(f"[model] {args.model}")
    client = genai.Client(api_key=get_api_key())
    throttle = RequestThrottle(args.request_interval)
    history = []
    question = args.question

    try:
        while True:
            if question is None:
                question = input("質問> ").strip()
            else:
                question = question.strip()

            if question.lower() in {"exit", "quit"}:
                break

            if question == "/model" or question.startswith("/model "):
                parts = question.split(maxsplit=1)
                if len(parts) == 1:
                    print("使い方: /model MODEL_ID")
                else:
                    args.model = parts[1]
                    print(f"[model] {args.model}")
                question = None
                continue

            if question:
                chat = client.chats.create(
                    model=args.model,
                    history=trim_history(history, args.max_history_turns),
                )
                print(f"[質問] {question}")
                try:
                    print(ask(question, chat, throttle, args.max_retries))
                except errors.APIError as error:
                    print(api_error_message(error.code, error.message or ""))
                    question = None
                    continue
                history = trim_history(chat.get_history(), args.max_history_turns)
                print()
            question = None
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()