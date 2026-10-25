"""Gemini API のレート制限を安全に観測する、回数制限付きスクリプト。

実行例:
    python stress_test.py
    python stress_test.py --models gemini-3.5-flash-lite gemini-2.5-flash
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, get_api_key

MAX_REQUESTS = 15
REQUEST_INTERVAL_SECONDS = 1.0
MAX_OUTPUT_TOKENS = 128
PROMPT = (
    "LoRA（Low-Rank Adaptation）の仕組みを、低ランク行列による重み更新 "
    "W' = W + BA（B ∈ R^(d×r), A ∈ R^(r×k), r ≪ min(d, k)）を使って説明してください。"
    "通常の全パラメータ微調整との違い、学習可能パラメータ数、推論時の特徴を "
    "日本語で簡潔に述べてください。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Gemini API のレート制限を最大{MAX_REQUESTS}回で観測します。"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[MODEL],
        metavar="MODEL",
        help="順に検証するモデルID（既定: %(default)s）",
    )
    return parser.parse_args()


def run_stress_test(client: genai.Client, models: list[str]) -> None:
    total_requests = 0

    for model in models:
        model_requests = 0
        print(f"\n[model] {model}")

        while total_requests < MAX_REQUESTS:
            if total_requests:
                time.sleep(REQUEST_INTERVAL_SECONDS)

            total_requests += 1
            model_requests += 1
            print(
                f"[request] model={model} "
                f"model_attempt={model_requests} total={total_requests}/{MAX_REQUESTS}"
            )

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=PROMPT,
                    config={"max_output_tokens": MAX_OUTPUT_TOKENS},
                )
            except errors.APIError as error:
                if error.code == 429:
                    print(
                        f"[rate-limit] HTTP 429: model={model} "
                        f"model_attempt={model_requests} "
                        f"total_attempt={total_requests}。検証を終了します。"
                    )
                    return
                raise

            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            print(
                f"[success] input_tokens={input_tokens} "
                f"output_tokens={output_tokens}"
            )

        if total_requests == MAX_REQUESTS:
            break

    print(f"\n[done] 429なし。送信リクエスト数: {total_requests}/{MAX_REQUESTS}")


def main() -> None:
    args = parse_args()
    client = genai.Client(api_key=get_api_key())
    run_stress_test(client, args.models)


if __name__ == "__main__":
    main()
