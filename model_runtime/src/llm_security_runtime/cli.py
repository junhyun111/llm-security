from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from llm_security.config import AppConfig

from .api import create_runtime_app
from .artifacts import assert_model_compatible, inspect_artifacts
from .paths import RuntimePaths, configure_process_environment
from .runner import analyze_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-security-runtime",
        description="Packaged Utility Router runtime for C/C++ vulnerability analysis",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Runtime .env path (default: model_runtime/.env)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "inspect", help="Validate router.pkl and candidate_ranker.pkl without API calls"
    )

    analyze = subparsers.add_parser(
        "analyze", help="Analyze one C/C++ file or project directory"
    )
    analyze.add_argument("source")
    analyze.add_argument("--output", default="analysis.json")

    serve = subparsers.add_parser("serve", help="Start the FastAPI backend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = RuntimePaths.discover(args.env_file)
    try:
        configure_process_environment(paths)
        metadata = inspect_artifacts(paths)
        if args.command == "inspect":
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return 0

        paths.require_env()
        config = AppConfig.from_env(paths.env_file)
        assert_model_compatible(metadata, config.model.expert_model)

        if args.command == "analyze":
            if not config.runtime.allow_paid_experiments:
                raise RuntimeError(
                    "Analysis calls OpenRouter. Set RUN_PAID_EXPERIMENTS=1 in .env "
                    "after checking the model and budget."
                )
            summary = analyze_source(
                Path(args.source),
                Path(args.output),
                config=config,
                paths=paths,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "serve":
            app, service = create_runtime_app(paths)
            try:
                uvicorn.run(app, host=args.host, port=args.port, reload=False)
            finally:
                service.close()
            return 0
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
