"""
STORM Wiki pipeline powered by Qwen models through Alibaba Cloud Model Studio.

Official docs:
    - Models: https://help.aliyun.com/zh/model-studio/models
    - OpenAI-compatible API: https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
    - Context cache: https://help.aliyun.com/zh/model-studio/context-cache

Environment variables:
    - DASHSCOPE_API_KEY: Alibaba Cloud Model Studio/DashScope API key
    - DASHSCOPE_API_BASE: Optional OpenAI-compatible API base URL
      Defaults to https://dashscope.aliyuncs.com/compatible-mode/v1
    - YDC_API_KEY, BING_SEARCH_API_KEY, SERPER_API_KEY, BRAVE_API_KEY, or TAVILY_API_KEY

The default model mix uses qwen3.6-flash for conversation/question tasks and
qwen3.7-plus for outline/article/polish tasks. LiteLLM local disk cache is on by
default. Provider-side implicit cache is handled by DashScope when supported;
use --cache-mode explicit to add cache_control markers to requests.
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from knowledge_storm import (
    STORMWikiLMConfigs,
    STORMWikiRunner,
    STORMWikiRunnerArguments,
)
from knowledge_storm.lm import QwenModel
from knowledge_storm.rm import (
    BingSearch,
    BraveRM,
    DuckDuckGoSearchRM,
    SearXNG,
    SerperRM,
    TavilySearchRM,
    YouRM,
)
from knowledge_storm.utils import load_api_key


def qwen_extra_body(thinking_mode: str, thinking_budget: int | None) -> dict | None:
    if thinking_mode == "default":
        return None

    extra_body = {"enable_thinking": thinking_mode == "on"}
    if thinking_mode == "on" and thinking_budget:
        extra_body["thinking_budget"] = thinking_budget
    return extra_body


def load_dotenv_file(env_file_path=".env", protected_keys=None):
    if not os.path.exists(env_file_path):
        return

    protected_keys = protected_keys or set()
    with open(env_file_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in protected_keys:
                os.environ[key] = value


def sanitize_topic(topic):
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)
    return topic or "unnamed_topic"


def build_retriever(args, engine_args):
    match args.retriever:
        case "bing":
            return BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "you":
            return YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        case "brave":
            return BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=engine_args.search_top_k,
            )
        case "duckduckgo":
            return DuckDuckGoSearchRM(
                k=engine_args.search_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            return SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            return TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=engine_args.search_top_k,
                include_raw_content=True,
            )
        case "searxng":
            return SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=engine_args.search_top_k
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose "bing", "you", '
                '"brave", "duckduckgo", "serper", "tavily", or "searxng".'
            )


def make_qwen_lm(model, max_tokens, args):
    return QwenModel(
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_base=os.getenv(
            "DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache=args.cache_mode != "off",
        provider_cache=args.cache_mode == "explicit",
        cache_prefix_chars=args.explicit_cache_prefix_chars or None,
        extra_body=qwen_extra_body(args.qwen_thinking, args.qwen_thinking_budget),
    )


def main(args):
    shell_env_keys = set(os.environ)
    if os.path.exists("secrets.toml"):
        load_api_key(toml_file_path="secrets.toml")
    load_dotenv_file(protected_keys=shell_env_keys)
    logger = logging.getLogger(__name__)

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError(
            "DASHSCOPE_API_KEY is not set. Add it to .env, secrets.toml, or the environment."
        )

    lm_configs = STORMWikiLMConfigs()

    conv_simulator_lm = make_qwen_lm(args.flash_model, 500, args)
    question_asker_lm = make_qwen_lm(args.flash_model, 500, args)
    outline_gen_lm = make_qwen_lm(args.plus_model, 400, args)
    article_gen_lm = make_qwen_lm(args.plus_model, 700, args)
    article_polish_lm = make_qwen_lm(args.plus_model, 4000, args)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    rm = build_retriever(args, engine_args)
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input("Topic: ")
    try:
        runner.run(
            topic=sanitize_topic(topic),
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=args.do_polish_article,
        )
        runner.post_run()
        runner.summary()
    except Exception as exc:
        logger.exception("An error occurred: %s", exc)
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/qwen",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Maximum number of threads to use for research and generation.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        required=True,
        help="The search engine API to use for retrieving information.",
    )
    parser.add_argument(
        "--flash-model",
        type=str,
        default="qwen3.6-flash",
        help="Cheaper/faster Qwen model for conversation and question asking.",
    )
    parser.add_argument(
        "--plus-model",
        type=str,
        default="qwen3.7-plus",
        help="Stronger Qwen model for outline, article generation, and polishing.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["implicit", "explicit", "off"],
        default="implicit",
        help=(
            "implicit keeps local LiteLLM cache and relies on DashScope automatic "
            "cache when available; explicit also adds cache_control markers; off "
            "disables local cache and explicit markers."
        ),
    )
    parser.add_argument(
        "--explicit-cache-prefix-chars",
        type=int,
        default=0,
        help=(
            "When --cache-mode explicit is used, mark only the first N characters "
            "of each prompt as cacheable. Use 0 to mark the full prompt."
        ),
    )
    parser.add_argument(
        "--qwen-thinking",
        choices=["default", "on", "off", "disable"],
        default="disable",
        help="Qwen thinking mode for both qwen3.6 and qwen3.7 model calls.",
    )
    parser.add_argument("--qwen-thinking-budget", type=int, default=4096)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature to use."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If True, polish the generated article.",
    )
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Top k search results to consider for each search query.",
    )

    main(parser.parse_args())
