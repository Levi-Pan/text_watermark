#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:PZW
# datetime:2026/11/13 10:45
# software:Vscode
# brief :解析真实模型水印实验参数并保存生成对照结果

import argparse
import json
import logging
import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from watermark.model_experiment import run_model_experiment


def main():
    """
    启动本地模型对照实验，原子写入 JSON 结果。

    :return: 无
    """
    parser = argparse.ArgumentParser(description="真实模型中文生成与软水印检测")
    parser.add_argument("--model-path", required=True, help="模型目录绝对路径")
    parser.add_argument("--output", required=True, help="结果 JSON 的绝对路径")
    parser.add_argument("--prompt", default="请写一篇关于城市公园如何改善居民生活的中文短文，约三百字。")
    parser.add_argument("--length", type=int, default=200, help="每组最多生成词元数")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", default="pzw-demo", help="实验名单密钥")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=4.0)
    arguments = parser.parse_args()
    if not os.path.isabs(arguments.output):
        parser.error("结果文件必须使用绝对路径")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    result = run_model_experiment(
        arguments.model_path, arguments.prompt, arguments.length, arguments.gamma,
        arguments.delta, arguments.key, arguments.seed, arguments.threshold,
    )
    output_directory = os.path.dirname(arguments.output)
    os.makedirs(output_directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=output_directory, delete=False) as output_file:
            temporary_path = output_file.name
            json.dump(result, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, arguments.output)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
    logging.getLogger(__name__).info("结果已保存：%s", arguments.output)


if __name__ == "__main__":
    main()
