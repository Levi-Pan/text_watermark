#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:PZW
# datetime:2026/11/11 10:40
# software:Vscode
# brief :运行无水印与软水印的模拟词元对照演示

import argparse
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from watermark.core import detect_tokens, generate_tokens


def main():
    """
    解析实验参数，生成两组模拟词元并记录检测结果。

    :return: 无
    """
    parser = argparse.ArgumentParser(description="第一步：模拟词表水印实验，无需 GPU")
    parser.add_argument("--length", type=int, default=200, help="生成词元数")
    parser.add_argument("--vocabulary-size", type=int, default=256, help="模拟词表大小")
    parser.add_argument("--gamma", type=float, default=0.5, help="绿名单比例")
    parser.add_argument("--delta", type=float, default=2.0, help="绿名单 logit 偏置")
    parser.add_argument("--key", default="pzw-demo", help="演示密钥")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--threshold", type=float, default=4.0, help="z 分数判定阈值")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info("教学模拟：均匀词表采样，尚未接入语言模型或中文分词器。")
    for label, delta in (("无水印", 0.0), ("有水印", arguments.delta)):
        token_ids = generate_tokens(
            arguments.length, arguments.vocabulary_size, arguments.gamma,
            delta, arguments.key, arguments.seed,
        )
        result = detect_tokens(
            token_ids, arguments.vocabulary_size, arguments.gamma,
            arguments.key, arguments.threshold,
        )
        logger.info("%s | delta=%.2f | 生成=%d | 去重后检测=%d | 绿名单=%d (%.1f%%)",
                    label, delta, result["generated_count"], result["evaluated_count"],
                    result["green_count"], result["green_ratio"] * 100)
        logger.info("z=%.3f | 单侧近似 p=%.3g | 检出水印=%s",
                    result["z_score"], result["p_value_approx"], result["detected"])


if __name__ == "__main__":
    main()
