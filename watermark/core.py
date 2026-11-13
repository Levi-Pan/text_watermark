#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:PZW
# datetime:2026/11/11 10:40
# software:Vscode
# brief :实现教学用绿名单划分、模拟软水印采样与统计检测

import hashlib
import math
import random


def make_green_list(previous_token, vocabulary_size, green_fraction, key):
    """
    根据前一个词元和密钥重建绿名单，使用局部随机数生成器。

    :param previous_token: 前一个词元的整数编号
    :param vocabulary_size: 模拟词表大小
    :param green_fraction: 绿名单比例，乘词表大小须为整数
    :param key: 控制名单划分的字符串密钥
    :return: 绿名单词元编号集合
    """
    if vocabulary_size < 2 or not 0 < green_fraction < 1:
        raise ValueError("词表大小至少为 2，绿名单比例必须介于 0 与 1 之间")
    green_count = round(vocabulary_size * green_fraction)
    if not math.isclose(green_count, vocabulary_size * green_fraction):
        raise ValueError("本阶段要求绿名单比例乘词表大小为整数")
    if not 0 <= previous_token < vocabulary_size:
        raise ValueError("前一个词元编号超出词表范围")
    seed_bytes = hashlib.sha256(f"{key}\0{previous_token}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(seed_bytes, "big"))
    return set(generator.sample(range(vocabulary_size), green_count))


def generate_tokens(token_count, vocabulary_size, green_fraction, delta, key, seed):
    """
    从均匀模拟词表采样，为绿名单词元添加 logit 偏置。

    :param token_count: 生成词元数，不含起始词元
    :param vocabulary_size: 模拟词表大小
    :param green_fraction: 绿名单比例
    :param delta: 非负且有限的偏置强度，零表示无水印
    :param key: 名单生成密钥
    :param seed: 独立控制词元采样的随机种子
    :return: 包含起始词元的整数编号列表
    """
    if token_count < 1 or not math.isfinite(delta) or delta < 0:
        raise ValueError("生成长度必须为正，偏置强度须为非负有限数")
    generator = random.Random(seed)
    token_ids = [0]
    for _ in range(token_count):
        green_list = make_green_list(token_ids[-1], vocabulary_size, green_fraction, key)
        # 统一减去 delta 避免指数溢出，归一化后的概率与加偏置等价。
        weights = [1.0 if token_id in green_list else math.exp(-delta)
                   for token_id in range(vocabulary_size)]
        token_ids.append(generator.choices(range(vocabulary_size), weights=weights, k=1)[0])
    return token_ids


def detect_tokens(token_ids, vocabulary_size, green_fraction, key, threshold):
    """
    统计绿名单命中并计算单侧正态近似，只统计首次出现的词元对。

    :param token_ids: 含起始词元的整数编号序列
    :param vocabulary_size: 与生成端一致的词表大小
    :param green_fraction: 与生成端一致的绿名单比例
    :param key: 与生成端一致的名单密钥
    :param threshold: 严格大于此值时报告检测到水印
    :return: 包含计数、z 分数、近似 p 值和判定的字典
    """
    if len(token_ids) < 2:
        raise ValueError("至少需要一个起始词元和一个待检测词元")
    if not math.isfinite(threshold):
        raise ValueError("检测阈值必须是有限数")
    if any(not isinstance(token_id, int) or not 0 <= token_id < vocabulary_size
           for token_id in token_ids):
        raise ValueError("词元编号必须是词表范围内的整数")
    seen_pairs = set()
    green_count = 0
    for previous_token, current_token in zip(token_ids, token_ids[1:]):
        green_list = make_green_list(previous_token, vocabulary_size, green_fraction, key)
        token_pair = (previous_token, current_token)
        if token_pair not in seen_pairs:
            seen_pairs.add(token_pair)
            green_count += current_token in green_list
    evaluated_count = len(seen_pairs)
    z_score = (green_count - green_fraction * evaluated_count) / math.sqrt(
        evaluated_count * green_fraction * (1 - green_fraction)
    )
    return {
        "generated_count": len(token_ids) - 1,
        "evaluated_count": evaluated_count,
        "green_count": green_count,
        "green_ratio": green_count / evaluated_count,
        "z_score": z_score,
        "p_value_approx": 0.5 * math.erfc(z_score / math.sqrt(2)),
        "detected": z_score > threshold,
    }
