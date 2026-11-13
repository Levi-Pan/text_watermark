#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:PZW
# datetime:2026/11/13 10:45
# software:Vscode
# brief :为本地语言模型注入绿名单偏置并进行中文生成对照检测

import hashlib
import logging
import math
import os
import platform
import time

from watermark.core import detect_tokens, make_green_list


class GreenListLogitsProcessor:
    """单序列采样处理器，复用检测端名单规则，并屏蔽无效词元。"""

    def __init__(self, vocabulary_size, green_fraction, delta, key, suppressed_ids):
        """
        保存偏置参数并验证名单划分。

        :param vocabulary_size: 模型输出词表大小
        :param green_fraction: 绿名单占模型输出词表的比例
        :param delta: 非负且有限的 logit 偏置
        :param key: 名单生成密钥
        :param suppressed_ids: 无效编号及不允许生成的特殊词元
        :return: 无
        """
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("水印强度必须是非负有限数")
        make_green_list(0, vocabulary_size, green_fraction, key)
        self.vocabulary_size = vocabulary_size
        self.green_fraction = green_fraction
        self.delta = delta
        self.key = key
        self.suppressed_ids = list(suppressed_ids)

    def __call__(self, input_ids, scores):
        """
        对当前生成位置增加绿名单偏置，再屏蔽禁用编号。

        :param input_ids: 一条序列的完整上下文编号，包含提示词
        :param scores: 当前步的模型输出 logit
        :return: 新的 logit 张量
        """
        if scores.shape != (1, self.vocabulary_size):
            raise ValueError("当前入口只支持单序列生成，且输出词表须保持一致")
        updated_scores = scores.clone()
        if self.delta > 0:
            green_list = make_green_list(
                input_ids[0, -1].item(), self.vocabulary_size, self.green_fraction, self.key,
            )
            updated_scores[0, list(green_list)] += self.delta
        updated_scores[0, self.suppressed_ids] = -math.inf
        return updated_scores


def run_model_experiment(model_path, prompt, max_new_tokens, green_fraction, delta,
                         key, seed, threshold):
    """
    离线加载本地模型，用同一提示和种子生成两组文本并检测。

    :param model_path: 本地模型目录的绝对路径
    :param prompt: 中文用户提示词
    :param max_new_tokens: 每组最多生成的词元数
    :param green_fraction: 绿名单比例
    :param delta: 有水印组的偏置强度
    :param key: 两端共用的名单密钥
    :param seed: 两次生成分别重设的随机种子
    :param threshold: z 分数严格大于此值时报告检出
    :return: 含环境参数、文本、词元编号和统计结果的字典
    """
    if not os.path.isabs(model_path) or not os.path.isdir(model_path):
        raise ValueError("模型路径必须是已经存在的本地绝对目录")
    if not prompt.strip() or max_new_tokens < 1 or not math.isfinite(threshold):
        raise ValueError("提示词不能为空，生成长度须为正，阈值须有限")
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("水印强度必须是非负有限数")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GenerationConfig, LogitsProcessorList, set_seed

    logger = logging.getLogger(__name__)
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA GPU，请激活服务器上的 watermark_env 环境")
    logger.info("正在从本地目录加载模型：%s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        use_safetensors=True, local_files_only=True,
    )
    model.eval()
    vocabulary_size = model.config.vocab_size
    # 输出矩阵可能包含分词器中不存在的填充编号，需在采样时屏蔽。
    valid_ids = set(tokenizer.get_vocab().values())
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not eos_ids or not set(eos_ids).issubset(valid_ids):
        raise ValueError("模型须提供有效的结束词元编号")
    if not valid_ids or min(valid_ids) < 0 or max(valid_ids) >= vocabulary_size:
        raise ValueError("分词器与模型输出词表不匹配")
    suppressed_ids = sorted(
        (set(range(vocabulary_size)) - valid_ids)
        | (set(tokenizer.all_special_ids) - set(eos_ids))
    )
    encoded_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(encoded_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("提示词与生成长度之和超过模型上下文上限")
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
        top_p=1.0, top_k=0, repetition_penalty=1.0, num_beams=1,
        eos_token_id=eos_ids, pad_token_id=eos_ids[0], use_cache=True,
    )
    result = {
        "model_path": model_path, "model_type": model.config.model_type,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "vocabulary_size": vocabulary_size, "gamma": green_fraction, "delta": delta,
        "seed": seed, "threshold": threshold, "max_new_tokens": max_new_tokens,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "key_fingerprint": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "prompt": prompt, "prompt_token_ids": inputs["input_ids"][0].tolist(), "samples": [],
    }
    for label, bias in (("无水印", 0.0), ("有水印", delta)):
        processor = GreenListLogitsProcessor(
            vocabulary_size, green_fraction, bias, key, suppressed_ids,
        )
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        started_at = time.perf_counter()
        logger.info("正在生成%s文本……", label)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, generation_config=generation_config,
                logits_processor=LogitsProcessorList([processor]),
            )
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started_at
        generated_ids = outputs[0, prompt_length:].tolist()
        content_ids = generated_ids.copy()
        if content_ids and content_ids[-1] in eos_ids:
            content_ids.pop()
        detection = None
        if content_ids:
            detection = detect_tokens(
                [result["prompt_token_ids"][-1]] + content_ids,
                vocabulary_size, green_fraction, key, threshold,
            )
        text = tokenizer.decode(content_ids, skip_special_tokens=True)
        result["samples"].append({
            "label": label, "delta": bias, "text": text,
            "generated_token_ids": generated_ids, "detection": detection,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        })
        logger.info("%s文本：\n%s", label, text)
        logger.info("检测结果：%s", detection)
        logger.info("耗时 %.1f 秒，峰值显存 %.2f GiB", elapsed_seconds,
                    result["samples"][-1]["peak_gpu_gib"])
    return result
