# 文本水印逐步实现

第一步实现绿名单划分、软水印采样和统计检测。只使用 Python 标准库，可在 Windows 或 Linux 上运行。

demo_run.py 是均匀模拟词表上的教学实验，词元为整数编号。model_run.py 提供本地真实语言模型的中文生成、水印注入与统计检测入口。单次演示结果需要进一步通过正式数据集和质量评测检验。

## 运行

代码统一通过 Git 仓库分发。服务器首次使用 git clone 获取代码，后续使用 git pull 更新。开发变更先提交并推送到仓库，再由服务器拉取。模型权重通过模型平台下载，实验环境与权重保留在服务器约定目录。

在 PowerShell 中执行下列命令。Python 创建的运行日志输出到终端。

```powershell
python "D:\毕业设计\text_watermark\app\demo_run.py"
python "D:\毕业设计\text_watermark\app\demo_run.py" --delta 0
python "D:\毕业设计\text_watermark\app\demo_run.py" --delta 1
```

同一次运行分别输出无水印和有水印的检测结果。默认生成 200 个词元，绿名单比例为 0.5，水印偏置为 2，判定阈值为 z > 4。可改变 seed 观察随机波动，同一组参数下结果应一致。delta=0 时两组结果完全相同。

## 核心步骤

1. 用前一个词元编号与演示密钥的 SHA-256 摘要构造局部随机数生成器的种子。
2. 从词表中抽取固定数量的词元作为绿名单。这个教学实现要求 gamma 乘词表大小为整数，以保证实际比例与检测公式一致。
3. 无水印时均匀采样。有水印时给绿名单的 logit 增加 delta，再按 softmax 概率采样。代码使用等价的缩放权重避免指数溢出。
4. 检测端重建绿名单，对重复的“前一个词元、当前词元”组合只计数一次，减少重复片段对统计量的影响。
5. 令去重后的检测数为 T、绿名单命中数为 G，计算 z = (G - gamma*T) / sqrt(T*gamma*(1-gamma))。

p 值使用单侧正态近似。名单之间的依赖、重复和有限词表会影响校准，正式实验须用独立无水印数据验证误报率。名单生成使用 Python random，正式模型实现时需固定软件版本和名单算法，并验证两端一致。

## 文件

- app/demo_run.py 是运行入口，处理参数和日志。
- watermark/core.py 是业务实现。make_green_list 重建名单，generate_tokens 模拟采样，detect_tokens 计算检测结果。

## 后续步骤

### 服务器真实模型实验

已验证模型加载环境为 Python 3.12.4、PyTorch 2.5.1 CUDA 12.4、Transformers 4.51.3、Accelerate 1.6.0。真实水印入口需要在服务器进一步运行验证。

```bash
source /data/watermark_env/bin/activate
git -C /data/text_watermark pull --ff-only
python /data/text_watermark/app/model_run.py --model-path /data/models/Qwen2.5-7B-Instruct --output /data/text_watermark/outputs/first_run.json
```

程序只读取本地模型，先生成无水印文本，再生成有水印文本，并保存两组文字、词元编号、检测结果和环境参数。每组分别重设相同的随机种子。首次对照固定温度为 1、top-p 为 1、top-k 为 0，以减少其他采样机制对偏置的影响。示例使用公开演示密钥。

水印和检测均基于模型输出词表大小划分绿名单，屏蔽分词器中不存在的编号以及非结束特殊词元。检测去除尾部结束词元，将提示词最后一个词元作为第一步的上下文，不把提示词本身计入检测样本。直接拿解码文本再次分词可能改变边界，当前结果以保存的原始编号为准。

把 --delta 设为 0 时，两组使用相同采样机制和种子，应产生相同结果。短文本或低熵文本可能无法检出。当前不计算困惑度，也不据单次结果宣称达到任务书检出率或误报率指标。统计阈值需在独立样本上校准。

JSON 使用原子替换方式保存，重复使用同一个 --output 路径会更新该文件；需要保留多次实验时使用不同文件名。结果、模型权重和凭据均由 .gitignore 排除。

之后构建数据集，开展质量、误报率与四类扰动评测。

参考论文为 Kirchenbauer 等的 A Watermark for Large Language Models，ICML 2023。该代码是依据机制独立编写的教学实现，名单算法与论文官方代码不保证互通。

https://proceedings.mlr.press/v202/kirchenbauer23a.html
