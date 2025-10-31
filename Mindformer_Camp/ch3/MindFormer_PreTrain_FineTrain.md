# MindSpore Transformers预训练与微调实践

## 环境与目录

* 建议目录：`/home/mindspore/work/demo`
* 建议 Python ≥ 3.9，已安装 MindSpore + MindFormers
* 可选：配置国内镜像

```bash
mkdir -p /home/mindspore/work/demo && cd /home/mindspore/work/demo
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com   # 可按需要注释/取消
```

## 准备权重、数据集与MindSpore Transformers

Qwen3-0.6B: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
Qwen3-1.7B: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

预训练数据集：[wikitext-2-v1](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/wikitext/wikitext-2-v1.zip)

> 预训练数据集参考[社区issue](https://gitee.com/mindspore/mindformers/issues/IBV35D)获取

微调数据集: [llm-wizard/alpaca-gpt4-data-zh](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh)

```bash
# 下载权重
mkdir -p /home/mindspore/work/demo
cd /home/mindspore/work/demo
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B
huggingface-cli download --resume-download Qwen/Qwen3-1.7B --local-dir Qwen3-1.7B

# 下载预训练数据集
wget https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip

# 下载微调数据集
huggingface-cli download --repo-type dataset --resume-download llm-wizard/alpaca-gpt4-data-zh --local-dir alpaca-gpt4-data-zh

# 下载MindSpore Transformers
git clone https://gitee.com/mindspore/mindformers.git
cd mindformers
git checkout 5a12973fb38bfd5b504240334492f4fb7ff7f7a6
pip install -r requirements.txt

# 升级MindSpore
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202509/20250917/master_20250917220006_52c46b3bfd9e9d50b2334d764afc80a6d7b56e90_newest/unified/aarch64/mindspore-2.7.1-cp39-cp39-linux_aarch64.whl

```

> 所有需要的权重和数据集都应当挂载到/home/mindspore/work/demo

最后的文件结构应为：

```plaintext
/home/mindspore/work/demo
├── Qwen3-0.6B/
├── Qwen3-1.7B/
├── wikitext-2-v1/
├── alpaca-gpt4-data-zh/
└── mindformers/
```

## 数据预处理

### wiki => json

脚本来源：[社区issue #ICOKGY](https://gitee.com/mindspore/mindformers/issues/ICOKGY)

```python
import json
import re
import os
import sys
import time

def clean_wikitext(string: str) -> str:
    """cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def print_progress(done_bytes, total_bytes, start_time, bar_len=40):
    """Simple text progress bar without extra libs."""
    if total_bytes == 0:
        return
    ratio = done_bytes / total_bytes
    filled = int(bar_len * ratio)
    elapsed = time.time() - start_time
    speed = done_bytes / elapsed if elapsed else 0  # bytes/s
    eta = (total_bytes - done_bytes) / speed if speed else 0

    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r[{bar}] {ratio:6.2%}  "
        f"{done_bytes/1024/1024:6.2f}MB / {total_bytes/1024/1024:6.2f}MB  "
        f"ETA {eta:5.1f}s"
    )
    sys.stdout.flush()


def gen_wiki_json(input_file, output_file):
    """generate wikitext-2/wikitext-103 json with a progress bar"""
    total_bytes = os.stat(input_file).st_size
    done_bytes = 0
    start_time = time.time()

    data_idx = 0
    out = open(output_file, 'w', encoding='utf-8')

    para_buffer = []
    def flush_para():
        nonlocal data_idx
        para = "".join(para_buffer).strip()
        if para and not para.startswith('='):
            cleaned = clean_wikitext(para)
            content = {"text": cleaned, "id": str(data_idx)}
            out.write(json.dumps(content, ensure_ascii=False) + "\n")
            data_idx += 1

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            done_bytes += len(line.encode('utf-8', errors='ignore'))
            # Paragraph split: empty line indicates paragraph boundary
            if line.strip() == "":
                flush_para()
                para_buffer = []
            else:
                para_buffer.append(line)

            # Update progress bar
            print_progress(done_bytes, total_bytes, start_time)

        # Flush the last paragraph if exists
        if para_buffer:
            flush_para()

    out.close()
    # final newline after progress bar
    sys.stdout.write("\nDone! Total paragraphs: {}\n".format(data_idx))
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate JSONL from WikiText files using clean_wikitext + gen_wiki_json."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw WikiText file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    gen_wiki_json(str(args.input), str(args.output))

```

将脚本复制到`/home/mindspore/work/demo/mindformers/gen_wiki_json.py`

```bash
cd /home/mindspore/work/demo/mindformers
python gen_wiki_json.py --input /home/mindspore/work/demo/wikitext-2/wiki.train.tokens  --output /home/mindspore/work/demo/wikitext-2/wiki.jsonl
```

### json => megatron

先取前1000条数据

```bash
head -n 1000 /home/mindspore/work/demo/wikitext-2/wiki.jsonl > /home/mindspore/work/demo/wikitext-2/wiki.less.jsonl
```

```shell
# cd /home/mindspore/work/demo/mindformers
mkdir -p /home/mindspore/work/demo/megatron_data
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /home/mindspore/work/demo/wikitext-2/wiki.less.jsonl \
  --output-prefix /home/mindspore/work/demo/megatron_data/wikitext-2-v1-qwen3_text_document \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /home/mindspore/work/demo/Qwen3-0.6B/ \
  --workers 64
```

## 准备Qwen3-0.6B模型的yaml

MindSpore Transformers源码仓库中提供了Qwen3-32B的预训练模型[yaml配置文件](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/pretrain_qwen3_32b_4k.yaml)，我们可以参考该文件来创建Qwen3-0.6B的yaml文件。

首先复制32B配置并重命名为0.6b的配置

```bash
cp configs/qwen3/pretrain_qwen3_32b_4k.yaml configs/qwen3/pretrain_qwen3_0_6b_4k.yaml
```

接下来需要对`configs/qwen3/pretrain_qwen3_0_6b_4k.yaml`做出以下修改：

```yaml
# Model configuration
model:
  model_config:
    # Configurations from Hugging Face
    ...
    hidden_size: 1024
    intermediate_size: 3072
    num_hidden_layers: 28
    num_attention_heads: 16
    num_key_value_heads: 8
    head_dim: 128
    ...
    # Configurations from MindFormers
    offset: 0
```

## 预训练

### 修改配置以启动预训练任务

修改训练epoch数

```yaml
# runner config
runner_config:
  epochs: 1
```

修改数据集配置

```yaml
# dataset
train_dataset: &train_dataset
  data_loader:
    ...
    sizes:
      - 400   # 训练集数据样本数
      - 0    # 测试集数据样本数，当前不支持配置
      - 0    # 评测集数据样本数，当前不支持配置
    config:  # GPTDataset配置项
      ...
      data_path:  # Megatron数据集采样比例以及路径
        - '1'
        - "/home/mindspore/work/demo/megatron_data/wikitext-2-v1-qwen3_text_document_text_document"
```

添加以下参数以开启TensorBoard监控

```yaml
monitor_config:
    monitor_on: True
    dump_path: './dump'
    target: ['layers.0', 'layers.1'] # 只监控第一、二层的参数
    invert: False
    step_interval: 1
    local_loss_format: ['tensorboard']
    device_local_loss_format: ['tensorboard']
    local_norm_format: ['tensorboard']
    device_local_norm_format: ['tensorboard']
    optimizer_state_format: null
    weight_state_format: null
    throughput_baseline: null
    print_struct: False
    check_for_global_norm: False
    global_norm_spike_threshold: 1.0
    global_norm_spike_count_threshold: 10

tensorboard:
    tensorboard_dir: 'worker/tensorboard'
    tensorboard_queue_size: 10
    log_loss_scale_to_tensorboard: True
    log_timers_to_tensorboard: True
```



添加以下配置以开启权重去冗余

```yaml
callbacks:
  - type: MFLossMonitor  # Prints training progress information
  - type: CheckpointMonitor  # Saves model weights during training
    ...
    save_checkpoint_steps: 50  # Interval steps for saving model weights
    keep_checkpoint_max: 2  # Maximum number of saved model weight files
    ...
    remove_redundancy: True
```

修改以下配置以切换并行配置

```yaml
parallel_config:
  data_parallel: &dp 2
  model_parallel: 2
  pipeline_stage: 2
  micro_batch_num: &micro_batch_num 2
```

### 启动预训练任务

```bash
rm -rf ouput/checkpoint
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_0_6b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train \
--recompute_config.recompute False"
```

### 启动断点续训的预训练任务

修改以下配置以开启断点续训功能

```yaml
resume_training: True
```

备份上一次任务日志，用于查看续训是否成功

```bash
mv ouput/msrun_log output/msrun_log_bak
```

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_0_6b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train \
--load_checkpoint "output/checkpoint_50_step" \
--recompute_config.recompute False"
```

## 全参微调

### 修改配置以启动全参微调任务

MindSpore Transformers源码仓库中提供了Qwen3类不同规模的微调模型[yaml配置文件](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)，不同规模的模型可以通过`--pretrained_model_dir /path/to/Qwen3-0.6B/`来指定。我们可以参考该文件来创建Qwen3-0.6B的全参微调的yaml文件。


需要对`configs/qwen3/finetune_qwen3.yaml`做出以下修改：

修改以下配置项以设置数据集大小

```yaml
# Dataset configuration
train_dataset: &train_dataset
  ...
  data_loader:
  type: HFDataLoader
    path: "llm-wizard/alpaca-gpt4-data-zh"
    ...
    # dataset process arguments
    handler:
      - type: take
        n: 1000
```

修改以下配置以切换并行配置

```yaml
parallel_config:
  data_parallel: &dp 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: &micro_batch_num 1
  ...
  use_seq_parallel: False
```

### 启动全参微调任务

启动Qwen3-0.6B的全参微调任务

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/finetune_qwen3.yaml \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode train \
--pretrained_model_dir /home/mindspore/work/demo/Qwen3-0.6B/ \
--recompute_config.recompute False"
```

## LoRA微调

MindSpore Transformers源码仓库中提供了Qwen3类不同规模的微调模型[yaml配置文件](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml)，不同规模的模型可以通过`--pretrained_model_dir /path/to/Qwen3-1.7B/`来指定。我们可以参考该文件来创建Qwen3-1.7B的LoRA微调的yaml文件。

首先复制微调配置并重命名为lora的配置

```bash
cp configs/qwen3/finetune_qwen3.yaml configs/qwen3/finetune_qwen3_lora.yaml
```

### 修改配置以启动LoRA微调任务

添加以下参数以启动LoRA微调

```yaml
# Model configuration
model:
  model_config:
    # Configurations from Hugging Face
    ...
    pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      lora_a_init: 'normal'
      lora_b_init: 'zeros'
      target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
      freeze_include: ['*']
      freeze_exclude: ['*lora*']
```

修改以下配置以切换并行配置

```yaml
parallel_config:
  data_parallel: &dp 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: &micro_batch_num 1
```

### 启动LoRA微调任务

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/finetune_qwen3_lora.yaml \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode train \
--pretrained_model_dir /home/mindspore/work/demo/Qwen3-1.7B/ \
--recompute_config.recompute False"
```

home/mindspore/work/tools/mindformers/output/msrun_log