## 工程目录

```bash
/home/mindspore/work/sig_lesson_6/mindformers/
```



## 词表下载

[Qwen/Qwen3-32B at main](https://huggingface.co/Qwen/Qwen3-32B/tree/main)



## 数据集处理

参考文档：https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html

- 转wiki为json格式（自行开发脚本）

```bash
python process_wiki_to_json.py
```

- 转megatron数据集1

```bash
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
--input /home/mindspore/work/sig_lesson_6/wiki.train.json \
--output-prefix /home/mindspore/work/sig_lesson_6/megatron_data \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-dir /home/mindspore/work/sig_lesson_6/Qwen3-32B
```

- 转megatron数据集2

```bash
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
--input /home/mindspore/work/sig_lesson_6/wiki.train.json \
--output-prefix /home/mindspore/work/sig_lesson_6/megatron_data2 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-dir /home/mindspore/work/sig_lesson_6/Qwen3-32B
```



## 上机演示

### 中断续训

**概述：**正常训练任务异常中断，需要基于保存的权重重新恢复训练任务。

#### ① 初始训练

**描述：**运行Qwen3【4层】训练， bs=2，dp2-tp2-pp2，一共训练20步，训练到第15步手动停止任务，本次训练会保存第10步权重；

拷贝`pretrain_qwen3_32b_4k.yaml`为`pretrain_qwen3_32b_4k_train_1-1.yaml`，并修改配置

```yaml
output_dir: './output_qwen3'
...
runner_config:
  epochs: 1
  batch_size: 2
...
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.5e-5
  lr_end: 1.5e-6
...
train_dataset: &train_dataset
  data_loader:
    sizes:
      - 160
    data_path:  # Sampling proportion and path for the Megatron dataset
      - "/home/mindspore/work/sig_lesson_6/megatron_data_text_document"
...
parallel_config:
  data_parallel: &dp 2  # Number of data parallel
  model_parallel: 2  # Number of model parallel
  pipeline_stage: 2  # Number of pipeline parallel
  micro_batch_num: 2
...
model:
  model_config:
    num_hidden_layers: 2
    offset: [-1, 1]
...
callbacks:
  - type: CheckpointMonitor  # Saves model weights during training
    save_checkpoint_steps: 10  # Interval steps for saving model weights
    keep_checkpoint_max: 3 
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_1-1.yaml"
```

训练日志：

```text
2025-10-16 11:25:12,147 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    1/   20], loss: 12.988943, per_step_time: 120509ms, lr: 1.5e-05, overflow cond: False, loss_scale: 1.0, global_norm: [20.29484], train_throughput_per_npu: 0.194T
2025-10-16 11:25:12,148 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    5.0% |██                                                | 0.00830 samples/s/p  0:38:09 }
2025-10-16 11:25:14,291 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:14,292 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    2/   20], loss: 11.152445, per_step_time: 2115ms, lr: 1.4916895e-05, overflow cond: False, loss_scale: 1.0, global_norm: [59.605118], train_throughput_per_npu: 11.044T
2025-10-16 11:25:14,293 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   10.0% |█████                                             | 0.47280 samples/s/p  0:00:38 }
2025-10-16 11:25:15,814 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:15,815 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    3/   20], loss: 11.315407, per_step_time: 1518ms, lr: 1.4669631e-05, overflow cond: False, loss_scale: 1.0, global_norm: [66.69845], train_throughput_per_npu: 15.381T
2025-10-16 11:25:15,816 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   15.0% |███████                                           | 0.65846 samples/s/p  0:00:25 }
2025-10-16 11:25:17,335 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:17,336 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    4/   20], loss: 11.152390, per_step_time: 1516ms, lr: 1.4264293e-05, overflow cond: False, loss_scale: 1.0, global_norm: [17.723818], train_throughput_per_npu: 15.401T
2025-10-16 11:25:17,337 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   20.0% |██████████                                        | 0.65931 samples/s/p  0:00:24 }
2025-10-16 11:25:18,852 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:18,853 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    5/   20], loss: 10.914505, per_step_time: 1512ms, lr: 1.3710864e-05, overflow cond: False, loss_scale: 1.0, global_norm: [18.08451], train_throughput_per_npu: 15.440T
2025-10-16 11:25:18,854 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   25.0% |████████████                                      | 0.66098 samples/s/p  0:00:22 }
2025-10-16 11:25:20,372 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:20,372 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    6/   20], loss: 10.134750, per_step_time: 1515ms, lr: 1.3022971e-05, overflow cond: False, loss_scale: 1.0, global_norm: [13.721817], train_throughput_per_npu: 15.416T
2025-10-16 11:25:20,374 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   30.0% |███████████████                                   | 0.65994 samples/s/p  0:00:21 }
2025-10-16 11:25:21,892 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:21,892 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    7/   20], loss: 10.236618, per_step_time: 1516ms, lr: 1.221755e-05, overflow cond: False, loss_scale: 1.0, global_norm: [72.45325], train_throughput_per_npu: 15.408T
2025-10-16 11:25:21,894 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   35.0% |█████████████████                                 | 0.65961 samples/s/p  0:00:19 }
2025-10-16 11:25:23,409 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:23,410 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    8/   20], loss: 10.729834, per_step_time: 1513ms, lr: 1.1314434e-05, overflow cond: False, loss_scale: 1.0, global_norm: [94.44166], train_throughput_per_npu: 15.436T
2025-10-16 11:25:23,411 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   40.0% |████████████████████                              | 0.66081 samples/s/p  0:00:18 }
2025-10-16 11:25:24,927 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:24,928 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    9/   20], loss: 9.902195, per_step_time: 1513ms, lr: 1.0335863e-05, overflow cond: False, loss_scale: 1.0, global_norm: [41.037834], train_throughput_per_npu: 15.431T
2025-10-16 11:25:24,929 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   45.0% |██████████████████████                            | 0.66058 samples/s/p  0:00:16 }
2025-10-16 11:25:26,446 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:25:26,447 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   10/   20], loss: 9.295567, per_step_time: 1514ms, lr: 9.305933e-06, overflow cond: False, loss_scale: 1.0, global_norm: [14.962573], train_throughput_per_npu: 15.421T
2025-10-16 11:25:26,448 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   50.0% |█████████████████████████                         | 0.66018 samples/s/p  0:00:15 }
2025-10-16 11:25:26,449 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1435] - INFO - ......Saving ckpt......
2025-10-16 11:25:26,449 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1494] - INFO - global_batch_size: 8
2025-10-16 11:25:26,449 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1495] - INFO - epoch_num: 10
2025-10-16 11:25:26,450 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1496] - INFO - step_num: 10
2025-10-16 11:25:26,450 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1497] - INFO - global_step: 10
2025-10-16 11:26:03,331 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1350] - INFO - Finish saving ckpt of epoch 10 step 1 using 36.791 seconds
2025-10-16 11:26:24,968 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:24,969 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   11/   20], loss: 9.158279, per_step_time: 21634ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [14.018612], train_throughput_per_npu: 1.080T
2025-10-16 11:26:24,970 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   55.0% |███████████████████████████                       | 0.04622 samples/s/p  0:03:14 }
2025-10-16 11:26:26,491 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:26,491 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   12/   20], loss: 8.932339, per_step_time: 1518ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [12.136359], train_throughput_per_npu: 15.388T
2025-10-16 11:26:26,492 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   60.0% |██████████████████████████████                    | 0.65875 samples/s/p  0:00:12 }
2025-10-16 11:26:28,013 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:28,014 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   13/   20], loss: 8.885098, per_step_time: 1518ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [21.226376], train_throughput_per_npu: 15.381T
2025-10-16 11:26:28,015 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   65.0% |████████████████████████████████                  | 0.65843 samples/s/p  0:00:10 }
2025-10-16 11:26:29,543 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:29,543 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   14/   20], loss: 8.515237, per_step_time: 1525ms, lr: 5.185564e-06, overflow cond: False, loss_scale: 1.0, global_norm: [16.667692], train_throughput_per_npu: 15.313T
2025-10-16 11:26:29,545 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   70.0% |███████████████████████████████████               | 0.65552 samples/s/p  0:00:09 }
2025-10-16 11:26:31,066 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:31,067 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   15/   20], loss: 8.407747, per_step_time: 1518ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [13.268136], train_throughput_per_npu: 15.380T
2025-10-16 11:26:31,068 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   75.0% |█████████████████████████████████████             | 0.65840 samples/s/p  0:00:07 }
2025-10-16 11:26:32,588 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:32,589 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   16/   20], loss: 8.467024, per_step_time: 1517ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.018916], train_throughput_per_npu: 15.393T
2025-10-16 11:26:32,590 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   80.0% |████████████████████████████████████████          | 0.65894 samples/s/p  0:00:06 }
2025-10-16 11:26:34,110 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:26:34,110 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   17/   20], loss: 8.410109, per_step_time: 1517ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.815596], train_throughput_per_npu: 15.392T
2025-10-16 11:26:34,112 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   85.0% |██████████████████████████████████████████        | 0.65892 samples/s/p  0:00:04 }
```



#### ② 续训方式1

**描述：**基于**最后保存完整的权重**续训，验证从11步开始续训且loss对齐。

拷贝`pretrain_qwen3_32b_4k_train_1-1.yaml`为`pretrain_qwen3_32b_4k_train_1-2.yaml`，并修改配置

```yaml
load_checkpoint: '/home/mindspore/work/sig_lesson_6/mindformers/output_qwen3/checkpoint'
resume_training: True
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_1-2.yaml"
```

训练日志：

```text
2025-10-16 11:31:26,626 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   11/   20], loss: 9.158279, per_step_time: 30823ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [14.018621], train_throughput_per_npu: 0.758T
2025-10-16 11:31:26,628 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   55.0% |███████████████████████████                       | 0.03244 samples/s/p  0:04:37 }
2025-10-16 11:31:28,530 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:28,530 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   12/   20], loss: 8.932501, per_step_time: 1873ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [12.136798], train_throughput_per_npu: 12.467T
2025-10-16 11:31:28,531 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   60.0% |██████████████████████████████                    | 0.53370 samples/s/p  0:00:14 }
2025-10-16 11:31:30,053 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:30,054 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   13/   20], loss: 8.885171, per_step_time: 1519ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [21.22979], train_throughput_per_npu: 15.375T
2025-10-16 11:31:30,055 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   65.0% |████████████████████████████████                  | 0.65819 samples/s/p  0:00:10 }
2025-10-16 11:31:31,575 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:31,576 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   14/   20], loss: 8.515329, per_step_time: 1517ms, lr: 5.185564e-06, overflow cond: False, loss_scale: 1.0, global_norm: [16.667099], train_throughput_per_npu: 15.391T
2025-10-16 11:31:31,577 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   70.0% |███████████████████████████████████               | 0.65886 samples/s/p  0:00:09 }
2025-10-16 11:31:33,095 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:33,096 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   15/   20], loss: 8.407721, per_step_time: 1515ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [13.270613], train_throughput_per_npu: 15.415T
2025-10-16 11:31:33,097 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   75.0% |█████████████████████████████████████             | 0.65989 samples/s/p  0:00:07 }
2025-10-16 11:31:34,618 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:34,618 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   16/   20], loss: 8.467200, per_step_time: 1518ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.019392], train_throughput_per_npu: 15.384T
2025-10-16 11:31:34,620 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   80.0% |████████████████████████████████████████          | 0.65856 samples/s/p  0:00:06 }
2025-10-16 11:31:36,139 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:36,140 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   17/   20], loss: 8.410226, per_step_time: 1516ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.815823], train_throughput_per_npu: 15.399T
2025-10-16 11:31:36,141 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   85.0% |██████████████████████████████████████████        | 0.65921 samples/s/p  0:00:04 }
2025-10-16 11:31:37,657 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:37,658 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   18/   20], loss: 8.403861, per_step_time: 1514ms, lr: 2.2357062e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.402732], train_throughput_per_npu: 15.428T
2025-10-16 11:31:37,659 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   90.0% |█████████████████████████████████████████████     | 0.66048 samples/s/p  0:00:03 }
2025-10-16 11:31:39,177 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:39,177 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   19/   20], loss: 8.319234, per_step_time: 1515ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.509191], train_throughput_per_npu: 15.416T
2025-10-16 11:31:39,179 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   95.0% |███████████████████████████████████████████████   | 0.65995 samples/s/p  0:00:01 }
2025-10-16 11:31:40,696 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:31:40,697 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   20/   20], loss: 8.276322, per_step_time: 1514ms, lr: 1.5831035e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.3199444], train_throughput_per_npu: 15.419T
2025-10-16 11:31:40,698 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   100.0% |██████████████████████████████████████████████████| 0.66009 samples/s/p  0:00:00 }
```





#### ③ 续训方式2

**描述：**指定**第10步权重**续训，验证从11步开始续训且loss对齐。

拷贝`pretrain_qwen3_32b_4k_train_1-2.yaml`为`pretrain_qwen3_32b_4k_train_1-3.yaml`，并修改配置

```yaml
resume_training: 'qwen3_rank_0-10_1.safetensors'
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_1-3.yaml"
```

训练日志：

```text
2025-10-16 11:36:43,214 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   11/   20], loss: 9.158279, per_step_time: 30686ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [14.018583], train_throughput_per_npu: 0.761T
2025-10-16 11:36:43,216 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   55.0% |███████████████████████████                       | 0.03259 samples/s/p  0:04:36 }
2025-10-16 11:36:44,759 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:44,759 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   12/   20], loss: 8.932548, per_step_time: 1515ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [12.136445], train_throughput_per_npu: 15.416T
2025-10-16 11:36:44,761 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   60.0% |██████████████████████████████                    | 0.65993 samples/s/p  0:00:12 }
2025-10-16 11:36:46,284 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:46,284 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   13/   20], loss: 8.884958, per_step_time: 1520ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [21.223118], train_throughput_per_npu: 15.361T
2025-10-16 11:36:46,286 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   65.0% |████████████████████████████████                  | 0.65758 samples/s/p  0:00:10 }
2025-10-16 11:36:47,843 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:47,844 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   14/   20], loss: 8.515385, per_step_time: 1555ms, lr: 5.185564e-06, overflow cond: False, loss_scale: 1.0, global_norm: [16.678844], train_throughput_per_npu: 15.019T
2025-10-16 11:36:47,845 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   70.0% |███████████████████████████████████               | 0.64297 samples/s/p  0:00:09 }
2025-10-16 11:36:49,365 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:49,365 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   15/   20], loss: 8.407721, per_step_time: 1516ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [13.271069], train_throughput_per_npu: 15.401T
2025-10-16 11:36:49,366 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   75.0% |█████████████████████████████████████             | 0.65929 samples/s/p  0:00:07 }
2025-10-16 11:36:50,918 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:50,919 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   16/   20], loss: 8.467037, per_step_time: 1549ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.019668], train_throughput_per_npu: 15.075T
2025-10-16 11:36:50,920 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   80.0% |████████████████████████████████████████          | 0.64536 samples/s/p  0:00:06 }
2025-10-16 11:36:52,439 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:52,440 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   17/   20], loss: 8.410301, per_step_time: 1516ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.817221], train_throughput_per_npu: 15.402T
2025-10-16 11:36:52,441 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   85.0% |██████████████████████████████████████████        | 0.65933 samples/s/p  0:00:04 }
2025-10-16 11:36:53,957 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:53,958 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   18/   20], loss: 8.403791, per_step_time: 1513ms, lr: 2.2357062e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.4035015], train_throughput_per_npu: 15.434T
2025-10-16 11:36:53,959 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   90.0% |█████████████████████████████████████████████     | 0.66070 samples/s/p  0:00:03 }
2025-10-16 11:36:55,475 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:55,476 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   19/   20], loss: 8.319315, per_step_time: 1514ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.506244], train_throughput_per_npu: 15.429T
2025-10-16 11:36:55,477 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   95.0% |███████████████████████████████████████████████   | 0.66050 samples/s/p  0:00:01 }
2025-10-16 11:36:57,009 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:36:57,010 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   20/   20], loss: 8.276365, per_step_time: 1529ms, lr: 1.5831035e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.316961], train_throughput_per_npu: 15.272T
2025-10-16 11:36:57,011 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   100.0% |██████████████████████████████████████████████████| 0.65377 samples/s/p  0:00:00 }
```



### 转换策略续训

**概述：**需要修改分布式策略或扩大/缩小集群规模继续训练任务，同时需要权重转换。

#### ① 修改策略续训

**描述：**修改配置为bs=2， dp2-tp1-pp2，指定加载第10步权重转换续训，任务会**在线合并分布式权重**，验证任务从第11步开始续训且loss对齐。

拷贝`pretrain_qwen3_32b_4k_train_1-3.yaml`为`pretrain_qwen3_32b_4k_train_2-1.yaml`，并修改配置

```yaml
output_dir: './output_qwen3_2'
src_strategy_path_or_dir: '/home/mindspore/work/sig_lesson_6/mindformers/output_qwen3/strategy'
auto_trans_ckpt: True
parallel_config:
  data_parallel: &dp 2
  model_parallel: 1
  pipeline_stage: 2
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_2-1.yaml" 4
```

训练日志：

```text
2025-10-16 11:45:42,086 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   11/   20], loss: 9.158377, per_step_time: 29870ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [14.0188265], train_throughput_per_npu: 2.686T
2025-10-16 11:45:42,087 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   55.0% |███████████████████████████                       | 0.06695 samples/s/p  0:04:28 }
2025-10-16 11:45:45,850 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:45,853 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   12/   20], loss: 8.932455, per_step_time: 3736ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [12.136663], train_throughput_per_npu: 21.473T
2025-10-16 11:45:45,854 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   60.0% |██████████████████████████████                    | 0.53519 samples/s/p  0:00:29 }
2025-10-16 11:45:48,589 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:48,590 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   13/   20], loss: 8.885075, per_step_time: 2732ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [21.233116], train_throughput_per_npu: 29.366T
2025-10-16 11:45:48,591 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   65.0% |████████████████████████████████                  | 0.73193 samples/s/p  0:00:19 }
2025-10-16 11:45:51,331 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:51,331 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   14/   20], loss: 8.515325, per_step_time: 2737ms, lr: 5.185564e-06, overflow cond: False, loss_scale: 1.0, global_norm: [16.676113], train_throughput_per_npu: 29.316T
2025-10-16 11:45:51,332 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   70.0% |███████████████████████████████████               | 0.73067 samples/s/p  0:00:16 }
2025-10-16 11:45:54,070 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:54,070 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   15/   20], loss: 8.407734, per_step_time: 2735ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [13.26971], train_throughput_per_npu: 29.338T
2025-10-16 11:45:54,071 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   75.0% |█████████████████████████████████████             | 0.73122 samples/s/p  0:00:13 }
2025-10-16 11:45:56,807 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:56,807 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   16/   20], loss: 8.466991, per_step_time: 2733ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.020935], train_throughput_per_npu: 29.360T
2025-10-16 11:45:56,808 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   80.0% |████████████████████████████████████████          | 0.73178 samples/s/p  0:00:10 }
2025-10-16 11:45:59,543 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:45:59,544 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   17/   20], loss: 8.410078, per_step_time: 2732ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.814938], train_throughput_per_npu: 29.361T
2025-10-16 11:45:59,545 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   85.0% |██████████████████████████████████████████        | 0.73180 samples/s/p  0:00:08 }
2025-10-16 11:46:02,280 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:46:02,281 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   18/   20], loss: 8.403811, per_step_time: 2733ms, lr: 2.2357062e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.4037285], train_throughput_per_npu: 29.359T
2025-10-16 11:46:02,282 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   90.0% |█████████████████████████████████████████████     | 0.73175 samples/s/p  0:00:05 }
2025-10-16 11:46:05,019 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:46:05,020 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   19/   20], loss: 8.319027, per_step_time: 2735ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.5093493], train_throughput_per_npu: 29.338T
2025-10-16 11:46:05,021 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   95.0% |███████████████████████████████████████████████   | 0.73123 samples/s/p  0:00:02 }
2025-10-16 11:46:07,765 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:46:07,766 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   20/   20], loss: 8.276321, per_step_time: 2741ms, lr: 1.5831035e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.3206153], train_throughput_per_npu: 29.266T
2025-10-16 11:46:07,767 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   100.0% |██████████████████████████████████████████████████| 0.72942 samples/s/p  0:00:00 }
```



#### ② 扩容续训

**描述：**修改配置为dp4-tp1-pp2，加载**合并权重**续训，验证任务从第5步开始续训且loss在合理范围内。

拷贝`pretrain_qwen3_32b_4k_train_2-1.yaml`为`pretrain_qwen3_32b_4k_train_2-2.yaml`，并修改配置

```yaml
load_checkpoint: '/home/mindspore/work/sig_lesson_6/mindformers/output_qwen3_2/unified_checkpoint'
src_strategy_path_or_dir: ''
resume_training: True
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_2-2.yaml"
```

训练日志：

```text
2025-10-16 11:51:40,802 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    6/   10], loss: 9.114973, per_step_time: 41029ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [13.064995], train_throughput_per_npu: 1.139T
2025-10-16 11:51:40,803 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   60.0% |██████████████████████████████                    | 0.04874 samples/s/p  0:02:44 }
2025-10-16 11:51:44,110 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:51:44,110 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    7/   10], loss: 8.812031, per_step_time: 3280ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [15.712417], train_throughput_per_npu: 14.241T
2025-10-16 11:51:44,111 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   70.0% |███████████████████████████████████               | 0.60966 samples/s/p  0:00:09 }
2025-10-16 11:51:46,709 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:51:46,710 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    8/   10], loss: 8.782402, per_step_time: 2595ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [21.09246], train_throughput_per_npu: 18.001T
2025-10-16 11:51:46,711 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   80.0% |████████████████████████████████████████          | 0.77059 samples/s/p  0:00:05 }
2025-10-16 11:51:49,314 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:51:49,314 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[    9/   10], loss: 8.804184, per_step_time: 2600ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [18.780453], train_throughput_per_npu: 17.969T
2025-10-16 11:51:49,315 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   90.0% |█████████████████████████████████████████████     | 0.76922 samples/s/p  0:00:02 }
2025-10-16 11:51:53,236 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 11:51:53,236 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/  1], step:[   10/   10], loss: 8.601419, per_step_time: 3917ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.595917], train_throughput_per_npu: 11.925T
2025-10-16 11:51:53,237 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   100.0% |██████████████████████████████████████████████████| 0.51048 samples/s/p  0:00:00 }
```



### 增量续训

**概述：**训练数据集边生产边训练，当前数据集训练结束后，加入新生产的训练数据集继续训练。

**前期准备：**准备两个数据集

#### ① 训练数据集1

**描述：**运行Qwen3【4层】训练，bs=2，dp2-tp2-pp2，一共训练20步，保存第20步权重。

拷贝`pretrain_qwen3_32b_4k_train_1-1.yaml`为`pretrain_qwen3_32b_4k_train_3-1.yaml`，并修改配置

```yaml
output_dir: './output_qwen3_3'
epochs: 10
total_steps: 40
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_3-1.yaml"
```

训练日志：

```text
2025-10-16 12:01:36,275 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/trainer/base_trainer.py:1449] - INFO - .........Starting Training Model..........
..2025-10-16 12:03:00,207 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:01,234 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:470] - INFO - Full model flops is 149716117487616, Shard model flops is 120029303537664.
2025-10-16 12:03:01,234 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    1/   20], loss: 12.988943, per_step_time: 76374ms, lr: 1.5e-05, overflow cond: False, loss_scale: 1.0, global_norm: [20.294798], train_throughput_per_npu: 0.306T
2025-10-16 12:03:01,236 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    0.5% |                                                  | 0.01309 samples/s/p  4:13:18 }
2025-10-16 12:03:03,298 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:03,299 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    2/   20], loss: 11.152285, per_step_time: 2038ms, lr: 1.497919e-05, overflow cond: False, loss_scale: 1.0, global_norm: [59.590164], train_throughput_per_npu: 11.457T
2025-10-16 12:03:03,300 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    1.0% |                                                  | 0.49044 samples/s/p  0:06:43 }
2025-10-16 12:03:04,820 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:04,821 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    3/   20], loss: 11.322637, per_step_time: 1517ms, lr: 1.4916895e-05, overflow cond: False, loss_scale: 1.0, global_norm: [66.926125], train_throughput_per_npu: 15.396T
2025-10-16 12:03:04,822 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    1.5% |                                                  | 0.65910 samples/s/p  0:04:58 }
2025-10-16 12:03:06,345 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:06,345 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    4/   20], loss: 11.163303, per_step_time: 1520ms, lr: 1.4813497e-05, overflow cond: False, loss_scale: 1.0, global_norm: [17.629583], train_throughput_per_npu: 15.366T
2025-10-16 12:03:06,346 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    2.0% |█                                                 | 0.65780 samples/s/p  0:04:57 }
2025-10-16 12:03:07,873 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:07,873 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    5/   20], loss: 10.909895, per_step_time: 1524ms, lr: 1.4669631e-05, overflow cond: False, loss_scale: 1.0, global_norm: [17.921074], train_throughput_per_npu: 15.328T
2025-10-16 12:03:07,874 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    2.5% |█                                                 | 0.65616 samples/s/p  0:04:57 }
2025-10-16 12:03:09,402 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:09,403 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    6/   20], loss: 10.105618, per_step_time: 1525ms, lr: 1.4486186e-05, overflow cond: False, loss_scale: 1.0, global_norm: [13.835648], train_throughput_per_npu: 15.313T
2025-10-16 12:03:09,404 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    3.0% |█                                                 | 0.65555 samples/s/p  0:04:55 }
2025-10-16 12:03:10,924 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:10,924 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    7/   20], loss: 9.938313, per_step_time: 1516ms, lr: 1.4264293e-05, overflow cond: False, loss_scale: 1.0, global_norm: [49.54612], train_throughput_per_npu: 15.399T
2025-10-16 12:03:10,925 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    3.5% |█                                                 | 0.65921 samples/s/p  0:04:52 }
2025-10-16 12:03:12,442 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:12,443 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    8/   20], loss: 11.121490, per_step_time: 1514ms, lr: 1.400532e-05, overflow cond: False, loss_scale: 1.0, global_norm: [103.26017], train_throughput_per_npu: 15.425T
2025-10-16 12:03:12,444 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    4.0% |██                                                | 0.66034 samples/s/p  0:04:50 }
2025-10-16 12:03:13,962 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:13,963 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[    9/   20], loss: 10.047550, per_step_time: 1515ms, lr: 1.3710864e-05, overflow cond: False, loss_scale: 1.0, global_norm: [59.293636], train_throughput_per_npu: 15.416T
2025-10-16 12:03:13,964 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    4.5% |██                                                | 0.65996 samples/s/p  0:04:49 }
2025-10-16 12:03:15,482 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:03:15,483 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   10/   20], loss: 9.394676, per_step_time: 1515ms, lr: 1.338274e-05, overflow cond: False, loss_scale: 1.0, global_norm: [17.683195], train_throughput_per_npu: 15.409T
2025-10-16 12:03:15,484 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    5.0% |██                                                | 0.65967 samples/s/p  0:04:48 }
2025-10-16 12:03:15,485 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1435] - INFO - ......Saving ckpt......
2025-10-16 12:03:15,485 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1494] - INFO - global_batch_size: 8
2025-10-16 12:03:15,485 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1495] - INFO - epoch_num: 10
2025-10-16 12:03:15,486 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1496] - INFO - step_num: 10
2025-10-16 12:03:15,486 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1497] - INFO - global_step: 10
2025-10-16 12:04:07,802 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1350] - INFO - Finish saving ckpt of epoch 10 step 1 using 52.230 seconds
2025-10-16 12:04:09,325 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:09,326 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   11/   20], loss: 9.335443, per_step_time: 1520ms, lr: 1.3022971e-05, overflow cond: False, loss_scale: 1.0, global_norm: [14.212884], train_throughput_per_npu: 15.368T
2025-10-16 12:04:09,327 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    5.5% |██                                                | 0.65787 samples/s/p  0:04:47 }
2025-10-16 12:04:10,850 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:10,850 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   12/   20], loss: 9.042027, per_step_time: 1519ms, lr: 1.2633773e-05, overflow cond: False, loss_scale: 1.0, global_norm: [20.529808], train_throughput_per_npu: 15.371T
2025-10-16 12:04:10,851 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    6.0% |███                                               | 0.65802 samples/s/p  0:04:45 }
2025-10-16 12:04:12,371 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:12,371 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   13/   20], loss: 8.725659, per_step_time: 1516ms, lr: 1.221755e-05, overflow cond: False, loss_scale: 1.0, global_norm: [12.70041], train_throughput_per_npu: 15.399T
2025-10-16 12:04:12,373 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    6.5% |███                                               | 0.65923 samples/s/p  0:04:43 }
2025-10-16 12:04:13,891 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:13,892 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   14/   20], loss: 8.437876, per_step_time: 1516ms, lr: 1.1776865e-05, overflow cond: False, loss_scale: 1.0, global_norm: [23.36068], train_throughput_per_npu: 15.406T
2025-10-16 12:04:13,893 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    7.0% |███                                               | 0.65952 samples/s/p  0:04:42 }
2025-10-16 12:04:15,412 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:15,412 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   15/   20], loss: 8.416552, per_step_time: 1515ms, lr: 1.1314434e-05, overflow cond: False, loss_scale: 1.0, global_norm: [17.947992], train_throughput_per_npu: 15.410T
2025-10-16 12:04:15,413 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    7.5% |███                                               | 0.65967 samples/s/p  0:04:40 }
2025-10-16 12:04:16,933 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:16,934 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   16/   20], loss: 8.436235, per_step_time: 1517ms, lr: 1.0833113e-05, overflow cond: False, loss_scale: 1.0, global_norm: [16.866219], train_throughput_per_npu: 15.397T
2025-10-16 12:04:16,935 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    8.0% |████                                              | 0.65914 samples/s/p  0:04:39 }
2025-10-16 12:04:18,452 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:18,453 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   17/   20], loss: 8.289206, per_step_time: 1515ms, lr: 1.0335863e-05, overflow cond: False, loss_scale: 1.0, global_norm: [8.778104], train_throughput_per_npu: 15.418T
2025-10-16 12:04:18,454 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    8.5% |████                                              | 0.66005 samples/s/p  0:04:37 }
2025-10-16 12:04:19,973 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:19,974 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   18/   20], loss: 8.255161, per_step_time: 1516ms, lr: 9.825755e-06, overflow cond: False, loss_scale: 1.0, global_norm: [12.741383], train_throughput_per_npu: 15.401T
2025-10-16 12:04:19,975 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    9.0% |████                                              | 0.65928 samples/s/p  0:04:36 }
2025-10-16 12:04:21,494 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:21,495 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   19/   20], loss: 8.107678, per_step_time: 1516ms, lr: 9.305933e-06, overflow cond: False, loss_scale: 1.0, global_norm: [9.425522], train_throughput_per_npu: 15.407T
2025-10-16 12:04:21,496 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -    9.5% |████                                              | 0.65957 samples/s/p  0:04:34 }
2025-10-16 12:04:23,022 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:04:23,022 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  1/ 10], step:[   20/   20], loss: 7.960324, per_step_time: 1522ms, lr: 8.779598e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.6081753], train_throughput_per_npu: 15.339T
2025-10-16 12:04:23,023 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   10.0% |█████                                             | 0.65666 samples/s/p  0:04:34 }
2025-10-16 12:04:23,024 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1435] - INFO - ......Saving ckpt......
2025-10-16 12:04:23,025 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1494] - INFO - global_batch_size: 8
2025-10-16 12:04:23,025 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1495] - INFO - epoch_num: 20
2025-10-16 12:04:23,025 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1496] - INFO - step_num: 20
2025-10-16 12:04:23,025 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1497] - INFO - global_step: 20
2025-10-16 12:04:52,819 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1350] - INFO - Finish saving ckpt of epoch 20 step 1 using 29.694 seconds
2025-10-16 12:05:04,739 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:05:04,740 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    1/   20], loss: 7.715164, per_step_time: 11916ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.416067], train_throughput_per_npu: 1.960T
2025-10-16 12:05:04,741 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   10.5% |█████                                             | 0.08392 samples/s/p  0:35:33 }
2025-10-16 12:05:06,263 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:05:06,263 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    2/   20], loss: 8.001792, per_step_time: 1519ms, lr: 7.720401e-06, overflow cond: False, loss_scale: 1.0, global_norm: [9.585871], train_throughput_per_npu: 15.375T
2025-10-16 12:05:06,265 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   11.0% |█████                                             | 0.65818 samples/s/p  0:04:30 }
2025-10-16 12:05:07,788 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:05:07,788 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    3/   20], loss: 7.790206, per_step_time: 1520ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [8.86793], train_throughput_per_npu: 15.365T
2025-10-16 12:05:07,789 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   11.5% |█████                                             | 0.65774 samples/s/p  0:04:29 }
```



#### ② 训练数据集2

**描述：**更换数据集2，基于最后保存的权重续训，验证从21步开始续训且loss在合理范围内，训练到第35步左右手动停止任务，保存第30步权重 。

拷贝`pretrain_qwen3_32b_4k_train_3-1.yaml`为`pretrain_qwen3_32b_4k_train_3-2.yaml`，并修改配置

```yaml
load_checkpoint: '/home/mindspore/work/sig_lesson_6/mindformers/output_qwen3_3/checkpoint'
resume_training: True
train_dataset: &train_dataset
  data_loader:
    sizes:
      - 120
    data_path:  # Sampling proportion and path for the Megatron dataset
      - "/home/mindspore/work/sig_lesson_6/megatron_data2_text_document"
```

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_3-2.yaml"
```

训练日志：

```text
2025-10-16 12:09:56,603 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    6/   15], loss: 7.757602, per_step_time: 29568ms, lr: 8.249999e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.5633774], train_throughput_per_npu: 0.790T
2025-10-16 12:09:56,605 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   14.0% |███████                                           | 0.03382 samples/s/p  1:03:34 }
2025-10-16 12:09:58,504 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:09:58,505 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    7/   15], loss: 7.776651, per_step_time: 1875ms, lr: 7.720401e-06, overflow cond: False, loss_scale: 1.0, global_norm: [10.859435], train_throughput_per_npu: 12.452T
2025-10-16 12:09:58,506 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   14.7% |███████                                           | 0.53305 samples/s/p  0:04:00 }
2025-10-16 12:10:00,024 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:00,025 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    8/   15], loss: 7.792886, per_step_time: 1516ms, lr: 7.1940667e-06, overflow cond: False, loss_scale: 1.0, global_norm: [8.898812], train_throughput_per_npu: 15.406T
2025-10-16 12:10:00,026 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   15.3% |███████                                           | 0.65952 samples/s/p  0:03:12 }
2025-10-16 12:10:01,546 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:01,547 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[    9/   15], loss: 7.823087, per_step_time: 1517ms, lr: 6.674243e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.0935783], train_throughput_per_npu: 15.393T
2025-10-16 12:10:01,548 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   16.0% |████████                                          | 0.65897 samples/s/p  0:03:11 }
2025-10-16 12:10:03,101 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:03,101 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   10/   15], loss: 7.658785, per_step_time: 1550ms, lr: 6.164134e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.165509], train_throughput_per_npu: 15.067T
2025-10-16 12:10:03,103 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   16.7% |████████                                          | 0.64499 samples/s/p  0:03:13 }
2025-10-16 12:10:04,624 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:04,625 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   11/   15], loss: 7.787768, per_step_time: 1519ms, lr: 5.666886e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.5175867], train_throughput_per_npu: 15.376T
2025-10-16 12:10:04,626 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   17.3% |████████                                          | 0.65824 samples/s/p  0:03:08 }
2025-10-16 12:10:06,145 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:06,146 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   12/   15], loss: 7.674144, per_step_time: 1516ms, lr: 5.185564e-06, overflow cond: False, loss_scale: 1.0, global_norm: [6.3638673], train_throughput_per_npu: 15.404T
2025-10-16 12:10:06,147 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   18.0% |█████████                                         | 0.65942 samples/s/p  0:03:06 }
2025-10-16 12:10:07,664 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:07,664 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   13/   15], loss: 7.570381, per_step_time: 1514ms, lr: 4.7231338e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.412529], train_throughput_per_npu: 15.425T
2025-10-16 12:10:07,666 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   18.7% |█████████                                         | 0.66032 samples/s/p  0:03:04 }
2025-10-16 12:10:09,183 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:09,183 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   14/   15], loss: 7.384876, per_step_time: 1514ms, lr: 4.28245e-06, overflow cond: False, loss_scale: 1.0, global_norm: [3.947785], train_throughput_per_npu: 15.424T
2025-10-16 12:10:09,184 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   19.3% |█████████                                         | 0.66029 samples/s/p  0:03:03 }
2025-10-16 12:10:10,702 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:10:10,702 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  2/ 10], step:[   15/   15], loss: 7.394458, per_step_time: 1514ms, lr: 3.866226e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.0848], train_throughput_per_npu: 15.423T
2025-10-16 12:10:10,703 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   20.0% |██████████                                        | 0.66023 samples/s/p  0:03:01 }
2025-10-16 12:10:10,704 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1435] - INFO - ......Saving ckpt......
2025-10-16 12:10:10,704 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1494] - INFO - global_batch_size: 8
2025-10-16 12:10:10,705 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1495] - INFO - epoch_num: 30
2025-10-16 12:10:10,705 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1496] - INFO - step_num: 30
2025-10-16 12:10:10,705 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1497] - INFO - global_step: 30
2025-10-16 12:11:15,846 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:1350] - INFO - Finish saving ckpt of epoch 10 step 1 using 65.056 seconds
2025-10-16 12:11:17,363 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:17,364 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    1/   15], loss: 7.398445, per_step_time: 1515ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.361766], train_throughput_per_npu: 15.418T
2025-10-16 12:11:17,365 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   20.7% |██████████                                        | 0.66005 samples/s/p  0:03:00 }
2025-10-16 12:11:18,887 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:18,888 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    2/   15], loss: 7.654553, per_step_time: 1519ms, lr: 3.11726e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.0824885], train_throughput_per_npu: 15.369T
2025-10-16 12:11:18,889 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   21.3% |██████████                                        | 0.65794 samples/s/p  0:02:59 }
2025-10-16 12:11:20,407 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:20,408 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    3/   15], loss: 7.471928, per_step_time: 1515ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.1529117], train_throughput_per_npu: 15.410T
2025-10-16 12:11:20,409 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   22.0% |███████████                                       | 0.65968 samples/s/p  0:02:57 }
2025-10-16 12:11:21,927 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:21,927 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    4/   15], loss: 7.276101, per_step_time: 1515ms, lr: 2.494678e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.9446826], train_throughput_per_npu: 15.414T
2025-10-16 12:11:21,929 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   22.7% |███████████                                       | 0.65986 samples/s/p  0:02:55 }
2025-10-16 12:11:23,448 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:23,448 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    5/   15], loss: 7.311204, per_step_time: 1516ms, lr: 2.2357062e-06, overflow cond: False, loss_scale: 1.0, global_norm: [3.83122], train_throughput_per_npu: 15.408T
2025-10-16 12:11:23,449 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   23.3% |███████████                                       | 0.65958 samples/s/p  0:02:54 }
2025-10-16 12:11:24,968 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:24,969 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    6/   15], loss: 7.245948, per_step_time: 1516ms, lr: 2.013813e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.297004], train_throughput_per_npu: 15.405T
2025-10-16 12:11:24,970 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   24.0% |████████████                                      | 0.65946 samples/s/p  0:02:52 }
2025-10-16 12:11:26,489 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:11:26,489 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    7/   15], loss: 7.283252, per_step_time: 1515ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.562732], train_throughput_per_npu: 15.410T
2025-10-16 12:11:26,490 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   24.7% |████████████                                      | 0.65969 samples/s/p  0:02:51 }
```



#### ③ 续训数据集2

**描述：**基于最后保存的权重续训，验证续训从30步开始，loss对齐。

运行命令：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/qwen3/pretrain_qwen3_32b_4k_train_3-2.yaml"
```

训练日志：

```text
2025-10-16 12:15:53,995 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    1/   15], loss: 7.398445, per_step_time: 30360ms, lr: 3.4770292e-06, overflow cond: False, loss_scale: 1.0, global_norm: [7.361754], train_throughput_per_npu: 0.769T
2025-10-16 12:15:53,996 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   20.7% |██████████                                        | 0.03294 samples/s/p  1:00:12 }
2025-10-16 12:15:55,874 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:15:55,874 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    2/   15], loss: 7.654472, per_step_time: 1853ms, lr: 3.11726e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.0821295], train_throughput_per_npu: 12.600T
2025-10-16 12:15:55,875 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   21.3% |██████████                                        | 0.53941 samples/s/p  0:03:38 }
2025-10-16 12:15:57,396 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:15:57,396 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    3/   15], loss: 7.471920, per_step_time: 1517ms, lr: 2.7891351e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.152682], train_throughput_per_npu: 15.389T
2025-10-16 12:15:57,397 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   22.0% |███████████                                       | 0.65878 samples/s/p  0:02:57 }
2025-10-16 12:15:58,921 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:15:58,921 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    4/   15], loss: 7.276098, per_step_time: 1520ms, lr: 2.494678e-06, overflow cond: False, loss_scale: 1.0, global_norm: [5.9442096], train_throughput_per_npu: 15.361T
2025-10-16 12:15:58,922 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   22.7% |███████████                                       | 0.65757 samples/s/p  0:02:56 }
2025-10-16 12:16:00,457 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:00,457 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    5/   15], loss: 7.311240, per_step_time: 1531ms, lr: 2.2357062e-06, overflow cond: False, loss_scale: 1.0, global_norm: [3.831543], train_throughput_per_npu: 15.252T
2025-10-16 12:16:00,458 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   23.3% |███████████                                       | 0.65292 samples/s/p  0:02:56 }
2025-10-16 12:16:02,019 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:02,020 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    6/   15], loss: 7.245981, per_step_time: 1558ms, lr: 2.013813e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.297491], train_throughput_per_npu: 14.989T
2025-10-16 12:16:02,021 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   24.0% |████████████                                      | 0.64165 samples/s/p  0:02:57 }
2025-10-16 12:16:03,540 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:03,540 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    7/   15], loss: 7.283454, per_step_time: 1516ms, lr: 1.8303688e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.561908], train_throughput_per_npu: 15.407T
2025-10-16 12:16:03,542 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   24.7% |████████████                                      | 0.65956 samples/s/p  0:02:51 }
2025-10-16 12:16:05,100 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:05,101 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    8/   15], loss: 7.327477, per_step_time: 1556ms, lr: 1.6865024e-06, overflow cond: False, loss_scale: 1.0, global_norm: [4.1615686], train_throughput_per_npu: 15.010T
2025-10-16 12:16:05,102 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   25.3% |████████████                                      | 0.64258 samples/s/p  0:02:54 }
2025-10-16 12:16:06,620 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:06,620 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[    9/   15], loss: 7.402902, per_step_time: 1514ms, lr: 1.5831035e-06, overflow cond: False, loss_scale: 1.0, global_norm: [3.878625], train_throughput_per_npu: 15.419T
2025-10-16 12:16:06,621 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   26.0% |█████████████                                     | 0.66009 samples/s/p  0:02:48 }
2025-10-16 12:16:08,139 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:374] - WARNING - pipeline stages: 2 > 1, the loss on the last card is valid.
2025-10-16 12:16:08,140 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:544] - INFO - { Epoch:[  3/ 10], step:[   10/   15], loss: 7.270339, per_step_time: 1514ms, lr: 1.5208075e-06, overflow cond: False, loss_scale: 1.0, global_norm: [2.8283603], train_throughput_per_npu: 15.419T
2025-10-16 12:16:08,141 - mindformers/home/mindspore/work/sig_lesson_6/mindformers/output/log[mindformers/core/callback/callback.py:560] - INFO -   26.7% |█████████████                                     | 0.66008 samples/s/p  0:02:46 }
```

