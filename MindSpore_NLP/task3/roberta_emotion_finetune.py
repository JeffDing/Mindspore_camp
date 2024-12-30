#!/usr/bin/env python
# coding: utf-8
"""
unset MULTI_NPU && python bert_imdb_finetune_cpu_mindnlp_trainer_npus_same.py
bash bert_imdb_finetune_npu_mindnlp_trainer.sh
"""

import mindspore
from mindnlp.engine import Trainer
from mindnlp.dataset import load_dataset
from mindspore.dataset import GeneratorDataset, transforms

from mindnlp.accelerate.utils.constants import accelerate_distributed_type
from mindnlp.accelerate.utils.dataclasses import DistributedType

# prepare dataset
class SentimentDataset:
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)

def main():
    """demo

    Returns:
        desc: _description_
    """
    imdb_ds = load_dataset('imdb', split=['train', 'test'])
    imdb_train = imdb_ds['train']
    imdb_train.get_dataset_size()

    from mindnlp.transformers import AutoTokenizer
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def process_dataset(source, tokenizer, max_seq_len=64, batch_size=32, shuffle=True):
        is_ascend = mindspore.get_context('device_target') == 'Ascend'

        column_names = ["label", "text_a"]

        dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
        # transforms
        type_cast_op = transforms.TypeCast(mindspore.int32)
        def tokenize_and_pad(text):
            if is_ascend:
                tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
            else:
                tokenized = tokenizer(text)
            return tokenized['input_ids'], tokenized['attention_mask']
        # map dataset
        dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a", output_columns=['input_ids', 'attention_mask'])
        dataset = dataset.map(operations=[type_cast_op], input_columns="label", output_columns='labels')
        # # batch dataset
        if is_ascend:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                             'attention_mask': (None, 0)})

        return dataset


    dataset_train = process_dataset(SentimentDataset("data/train.tsv"), tokenizer)
    dataset_val = process_dataset(SentimentDataset("data/dev.tsv"), tokenizer)
    dataset_test = process_dataset(SentimentDataset("data/test.tsv"), tokenizer, shuffle=False)

    next(dataset_train.create_tuple_iterator())

    from mindnlp.transformers import AutoModelForSequenceClassification

    # set bert config and define parameters for training
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    from mindnlp.engine import TrainingArguments

    training_args = TrainingArguments(
        output_dir="roberta_emotion_npu",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=2.0,
        learning_rate=2e-5
    )
    training_args = training_args.set_optimizer(name="adamw", beta1=0.8) # 手动指定优化器，OptimizerNames.SGD

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
    )
    print("Start training")
    trainer.train()

if __name__ == '__main__':
    main()

