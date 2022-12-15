from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from transformers import RobertaTokenizerFast, DataCollatorForTokenClassification

class PhrExtDataLoader:
    def __init__(self, pretrained_ck: str, max_length: int):
        dataset = load_dataset('conll2003').remove_columns(['id', 'pos_tags', 'ner_tags'])
        self.tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_ck, add_prefix_space=True)
        self.max_length = max_length
        self.dataset = dataset.map(
            self.__tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    def __align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def __tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )
        all_labels = examples["chunk_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.__align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items()}
        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test", "valid"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
