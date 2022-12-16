import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from PhrExt import PhrExtModel, PhrExtConfig, LabelConverter
import evaluate
import numpy as np

class LitPhrExt(pl.LightningModule):
    def __init__(self, pretrained_ck: str, layers_use_from_last: int, method_for_layers: str, lr: float):
        super(LitPhrExt, self).__init__()
        label_converter = LabelConverter()
        self.id2label = label_converter.id2label
        config = PhrExtConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            layers_use_from_last=layers_use_from_last,
            method_for_layers=method_for_layers,
            id2label={i: label for i, label in enumerate(self.id2label)},
            label2id=label_converter.label2id)
        self.model = PhrExtModel(config)
        self.num_labels = config.num_labels
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.valid_metric = evaluate.load('seqeval')
        self.test_metric = evaluate.load('seqeval')    
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def __postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[ self.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        predictions = logits.argmax(dim=-1)

        decoded_preds, decoded_labels = self.__postprocess(predictions, labels)
        self.valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.valid_metric.compute()
        self.log('valid/precision', results['overall_precision'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid/recall', results['overall_recall'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid/f1', results['overall_f1'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid/accuracy', results['overall_accuracy'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        predictions = logits.argmax(dim=-1)

        decoded_preds, decoded_labels = self.__postprocess(predictions, labels)
        self.test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def test_epoch_end(self, outputs):
        results = self.test_metric.compute()
        self.log('test/precision', results['overall_precision'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('test/recall', results['overall_recall'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('test/f1', results['overall_f1'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('test/accuracy', results['overall_accuracy'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer