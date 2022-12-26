from transformers import PreTrainedModel, RobertaModel, BertModel
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from PhrExt import PhrExtConfig

class PhrExtPreTrainedModel(PreTrainedModel):
    config_class = PhrExtConfig
    base_model_prefix = "phrext"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (MarianDecoder, MarianEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs

class PhrExtModel(PhrExtPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: PhrExtConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layers_use_from_last = config.layers_use_from_last
        self.method_for_layers = config.method_for_layers

        if config.pretrained_ck == "":
            self.roberta = RobertaModel(config, add_pooling_layer=False)
        else:
            self.roberta = RobertaModel.from_pretrained(config.pretrained_ck, add_pooling_layer=False, output_hidden_states=True)
            config.pretrained_ck = ""

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.normalize = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size//2, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        list_sequence_output = outputs.hidden_states[(-1)*self.layers_use_from_last:]
        if self.method_for_layers == 'sum':
            sequence_output = torch.stack(list_sequence_output).sum(0)
        else:
            sequence_output = torch.stack(list_sequence_output).mean(0)

        intermidiate = self.act(self.normalize(self.dropout1(sequence_output)))
        logits = self.classifier(self.dropout1(intermidiate))
        return logits