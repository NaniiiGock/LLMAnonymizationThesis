# model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pad_sequence

class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]

        # Select only original tokens from subword-level output
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)
        ]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        padded_sequence_output = self.dropout(padded_sequence_output)

        logits = self.classifier(padded_sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_mask = labels.ne(-100)
            active_logits = logits[loss_mask]
            active_labels = labels[loss_mask]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs

