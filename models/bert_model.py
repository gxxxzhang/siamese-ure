import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel


class BertForsiaURE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.output_emebedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, e1_pos=None, e2_pos=None,
                sentence_data=None):

        if sentence_data is not None:
            device = torch.device("cuda")
            input_ids = sentence_data[0].to(device)
            attention_mask = sentence_data[1].to(device)
            labels = sentence_data[2].to(device)
            e1_pos = sentence_data[3].to(device)
            e2_pos = sentence_data[4].to(device)


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            # data augmentation part
            e_pos_outputs.append(e_pos_output_i)
        e_pos_output = torch.stack(e_pos_outputs)

        self.output_emebedding = e_pos_output

        e_pos_output = self.dropout(e_pos_output)
        return e_pos_output


class RelationClassification(BertForsiaURE):
    def __init__(self, config):
        super().__init__(config)

