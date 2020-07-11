# /usr/bin/env python
# coding=utf-8
"""model"""

import torch
import torch.nn as nn
from transformers import ElectraPreTrainedModel, ElectraModel, BertPreTrainedModel, RobertaModel

"""表示标签开始和结束，用于CRF"""
START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"


def log_sum_exp(tensor: torch.Tensor,
                dim: int = -1,
                keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class CRFLayer(nn.Module):
    def __init__(self, tag_size, params):
        super(CRFLayer, self).__init__()

        # transition[i][j] means transition probability from j to i
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size), requires_grad=True)
        self.tags = params.tags
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        # 重置transition参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置transition参数
        """
        nn.init.xavier_normal_(self.transition)
        # initialize START_TAG, END_TAG probability in log space
        # 从i到start和从end到i的score都应该为负
        self.transition.detach()[self.tag2idx[START_TAG], :] = -10000
        self.transition.detach()[:, self.tag2idx[END_TAG]] = -10000

    def forward(self, feats, mask):
        """求total scores of all the paths
        Arg:
          feats: tag概率分布. (seq_len, batch_size, tag_size)
          mask: 填充. (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize alpha to zero in log space
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        # alpha in START_TAG is 1
        alpha[:, self.tag2idx[START_TAG]] = 0

        # 取当前step的emit score
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1)
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            transition_score = self.transition.unsqueeze(0)  # (1, tag_size, tag_size)
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size)
            alpha_score = alpha_score + transition_score + emit_score  # (batch_size, tag_size, tag_size)
            # log_sum_exp along current_tag dimension to get next_tag alpha
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            # 累加每次的alpha
            alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * torch.logical_not(mask_t)  # (batch_size, tag_size)
        # arrive at END_TAG
        alpha = alpha + self.transition[self.tag2idx[END_TAG]].unsqueeze(0)  # (batch_size, tag_size)

        return log_sum_exp(alpha, -1)  # (batch_size, )

    def score_sentence(self, feats, tags, mask):
        """求gold score
        Arg:
          feats: (seq_len, batch_size, tag_size)
          tags: (seq_len, batch_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=self.tag2idx[START_TAG]), tags],
                         0)  # (seq_len + 1, batch_size)
        # 取一个step
        for t, feat in enumerate(feats):
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])  # (batch_size,)
            transition_score = torch.stack(
                [self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])  # (batch_size,)
            # 累加
            scores += (emit_score + transition_score) * mask[t]
        # 到end的score
        transition_to_end = torch.stack(
            [self.transition[self.tag2idx[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in
             enumerate(tags.transpose(0, 1))])
        scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        """维特比算法，解码最佳路径
        :param feats: (seq_len, batch_size, tag_size)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize scores in log space
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, self.tag2idx[START_TAG]] = 0
        pointers = []

        # forward
        # 取一个step
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # (bat, 1, tag_size) + (1, tag_size, tag_size)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * torch.logical_not(mask_t)
        pointers = torch.stack(pointers, 0)  # (seq_len, batch_size, tag_size)
        scores += self.transition[self.tag2idx[END_TAG]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )

        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist()  # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (tag_size, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
        return best_path


class BiLSTMCRF(nn.Module):
    def __init__(self, tag_size, embedding_size, hidden_size, num_layers, dropout, with_ln):
        super(BiLSTMCRF, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # set multi-lstm dropout
        self.multi_dropout = 0. if num_layers == 1 else dropout
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=self.multi_dropout,
                              bidirectional=True)

        self.with_ln = with_ln
        if with_ln:
            self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
        # 标签动态权重
        # self.tag_dy_weight = nn.Parameter(torch.ones((1, 1, tag_size)),
        #                                   requires_grad=True)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.hidden2tag.weight)
        # nn.init.xavier_normal_(self.tag_dy_weight)

    def get_lstm_features(self, embed, mask):
        """
        :param seq: (seq_len, batch_size, embedding_size)
        :param mask: (seq_len, batch_size)
        :return lstm_features: (seq_len, batch_size, tag_size)
        """
        embed = self.dropout(embed)
        max_len, _, __ = embed.size()
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed)  # (seq_len, batch_size, hidden_size*2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len)
        lstm_output = lstm_output * mask.unsqueeze(-1)
        if self.with_ln:
            lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        # lstm_features *= self.tag_dy_weight
        # print(self.tag_dy_weight)
        return lstm_features


class ElectraForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        # 实体类别数
        self.num_labels = len(params.tags)
        # electra
        self.bert = RobertaModel(config)

        # 动态权重
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)
        nn.init.xavier_normal_(self.dym_weight)

        self.bilstm = BiLSTMCRF(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                                num_layers=params.lstm_layer,
                                dropout=params.drop_prob, with_ln=True)
        # crf
        self.crf = CRFLayer(self.num_labels, params)

        # for dym's dense
        # self.dym_dense = nn.Linear(config.hidden_size, 1)
        # self.dym_output = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(params.drop_prob))
        # utils.initial_parameter([self.dym_dense, self.dym_output])

        self.init_weights()

    def get_dym_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[2][1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    # def get_dym_layer(self, outputs):
    #     """Bert output融合(seq_len维度)
    #     :param outputs: bert output.
    #     :return: pooled_output: (batch_size, seq_len, hidden_size)
    #     """
    #     # get all outputs
    #     all_encoder_layers = outputs[1][1:]  # (12, batch_size, sequence_length, hidden_size)
    #     # get distribution
    #     layer_logits = [self.dym_dense(out) for out in all_encoder_layers]  # (batch_size, sequence_length, 1)
    #     layer_logits = torch.cat(layer_logits, dim=2)  # (batch_size, sequence_length, 12)
    #     layer_dist = F.softmax(layer_logits, dim=-1)  # (batch_size, sequence_length, 12)
    #
    #     # concat outputs
    #     # (batch_size, sequence_length, 12, hidden_size)
    #     seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
    #     # get all distribution about seq_len
    #     # (batch_size, seq_len, 1, 12) * (batch_size, seq_len, 12, hidden_size)
    #     pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
    #     pooled_output = torch.squeeze(pooled_output, 2)  # (batch_size, seq_len, hidden_size)
    #     pooled_output = self.dym_output(pooled_output)
    #     return pooled_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        """
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: 各元素的值为0或1，避免在padding的token上计算attention。(batch_size, seq_len)
        :param token_type_ids: 就是token对应的句子类型id，值为0或1。为空自动生成全0。(batch_size, seq_len)
        :param position_ids: 位置编码。为空自动根据句子长度生成。(batch_size, sequence_length)
        :param head_mask: 各元素的值为0或1。为空自动生成全1，即不mask。(num_heads,) or (num_layers, num_heads)
        :param inputs_embeds: 与input_ids互斥。(batch_size, seq_len, embedding_dim)
        :param labels: (batch_size, seq_len)

    Returns:
        loss: scores对应的交叉熵损失
            returned when ``labels`` is provided)
            Classification loss.
        scores: (batch_size, sequence_length, config.num_labels)
            Classification scores (before SoftMax).
        hidden_states (Tuple): embedding层的输出和各层encoder的输出
            one for the output of the embeddings + one for the output of each layer
            returned when ``config.output_hidden_states=True`` (batch_size, sequence_length, hidden_size)
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (Tuple): 各层encoder中，各attention head的self-attention概率
            returned when ``config.output_attentions=True`` (batch_size, num_heads, sequence_length, sequence_length)
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        # pretrain model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        # middle
        sequence_output = self.get_dym_layer(outputs)  # (batch_size, seq_len, hidden_size[embedding_dim])
        # (seq_len, batch_size, tag_size)
        lstm_feats = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))

        # CRF
        if labels is not None:
            # total scores
            forward_score = self.crf(lstm_feats, attention_mask.transpose(1, 0))
            gold_score = self.crf.score_sentence(lstm_feats, labels.transpose(1, 0),
                                                 attention_mask.transpose(1, 0))
            loss = (forward_score - gold_score).sum()
            return loss
        else:
            # 维特比算法
            best_paths = self.crf.viterbi_decode(lstm_feats, attention_mask.transpose(1, 0))
            return best_paths


if __name__ == '__main__':
    from transformers import ElectraConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    config = ElectraConfig.from_pretrained(params.bert_model_dir / 'config.json', output_hidden_states=True)
    model = ElectraForTokenClassification.from_pretrained(params.bert_model_dir,
                                                          config=config, params=params)

    for n, _ in model.named_parameters():
        print(n)
