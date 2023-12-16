from __future__ import absolute_import
import torch
import torch.nn as nn
from beam_algo import Beam

class GitCommitGeneratorModel(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, config, beam_size, max_length, sos_id, eos_id):
        super(GitCommitGeneratorModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.tokenizer = tokenizer
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048))) #
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tie_weights()


    def generate(self, source_ids, source_mask,target_ids, target_mask):
        # encode the source sequence
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_outputs = outputs[0].permute(1,0,2).contiguous() # convert to batch_size, seq_len, hidden_size

        attn_mask = -10e4 * (1 - self.bias[:target_ids.size(1), :target_ids.size(1)])
        target_embeddings = self.encoder.embeddings(target_ids).permute(1,0,2).contiguous() # convert to batch_size, seq_len, hidden_size

        output = self.decoder(target_embeddings, encoder_outputs, tgt_mask=attn_mask, memory_key_padding_mask= (1 - source_mask).bool())
        hidden_states = torch.tanh(self.dense(output)).permute(1,0,2).contiguous() 
        lm_logits = self.lm_head(hidden_states)

        # shift decoder target ids to the right, so that it predicts the next token
        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss], shift_labels.view(-1)[active_loss])

        final_result = loss, loss*active_loss.sum(), active_loss.sum()
        return final_result


    def predict(self, source_ids, source_mask, target_ids, target_mask):
        predictions = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_outputs = outputs[0].permute(1,0,2).contiguous()
        for i in range(source_ids.shape[0]):
            context = encoder_outputs[:, i:i+1]
            context_mask = source_mask[i:i+1, :]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context = context.repeat(1, self.beam_size, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)

            for _ in range(self.max_length):
                if beam.done():
                    break

                attn_mask = -10e4 * (1 - self.bias[:input_ids.size(1), :input_ids.size(1)])
                target_embeddings = self.encoder.embeddings(input_ids).permute(1,0,2).contiguous()
                output = self.decoder(target_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask= (1 - context_mask).bool())
                hidden_states = torch.tanh(self.dense(output)).permute(1,0,2).contiguous()
                softmax_output = self.softmax(self.lm_head(hidden_states)).data
                beam.advance(softmax_output)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat([input_ids, beam.getCurrentState()], -1)

            hyps = beam.getHyp(beam.getFinal())
            preds = beam.buildTargetTokens(hyps[0])[:self.beam_size]
            preds = [torch.cat([x.view(-1) for x in p]+[zero]*(self.beam_size-len(p))).view(1, -1) for p in preds]
            predictions.append(torch.cat(preds, 0).unsqueeze(0))

        predictions = torch.cat(predictions, 0)
        return predictions


    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
        """
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, source_ids, target_ids=None, source_mask=None, target_mask=None):
        if target_ids is not None:
            return self.generate(source_ids, source_mask, target_ids, target_mask)
        else:
            return self.predict(source_ids, source_mask, target_ids, target_mask)