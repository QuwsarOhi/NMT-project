from torchtext.prototype.models import T5_SMALL_GENERATION, T5Transform, T5Model
from torch import Tensor
from torch.nn import Module
from typing import List, Optional
import torch
import torch.nn.functional as F


# Link: https://pytorch.org/text/main/tutorials/t5_demo.html#model-preparation


class T5(Module):

    def __init__(self, max_seq_len, eos_idx, padding_idx, beam_size, t5_trans_path=None):
        
        super().__init__()

        # Model fetch
        self.t5_gen = T5_SMALL_GENERATION
        self.model = self.t5_gen.get_model()
        
        # Setting model-specific data transform
        self.t5_trans_path = t5_trans_path
        if self.t5_trans_path is None:
            self.t5_trans_path = "https://download.pytorch.org/models/text/t5_tokenizer_base.model"
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.beam_size = beam_size

        # encoder tokenizer
        self.transform = T5Transform(
            sp_model_path=self.t5_trans_path,
            max_seq_len=max_seq_len,
            eos_idx=eos_idx,
            padding_idx=self.padding_idx,
        )

    
    def forward(self, input:List[str], target:Optional[List[str]]=None) -> Tensor:

        encoder_tokens = self.transform(input)

        output = self.generate(encoder_tokens=encoder_tokens, 
                               eos_idx=self.eos_idx, 
                               model=self.model,
                               beam_size=self.beam_size)

        #print('encoder_tokens:', encoder_tokens)
        
        
        if target is not None:
            target_tokens = self.transform(target)
            #print('target_tokens:', target_tokens)
            for out, tar in zip(output, target_tokens):
                print('out:', out)
                print('target:', tar)
        #else:
        #    print('output:', output)

        return output

    
    def predict(self, input:List[str]) -> List[str]:

        output = self.forward(input)
        decoded_output = self.transform.decode(output.tolist())

        return decoded_output


    def generate(self, encoder_tokens: Tensor, eos_idx: int, model: T5Model, beam_size: int) -> Tensor:

        # pass tokens through encoder
        bsz = encoder_tokens.size(0)
        encoder_padding_mask = encoder_tokens.eq(model.padding_idx)
        encoder_embeddings = model.dropout1(model.token_embeddings(encoder_tokens))
        encoder_output = model.encoder(encoder_embeddings, tgt_key_padding_mask=encoder_padding_mask)[0]

        encoder_output = model.norm1(encoder_output)
        encoder_output = model.dropout2(encoder_output)

        # initialize decoder input sequence; T5 uses padding index as starter index to decoder sequence
        decoder_tokens = torch.ones((bsz, 1), dtype=torch.long) * model.padding_idx
        scores = torch.zeros((bsz, beam_size))

        # mask to keep track of sequences for which the decoder has not produced an end-of-sequence token yet
        incomplete_sentences = torch.ones(bsz * beam_size, dtype=torch.long)

        # iteratively generate output sequence until all sequences in the batch have generated the end-of-sequence token
        for step in range(model.config.max_seq_len):

            if step == 1:
                # duplicate and order encoder output so that each beam is treated as its own independent sequence
                new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
                new_order = new_order.to(encoder_tokens.device).long()
                encoder_output = encoder_output.index_select(0, new_order)
                encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)

            # causal mask and padding mask for decoder sequence
            tgt_len = decoder_tokens.shape[1]
            decoder_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.float64), diagonal=1).bool()
            decoder_padding_mask = decoder_tokens.eq(model.padding_idx)

            # T5 implemention uses padding idx to start sequence. Want to ignore this when masking
            decoder_padding_mask[:, 0] = False

            # pass decoder sequence through decoder
            decoder_embeddings = model.dropout3(model.token_embeddings(decoder_tokens))
            decoder_output = model.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=decoder_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask,
            )[0]

            decoder_output = model.norm2(decoder_output)
            decoder_output = model.dropout4(decoder_output)
            decoder_output = decoder_output * (model.config.embedding_dim ** -0.5)
            decoder_output = model.lm_head(decoder_output)

            print(decoder_output.shape)

            decoder_tokens, scores, incomplete_sentences = self.beam_search(
                beam_size, step + 1, bsz, decoder_output, decoder_tokens, scores, incomplete_sentences
            )
            # ignore newest tokens for sentences that are already complete
            decoder_tokens[:, -1] *= incomplete_sentences

            # update incomplete_sentences to remove those that were just ended
            incomplete_sentences = incomplete_sentences - (decoder_tokens[:, -1] == eos_idx).long()

            # early stop if all sentences have been ended
            if (incomplete_sentences == 0).all():
                break

        # take most likely sequence
        decoder_tokens = decoder_tokens.view(bsz, beam_size, -1)[:, 0, :]
        return decoder_tokens
    

    def beam_search(
            self,
            beam_size: int,
            step: int,
            bsz: int,
            decoder_output: Tensor,
            decoder_tokens: Tensor,
            scores: Tensor,
            incomplete_sentences: Tensor,
        ):

        probs = F.log_softmax(decoder_output[:, -1], dim=-1)
        top = torch.topk(probs, beam_size)

        # N is number of sequences in decoder_tokens, L is length of sequences, B is beam_size
        # decoder_tokens has shape (N,L) -> (N,B,L)
        # top.indices has shape (N,B) - > (N,B,1)
        # x has shape (N,B,L+1)
        # note that when step == 1, N = batch_size, and when step > 1, N = batch_size * beam_size
        x = torch.cat([decoder_tokens.unsqueeze(1).repeat(1, beam_size, 1), top.indices.unsqueeze(-1)], dim=-1)

        # beams are first created for a given sequence
        if step == 1:
            # x has shape (batch_size, B, L+1) -> (batch_size * B, L+1)
            # new_scores has shape (batch_size,B)
            # incomplete_sentences has shape (batch_size * B) = (N)
            new_decoder_tokens = x.view(-1, step + 1)
            new_scores = top.values
            new_incomplete_sentences = incomplete_sentences

        # beams already exist, want to expand each beam into possible new tokens to add
        # and for all expanded beams beloning to the same sequences, choose the top k
        else:
            # scores has shape (batch_size,B) -> (N,1) -> (N,B)
            # top.values has shape (N,B)
            # new_scores has shape (N,B) -> (batch_size, B^2)
            new_scores = (scores.view(-1, 1).repeat(1, beam_size) + top.values).view(bsz, -1)

            # v, i have shapes (batch_size, B)
            v, i = torch.topk(new_scores, beam_size)

            # x has shape (N,B,L+1) -> (batch_size, B, L+1)
            # i has shape (batch_size, B) -> (batch_size, B, L+1)
            # new_decoder_tokens has shape (batch_size, B, L+1) -> (N, L)
            x = x.view(bsz, -1, step + 1)
            new_decoder_tokens = x.gather(index=i.unsqueeze(-1).repeat(1, 1, step + 1), dim=1).view(-1, step + 1)

            # need to update incomplete sentences in case one of the beams was kicked out
            # y has shape (N) -> (N, 1) -> (N, B) -> (batch_size, B^2)
            y = incomplete_sentences.unsqueeze(-1).repeat(1, beam_size).view(bsz, -1)

            # now can use i to extract those beams that were selected
            # new_incomplete_sentences has shape (batch_size, B^2) -> (batch_size, B) -> (N, 1) -> N
            new_incomplete_sentences = y.gather(index=i, dim=1).view(bsz * beam_size, 1).squeeze(-1)

            # new_scores has shape (batch_size, B)
            new_scores = v

        return new_decoder_tokens, new_scores, new_incomplete_sentences



if __name__ == '__main__':

    model = T5(max_seq_len=12, eos_idx=1, padding_idx=0, beam_size=1)

    inputs = [
        "translate English to German: Thank you so much, Chris.",
        "translate English to German: I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.",
        "translate German to English: Es ist mir wirklich eine Ehre, zweimal auf dieser Bühne stehen zu dürfen. Tausend Dank dafür.",
    ]

    targets = [
        "Vielen Dank, Chris.",
        "Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen für die vielen netten Kommentare zu meiner Rede vorgestern Abend.",
        "And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful.",
    ]


    model.forward(inputs, targets)

    outputs = model.predict(inputs)
    for (inp, out), tar in zip(zip(inputs, outputs), targets):
        print(f"Input: {inp}\nOutput: {out}\nTarget: {tar}\n")

