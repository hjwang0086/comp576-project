import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .transformer import TransformerEncoder, TransformerEncoderLayer


class MatchERT(nn.Module):
    def __init__(self, d_global, d_model, seq_len, d_K, nhead, num_encoder_layers, dim_feedforward, dropout, 
                activation, normalize_before, use_bottleneck=False, use_duplicate=False):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        # add bottleneck layer
        self.bottleneck = nn.Linear(128, d_model) if use_bottleneck else None
        self.use_duplicate = use_duplicate

        encoder_layer = TransformerEncoderLayer(d_model, seq_len, d_K, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.pos_encoder = PositionEmbeddingSine(d_model//2, normalize=True, scale=2.0)
        self.remap = nn.Linear(d_global, d_model)
        self.scale_encoder = nn.Embedding(7, d_model)
        self.seg_encoder = nn.Embedding(6, d_model)
        self.classifier = nn.Linear(d_model, 1)
        self._reset_parameters()
        self.d_model = d_model
        self.seq_len = seq_len
        self.d_K = d_K
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # Combine pretrain and finetune script together
    def forward(self, 
            src_global, src_local, src_mask, src_scales, src_positions,
            tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions,
            normalize=True, as_pretrained=False):
        
        if self.bottleneck:
            src_local = self.bottleneck(src_local)
            tgt_local = self.bottleneck(tgt_local)
        elif self.use_duplicate:
            src_local = src_local.repeat(1, 1, 2) # repeat along chn axis
            tgt_local = tgt_local.repeat(1, 1, 2) # repeat along chn axis
        
        if as_pretrained: # all embedded in src_local
            return self.forward_pretrain(src_local, src_mask, src_scales, src_positions, normalize)
        else:
            return self.forward_finetune(src_global, src_local, src_mask, src_scales, src_positions,
                tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions,
                normalize=normalize
            )

    # Daniel: added pretrained script
    def forward_pretrain(self, src_local, src_mask, src_scales, src_positions, normalize=True):

        if normalize:
            src_local  = F.normalize(src_local,  p=2, dim=-1)
        bsize, slen, fsize = src_local.size()
        
        ##################################################################################################################
        ## The final model does not use position embeddings
        src_local = src_local + self.scale_encoder(src_scales) # + self.pos_encoder(src_positions)
        src_local = src_local + self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long))
        ##################################################################################################################

        input_feats = src_local.permute(1,0,2)
        input_mask = src_mask

        outputs = self.encoder(input_feats, src_key_padding_mask=input_mask).permute(1,0,2)

        # take top-16 tokens
        outputs = outputs[:,:2048//128]
        outputs = outputs.reshape(-1, 2048)

        return outputs


    # original forward script
    def forward_finetune(self, 
            src_global, src_local, src_mask, src_scales, src_positions,
            tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions,
            normalize=True):
        # src: bsize, slen, fsize
        # tgt: bsize, slen, fsize
        src_global = self.remap(src_global)
        tgt_global = self.remap(tgt_global)
        if normalize:
            src_global = F.normalize(src_global, p=2, dim=-1)
            tgt_global = F.normalize(tgt_global, p=2, dim=-1)
            src_local  = F.normalize(src_local,  p=2, dim=-1)
            tgt_local  = F.normalize(tgt_local,  p=2, dim=-1)
        bsize, slen, fsize = src_local.size()

        ##################################################################################################################
        ## The final model does not use position embeddings
        src_local = src_local + self.scale_encoder(src_scales) # + self.pos_encoder(src_positions)
        tgt_local = tgt_local + self.scale_encoder(tgt_scales) # + self.pos_encoder(tgt_positions)
        ##################################################################################################################

        ##################################################################################################################
        src_local  = src_local + self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long))
        tgt_local  = tgt_local + self.seg_encoder(src_local.new_ones((bsize, 1), dtype=torch.long))
        cls_embed  = self.seg_encoder(2 * src_local.new_ones((bsize, 1), dtype=torch.long))
        sep_embed  = self.seg_encoder(3 * src_local.new_ones((bsize, 1), dtype=torch.long))
        src_global = src_global.unsqueeze(1) + self.seg_encoder(4 * src_local.new_ones((bsize, 1), dtype=torch.long))
        tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(5 * src_local.new_ones((bsize, 1), dtype=torch.long))
        ##################################################################################################################
        
        # input_feats = torch.cat([cls_embed, src_global, src_local, sep_embed, tgt_global, tgt_local], 1).permute(1,0,2)
        # input_mask = torch.cat([
        #     src_local.new_zeros((bsize, 2), dtype=torch.bool),
        #     src_mask,
        #     src_local.new_zeros((bsize, 2), dtype=torch.bool),
        #     tgt_mask
        # ], 1)

        # ablation: remove global feature: better!
        input_feats = torch.cat([cls_embed, src_local, sep_embed, tgt_local], 1).permute(1,0,2)
        input_mask = torch.cat([
            src_local.new_zeros((bsize, 1), dtype=torch.bool),
            src_mask,
            src_local.new_zeros((bsize, 1), dtype=torch.bool),
            tgt_mask
        ], 1)
        
        logits = self.encoder(input_feats, src_key_padding_mask=input_mask)
        logits = logits[0]
        return self.classifier(logits).view(-1)