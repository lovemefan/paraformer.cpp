```
SeacoParaformer(
  (specaug): SpecAugLFR(
    (freq_mask): MaskAlongAxisLFR(mask_width_range=[0, 30], num_mask=1, axis=freq)
    (time_mask): MaskAlongAxisLFR(mask_width_range=[0, 12], num_mask=1, axis=time)
  )
  (encoder): SANMEncoder(
    (embed): SinusoidalPositionEncoder()
    (encoders0): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=560, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((560,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (encoders): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (5): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (6): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (7): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (8): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (9): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (10): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (11): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (12): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (13): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (14): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (15): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (16): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (17): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (18): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (19): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (20): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (21): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (22): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (23): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (24): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (25): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (26): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (27): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (28): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (29): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (30): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (31): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (32): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (33): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (34): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (35): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (36): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (37): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (38): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (39): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (40): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (41): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (42): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (43): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (44): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (45): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (46): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (47): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (48): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (linear_q_k_v): Linear(in_features=512, out_features=1536, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (after_norm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): ParaformerSANMDecoder(
    (embed): Sequential(
      (0): Embedding(11666, 512)
    )
    (after_norm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
    (output_layer): Linear(in_features=512, out_features=11666, bias=True)
    (decoders): MultiSequential(
      (0): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (5): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (6): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (7): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (8): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (9): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (10): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (11): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (12): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (13): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (14): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (15): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(11,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (decoders3): MultiSequential(
      (0): DecoderLayerSANM(
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=2048, bias=True)
          (w_2): Linear(in_features=2048, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((2048,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (criterion_att): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
  (predictor): CifPredictorV2(
    (pad): ConstantPad1d(padding=(1, 1), value=0)
    (cif_conv1d): Conv1d(512, 512, kernel_size=(3,), stride=(1,))
    (cif_output): Linear(in_features=512, out_features=1, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (criterion_pre): mae_loss(
    (criterion): L1Loss()
  )
  (bias_encoder): LSTM(512, 512, num_layers=2, batch_first=True)
  (seaco_decoder): ParaformerSANMDecoder(
    (embed): None
    (after_norm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
    (decoders): MultiSequential(
      (0): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (5): DecoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMDecoder(
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(512, 512, kernel_size=(21,), stride=(1,), groups=512, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 10), value=0.0)
        )
        (src_attn): MultiHeadedAttentionCrossAtt(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k_v): Linear(in_features=512, out_features=1024, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (decoders3): MultiSequential(
      (0): DecoderLayerSANM(
        (feed_forward): PositionwiseFeedForwardDecoderSANM(
          (w_1): Linear(in_features=512, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=512, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
          (norm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
        )
        (norm1): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (hotword_output_layer): Linear(in_features=512, out_features=11666, bias=True)
  (criterion_seaco): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
)
```

## 参数

| key                                                  | n dims     | pre type      | convert type | param size |
|------------------------------------------------------|------------|---------------|--------------|------------|
| encoder.encoders0.0.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders0.0.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders0.0.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 860160     |
| encoder.encoders0.0.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders0.0.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders0.0.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders0.0.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders0.0.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders0.0.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders0.0.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 560        |
| encoder.encoders0.0.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 560        |
| encoder.encoders0.0.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders0.0.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.0.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.0.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.0.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.0.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.0.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.0.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.0.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.0.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.1.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.1.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.1.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.1.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.1.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.1.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.1.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.1.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.2.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.2.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.2.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.2.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.2.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.2.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.2.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.2.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.3.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.3.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.3.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.3.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.3.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.3.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.3.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.3.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.4.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.4.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.4.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.4.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.4.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.4.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.4.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.4.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.5.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.5.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.5.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.5.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.5.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.5.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.5.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.5.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.6.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.6.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.6.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.6.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.6.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.6.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.6.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.6.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.7.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.7.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.7.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.7.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.7.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.7.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.7.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.7.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.8.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.8.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.8.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.8.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.8.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.8.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.8.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.8.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.self_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.9.self_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.self_attn.linear_q_k_v.weight     | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.9.self_attn.linear_q_k_v.bias       | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.9.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.9.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.9.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.9.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.9.feed_forward.w_2.bias             | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.9.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.10.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.10.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.10.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.10.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.10.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.10.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.10.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.10.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.11.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.11.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.11.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.11.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.11.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.11.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.11.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.11.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.12.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.12.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.12.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.12.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.12.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.12.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.12.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.12.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.13.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.13.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.13.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.13.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.13.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.13.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.13.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.13.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.14.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.14.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.14.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.14.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.14.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.14.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.14.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.14.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.15.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.15.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.15.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.15.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.15.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.15.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.15.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.15.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.16.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.16.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.16.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.16.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.16.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.16.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.16.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.16.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.17.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.17.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.17.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.17.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.17.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.17.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.17.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.17.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.18.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.18.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.18.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.18.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.18.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.18.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.18.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.18.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.19.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.19.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.19.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.19.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.19.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.19.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.19.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.19.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.20.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.20.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.20.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.20.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.20.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.20.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.20.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.20.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.21.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.21.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.21.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.21.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.21.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.21.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.21.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.21.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.22.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.22.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.22.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.22.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.22.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.22.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.22.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.22.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.23.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.23.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.23.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.23.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.23.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.23.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.23.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.23.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.24.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.24.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.24.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.24.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.24.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.24.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.24.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.24.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.25.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.25.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.25.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.25.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.25.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.25.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.25.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.25.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.26.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.26.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.26.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.26.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.26.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.26.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.26.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.26.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.27.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.27.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.27.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.27.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.27.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.27.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.27.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.27.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.28.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.28.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.28.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.28.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.28.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.28.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.28.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.28.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.29.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.29.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.29.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.29.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.29.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.29.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.29.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.29.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.30.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.30.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.30.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.30.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.30.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.30.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.30.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.30.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.31.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.31.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.31.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.31.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.31.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.31.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.31.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.31.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.32.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.32.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.32.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.32.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.32.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.32.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.32.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.32.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.33.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.33.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.33.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.33.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.33.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.33.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.33.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.33.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.34.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.34.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.34.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.34.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.34.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.34.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.34.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.34.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.35.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.35.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.35.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.35.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.35.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.35.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.35.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.35.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.36.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.36.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.36.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.36.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.36.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.36.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.36.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.36.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.37.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.37.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.37.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.37.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.37.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.37.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.37.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.37.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.38.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.38.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.38.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.38.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.38.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.38.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.38.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.38.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.39.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.39.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.39.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.39.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.39.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.39.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.39.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.39.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.40.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.40.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.40.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.40.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.40.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.40.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.40.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.40.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.41.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.41.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.41.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.41.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.41.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.41.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.41.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.41.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.42.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.42.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.42.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.42.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.42.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.42.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.42.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.42.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.43.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.43.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.43.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.43.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.43.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.43.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.43.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.43.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.44.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.44.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.44.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.44.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.44.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.44.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.44.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.44.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.45.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.45.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.45.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.45.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.45.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.45.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.45.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.45.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.46.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.46.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.46.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.46.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.46.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.46.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.46.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.46.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.47.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.47.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.47.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.47.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.47.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.47.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.47.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.47.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.self_attn.linear_out.weight      | n_dims = 2 | torch.float32 | float16      | 262144     |
| encoder.encoders.48.self_attn.linear_out.bias        | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.self_attn.linear_q_k_v.weight    | n_dims = 2 | torch.float32 | float16      | 786432     |
| encoder.encoders.48.self_attn.linear_q_k_v.bias      | n_dims = 1 | torch.float32 | float32      | 1536       |
| encoder.encoders.48.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| encoder.encoders.48.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.48.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| encoder.encoders.48.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| encoder.encoders.48.feed_forward.w_2.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.encoders.48.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.after_norm.weight                            | n_dims = 1 | torch.float32 | float32      | 512        |
| encoder.after_norm.bias                              | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.embed.0.weight                               | n_dims = 2 | torch.float32 | float16      | 4302848    |
| decoder.after_norm.weight                            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.after_norm.bias                              | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.output_layer.weight                          | n_dims = 2 | torch.float32 | float16      | 4302848    |
| decoder.output_layer.bias                            | n_dims = 1 | torch.float32 | float32      | 8404       |
| decoder.decoders.0.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.0.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.0.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.0.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.0.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.0.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.0.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.0.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.0.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.0.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.0.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.0.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.1.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.1.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.1.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.1.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.1.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.1.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.1.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.1.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.1.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.1.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.1.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.2.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.2.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.2.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.2.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.2.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.2.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.2.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.2.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.2.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.2.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.2.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.3.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.3.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.3.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.3.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.3.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.3.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.3.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.3.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.3.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.3.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.3.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.4.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.4.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.4.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.4.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.4.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.4.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.4.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.4.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.4.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.4.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.4.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.5.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.5.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.5.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.5.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.5.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.5.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.5.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.5.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.5.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.5.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.5.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.6.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.6.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.6.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.6.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.6.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.6.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.6.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.6.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.6.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.6.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.6.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.7.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.7.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.7.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.7.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.7.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.7.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.7.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.7.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.7.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.7.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.7.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.8.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.8.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.8.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.8.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.8.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.8.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.8.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.8.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.8.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.8.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.8.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.self_attn.fsmn_block.weight       | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.9.src_attn.linear_q.weight          | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.9.src_attn.linear_q.bias            | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.src_attn.linear_k_v.weight        | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.9.src_attn.linear_k_v.bias          | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.9.src_attn.linear_out.weight        | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.9.src_attn.linear_out.bias          | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.feed_forward.w_1.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.9.feed_forward.w_1.bias             | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.9.feed_forward.w_2.weight           | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.9.feed_forward.norm.weight          | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.9.feed_forward.norm.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.9.norm1.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.norm1.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.norm2.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.norm2.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.norm3.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.9.norm3.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.10.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.10.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.10.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.10.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.10.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.10.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.10.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.10.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.10.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.10.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.10.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.11.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.11.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.11.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.11.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.11.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.11.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.11.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.11.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.11.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.11.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.11.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.12.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.12.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.12.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.12.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.12.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.12.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.12.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.12.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.12.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.12.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.12.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.13.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.13.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.13.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.13.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.13.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.13.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.13.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.13.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.13.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.13.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.13.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.14.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.14.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.14.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.14.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.14.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.14.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.14.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.14.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.14.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.14.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.14.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.self_attn.fsmn_block.weight      | n_dims = 2 | torch.float32 | float16      | 5632       |
| decoder.decoders.15.src_attn.linear_q.weight         | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.15.src_attn.linear_q.bias           | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.src_attn.linear_k_v.weight       | n_dims = 2 | torch.float32 | float16      | 524288     |
| decoder.decoders.15.src_attn.linear_k_v.bias         | n_dims = 1 | torch.float32 | float32      | 1024       |
| decoder.decoders.15.src_attn.linear_out.weight       | n_dims = 2 | torch.float32 | float16      | 262144     |
| decoder.decoders.15.src_attn.linear_out.bias         | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.15.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.15.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders.15.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.15.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders.15.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.norm2.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.norm2.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.norm3.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders.15.norm3.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders3.0.feed_forward.w_1.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders3.0.feed_forward.w_1.bias            | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders3.0.feed_forward.w_2.weight          | n_dims = 2 | torch.float32 | float16      | 1048576    |
| decoder.decoders3.0.feed_forward.norm.weight         | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders3.0.feed_forward.norm.bias           | n_dims = 1 | torch.float32 | float32      | 2048       |
| decoder.decoders3.0.norm1.weight                     | n_dims = 1 | torch.float32 | float32      | 512        |
| decoder.decoders3.0.norm1.bias                       | n_dims = 1 | torch.float32 | float32      | 512        |
| predictor.cif_conv1d.weight                          | n_dims = 3 | torch.float32 | float32      | 786432     |
| predictor.cif_conv1d.bias                            | n_dims = 1 | torch.float32 | float32      | 512        |
| predictor.cif_output.weight                          | n_dims = 1 | torch.float32 | float32      | 512        |
| predictor.cif_output.bias                            | n_dims = 1 | torch.float32 | float32      | 1          |
| bias_encoder.weight_ih_l0                            | n_dims = 2 | torch.float32 | float32      | 1048576    |
| bias_encoder.weight_hh_l0                            | n_dims = 2 | torch.float32 | float32      | 1048576    |
| bias_encoder.bias_ih_l0                              | n_dims = 1 | torch.float32 | float32      | 2048       |
| bias_encoder.bias_hh_l0                              | n_dims = 1 | torch.float32 | float32      | 2048       |
| bias_encoder.weight_ih_l1                            | n_dims = 2 | torch.float32 | float32      | 1048576    |
| bias_encoder.weight_hh_l1                            | n_dims = 2 | torch.float32 | float32      | 1048576    |
| bias_encoder.bias_ih_l1                              | n_dims = 1 | torch.float32 | float32      | 2048       |
| bias_encoder.bias_hh_l1                              | n_dims = 1 | torch.float32 | float32      | 2048       |
| hotword_output_layer.weight                          | n_dims = 2 | torch.float32 | float16      | 4302848    |
| hotword_output_layer.bias                            | n_dims = 1 | torch.float32 | float32      | 8404       |
| seaco_decoder.after_norm.weight                      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.after_norm.bias                        | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.0.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.0.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.0.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.0.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.0.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.0.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.0.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.0.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.0.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.0.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.0.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.1.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.1.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.1.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.1.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.1.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.1.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.1.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.1.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.1.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.1.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.1.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.2.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.2.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.2.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.2.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.2.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.2.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.2.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.2.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.2.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.2.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.2.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.3.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.3.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.3.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.3.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.3.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.3.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.3.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.3.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.3.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.3.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.3.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.4.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.4.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.4.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.4.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.4.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.4.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.4.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.4.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.4.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.4.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.4.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.self_attn.fsmn_block.weight | n_dims = 2 | torch.float32 | float16      | 10752      |
| seaco_decoder.decoders.5.src_attn.linear_q.weight    | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.5.src_attn.linear_q.bias      | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.src_attn.linear_k_v.weight  | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.5.src_attn.linear_k_v.bias    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.5.src_attn.linear_out.weight  | n_dims = 2 | torch.float32 | float16      | 262144     |
| seaco_decoder.decoders.5.src_attn.linear_out.bias    | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.feed_forward.w_1.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.5.feed_forward.w_1.bias       | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.5.feed_forward.w_2.weight     | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders.5.feed_forward.norm.weight    | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.5.feed_forward.norm.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders.5.norm1.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.norm1.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.norm2.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.norm2.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.norm3.weight                | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders.5.norm3.bias                  | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders3.0.feed_forward.w_1.weight    | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders3.0.feed_forward.w_1.bias      | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders3.0.feed_forward.w_2.weight    | n_dims = 2 | torch.float32 | float16      | 524288     |
| seaco_decoder.decoders3.0.feed_forward.norm.weight   | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders3.0.feed_forward.norm.bias     | n_dims = 1 | torch.float32 | float32      | 1024       |
| seaco_decoder.decoders3.0.norm1.weight               | n_dims = 1 | torch.float32 | float32      | 512        |
| seaco_decoder.decoders3.0.norm1.bias                 | n_dims = 1 | torch.float32 | float32      | 512        |
| predictor.upsample_cnn.weight                        | n_dims = 3 | torch.float32 | float32      | 786432     |
| predictor.upsample_cnn.bias                          | n_dims = 1 | torch.float32 | float32      | 512        |
| predictor.blstm.weight_ih_l0                         | n_dims = 2 | torch.float32 | float32      | 1048576    |
| predictor.blstm.weight_hh_l0                         | n_dims = 2 | torch.float32 | float32      | 1048576    |
| predictor.blstm.bias_ih_l0                           | n_dims = 1 | torch.float32 | float32      | 2048       |
| predictor.blstm.bias_hh_l0                           | n_dims = 1 | torch.float32 | float32      | 2048       |
| predictor.blstm.weight_ih_l0_reverse                 | n_dims = 2 | torch.float32 | float32      | 1048576    |
| predictor.blstm.weight_hh_l0_reverse                 | n_dims = 2 | torch.float32 | float32      | 1048576    |
| predictor.blstm.bias_ih_l0_reverse                   | n_dims = 1 | torch.float32 | float32      | 2048       |
| predictor.blstm.bias_hh_l0_reverse                   | n_dims = 1 | torch.float32 | float32      | 2048       |
| predictor.cif_output2.weight                         | n_dims = 1 | torch.float32 | float32      | 1024       |
| predictor.cif_output2.bias                           | n_dims = 1 | torch.float32 | float32      | 1          |

## 总结

与前一代contextual热词模型的区别：

1. decoder 由原来的14层变为现有的15层
2. bias encoder由原来的一层LSTM变成2层LSTM， 删掉了LSTM的embedding
3. 将原有的contextual_decoder替换为ParaformerSANMDecoder，区别在于DecoderLayerSANM从1层加深到6层
