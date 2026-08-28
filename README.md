# Multimodal Autoregressive Image Generator from Scratch in JAX

Build a text-conditioned image generator end to end in JAX by training a VQ-VAE to turn tiny images into discrete tokens, then an autoregressive transformer that models text-prefixed token sequences. You implement every piece from patch encoding and vector quantization to multi-head attention and classifier-free guided sampling.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** generate_toy_images
- [x] **2.** assign_image_labels
- [x] **3.** normalize_image_batch
- [x] **4.** split_image_into_patches
- [x] **5.** flatten_patches
- [x] **6.** init_patch_encoder
- [x] **7.** encode_patches
- [x] **8.** init_patch_decoder
- [x] **9.** decode_latents
- [x] **10.** reassemble_patches_into_image
- [x] **11.** init_codebook
- [x] **12.** squared_distance_to_codebook
- [x] **13.** grid_distances_to_codebook
- [x] **14.** assign_nearest_codes
- [x] **15.** lookup_codebook_vectors
- [x] **16.** straight_through_quantize
- [x] **17.** codebook_loss
- [x] **18.** commitment_loss
- [x] **19.** reconstruction_loss
- [x] **20.** total_vqvae_loss
- [x] **21.** vqvae_loss_and_grads
- [x] **22.** apply_vqvae_update
- [x] **23.** encode_image_to_tokens
- [x] **24.** flatten_token_grid
- [x] **25.** reshape_tokens_to_grid
- [ ] **26.** build_char_vocab
- [ ] **27.** encode_label_to_ids
- [ ] **28.** form_multimodal_sequence
- [ ] **29.** init_token_embedding
- [ ] **30.** init_positional_embedding
- [ ] **31.** lookup_token_embeddings
- [ ] **32.** add_positional_embeddings
- [ ] **33.** build_causal_mask
- [ ] **34.** layer_norm
- [ ] **35.** init_attention_params
- [ ] **36.** project_qkv
- [ ] **37.** reshape_to_heads
- [ ] **38.** scaled_dot_product_scores
- [ ] **39.** add_causal_mask_to_scores
- [ ] **40.** attention_weights_softmax
- [ ] **41.** weighted_sum_of_values
- [ ] **42.** merge_heads_and_project
- [ ] **43.** init_feedforward_params
- [ ] **44.** feedforward_mlp
- [ ] **45.** transformer_block
- [ ] **46.** transformer_backbone
- [ ] **47.** init_output_projection
- [ ] **48.** project_to_logits
- [ ] **49.** image_token_cross_entropy
- [ ] **50.** transformer_loss_and_grads
- [ ] **51.** apply_transformer_update
- [ ] **52.** drop_text_prefix
- [ ] **53.** combine_guided_logits
- [ ] **54.** logits_to_probabilities
- [ ] **55.** top_k_filter_logits
- [ ] **56.** sample_token_index
- [ ] **57.** generate_image_tokens
- [ ] **58.** decode_tokens_to_image
- [ ] **59.** next_token_accuracy
- [ ] **60.** average_reconstruction_error
- [ ] **61.** nearest_neighbor_distance_to_dataset
- [ ] **62.** train_vqvae_on_toy_images
- [ ] **63.** train_transformer_on_token_sequences
- [ ] **64.** generate_image_from_label

---

Built on Deep-ML.
