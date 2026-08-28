"""
Multimodal Autoregressive Image Generator from Scratch in JAX scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Multimodal autoregressive image generator in JAX."""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from solution import (
    generate_toy_images, assign_image_labels, normalize_image_batch,
    split_image_into_patches, flatten_patches, init_patch_encoder, encode_patches,
    init_patch_decoder, decode_latents, reassemble_patches_into_image,
    init_codebook, squared_distance_to_codebook, grid_distances_to_codebook,
    assign_nearest_codes, lookup_codebook_vectors, straight_through_quantize,
    codebook_loss, commitment_loss, reconstruction_loss, total_vqvae_loss,
    vqvae_loss_and_grads, apply_vqvae_update, encode_image_to_tokens,
    flatten_token_grid, reshape_tokens_to_grid, build_char_vocab,
    encode_label_to_ids, form_multimodal_sequence, init_token_embedding,
    init_positional_embedding, lookup_token_embeddings, add_positional_embeddings,
    build_causal_mask, layer_norm, init_attention_params, project_qkv,
    reshape_to_heads, scaled_dot_product_scores, add_causal_mask_to_scores,
    attention_weights_softmax, weighted_sum_of_values, merge_heads_and_project,
    init_feedforward_params, feedforward_mlp, transformer_block,
    transformer_backbone, init_output_projection, project_to_logits,
    image_token_cross_entropy, transformer_loss_and_grads, apply_transformer_update,
    drop_text_prefix, combine_guided_logits, logits_to_probabilities,
    top_k_filter_logits, sample_token_index, generate_image_tokens,
    decode_tokens_to_image, next_token_accuracy, average_reconstruction_error,
    nearest_neighbor_distance_to_dataset, train_vqvae_on_toy_images,
    train_transformer_on_token_sequences, generate_image_from_label,
)


def main():
    np.random.seed(0)
    key = jax.random.PRNGKey(0)

    # train_vqvae_on_toy_images hardcodes patch_size=2, so align here.
    image_size, patch_size = 8, 2
    grid_h = grid_w = image_size // patch_size
    num_image_tokens = grid_h * grid_w
    patch_dim = patch_size * patch_size
    latent_dim, num_codes = 8, 16
    # train_transformer_on_token_sequences hardcodes num_heads=1 internally.
    text_len, d_model, num_heads, d_ff = 6, 32, 1, 64

    k_data, k_enc, k_dec, k_cb, key = jax.random.split(key, 5)
    images = generate_toy_images(k_data, num_images=16, image_size=image_size)
    images = normalize_image_batch(images)
    labels = assign_image_labels(images)
    vocab = build_char_vocab(labels)
    print("num images:", len(images), "| labels sample:", labels[:3])
    print("char vocab size:", len(vocab))

    vqvae_params = {
        "encoder": init_patch_encoder(k_enc, patch_dim, latent_dim),
        "decoder": init_patch_decoder(k_dec, latent_dim, patch_dim),
        "codebook": init_codebook(k_cb, num_codes, latent_dim),
    }
    vq_opt = optax.adam(1e-2)
    vq_state = vq_opt.init(vqvae_params)
    vqvae_params, codebook, vq_state, vq_losses = train_vqvae_on_toy_images(
        images, vqvae_params, vqvae_params["codebook"], vq_state, vq_opt,
        num_steps=20
    )
    # trainer strips codebook out of params; put it back for encode_image_to_tokens
    vqvae_params = {"encoder": vqvae_params["encoder"],
                    "decoder": vqvae_params["decoder"],
                    "codebook": codebook}
    print("VQ-VAE loss (first->last):",
          f"{float(vq_losses[0]):.4f} -> {float(vq_losses[-1]):.4f}")

    text_vocab_size = len(vocab)
    image_token_offset = text_vocab_size
    full_vocab_size = text_vocab_size + num_codes + 1

    sequences = []
    for img, label in zip(images, labels):
        grid = encode_image_to_tokens(img, vqvae_params, patch_size)
        image_tokens = flatten_token_grid(grid)
        text_ids = list(encode_label_to_ids(label, vocab))[:text_len]
        text_ids = text_ids + [0] * (text_len - len(text_ids))
        seq = form_multimodal_sequence(
            jnp.array(text_ids), image_tokens, image_token_offset
        )
        sequences.append(seq)
    batch_sequences = jnp.stack(sequences)
    print("token sequence shape:", np.asarray(batch_sequences).shape)
    print("example sequence:", np.asarray(batch_sequences[0]).tolist())

    seq_len = text_len + num_image_tokens
    k_tok, k_pos, k_out, key = jax.random.split(key, 4)
    blocks = []
    for _ in range(2):
        k_a, k_f, key = jax.random.split(key, 3)
        blocks.append({
            "ln1_scale": jnp.ones(d_model),
            "ln1_shift": jnp.zeros(d_model),
            "attn": init_attention_params(k_a, d_model),
            "ln2_scale": jnp.ones(d_model),
            "ln2_shift": jnp.zeros(d_model),
            "ff": init_feedforward_params(k_f, d_model, d_ff),
        })
    tparams = {
        "token_embedding": init_token_embedding(k_tok, full_vocab_size, d_model),
        "positional_embedding": init_positional_embedding(k_pos, seq_len, d_model),
        "blocks": blocks,
        "output": init_output_projection(k_out, d_model, full_vocab_size),
    }
    t_opt = optax.adam(1e-3)
    t_state = t_opt.init(tparams)
    tparams, t_state, t_losses = train_transformer_on_token_sequences(
        batch_sequences, tparams, t_state, t_opt, text_len, num_steps=30
    )
    print("transformer loss (first->last):",
          f"{float(t_losses[0]):.4f} -> {float(t_losses[-1]):.4f}")

    causal_mask = build_causal_mask(seq_len)
    acc = next_token_accuracy(tparams, batch_sequences, causal_mask, num_heads, text_len)
    print("next-token accuracy (image positions):", f"{float(acc):.4f}")

    # ---- end-to-end conditional generation ----
    k_gen, key = jax.random.split(key)
    tparams_gen = dict(tparams)
    tparams_gen["num_heads"] = num_heads  # generate_image_from_label reads this
    # generate_image_from_label unpacks grid_shape as (grid_size, patch_size)
    # and decode_tokens_to_image reads decoder_params['decoder'].
    gen_image = generate_image_from_label(
        labels[0], vocab, tparams_gen, codebook, vqvae_params,
        grid_shape=(grid_h, patch_size), text_len=text_len,
        num_image_tokens=num_image_tokens, key=k_gen,
        guidance_scale=3.0, temperature=1.0, top_k=8,
    )
    nn_dist = nearest_neighbor_distance_to_dataset(gen_image, images)
    print("generated image shape:", np.asarray(gen_image).shape)
    print("generated image[0]:", np.asarray(gen_image).round(4).tolist()[0])
    print("nearest-neighbor distance to dataset:", f"{float(nn_dist):.4f}")


if __name__ == "__main__":
    main()
