"""
Multimodal Autoregressive Image Generator from Scratch in JAX

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - generate_toy_images
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def generate_toy_images(key, num_images, image_size):
    """Generate tiny grayscale images containing one randomly placed bright square.

    Args:
        key: JAX PRNG key.
        num_images: Number of images to generate.
        image_size: Height and width of each square image.

    Returns:
        jnp.ndarray: Array of shape
            (num_images, image_size, image_size)
            with values in [0, 1].
    """
    square_size = image_size // 2

    # Split the key so every image gets its own independent random key.
    keys = jax.random.split(key, num_images)

    def make_image(image_key):
        # Valid top-left coordinates range from 0 to
        # image_size - square_size, inclusive.
        max_position = image_size - square_size

        position_key_y, position_key_x = jax.random.split(image_key)

        top = jax.random.randint(
            position_key_y,
            shape=(),
            minval=0,
            maxval=max_position + 1,
        )

        left = jax.random.randint(
            position_key_x,
            shape=(),
            minval=0,
            maxval=max_position + 1,
        )

        # Start with an all-zero image.
        image = jnp.zeros((image_size, image_size), dtype=jnp.float32)

        # Place the bright square.
        image = image.at[
            top:top + square_size,
            left:left + square_size
        ].set(1.0)

        return image

    # Generate and stack all images into one JAX array.
    return jnp.stack([make_image(k) for k in keys])

# Step 2 - assign_image_labels
def assign_image_labels(images):
    """Assign 'left' or 'right' labels based on pixel mass.

    Args:
        images: JAX array of shape
            (num_images, image_size, image_size).

    Returns:
        list[str]: One label per image. Ties are labeled 'left'.
    """
    image_size = images.shape[-1]
    midpoint = image_size // 2

    # Calculate total brightness in the left and right halves.
    left_mass = jnp.sum(images[:, :, :midpoint], axis=(1, 2))
    right_mass = jnp.sum(images[:, :, midpoint:], axis=(1, 2))

    # Exact ties are resolved in favor of "left".
    return [
        "left" if float(left) >= float(right) else "right"
        for left, right in zip(left_mass, right_mass)
    ]

# Step 3 - normalize_image_batch
def normalize_image_batch(images):
    """Rescale image values from [0, 1] to [-1, 1].

    Args:
        images: JAX array containing image pixel values in [0, 1].

    Returns:
        jnp.ndarray: Images with the same shape, rescaled to [-1, 1].
    """
    return 2.0 * jnp.asarray(images) - 1.0

# Step 4 - split_image_into_patches
def split_image_into_patches(image, patch_size):
    """Split a grayscale image into a grid of non-overlapping square patches.

    Args:
        image: JAX array of shape (H, W).
        patch_size: Size of each square patch. Must evenly divide H and W.

    Returns:
        jnp.ndarray: Array of shape
            (num_patches_h, num_patches_w, patch_size, patch_size).
    """
    image = jnp.asarray(image)

    height, width = image.shape

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    return image.reshape(
        num_patches_h,
        patch_size,
        num_patches_w,
        patch_size,
    ).transpose(0, 2, 1, 3)

# Step 5 - flatten_patches
def flatten_patches(patches):
    """Flatten each patch in a patch grid into a 1D vector.

    Args:
        patches: JAX array whose first two dimensions are the patch grid
                 (gh, gw), followed by the patch-element dimensions.

    Returns:
        jnp.ndarray: Shape (gh * gw, product of all remaining dimensions).
    """
    gh, gw = patches.shape[:2]

    return patches.reshape(gh * gw, -1)

# Step 6 - init_patch_encoder
def init_patch_encoder(key, patch_dim, latent_dim):
    """Initialize the linear patch encoder weights.

    Args:
        key: JAX PRNG key.
        patch_dim: Number of pixels in a flattened patch.
        latent_dim: Size of the latent representation.

    Returns:
        jnp.ndarray: Weight matrix of shape (patch_dim, latent_dim).
    """
    return (
        jax.random.normal(
            key,
            shape=(patch_dim, latent_dim),
            dtype=jnp.float32,
        )
        / jnp.sqrt(patch_dim)
    )

# Step 7 - encode_patches
def encode_patches(flat_patches, encoder_weight):
    # Project each flattened patch into the latent space.
    return flat_patches @ encoder_weight

# Step 8 - init_patch_decoder
def init_patch_decoder(key, latent_dim, patch_dim):
    # Sample decoder weights and scale by 1/sqrt(latent_dim).
    return (
        jax.random.normal(
            key,
            shape=(latent_dim, patch_dim),
            dtype=jnp.float32,
        )
        / jnp.sqrt(latent_dim)
    )

# Step 9 - decode_latents
def decode_latents(latents, decoder_weight):
    # Project each latent vector back to the flat patch space.
    return latents @ decoder_weight

# Step 10 - reassemble_patches_into_image
def reassemble_patches_into_image(flat_patches, grid_h, grid_w, patch_size):
    # Reshape flat patches into a row-major patch grid.
    patches = flat_patches.reshape(
        grid_h,
        grid_w,
        patch_size,
        patch_size,
    )

    # Move patch dimensions next to their corresponding grid dimensions,
    # then merge them into the full image dimensions.
    image = patches.transpose(0, 2, 1, 3)

    return image.reshape(
        grid_h * patch_size,
        grid_w * patch_size,
    )

# Step 11 - init_codebook
def init_codebook(key, num_codes, latent_dim):
    # Initialize the codebook with small, zero-centered random values.
    return jax.random.normal(
        key,
        shape=(num_codes, latent_dim),
        dtype=jnp.float32,
    ) * 0.1

# Step 12 - squared_distance_to_codebook
def squared_distance_to_codebook(latent, codebook):
    # Compute squared Euclidean distance to every codebook vector.
    return jnp.sum((codebook - latent) ** 2, axis=1)

# Step 13 - grid_distances_to_codebook
def grid_distances_to_codebook(latents, codebook):
    # Compute squared Euclidean distances:
    # (P, 1, D) - (1, K, D) -> (P, K, D)
    # Then sum over the latent dimension.
    return jnp.sum(
        (latents[:, None, :] - codebook[None, :, :]) ** 2,
        axis=-1,
    )

# Step 14 - assign_nearest_codes
def assign_nearest_codes(distances):
    # Return the nearest codebook index for each latent.
    return jnp.argmin(distances, axis=1)

# Step 15 - lookup_codebook_vectors
def lookup_codebook_vectors(indices, codebook):
    # Look up the codebook vector corresponding to each token index.
    return codebook[indices]

# Step 16 - straight_through_quantize
def straight_through_quantize(latents, quantized):
    # Forward pass: quantized
    # Backward pass: gradient flows through latents.
    return latents + jax.lax.stop_gradient(quantized - latents)

# Step 17 - codebook_loss
def codebook_loss(latents, quantized):
    # Stop gradients from flowing into the encoder.
    stopped_latents = jax.lax.stop_gradient(latents)

    # Mean squared error over all patches and latent dimensions.
    return jnp.mean((stopped_latents - quantized) ** 2)

# Step 18 - commitment_loss
def commitment_loss(latents, quantized):
    # Stop gradients from flowing into the codebook.
    stopped_quantized = jax.lax.stop_gradient(quantized)

    # Mean squared error over all patches and latent dimensions.
    return jnp.mean((latents - stopped_quantized) ** 2)

# Step 19 - reconstruction_loss
def reconstruction_loss(image, reconstruction):
    # Mean squared error over all pixels.
    return jnp.mean((image - reconstruction) ** 2)

# Step 20 - total_vqvae_loss
def total_vqvae_loss(recon_loss, cb_loss, commit_loss, commitment_weight):
    return recon_loss + cb_loss + commitment_weight * commit_loss

# Step 21 - vqvae_loss_and_grads
def vqvae_loss_and_grads(params, image_batch, patch_size, commitment_weight):

    def loss_fn(params):
        encoder_weight = params["encoder"]
        decoder_weight = params["decoder"]
        codebook = params["codebook"]

        def image_loss(image):
            # Split image into patches.
            patches = split_image_into_patches(image, patch_size)

            # Flatten the patch grid.
            flat_patches = flatten_patches(patches)

            # Encode patches into latent vectors.
            latents = encode_patches(flat_patches, encoder_weight)

            # Compute distances to every codebook vector.
            distances = grid_distances_to_codebook(latents, codebook)

            # Select the nearest codebook entry for each latent.
            indices = assign_nearest_codes(distances)

            # Look up the corresponding quantized vectors.
            quantized = lookup_codebook_vectors(indices, codebook)

            # Straight-through estimator.
            quantized_st = straight_through_quantize(latents, quantized)

            # Decode the quantized latents back into flat patches.
            decoded_patches = decode_latents(
                quantized_st,
                decoder_weight,
            )

            # Reassemble patches into the reconstructed image.
            grid_h, grid_w = patches.shape[:2]

            reconstruction = reassemble_patches_into_image(
                decoded_patches,
                grid_h,
                grid_w,
                patch_size,
            )

            # Compute the three VQ-VAE loss terms.
            recon_loss = reconstruction_loss(
                image,
                reconstruction,
            )

            cb_loss = codebook_loss(
                latents,
                quantized,
            )

            commit_loss = commitment_loss(
                latents,
                quantized,
            )

            return total_vqvae_loss(
                recon_loss,
                cb_loss,
                commit_loss,
                commitment_weight,
            )

        # Average the loss across the image batch.
        losses = jax.vmap(image_loss)(image_batch)
        return jnp.mean(losses)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    return loss, grads

# Step 22 - apply_vqvae_update
def apply_vqvae_update(params, grads, opt_state, optimizer):
    # Compute parameter updates and the new optimizer state.
    updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        params,
    )

    # Apply the updates while preserving the parameter pytree structure.
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state

# Step 23 - encode_image_to_tokens
def encode_image_to_tokens(image, params, patch_size):
    # Split the image into patches.
    patches = split_image_into_patches(image, patch_size)

    # Flatten each patch into a vector.
    flat_patches = flatten_patches(patches)

    # Encode patches into latent vectors.
    latents = encode_patches(flat_patches, params["encoder"])

    # Compute distances from each latent to every codebook vector.
    distances = grid_distances_to_codebook(
        latents,
        params["codebook"],
    )

    # Assign each patch to its nearest codebook entry.
    indices = assign_nearest_codes(distances)

    # Restore the original patch-grid layout.
    grid_h, grid_w = patches.shape[:2]

    return indices.reshape(grid_h, grid_w)

# Step 24 - flatten_token_grid
def flatten_token_grid(token_grid):
    # Flatten the 2D token grid into a 1D row-major sequence.
    return token_grid.reshape(-1)

# Step 25 - reshape_tokens_to_grid
def reshape_tokens_to_grid(token_sequence, grid_h, grid_w):
    # Restore the 1D row-major token sequence to a 2D grid.
    return token_sequence.reshape(grid_h, grid_w)

# Step 26 - build_char_vocab
def build_char_vocab(labels):
    # Collect every unique character and sort for deterministic ordering.
    characters = sorted(set(char for label in labels for char in label))

    return {char: idx for idx, char in enumerate(characters)}

# Step 27 - encode_label_to_ids
def encode_label_to_ids(label, char_vocab):
    # Map each character to its vocabulary ID.
    return jnp.array(
        [char_vocab[char] for char in label],
        dtype=jnp.int32,
    )

# Step 28 - form_multimodal_sequence
def form_multimodal_sequence(text_ids, image_tokens, image_token_offset):
    # Shift image tokens into their separate ID range.
    shifted_image_tokens = image_tokens + image_token_offset

    # Place text tokens first, followed by image tokens.
    return jnp.concatenate([text_ids, shifted_image_tokens])

# Step 29 - init_token_embedding
def init_token_embedding(key, vocab_size, embed_dim):
    # Initialize a small random embedding table.
    return jax.random.normal(
        key,
        shape=(vocab_size, embed_dim),
        dtype=jnp.float32,
    ) * 0.02

# Step 30 - init_positional_embedding
def init_positional_embedding(key, max_seq_len, embed_dim):
    # Initialize a small random learned positional embedding table.
    return jax.random.normal(
        key,
        shape=(max_seq_len, embed_dim),
        dtype=jnp.float32,
    ) * 0.02

# Step 31 - lookup_token_embeddings
def lookup_token_embeddings(token_embedding, token_ids):
    # Select the embedding vector for each token ID.
    return token_embedding[token_ids]

# Step 32 - add_positional_embeddings
def add_positional_embeddings(token_embeds, positional_embedding):
    # Use only the positional embeddings needed for this sequence.
    seq_len = token_embeds.shape[0]
    return token_embeds + positional_embedding[:seq_len]

# Step 33 - build_causal_mask
def build_causal_mask(seq_len):
    # Allow attention to the current and previous positions.
    return jnp.where(
        jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32)) == 1.0,
        0.0,
        -1e9,
    )

# Step 34 - layer_norm
def layer_norm(x, scale, shift, eps=1e-5):
    # Compute mean and variance over the last (feature) axis.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)

    # Normalize and apply learned affine parameters.
    normalized = (x - mean) / jnp.sqrt(variance + eps)

    return normalized * scale + shift

# Step 35 - init_attention_params
def init_attention_params(key, d_model):
    # Split the PRNG key so each projection gets independent weights.
    wq_key, wk_key, wv_key, wo_key = jax.random.split(key, 4)

    return {
        "wq": jax.random.normal(
            wq_key, (d_model, d_model), dtype=jnp.float32
        ) * 0.02,
        "wk": jax.random.normal(
            wk_key, (d_model, d_model), dtype=jnp.float32
        ) * 0.02,
        "wv": jax.random.normal(
            wv_key, (d_model, d_model), dtype=jnp.float32
        ) * 0.02,
        "wo": jax.random.normal(
            wo_key, (d_model, d_model), dtype=jnp.float32
        ) * 0.02,
    }

# Step 36 - project_qkv
def project_qkv(x, attn_params):
    # Project x into queries, keys, and values.
    q = x @ attn_params["wq"]
    k = x @ attn_params["wk"]
    v = x @ attn_params["wv"]

    return q, k, v

# Step 37 - reshape_to_heads
def reshape_to_heads(matrix, num_heads):
    seq_len, d_model = matrix.shape
    d_head = d_model // num_heads

    return matrix.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)

# Step 38 - scaled_dot_product_scores
def scaled_dot_product_scores(q_heads, k_heads):
    d_head = q_heads.shape[-1]
    return jnp.matmul(q_heads, jnp.swapaxes(k_heads, -1, -2)) / jnp.sqrt(d_head)

# Step 39 - add_causal_mask_to_scores
def add_causal_mask_to_scores(scores, causal_mask):
    # Broadcast the causal mask across all attention heads.
    return scores + causal_mask

# Step 40 - attention_weights_softmax
def attention_weights_softmax(masked_scores):
    # Numerically stable softmax over the key axis.
    shifted = masked_scores - jnp.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(shifted)
    return exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

# Step 41 - weighted_sum_of_values
def weighted_sum_of_values(attn_weights, v_heads):
    # Combine value vectors using the attention weights for each head.
    return jnp.matmul(attn_weights, v_heads)

# Step 42 - merge_heads_and_project
def merge_heads_and_project(head_outputs, attn_params):
    # Move heads after the sequence dimension, then merge them into d_model.
    merged = head_outputs.transpose(1, 0, 2).reshape(
        head_outputs.shape[1], -1
    )

    # Apply the output projection.
    return merged @ attn_params["wo"]

# Step 43 - init_feedforward_params
def init_feedforward_params(key, d_model, d_ff):
    # Split the key so w1 and w2 use independent random draws.
    key_w1, key_w2 = jax.random.split(key)

    w1 = 0.02 * jax.random.normal(key_w1, (d_model, d_ff))
    w2 = 0.02 * jax.random.normal(key_w2, (d_ff, d_model))

    return {
        "w1": w1,
        "w2": w2,
    }

# Step 44 - feedforward_mlp
def feedforward_mlp(x, ff_params):
    # Expand to d_ff.
    hidden = x @ ff_params["w1"]

    # Apply GELU activation.
    hidden = jax.nn.gelu(hidden)

    # Project back to d_model.
    return hidden @ ff_params["w2"]

# Step 45 - transformer_block
def transformer_block(x, block_params, causal_mask, num_heads):
    # First pre-norm.
    norm_x = layer_norm(
        x,
        block_params["ln1_scale"],
        block_params["ln1_shift"],
    )

    # Multi-head causal self-attention.
    q, k, v = project_qkv(norm_x, block_params["attn"])
    q_heads = reshape_to_heads(q, num_heads)
    k_heads = reshape_to_heads(k, num_heads)
    v_heads = reshape_to_heads(v, num_heads)

    scores = scaled_dot_product_scores(q_heads, k_heads)
    scores = add_causal_mask_to_scores(scores, causal_mask)
    attn_weights = attention_weights_softmax(scores)
    head_outputs = weighted_sum_of_values(attn_weights, v_heads)
    attn_output = merge_heads_and_project(
        head_outputs,
        block_params["attn"],
    )

    # First residual connection.
    x = x + attn_output

    # Second pre-norm.
    norm_x = layer_norm(
        x,
        block_params["ln2_scale"],
        block_params["ln2_shift"],
    )

    # Feed-forward MLP.
    mlp_output = feedforward_mlp(
        norm_x,
        block_params["ff"],
    )

    # Second residual connection.
    return x + mlp_output

# Step 46 - transformer_backbone
def transformer_backbone(x, blocks_params, causal_mask, num_heads):
    for block_params in blocks_params:
        x = transformer_block(
            x,
            block_params,
            causal_mask,
            num_heads,
        )

    return x

# Step 47 - init_output_projection
def init_output_projection(key, d_model, vocab_size):
    w_out = 0.02 * jax.random.normal(key, (d_model, vocab_size))
    b_out = jnp.zeros((vocab_size,), dtype=jnp.float32)

    return {
        "w_out": w_out,
        "b_out": b_out,
    }

# Step 48 - project_to_logits
def project_to_logits(hidden_states, output_params):
    # Project hidden states to vocabulary logits.
    return hidden_states @ output_params["w_out"] + output_params["b_out"]

# Step 49 - image_token_cross_entropy
def image_token_cross_entropy(logits, target_ids, image_start_index):
    # Use higher precision for the cross-entropy calculation.
    logits = jnp.asarray(logits, dtype=jnp.float64)

    image_logits = logits[image_start_index - 1:-1]
    image_targets = target_ids[image_start_index:]

    log_probs = jax.nn.log_softmax(image_logits, axis=-1)

    losses = -log_probs[
        jnp.arange(image_targets.shape[0]),
        image_targets,
    ]

    return jnp.mean(losses)

# Step 50 - transformer_loss_and_grads
def transformer_loss_and_grads(
    params,
    batch_sequences,
    causal_mask,
    num_heads,
    image_start_index,
):
    def loss_fn(params):
        def sequence_loss(sequence):
            # Look up token embeddings.
            hidden_states = lookup_token_embeddings(
                params["token_embedding"],
                sequence,
            )

            # Add positional embeddings.
            hidden_states = add_positional_embeddings(
                hidden_states,
                params["positional_embedding"],
            )

            # Run the transformer backbone.
            hidden_states = transformer_backbone(
                hidden_states,
                params["blocks"],
                causal_mask,
                num_heads,
            )

            # Project hidden states to vocabulary logits.
            logits = project_to_logits(
                hidden_states,
                params["output"],
            )

            # Compute loss only on image-token predictions.
            return image_token_cross_entropy(
                logits,
                sequence,
                image_start_index,
            )

        # Compute one loss per sequence.
        per_sequence_losses = jax.vmap(sequence_loss)(batch_sequences)

        # Mean loss over the batch.
        return jnp.mean(per_sequence_losses)

    # Compute loss and gradients with respect to the full parameter pytree.
    loss, grads = jax.value_and_grad(loss_fn)(params)

    return loss, grads

# Step 51 - apply_transformer_update
def apply_transformer_update(params, grads, opt_state, optimizer):
    # Compute parameter updates and the new optimizer state.
    updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        params,
    )

    # Apply the updates to the current parameters.
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state

# Step 52 - drop_text_prefix
def drop_text_prefix(
    sequence,
    key,
    image_start_index,
    drop_prob,
    null_token_id,
):
    # Make one Bernoulli draw for the entire text prefix.
    drop = jax.random.bernoulli(key, drop_prob)

    # Replace only the text-prefix positions when dropping.
    dropped_sequence = sequence.at[:image_start_index].set(null_token_id)

    return jnp.where(drop, dropped_sequence, sequence).astype(jnp.int32)

# Step 53 - combine_guided_logits
def combine_guided_logits(cond_logits, uncond_logits, guidance_scale):
    return uncond_logits + guidance_scale * (cond_logits - uncond_logits)

# Step 54 - logits_to_probabilities
def logits_to_probabilities(logits, temperature):
    # Scale logits by temperature.
    scaled_logits = logits / temperature

    # Numerically stable softmax.
    shifted_logits = scaled_logits - jnp.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted_logits)

    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 55 - top_k_filter_logits
def top_k_filter_logits(logits, k):
    # Find the k-th largest logit.
    kth_value = jnp.sort(logits)[-k]

    # Keep logits greater than or equal to the threshold.
    return jnp.where(
        logits >= kth_value,
        logits,
        -1e9,
    )

# Step 56 - sample_token_index
def sample_token_index(probabilities, key):
    # Sample one token ID from the given probability distribution.
    return jax.random.categorical(
        key,
        jnp.log(probabilities),
    )

# Step 57 - generate_image_tokens
def generate_image_tokens(
    params,
    text_prefix,
    key,
    num_image_tokens,
    num_heads,
    null_prefix,
    guidance_scale,
    temperature,
    top_k,
):
    generated_tokens = jnp.array([], dtype=jnp.int32)

    for _ in range(num_image_tokens):
        key, sample_key = jax.random.split(key)

        # Conditional sequence: text prefix + generated image tokens.
        cond_sequence = jnp.concatenate(
            [text_prefix, generated_tokens]
        )

        # Unconditional sequence: null prefix + same generated tokens.
        uncond_sequence = jnp.concatenate(
            [null_prefix, generated_tokens]
        )

        # Run the transformer on the conditional sequence.
        cond_embeds = lookup_token_embeddings(
            params["token_embedding"],
            cond_sequence,
        )
        cond_embeds = add_positional_embeddings(
            cond_embeds,
            params["positional_embedding"],
        )
        cond_hidden = transformer_backbone(
            cond_embeds,
            params["blocks"],
            build_causal_mask(cond_sequence.shape[0]),
            num_heads,
        )
        cond_logits = project_to_logits(
            cond_hidden,
            params["output"],
        )[-1]

        # Run the transformer on the unconditional sequence.
        uncond_embeds = lookup_token_embeddings(
            params["token_embedding"],
            uncond_sequence,
        )
        uncond_embeds = add_positional_embeddings(
            uncond_embeds,
            params["positional_embedding"],
        )
        uncond_hidden = transformer_backbone(
            uncond_embeds,
            params["blocks"],
            build_causal_mask(uncond_sequence.shape[0]),
            num_heads,
        )
        uncond_logits = project_to_logits(
            uncond_hidden,
            params["output"],
        )[-1]

        # Classifier-free guidance.
        guided_logits = combine_guided_logits(
            cond_logits,
            uncond_logits,
            guidance_scale,
        )

        # Restrict sampling to the top-k logits.
        filtered_logits = top_k_filter_logits(
            guided_logits,
            top_k,
        )

        # Convert logits to probabilities using temperature.
        probabilities = logits_to_probabilities(
            filtered_logits,
            temperature,
        )

        # Sample the next image token.
        next_token = sample_token_index(
            probabilities,
            sample_key,
        )

        # Append it to the generated image-token sequence.
        generated_tokens = jnp.concatenate(
            [
                generated_tokens,
                jnp.array([next_token], dtype=jnp.int32),
            ]
        )

    return generated_tokens

# Step 58 - decode_tokens_to_image
def decode_tokens_to_image(
    image_tokens,
    codebook,
    decoder_params,
    grid_size,
    patch_size,
):
    # Restore the 2D token grid.
    token_grid = reshape_tokens_to_grid(
        image_tokens,
        grid_size,
        grid_size,
    )

    # Flatten the grid for codebook lookup.
    token_indices = flatten_token_grid(token_grid)

    # Look up the corresponding codebook latent vectors.
    latents = lookup_codebook_vectors(
        token_indices,
        codebook,
    )

    # Decode latent vectors back into flat image patches.
    flat_patches = decode_latents(
        latents,
        decoder_params["decoder"],
    )

    # Stitch the decoded patches into the full image.
    return reassemble_patches_into_image(
        flat_patches,
        grid_size,
        grid_size,
        patch_size,
    )

# Step 59 - next_token_accuracy
def next_token_accuracy(
    params,
    batch_sequences,
    causal_mask,
    num_heads,
    image_start_index,
):
    accuracies = []

    for sequence in batch_sequences:
        # Look up token embeddings.
        hidden_states = lookup_token_embeddings(
            params["token_embedding"],
            sequence,
        )

        # Add positional embeddings.
        hidden_states = add_positional_embeddings(
            hidden_states,
            params["positional_embedding"],
        )

        # Run the transformer backbone.
        hidden_states = transformer_backbone(
            hidden_states,
            params["blocks"],
            causal_mask,
            num_heads,
        )

        # Project to vocabulary logits.
        logits = project_to_logits(
            hidden_states,
            params["output"],
        )

        # Logits at t predict the token at t + 1.
        image_logits = logits[image_start_index - 1:-1]
        image_targets = sequence[image_start_index:]

        # Greedy next-token predictions.
        predictions = jnp.argmax(image_logits, axis=-1)

        # Accuracy for this sequence.
        accuracies.append(
            jnp.mean(predictions == image_targets)
        )

    # Average accuracy across the batch.
    return jnp.mean(jnp.stack(accuracies))

# Step 60 - average_reconstruction_error
def average_reconstruction_error(
    encoder_params,
    decoder_params,
    codebook,
    image_batch,
    patch_size,
):
    errors = []

    for image in image_batch:
        # Normalize the image to the [-1, 1] space used by the VQ-VAE.
        normalized_image = normalize_image_batch(image)

        # Split into patches and flatten them.
        patches = split_image_into_patches(
            normalized_image,
            patch_size,
        )
        flat_patches = flatten_patches(patches)

        # Encode patches.
        latents = encode_patches(
            flat_patches,
            encoder_params["weight"],
        )

        # Find the nearest codebook entry for each latent.
        distances = grid_distances_to_codebook(
            latents,
            codebook,
        )
        indices = assign_nearest_codes(distances)

        # Look up the quantized latent vectors.
        quantized = lookup_codebook_vectors(
            indices,
            codebook,
        )

        # Decode the quantized latents.
        decoded_patches = decode_latents(
            quantized,
            decoder_params["weight"],
        )

        # Reassemble the reconstructed image.
        grid_h, grid_w = patches.shape[:2]
        reconstruction = reassemble_patches_into_image(
            decoded_patches,
            grid_h,
            grid_w,
            patch_size,
        )

        # Compute reconstruction error in normalized space.
        error = reconstruction_loss(
            normalized_image,
            reconstruction,
        )

        errors.append(error)

    return jnp.mean(jnp.stack(errors))

# Step 61 - nearest_neighbor_distance_to_dataset (not yet solved)
# TODO: implement

# Step 62 - train_vqvae_on_toy_images (not yet solved)
# TODO: implement

# Step 63 - train_transformer_on_token_sequences (not yet solved)
# TODO: implement

# Step 64 - generate_image_from_label (not yet solved)
# TODO: implement

