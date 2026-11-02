import gc
import logging
import math


logger = logging.getLogger("promptrelay_lightx2v_patch")


def _find_subsequence(haystack, needle, start_pos=0):
    if not needle:
        return None
    max_start = len(haystack) - len(needle)
    for start in range(start_pos, max_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    for start in range(0, min(start_pos, max_start + 1)):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def _tokenize_without_padding(tokenizer, text, add_special_tokens):
    ids = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding=False,
        return_mask=False,
    )[0]
    return ids.tolist()


def build_prompt_relay_q_token_idx(tokenizer, prompt, relay_spec, config, input_info):
    import torch

    local_prompts = relay_spec.get("local_prompts") or []
    intervals = relay_spec.get("intervals") or []
    if not local_prompts:
        return None
    if len(local_prompts) != len(intervals):
        raise ValueError("Prompt-Relay local_prompts and intervals must have the same length.")

    latent_shape = getattr(input_info, "latent_shape", None)
    if not latent_shape or len(latent_shape) < 4:
        raise ValueError("input_info.latent_shape is required before building Prompt-Relay routing.")

    latent_frames = int(latent_shape[1])
    latent_h = int(latent_shape[2])
    latent_w = int(latent_shape[3])
    patch_h = int(config["patch_size"][1])
    patch_w = int(config["patch_size"][2])
    tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)

    full_ids = _tokenize_without_padding(tokenizer, prompt, add_special_tokens=True)
    token_spans = []
    cursor = 0
    for local_prompt in local_prompts:
        local_ids = _tokenize_without_padding(tokenizer, local_prompt, add_special_tokens=False)
        match = _find_subsequence(full_ids, local_ids, cursor)
        if match is None:
            raise ValueError(
                "Local prompt was not found in the tokenized full prompt. "
                f"It may have been truncated by text_len={config['text_len']}: {local_prompt!r}"
            )
        token_spans.append(match)
        cursor = match[1]

    q_token_idx = []
    sigma = float(relay_spec.get("sigma", 0.1448))
    window_offset = int(relay_spec.get("window_offset", 2))
    min_window = int(relay_spec.get("min_window", 0))

    for interval, span in zip(intervals, token_spans):
        frame_start = int(math.floor(float(interval["start"]) * latent_frames))
        frame_end = int(math.ceil(float(interval["end"]) * latent_frames))
        frame_start = max(0, min(frame_start, latent_frames - 1))
        frame_end = max(frame_start + 1, min(frame_end, latent_frames))
        segment_len = frame_end - frame_start
        window = max(min_window, segment_len // 2 - window_offset)
        token_start, token_end = span
        q_token_idx.append(
            {
                "window": window,
                "sigma": sigma,
                "midpoint": (frame_start + frame_end) // 2,
                "tokens_per_frame": tokens_per_frame,
                "local_token_idx": torch.tensor(list(range(token_start, token_end)), dtype=torch.long),
            }
        )

    return q_token_idx


def build_temporal_cost(q_token_idx, lq, lk, device, dtype):
    import torch

    offset = torch.zeros(lq, lk, device=device, dtype=dtype)
    tokens_per_frame = int(q_token_idx[0]["tokens_per_frame"])
    query_frames = torch.arange(lq, device=device, dtype=torch.long) // tokens_per_frame

    for segment in q_token_idx:
        local = segment["local_token_idx"].to(device=device, dtype=torch.long)
        local = local[(local >= 0) & (local < lk)]
        if local.numel() == 0:
            continue

        window = float(segment["window"])
        sigma = torch.tensor(float(segment["sigma"]), dtype=torch.float32, device=device)
        midpoint = torch.tensor(float(segment["midpoint"]), dtype=torch.float32, device=device)
        distance = (query_frames.float()[:, None] - midpoint).abs()
        cost = (torch.relu(distance - window) ** 2) / (2 * sigma**2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def _prompt_relay_cache_key(q_token_idx, lq, lk, device, dtype):
    segments = []
    for segment in q_token_idx:
        local = tuple(int(v) for v in segment["local_token_idx"].tolist())
        segments.append(
            (
                int(segment["window"]),
                round(float(segment["sigma"]), 8),
                int(segment["midpoint"]),
                int(segment["tokens_per_frame"]),
                local,
            )
        )
    return (lq, lk, str(device), str(dtype), tuple(segments))


def _get_cached_temporal_cost(transformer_infer, q_token_idx, lq, lk, device, dtype):
    key = _prompt_relay_cache_key(q_token_idx, lq, lk, device, dtype)
    cached = getattr(transformer_infer, "_prompt_relay_cost_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    cost = build_temporal_cost(q_token_idx, lq, lk, device, dtype)
    transformer_infer._prompt_relay_cost_cache = (key, cost)
    return cost


def prompt_relay_chunked_attention(q, k, v, temporal_cost, chunk_size):
    import torch

    q_heads = q.transpose(0, 1)
    k_heads = k.transpose(0, 1)
    v_heads = v.transpose(0, 1)
    heads, lq, head_dim = q_heads.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(heads, lq, head_dim, device=q.device, dtype=q.dtype)
    k_t = k_heads.transpose(-2, -1).float()

    for start in range(0, lq, chunk_size):
        end = min(start + chunk_size, lq)
        logits = torch.matmul(q_heads[:, start:end, :].float(), k_t) * scale
        logits = logits - temporal_cost[start:end].unsqueeze(0).float()
        attn = torch.softmax(logits, dim=-1).to(v_heads.dtype)
        out[:, start:end, :] = torch.matmul(attn, v_heads)

    return out.transpose(0, 1).reshape(lq, heads * head_dim)


def install_prompt_relay_patch(chunk_size=64):
    import torch
    import torch.distributed as dist

    from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_runner import WanRunner
    from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
    from lightx2v_platform.base.global_var import AI_DEVICE

    torch_device_module = getattr(torch, AI_DEVICE)

    if getattr(WanRunner.run_text_encoder, "_prompt_relay_patched", False):
        return

    original_transformer_infer = WanTransformerInfer.infer

    def patched_run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        neg_prompt = input_info.negative_prompt
        relay_spec = getattr(input_info, "prompt_relay", None)
        q_token_idx = None
        if relay_spec is not None:
            q_token_idx = build_prompt_relay_q_token_idx(
                tokenizer=self.text_encoders[0].tokenizer,
                prompt=prompt,
                relay_spec=relay_spec,
                config=self.config,
                input_info=input_info,
            )

        if self.config.get("enable_cfg", False) and self.config["cfg_parallel"]:
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            cfg_p_rank = dist.get_rank(cfg_p_group)
            if cfg_p_rank == 0:
                context = self.text_encoders[0].infer([prompt])
                context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
                text_encoder_output = {"context": context}
            else:
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
                text_encoder_output = {"context_null": context_null}
        else:
            context = self.text_encoders[0].infer([prompt])
            context = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context])
            if self.config.get("enable_cfg", False):
                context_null = self.text_encoders[0].infer([neg_prompt])
                context_null = torch.stack([torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))]) for u in context_null])
            else:
                context_null = None
            text_encoder_output = {
                "context": context,
                "context_null": context_null,
            }

        if q_token_idx is not None:
            text_encoder_output["prompt_relay_q_token_idx"] = q_token_idx

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return text_encoder_output

    def patched_pre_infer(self, weights, inputs, kv_start=0, kv_end=0):
        out = self._prompt_relay_original_pre_infer(weights, inputs, kv_start, kv_end)
        q_token_idx = inputs.get("text_encoder_output", {}).get("prompt_relay_q_token_idx")
        if q_token_idx is not None and self.scheduler.infer_condition:
            out.conditional_dict["prompt_relay_q_token_idx"] = q_token_idx
        return out

    def patched_transformer_infer(self, weights, pre_infer_out):
        self.prompt_relay_q_token_idx = pre_infer_out.conditional_dict.get("prompt_relay_q_token_idx")
        self.prompt_relay_chunk_size = int(chunk_size)
        try:
            return original_transformer_infer(self, weights, pre_infer_out)
        finally:
            self.prompt_relay_q_token_idx = None

    def patched_infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        infer_dtype = GET_DTYPE()
        sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        if sensitive_layer_dtype != infer_dtype:
            x = x.to(sensitive_layer_dtype) + y_out.to(sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        norm3_out = phase.norm3.apply(x)
        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if sensitive_layer_dtype != infer_dtype:
            context = context.to(infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(infer_dtype)

        heads, head_dim = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, heads, head_dim)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, heads, head_dim)
        v = phase.cross_attn_v.apply(context).view(-1, heads, head_dim)

        if self.cross_attn_cu_seqlens_q is None:
            self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)

        q_token_idx = getattr(self, "prompt_relay_q_token_idx", None)
        if q_token_idx:
            temporal_cost = _get_cached_temporal_cost(self, q_token_idx, q.size(0), k.size(0), q.device, q.dtype)
            attn_out = prompt_relay_chunked_attention(
                q=q,
                k=k,
                v=v,
                temporal_cost=temporal_cost,
                chunk_size=max(1, int(getattr(self, "prompt_relay_chunk_size", chunk_size))),
            )
        else:
            attn_out = phase.cross_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
            )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, heads, head_dim)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, heads, head_dim)

            if self.cross_attn_cu_seqlens_kv_img is None:
                self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32)

            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out

    WanRunner.run_text_encoder = patched_run_text_encoder
    WanRunner.run_text_encoder._prompt_relay_patched = True

    if not hasattr(WanPreInfer, "_prompt_relay_original_pre_infer"):
        WanPreInfer._prompt_relay_original_pre_infer = WanPreInfer.infer
    WanPreInfer.infer = patched_pre_infer

    WanTransformerInfer.infer = patched_transformer_infer
    WanTransformerInfer.infer_cross_attn = patched_infer_cross_attn
    logger.info("Prompt-Relay runtime patch installed for LightX2V Wan TI2V/I2V.")
