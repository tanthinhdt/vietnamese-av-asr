import pickle
import torch
import onnxruntime as ort
import numpy as np


def load_components(metadata_dict: dict):
    component_dict = dict()

    with open(metadata_dict['saved_cfg_path'], 'rb') as f:
        component_dict['cfg'] = pickle.load(f)

    avhubert_llm_layyer = torch.nn.Linear(
        in_features=1024,
        out_features=2560,
        bias=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    avhubert_llm_layyer.load_state_dict(
        torch.load(metadata_dict['avfeat_to_llm_path'])
    )
    avhubert_llm_layyer.eval()
    component_dict['projector'] = avhubert_llm_layyer

    component_dict['ins'] = torch.load(metadata_dict['ins_path'])

    component_dict['encoder'] = ort.InferenceSession(
        path_or_bytes=metadata_dict['encoder_path'],
        providers=['CPUExecutionProvider'],
    )

    component_dict['decoder'] = ort.InferenceSession(
        path_or_bytes=metadata_dict['decoder_path'],
        providers=['CPUExecutionProvider'],
    )

    return component_dict

def avhubert_llm_run(
        tokenizer,
        encoder_ort: ort.InferenceSession,
        decoder_ort: ort.InferenceSession,
        avfeat_to_llm: torch.nn.Module,
        instruction_embedding: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
        cluster_counts: torch.Tensor,

) -> torch.Tensor:
    encoder_input_feed = dict()
    if video is not None:
        encoder_input_feed["video"] = video.numpy()
    if audio is not None:
        encoder_input_feed["audio"] = audio.numpy()

    encoder_output = torch.from_numpy(encoder_ort.run(
        None,
        input_feed=encoder_input_feed,
    )[0])

    encoder_output = avfeat_to_llm(encoder_output)

    results_tensor = []
    start_idx = 0

    for clutser_num in cluster_counts:
        end_idx = start_idx + clutser_num
        slice = encoder_output[:, start_idx:end_idx, :]
        mean_tensor = torch.mean(slice, dim=1, keepdim=True)
        results_tensor.append(mean_tensor)
        start_idx = end_idx

    assert (cluster_counts.sum().item() == encoder_output.size()[1])

    reduced_enc_out = torch.cat(results_tensor, dim=1).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    llm_input = torch.cat((instruction_embedding, reduced_enc_out), dim=1).detach()

    # config = PretrainedConfig.from_pretrained('vilm/vinallama-2.7b')
    # ort_model = ORTModelForCausalLM(model=decoder_ort, config=config, use_cache=False, use_io_binding=False)
    # inputs = {
    #     'inputs_embeds': llm_input,
    #     'top_p': top_p,
    #     'num_beams': num_beams,
    #     'max_new_tokens': max_length,
    #     'min_length': min_length,
    #     'repetition_penalty': repetition_penalty,
    #     'do_sample': True,
    #     'length_penalty': length_penalty,
    # }
    # decoder_output = ort_model.generate(
    #     **inputs
    # )

    decoder_input_feed = {
        "llm_input": llm_input.numpy(),
    }

    decoder_output = torch.from_numpy(decoder_ort.run(
        None,
        input_feed=decoder_input_feed
    )[0])

    best_hypo = torch.argmax(decoder_output, dim=-1)

    return best_hypo

# def beam_search(logits, tokenizer, num_beams=20, max_length=30, min_length=1, top_p=0.9, repetition_penalty=1.0,
#                 length_penalty=0.0):
#     """
#     Perform beam search given the logits from the forward method.
#
#     Args:
#     logits (np.array): Logits output from the forward method of shape (batch_size, sequence_length, vocab_size)
#     num_beams (int): Number of beams for beam search.
#     max_length (int): Maximum length of the generated sequences.
#     min_length (int): Minimum length of the generated sequences.
#     top_p (float): Top-p sampling for nucleus sampling.
#     repetition_penalty (float): Penalty for repeated words.
#     length_penalty (float): Penalty to apply based on the length of the sequence.
#
#     Returns:
#     np.array: Generated token IDs.
#     """
#
#     batch_size, sequence_length, vocab_size = logits.shape
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Initialize the sequences with the initial input_ids
#     sequences = torch.full((batch_size * num_beams, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
#
#     # Initialize scores
#     scores = torch.zeros((batch_size * num_beams, 1), dtype=torch.float, device=device)
#
#     # Convert logits to log-probabilities
#     logits = torch.tensor(logits, device=device)
#     log_probs = torch.log_softmax(logits, dim=-1)
#
#     for step in range(max_length):
#         # Compute log_probs for all beams
#         log_probs_for_beams = log_probs.unsqueeze(1).expand(batch_size, num_beams, sequence_length, vocab_size)
#
#         # Apply repetition penalty
#         if repetition_penalty != 1.0:
#             for i in range(num_beams):
#                 log_probs_for_beams[:, i] /= torch.gather(sequences[:, i], 1, sequences[:, i]) * repetition_penalty
#
#         # Compute new scores and add length penalty
#         new_scores = scores.unsqueeze(-1) + log_probs_for_beams.view(batch_size * num_beams, -1)
#
#         if length_penalty != 0.0:
#             new_scores /= (step + 1) ** length_penalty
#
#         # Select top-k candidates
#         topk_scores, topk_indices = torch.topk(new_scores, num_beams, dim=-1, largest=True, sorted=True)
#
#         # Update sequences and scores
#         topk_beams = topk_indices // vocab_size
#         topk_tokens = topk_indices % vocab_size
#
#         sequences = torch.cat(
#             [torch.gather(sequences, 1, topk_beams.unsqueeze(-1).expand(batch_size * num_beams, step + 1)),
#              topk_tokens.unsqueeze(-1)], dim=-1)
#         scores = topk_scores
#
#         # Check for completion (i.e., if all sequences end with EOS token)
#         if step >= min_length and (topk_tokens == tokenizer.eos_token_id).all():
#             break
#
#     return sequences