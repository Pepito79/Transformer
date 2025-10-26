from pathlib import Path
from config import get_config, get_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset_py import BillingualDataset
import torch
from train import get_or_build_tokenizer
import sys

def translate(sentence: str):
    config = get_config()
    ds = load_dataset(f"opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split='all')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer_src = get_or_build_tokenizer(config, ds, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config["lang_tgt"])

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                              config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    model_filename = get_weights_file_path(config, "19")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    seq_len = config['seq_len']
    model.eval()

    with torch.no_grad():
        # Encode source sentence to ids
        enc = tokenizer_src.encode(sentence)
        src_ids = enc.ids

        # Truncate if too long (reserve 2 tokens for SOS/EOS)
        max_payload = seq_len - 2
        if len(src_ids) > max_payload:
            src_ids = src_ids[:max_payload]

        # Build source tensor with SOS, ids, EOS, PAD to fixed seq_len
        src_tensor = torch.tensor(
            [tokenizer_src.token_to_id('[SOS]')] + src_ids + [tokenizer_src.token_to_id('[EOS]')] +
            [tokenizer_src.token_to_id('[PAD]')] * (seq_len - (2 + len(src_ids))),
            dtype=torch.long
        ).unsqueeze(0).to(device)  # <-- important: add batch dim (1, seq_len)

        # Build source mask: 1 where not PAD, 0 where PAD
        pad_id_src = tokenizer_src.token_to_id('[PAD]')
        source_mask = (src_tensor != pad_id_src).unsqueeze(1).unsqueeze(2).int().to(device)
        # shape: (batch=1, 1, 1, seq_len) -- broadcastable with attention score shape

        encoder_output = model.encode(src_tensor, source_mask)

        # Initialize decoder input with SOS token (batch dim)
        sos_id = tokenizer_tgt.token_to_id('[SOS]')
        eos_id = tokenizer_tgt.token_to_id('[EOS]')
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long).to(device)  # shape (1,1)

        print(f"SOURCE: {sentence}")
        print("PREDICTED: ", end='')

        for _ in range(seq_len - 1):
            # Build causal mask for decoder self-attention: 1 = allowed, 0 = blocked
            L = decoder_input.size(1)
            # tril gives lower triangular ones, shape (L,L) with 1 on diag and below
            causal = torch.tril(torch.ones((L, L), dtype=torch.uint8, device=device))
            # expand to (batch, 1, L, L) to match attention broadcasting
            decoder_mask = causal.unsqueeze(0).unsqueeze(1)  # (1,1,L,L)

            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            prob = model.project(out[:, -1])  # (batch, vocab)
            _, next_word = torch.max(prob, dim=1)  # next_word is tensor([id])

            next_token_id = next_word.item()
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]], device=device)], dim=1)

            token_str = tokenizer_tgt.decode([next_token_id])
            print(token_str, end=' ')

            if next_token_id == eos_id:
                break

    # Return full decoded sequence as string (including SOS/EOS maybe -- you can clean)
    return tokenizer_tgt.decode(decoder_input.squeeze(0).tolist())
