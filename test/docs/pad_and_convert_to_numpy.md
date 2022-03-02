### Megatron源码阅读笔记——`pad_and_convert_to_numpy`

https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/t5_dataset.py#L160-L230

```python
def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id,
                             max_seq_length, max_seq_length_dec,
                             masked_spans=None, bos_id=None,
                             eos_id=None, sentinel_tokens=None):
    """Pad sequences and convert them to numpy."""

    # 这个部分没有理解为什么要用sentinel_tokens
    sentinel_tokens = collections.deque(sentinel_tokens)
    t5_input = []
    
    # decoder开头一个begin of sequence
    (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
    (start_index, end_index) = (0, None)
    
    # masked_spans是一个[masked_indexes, masked_labels]的集合
    for span in masked_spans:
        flag = sentinel_tokens.popleft()

        # Append the same tokens in decoder input and output
        t5_decoder_in.append(flag)
        t5_decoder_in.extend(span.label)
        t5_decoder_out.append(flag)
        t5_decoder_out.extend(span.label)

    # 这里end_index是当前span中被masked的第一个token的index
        end_index = span.index[0]
        t5_input.extend(tokens[start_index: end_index])
    # 我也不知道这里flag是啥意思
        t5_input.append(flag)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add <eos> token to the t5_decoder_out
    t5_decoder_out.append(eos_id)

    # Add the remaining tokens to the t5 input
    t5_input.extend(tokens[start_index:])

    # assert (len(t5_input) - len(masked_spans)) + \
    #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

    # Some checks.

    # Encoder-side padding mask.
    num_tokens = len(t5_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(t5_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(t5_decoder_in)
    padding_length_dec = max_seq_length_dec - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in)

   	# 这里用masked掉的tokens当成labels，用作预训练
    # Labels mask.
    labels = t5_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokens_dec_in, labels, enc_mask, \
           dec_mask, enc_dec_mask, loss_mask
```

