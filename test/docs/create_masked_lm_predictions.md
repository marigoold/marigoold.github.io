### Megatron 源码阅读笔记 —— `create_masked_lm_predictions`

代码地址在 https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/dataset_utils.py#L181-L380

```python
def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""
        

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    
    # token_boundary是一个长度和tokens相同的list，保存的值只有1和0
    # 1表示对应位置的token是一个词语的开头piece，0表示对应位置的token不是一个词语的开头，比如 ##ing这样
    # 之所以要这么记录，是因为whole word mask需要把整个word进行mask
    # 如果随机选到了某个word piece进行mask，那么和它属于同一个词的其他piece都要mask
    token_boundary = [0] * len(tokens)
  
    # 下面这个循环用来修改token_boundary这个list中每个元素的值
    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
                
    # is_start_piece用来判断是否是开头的word piece，判断方法就看token对应的word piece是否以##开头
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

	# output_tokens是输入tokens的一个副本，用来输出mask之后的token
    output_tokens = list(tokens)

    # 保存mask掉数据的位置和对应的label（label其实就是token_id）
    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

	# 这里计算要mask的数量，有一个max_predictions_per_seq用来约束最大值
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

	# ngrams保存被mask的word序列最大长度（不是word piece）
	# 比如ngrams==3，那么最多可以连续mask三个单词
    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
            
	# 这里pval是不同长度单词序列被mask的概率，比如ngrams==3，那么概率就是(1, 1/2, 1/3)，归一化就是(6/11, 3/11, 2/11)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

	# 这里ngram_indexes保存所有长度==[1, max_ngrams)的单词序列
	# 比如tokens=[[1], [2, 3], [4]]，max_ngrams==3，
	# 那么ngram_indexes==[[[1]], [[1], [2, 3]], [[1], [2, 3], [4]], [[2, 3]], [[2, 3], [4]], [[4]]]
    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

	# 这里shuffle为了之后随机采样做准备，先shuffle，再拿前n个，就相当于随机采样n个
    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
                
	# covered_indexes保存被mask过的indexes
    covered_indexes = set()
    for cand_index_set in ngram_indexes:

	# 如果mask掉的单词数量超过预期就结束mask过程
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
                
	# 这个循环确实没看懂在干啥
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue
                        
	# 根据之前的pval选择一个ngrams的值来选择mask的单词序列的长度
        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

	# 把cand_index_set[0: n]的所有元素concat起来
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)
```

