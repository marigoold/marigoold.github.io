### GPTDataset

函数地址：

https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/gpt_dataset.py#L139-L187

函数功能：

创建适用于GPT模型的dataset。

```python
class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
	# doc_idx是indexed_dataset所有document复制num_epochs遍之后打乱的结果，shape=(len(document) * num_epochs, )
    # sample_idx是[doc_idx_index, offset]的list，shape=(num_samples+1, 2)，每两行之间是一个sample
    # 这里多说一句，sample_idx是个区间端点的idx，所以要取一个sample需要用两个元素
    # 比如取第3个sample（从0开始计数），应该用sample_idx[3]和sample_idx[4]来取，
    # 如果sample_idx[3] = [4, 10], sample_idx[4] = [5, 2]
    # 那么第三个sample就是indexed_dataest中第4个文档第10个词开始，到第5个文档第2个词结束（不包含第2个词）
    # shuffle_idx应该是shape=(sample_idx.shape[0], )，单纯用于打乱
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
	# 这里其实没看懂length为什么要+1
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'text': np.array(sample, dtype=np.int64)}
```



### build_sample_idx

函数地址：

https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/helpers.cpp#L99-L185

函数功能：

把indexed_dataset的所有document铺平，按照给定的seq_length切开，生成不同的训练sample，输出的sample_idx是一个n\*2的数组。每一行的第一列是该sample对应的doc_idx，第二列是该sample开头位置在该doc_idx下doc对应的offset。

```c++
py::array build_sample_idx(const py::array_t<int32_t> &sizes_,
                           const py::array_t<int32_t> &doc_idx_,
                           const int32_t seq_length, const int32_t num_epochs,
                           const int64_t tokens_per_epoch) {
  /* Sample index (sample_idx) is used for gpt2 like dataset for which
     the documents are flattened and the samples are built based on this
     1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
     where [..., 0] contains the index into `doc_idx` and [..., 1] is the
     starting offset in that document.*/

  // 这里输入sizes_参数是每个sentence的长度，doc_idx_是每个doc开头的索引
  // Consistency checks.
  assert(seq_length > 1);
  assert(num_epochs > 0);
  assert(tokens_per_epoch > 1);

  // Remove bound checks.
  auto sizes = sizes_.unchecked<1>();
  auto doc_idx = doc_idx_.unchecked<1>();

  // Mapping and it's length (1D).
  int64_t num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length;
  int32_t *sample_idx = new int32_t[2 * (num_samples + 1)];

  cout << "    using:" << endl << std::flush;
  cout << "     number of documents:       " << doc_idx_.shape(0) / num_epochs
       << endl
       << std::flush;
  cout << "     number of epochs:          " << num_epochs << endl
       << std::flush;
  cout << "     sequence length:           " << seq_length << endl
       << std::flush;
  cout << "     total number of samples:   " << num_samples << endl
       << std::flush;

  // Index into sample_idx.
  int64_t sample_index = 0;
  // Index into doc_idx.
  int64_t doc_idx_index = 0;
  // Begining offset for each document.
  int32_t doc_offset = 0;
  // Start with first document and no offset.
  sample_idx[2 * sample_index] = doc_idx_index;
  sample_idx[2 * sample_index + 1] = doc_offset;
  ++sample_index;

  while (sample_index <= num_samples) {
    // Start with a fresh sequence.
    int32_t remaining_seq_length = seq_length + 1;
    while (remaining_seq_length != 0) {
      // Get the document length.
      auto doc_id = doc_idx[doc_idx_index];
      auto doc_length = sizes[doc_id] - doc_offset;
      // And add it to the current sequence.
      remaining_seq_length -= doc_length;
      // If we have more than a full sequence, adjust offset and set
      // remaining length to zero so we return from the while loop.
      // Note that -1 here is for the same reason we have -1 in
      // `_num_epochs` calculations.
	// 这里当remaining_seq_length <= 0的时候，doc_idx_index不变，但是offset要增加到当前doc的最后一个token（why）
      if (remaining_seq_length <= 0) {
        doc_offset += (remaining_seq_length + doc_length - 1);
        remaining_seq_length = 0;
      } else {
        // Otherwise, start from the begining of the next document.
        ++doc_idx_index;
        doc_offset = 0;
      }
    }
    // Record the sequence.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;
  }

  // Method to deallocate memory.
  py::capsule free_when_done(sample_idx, [](void *mem_) {
    int32_t *mem = reinterpret_cast<int32_t *>(mem_);
    delete[] mem;
  });

  // Return the numpy array.
  const auto byte_size = sizeof(int32_t);
  return py::array(std::vector<int64_t>{num_samples + 1, 2}, // shape
                   {2 * byte_size, byte_size}, // C-style contiguous strides
                   sample_idx,                 // the data pointer
                   free_when_done);            // numpy array references
}

```



### _build_doc_idx

函数地址：

https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/gpt_dataset.py#L346-L359

函数功能：

把documents. (np.ndarray) 复制num_epochs份，然后shuffle，虽然我不知道为什么这么做。打乱之后，同一个doc可能在一个epoch训练多次。

```python
def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
    # doc_idx是一个2维np.ndarray，一共num_epochs行，每一行是np.arange(len(documents))
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
    # flatten之后再打乱，比如[[1, 2, 3], [1, 2, 3]]变成[2, 3, 2, 1, 3, 1]
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))
```



### _build_index_mappings

函数地址：

https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/data/gpt_dataset.py#L190-L323

函数功能：

把indexed_dataset中的文档抽取成samples的indices。

代码解析（删掉了一些简单的的代码）：

```python
def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

           
            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
	# 生成打乱后的doc_idx, shape是(len(documents) * epochs, )
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
	# 这里可以见上面的代码解析，把doc_idx和sizes一起生成sample_idx，sample_idx是[doc_idx_index, offset]的list
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
	# 这里应该就是单纯的打乱，所以不看代码实现了
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))


    return doc_idx, sample_idx, shuffle_idx
```

