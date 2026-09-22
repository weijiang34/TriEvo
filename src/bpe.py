_WORKER_TOKENIZER = None

def _init_tokenizer_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None,
):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer(vocab, merges, special_tokens)
        
def _worker_shard_process(file_path, start_offset, end_offset, worker_id, output_dir): # 去掉最后一个参数
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Worker not initialized")

    token_ids = []
    with open(file_path, 'rb') as f:
        f.seek(start_offset)
        if start_offset != 0: f.readline()
        
        while f.tell() < end_offset:
            line = f.readline().decode('utf-8', errors='ignore')
            if not line: break
            # 关键：使用 initializer 创建的全局实例
            token_ids.extend(_WORKER_TOKENIZER.encode(line))
            
    # 写入二进制文件
    output_path = os.path.join(output_dir, f"part_{worker_id}.bin")
    with open(output_path, 'ab') as f_out:
        np.array(token_ids, dtype=np.uint16).tofile(f_out)
    return len(token_ids)

class BPETokenizer():
    def __init__(self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        '''Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:'''
        self.vocab = dict(vocab)
        self.merges = merges
        self.merge_rank = {merge: idx for idx, merge in enumerate(merges)}

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_token_set = set(self.special_tokens)
        self.special_token_bytes = {token: token.encode('utf-8') for token in self.special_tokens}

        for token in self.special_tokens:
            token_bytes = self.special_token_bytes[token]
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes

        self.vocab_bytes2id = {v: k for k, v in self.vocab.items()}
        self.gpt2_re = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_re = re.compile(f"({'|'.join(re.escape(t) for t in self.special_tokens)})") if self.special_tokens else None
        self.pretoken_cache: dict[bytes, tuple[int, ...]] = {}
        self.byte_tokens = tuple(bytes([i]) for i in range(256))
    
    @classmethod
    def from_files(cls,
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens. 
        This method should accept the following additional parameters:
        '''
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def _merge_pretoken_bytes(self, pt: bytes) -> list[bytes]:
        if not pt:
            return []

        symbols = [bytes([b]) for b in pt]
        length = len(symbols)
        prev = [-1] + list(range(length - 1))
        next = list(range(1, length)) + [-1]
        alive = [True] * length
        heap = [
            (self.merge_rank[pair], left_idx, left_idx + 1, pair)
            for left_idx, pair in enumerate(zip(symbols, symbols[1:]))
            if pair in self.merge_rank
        ]
        heapq.heapify(heap)

        while heap:
            rank, left_idx, right_idx, pair = heapq.heappop(heap)
            if not alive[left_idx] or not alive[right_idx]:
                continue
            if next[left_idx] != right_idx:
                continue
            if (symbols[left_idx], symbols[right_idx]) != pair:
                continue

            merged_symbol = pair[0] + pair[1]
            symbols[left_idx] = merged_symbol
            alive[right_idx] = False

            next_idx = next[right_idx]
            next[left_idx] = next_idx
            if next_idx != -1:
                prev[next_idx] = left_idx
                new_pair = (merged_symbol, symbols[next_idx])
                new_rank = self.merge_rank.get(new_pair)
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, left_idx, next_idx, new_pair))

            prev_idx = prev[left_idx]
            if prev_idx != -1:
                new_pair = (symbols[prev_idx], merged_symbol)
                new_rank = self.merge_rank.get(new_pair)
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, prev_idx, left_idx, new_pair))

        merged_tokens = []
        current_idx = 0
        while current_idx != -1:
            if alive[current_idx]:
                merged_tokens.append(symbols[current_idx])
            current_idx = next[current_idx]
        return merged_tokens
        
    def encode(self, text: str) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        text_blocks = self.special_re.split(text) if self.special_re else [text]
        token_ids = []

        for block in text_blocks:
            if not block:
                continue

            if block in self.special_token_set:
                token_ids.append(self.vocab_bytes2id[self.special_token_bytes[block]])
                continue

            for pt in self.gpt2_re.findall(block):
                pt_bytes = pt.encode('utf-8')
                cached_ids = self.pretoken_cache.get(pt_bytes)
                if cached_ids is not None:
                    token_ids.extend(cached_ids)
                    continue

                merged_tokens = self._merge_pretoken_bytes(pt_bytes)
                ids = tuple(self.vocab_bytes2id[token] for token in merged_tokens)
                self.pretoken_cache[pt_bytes] = ids
                token_ids.extend(ids)

        return token_ids
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        for line in iterable:
            yield from self.encode(line)

    # multiprocessing encoding iterably
    def encode_file_parallel(self, doc_path, output_path, num_processes=8):
        file_size = os.path.getsize(doc_path)
        chunk_size = file_size // num_processes
        task_args = []
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        # 计算每个进程负责的起止字节位置
        for i in range(num_processes):
            start = i * chunk_size
            end = file_size if i == num_processes - 1 else (i + 1) * chunk_size
            task_args.append((doc_path, start, end, i, dirname))

        with mp.Pool(
            processes=num_processes,
            initializer=_init_tokenizer_worker, # 关键：在这里初始化
            initargs=(self.vocab, self.merges, self.special_tokens)
        ) as pool:
            # 主进程瞬间发完所有任务，因为它只发了几个数字和路径
            results = pool.starmap(_worker_shard_process, task_args)
            # 合并分开存储的二进制文件
            with open(output_path, 'wb') as f_out:
                for i in range(num_processes):
                    with open(os.path.join(dirname, f"part_{i}.bin"), 'rb') as f_in:
                        f_out.write(f_in.read())
                    os.remove(os.path.join(dirname, f"part_{i}.bin"))
        
        print(f"Total Tokens: {sum(results)}")
        return sum(results)

    def decode(self, ids: list[int]) -> str:
        '''Decode a sequence of token IDs into text.'''
        tokens = [self.vocab[id] for id in ids]
        text = b''.join(tokens).decode('utf-8', errors='replace')
        return text

import os
import regex as re
from collections import defaultdict, Counter
from typing import BinaryIO
import multiprocessing as mp
import heapq

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            #   contractions   words(EN/CN)   numbers     special chars   whitespace
GPT2_RE = re.compile(GPT2_PAT)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Helper functions
    def reverse_lex_key(token_bytes: bytes):
        return tuple(255 - b for b in token_bytes) + (256, )
    
    def make_heap_item(pair, freq, vocab):
        return (
            -freq,
            reverse_lex_key(vocab[pair[0]]),
            reverse_lex_key(vocab[pair[1]]),
            pair
        )
    
    def pop_best_pair(bp_heap, bp_freq):
        while bp_heap:
            neg_freq, neg_id0, neg_id1, pair = heapq.heappop(bp_heap)
            freq = -neg_freq
            current_freq = bp_freq.get(pair, 0)
            if freq == current_freq:
                return pair
        return None
    
    def push_pair(bp_heap, bp_freq, pair, vocab):
        freq = bp_freq.get(pair, 0)
        if freq > 0:
            heapq.heappush(bp_heap, make_heap_item(pair, freq, vocab))
    
    def build_bp_heap(bp_freq, vocab):
        bp_heap = [make_heap_item(pair, freq, vocab) for pair, freq in bp_freq.items() if freq > 0]
        heapq.heapify(bp_heap)
        return bp_heap
    
    def reorder_vocab_specials_first(vocab, special_tokens):
        len_S = len(special_tokens)
        special_token_bytes = [t.encode('utf-8') for t in special_tokens]
        for i in range(len_S + 256):
            if i < len_S:
                vocab[i] = special_token_bytes[i]
            else:
                vocab[i] = bytes([i - len_S])
        return vocab
    
    # 0. Init vocab and merges (variables should return)
    # init vocab: { id: bytes, ... }, e.g. { 399: b'<|endoftext|>', ... }
    init_vocab = {i: bytes([i]) for i in range(256)}
    init_vocab.update({i: t.encode('utf-8') for i, t in enumerate(special_tokens, start=len(init_vocab))})  # add special tokens to vocab
    merges = [] # record merges of byte-pairs
    
    # 1. Pre-processing
    pt_freq = pre_tokenize_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        cpu_percent=kwargs.get('cpu_percent', 0.8),
        num_processes=kwargs.get('num_processes', 1),
    )
    pt_idx = {idx: (pt, freq_of_pt) for idx, (pt, freq_of_pt) in enumerate(pt_freq.items())}  # pt_id -> pt_sequence: { idx: (id0, id1, ...), ... }
    
    # initialization, full-scale scanning
    bp_freq = defaultdict(int)  # byte-pair frequency: { (id0, id1): freq }
    inv_idx = defaultdict(lambda: defaultdict(int)) # inverted-index: { (id0, id1): { (id0, id1, id2, ...): freq_in_pt, ... } }
    for idx, (pt, freq_of_pt) in pt_idx.items():    # O(N)
        for pair in zip(pt, pt[1:]):
            bp_freq[pair] += freq_of_pt
            inv_idx[pair][idx] += 1
    bp_heap = build_bp_heap(bp_freq, init_vocab)

    # 2. BPE Training, V: vocab size, N: training text, L: average pre-token length
    while len(init_vocab) < vocab_size: # O(V)
        if len(bp_freq) == 0:   # if no pair to merge, break
            print(f"Length of bp_freq is 0, breaking loop.")
            break
        
        # find max pair
        max_bp = pop_best_pair(bp_heap, bp_freq)
        if max_bp is None or bp_freq[max_bp] <= 0:   # if max pair freq is 0, break
            print(f"Max pair freq is {bp_freq.get(max_bp, 0)}, breaking loop.")
            break
        max_id0, max_id1 = max_bp
        # append merge and vocab
        merges.append((init_vocab[max_id0], init_vocab[max_id1]))  # record merge as bytes
        new_id = len(init_vocab)
        init_vocab[new_id] = init_vocab[max_id0] + init_vocab[max_id1]
        
        related_pt_ids = dict(inv_idx[max_bp])  # related pt, (prev, new_id, succ)
        for j, (pt_id, freq_in_pt) in enumerate(related_pt_ids.items()):
            old_pt_seq, freq_of_pt = pt_idx[pt_id]
            new_pt_seq = []
            i = 0
            L = len(old_pt_seq)
            prev_merged = False   # flag to indicate whether the previous pair has been merged, to avoid double counting in overlapping pairs like (max_id0, max_id1, max_id1)
            # get new pt and update bp_freq and inv_idx
            while i < L:
                if i < L -1 and old_pt_seq[i] == max_id0 and old_pt_seq[i+1] == max_id1:  # if pair matches max pair
                    new_pt_seq.append(new_id)
                    # decrease freq of (prev, max_id0)
                    if i > 0 and not prev_merged:
                        prev_old_pair = (old_pt_seq[i-1], max_id0)
                        bp_freq[prev_old_pair] -= freq_of_pt
                        if bp_freq[prev_old_pair] <= 0:
                            del bp_freq[prev_old_pair]
                        else:
                            push_pair(bp_heap, bp_freq, prev_old_pair, init_vocab)
                        inv_idx[prev_old_pair][pt_id] -= 1
                        if inv_idx[prev_old_pair][pt_id] == 0:
                            del inv_idx[prev_old_pair][pt_id]
                    # decrease freq of (max_id1, succ)
                    if i < L - 2:
                        succ_pair = (max_id1, old_pt_seq[i+2])
                        bp_freq[succ_pair] -= freq_of_pt
                        if bp_freq[succ_pair] <= 0:
                            del bp_freq[succ_pair]
                        else:
                            push_pair(bp_heap, bp_freq, succ_pair, init_vocab)
                        inv_idx[succ_pair][pt_id] -= 1
                        if inv_idx[succ_pair][pt_id] == 0:
                            del inv_idx[succ_pair][pt_id]
                    prev_merged = True
                    i += 2
                else:
                    new_pt_seq.append(old_pt_seq[i])
                    prev_merged = False
                    i += 1
            new_pt_seq = tuple(new_pt_seq)
            pt_idx[pt_id] = (new_pt_seq, freq_of_pt)  # update pre-token index

            # update new pairs in bp_freq and inv_idx
            for idx, pair in enumerate(zip(new_pt_seq, new_pt_seq[1:])):
                if new_id in pair:  # (prev, new_id) or (new_id, succ)
                    bp_freq[pair] += freq_of_pt
                    push_pair(bp_heap, bp_freq, pair, init_vocab)
                    inv_idx[pair][pt_id] += 1
                    
        # del max_bp from bp_freq and inv_idx
        del bp_freq[max_bp]
        del inv_idx[max_bp]
        if len(bp_heap) > 3 * len(bp_freq):
            bp_heap = build_bp_heap(bp_freq, init_vocab)
    
    # reconstructed_bp_freq = defaultdict(int)
    # for _, (seq, freq_of_pt) in pt_idx.items():
    #     for pair in zip(seq, seq[1:]):
    #         reconstructed_bp_freq[pair] += freq_of_pt

    # assert dict(bp_freq) == dict(reconstructed_bp_freq), (
    #     f"bp_freq mismatch at step {len(merges)}\n"
    #     f"selected max_bp={max_bp}, freq={bp_freq.get(max_bp, None)}\n"
    #     f"extra_or_wrong={ {k: (bp_freq.get(k), reconstructed_bp_freq.get(k)) for k in set(bp_freq) | set(reconstructed_bp_freq) if bp_freq.get(k) != reconstructed_bp_freq.get(k)} }"
    # )
    reordered_vocab = reorder_vocab_specials_first(init_vocab, special_tokens)
    return reordered_vocab, merges

# multiprocessing worker
def process_chunk_worker(input_path, start, end, special_tokens):
    special_pat = "|".join(re.escape(t) for t in special_tokens)
    special_re = re.compile(f"({special_pat})") if special_tokens else None     # special token regex pattern, keep special tokens in the split result
    special_token_to_id = {token: 256 + i for i, token in enumerate(special_tokens)}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)

    chunk_text = chunk_data.decode("utf-8", errors="ignore")
    parts = special_re.split(chunk_text) if special_re is not None else [chunk_text]

    pt_freq_counter = Counter()
    for part in parts:
        if not part:
            continue

        if part in special_token_to_id:
            pt_freq_counter[(special_token_to_id[part],)] += 1
            continue

        for token in GPT2_RE.findall(part):
            pt_freq_counter[tuple(token.encode("utf-8"))] += 1

    return pt_freq_counter
    
def pre_tokenize_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    **kwargs,
) -> dict[tuple[int, ...], int]:
    """Pre-tokenize the input text and return the frequency of each pre-token.

    Args:
        input_path (str | os.PathLike): The input text or path to the text file to be pre-tokenized.
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        dict[tuple[int, ...], int]: A dictionary mapping each pre-token (represented as a tuple of byte IDs) to its frequency in the input text.
    """
    
    # chunk file
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_tokens: list[bytes],
    ):
        assert [isinstance(t, bytes) for t in split_special_tokens], "Must represent special tokens as a list of bytestrings"
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        mini_chunk_size = 4096
        if file_size <= mini_chunk_size:
            return [0, file_size]
        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_ats = []
                for t in split_special_tokens:
                    found_ats.append(mini_chunk.find(t))
                if len(found_ats) == 0:
                    break
                found_at = min([pos for pos in found_ats if pos != -1], default=-1)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        return sorted(set(chunk_boundaries))

    pt_freq_counter = Counter()
    
    with open(input_path, 'rb') as f:
        cpu_percent = kwargs.get('cpu_percent', 1)
        num_processes = max(1, int((os.cpu_count() or 4) * cpu_percent)) if kwargs.get('num_processes') is None else kwargs['num_processes']
        print(f"File size: {os.path.getsize(input_path)} bytes, using {num_processes} processes for pre-tokenization.")
        boundaries = find_chunk_boundaries(f, num_processes, [t.encode('utf-8') for t in special_tokens])
        if len(boundaries) == 2:
            # print(f"File is small enough to process in a single chunk, skipping multiprocessing.")
            pt_freq_counter.update(process_chunk_worker(input_path, boundaries[0], boundaries[-1], special_tokens))
        else:
            # multiprocessing pool
            with mp.Pool(processes=num_processes) as pool:
                results = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    results.append(pool.apply_async(process_chunk_worker, args=(input_path, start, end, special_tokens)))
                pool.close()
                pool.join()
                for r in results:
                    pt_freq_counter.update(r.get())
                
    return dict(pt_freq_counter)