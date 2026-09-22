import torch
from itertools import islice, product
from typing import Dict, Iterable, Iterator, List, Tuple, Union

class KmerTokenizer:
    """Character-independent DNA tokenizer based on overlapping k-mers."""

    def __init__(
        self, 
        kmer_size: int = 3, 
        stride: int = 1,
        add_reverse_complement: bool = False,
    ):
        if kmer_size < 1 or stride < 1 or stride > kmer_size:
            raise ValueError("kmer_size must be >= stride >= 1")

        self.kmer_size = kmer_size
        self.stride = stride
        self.add_reverse_complement = add_reverse_complement
        self._complement_table = str.maketrans("ATCGN", "TAGCN")

        kmers = ("".join(kmer) for kmer in product("ATCG", repeat=kmer_size))
        self.token_to_id: Dict[str, int] = {
            kmer: token_id for token_id, kmer in enumerate(kmers)
        }
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        self.token_to_id.update({
            token: len(self.token_to_id) + offset
            for offset, token in enumerate(special_tokens)
        })
        self.id_to_token: Dict[int, str] = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

        self.pad_token_id = self.token_to_id["[PAD]"]
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.bos_token_id = self.token_to_id["[BOS]"]
        self.eos_token_id = self.token_to_id["[EOS]"]
        self.vocab_size = len(self.token_to_id)

    def reverse_complement(self, sequence: str) -> str:
        """Return the reverse-complement DNA sequence."""
        return sequence.translate(self._complement_table)[::-1]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert a DNA sequence into stride-spaced k-mer token IDs."""
        return [token_id for block in self.encode_iter(
            text, add_bos, add_eos
        ) for token_id in block]

    def encode_iter(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        block_size: int = 8192,
    ) -> Iterator[List[int]]:
        """Yield k-mer IDs in bounded blocks instead of one large list."""
        if block_size < 1:
            raise ValueError("block_size must be positive")

        sequence = "".join(text.upper().split())
        sequences = [sequence]
        if self.add_reverse_complement:
            sequences.append(self.reverse_complement(sequence))
        
        block = []
        for seq in sequences:
            if add_bos:
                if len(block) == block_size:
                    yield block
                    block = []
                block.append(self.bos_token_id)

            for start in range(0, len(seq) - self.kmer_size + 1, self.stride):
                if len(block) == block_size:
                    yield block
                    block = []
                block.append(self.token_to_id.get(
                    seq[start:start + self.kmer_size], self.unk_token_id
                ))

            if add_eos:
                if len(block) == block_size:
                    yield block
                    block = []
                block.append(self.eos_token_id)

        if block:
            yield block

    def decode(
        self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False
    ) -> str:
        """Reconstruct a sequence from k-mer IDs, preserving k-mer overlap."""
        return "".join(self.decode_iter(ids, skip_special_tokens))

    def decode_iter(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> Iterator[str]:
        """Yield decoded pieces without accumulating the complete sequence."""
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()

        special_tokens = {"[PAD]", "[BOS]", "[EOS]", "[UNK]"}
        previous_was_kmer = False
        for idx in ids:
            token = self.id_to_token.get(int(idx), "[UNK]")
            if token in special_tokens:
                if not skip_special_tokens:
                    yield token
                previous_was_kmer = False
            elif previous_was_kmer:
                yield token[self.kmer_size - self.stride:]
            else:
                yield token
            previous_was_kmer = token not in special_tokens

    def __len__(self) -> int:
        return self.vocab_size


def build_tokenizer(config) -> KmerTokenizer:
    """Build the configured tokenizer and keep construction in one place."""
    if config.kind != "kmer":
        raise ValueError(f"Unsupported tokenizer kind: {config.kind}")
    return KmerTokenizer(
        kmer_size=config.kmer_size,
        stride=config.stride,
        add_reverse_complement=getattr(config, "add_reverse_complement", False),
    )

import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import numpy.typing as npt
from Bio import SeqIO


def _encode_sequences(
    tokenizer,
    sequences: Iterable[str],
    encode_params: Dict = None,
    dtype=np.uint16,
) -> npt.NDArray:
    """Encode and flatten a sequence batch through the tokenizer API."""
    encode_params = encode_params or {}
    encoded_sequences = [
        np.asarray(tokenizer.encode(sequence, **encode_params), dtype=dtype)
        for sequence in sequences
    ]
    return (
        np.concatenate(encoded_sequences)
        if encoded_sequences
        else np.empty(0, dtype=dtype)
    )


def _encode_fasta_chunk_to_bin(args: Tuple) -> Tuple[str, int]:
    """Encode one FASTA chunk and write its concatenated IDs to a temp file."""
    (
        records,
        tokenizer,
        encode_params,
        temp_path,
        dtype,
    ) = args
    encoded = _encode_sequences(tokenizer, records, encode_params, dtype)
    encoded.tofile(temp_path)
    return temp_path, len(encoded)


def encode_from_fasta_to_bin(
    fasta_path: str,
    tokenizer_type,
    tokenizer_params: Dict,
    output_path: str,
    encode_params: Dict = None,
    num_workers: int = 8,
    batch_size: int = 256,
    dtype=np.uint16,
) -> int:
    """Multiprocess-encode FASTA records into one flat uint16 binary file.

    Each worker concatenates its records into a temporary binary file. The main
    process then merges temporary files in FASTA order and removes them.
    """
    encode_params = encode_params or {}
    if num_workers < 1 or batch_size < 1:
        raise ValueError("num_workers and batch_size must be positive")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    open(output_path, "wb").close()
    tokenizer = tokenizer_type(**tokenizer_params)
    records = (str(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))

    def chunks():
        while batch := list(islice(records, batch_size)):
            yield batch

    total_tokens = 0
    with tempfile.TemporaryDirectory(prefix="fasta_encode_", dir=output_dir) as temp_dir:
        worker_args = (
            (
                batch,
                tokenizer,
                encode_params,
                os.path.join(temp_dir, f"chunk_{chunk_id:08d}.bin"),
                dtype,
            )
            for chunk_id, batch in enumerate(chunks())
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for temp_path, token_count in executor.map(
                _encode_fasta_chunk_to_bin, worker_args
            ):
                with open(temp_path, "rb") as source, open(output_path, "ab") as target:
                    shutil.copyfileobj(source, target)
                total_tokens += token_count

    return total_tokens


def split_fasta_to_flat_numpy(
    fasta_path: str,
    out_dir: str,
    tokenizer,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 8,
    batch_size: int = 256,
    encode_params: Dict = None,
    keep_temp_fasta: bool = True,
) -> Tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
    """
    划分 FASTA 后，依次调用 encode_from_fasta_to_bin 生成三个二进制文件。

    ``keep_temp_fasta=True`` 时保留拆分后的 FASTA 文件；设为 ``False``
    时在函数结束后自动删除它们。
    """
    if num_workers < 1 or batch_size < 1:
        raise ValueError("num_workers and batch_size must be positive")
    os.makedirs(out_dir, exist_ok=True)
    if encode_params is None:
        encode_params = {"add_bos": True, "add_eos": True}

    print("正在从 FASTA 中提取序列...")
    records = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(fasta_path, "fasta")
        if len(record.seq) > 0
    ]
    random.seed(random_seed)
    random.shuffle(records)

    total_count = len(records)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }
    tokenizer_params = {
        "kmer_size": tokenizer.kmer_size,
        "stride": tokenizer.stride,
        "add_reverse_complement": tokenizer.add_reverse_complement,
    }

    temp_dir = tempfile.mkdtemp(prefix="fasta_split_", dir=out_dir)
    try:
        encoded_paths = {}
        split_stats = {}
        for split_name, split_records in splits.items():
            split_fasta = os.path.join(temp_dir, f"{split_name}.fasta")
            with open(split_fasta, "w") as output:
                for record_id, sequence in split_records:
                    output.write(f">{record_id}\n{sequence}\n")

            output_path = os.path.join(out_dir, f"{split_name}_vir.bin")
            token_count = encode_from_fasta_to_bin(
                fasta_path=split_fasta,
                tokenizer_type=type(tokenizer),
                tokenizer_params=tokenizer_params,
                output_path=output_path,
                encode_params=encode_params,
                num_workers=num_workers,
                batch_size=batch_size,
                dtype=np.uint16,
            )
            encoded_paths[split_name] = output_path
            split_stats[split_name] = {
                "bases": sum(len(sequence) for _, sequence in split_records),
                "tokens": token_count,
            }
    finally:
        if not keep_temp_fasta:
            shutil.rmtree(temp_dir)

    with open(os.path.join(out_dir, "dataset_info.csv"), "w") as f:
        f.write(f"#train:val:test={train_ratio}:{val_ratio}:{test_ratio}\n")
        f.write(
            f"#add_bos: {encode_params.get('add_bos', False)}, "
            f"add_eos: {encode_params.get('add_eos', False)}, "
            f"add_rc: {tokenizer.add_reverse_complement}\n"
        )
        f.write(f"#kmer_size: {tokenizer.kmer_size}, stride: {tokenizer.stride}\n")
        f.write(f"split_name,bases,tokens\n")
        for split_name in ["train", "val", "test"]:
            stats = split_stats[split_name]
            f.write(f"{split_name},{stats['bases']},{stats['tokens']}\n")

    return tuple(
        np.fromfile(encoded_paths[name], dtype=np.uint16)
        for name in ("train", "val", "test")
    )




def main():
    rep_ani_95_path = "/home/wjiang34/LLM/TriEvo/dataset/refseq_viral/cluster/rep_contigs_ani_95.fasta"
    raw_path = "/home/wjiang34/LLM/TriEvo/dataset/refseq_viral/ncbi_dataset/data/genomic.fna"
    fasta_path = rep_ani_95_path
    add_rc = True

    tokenizers = [
        # (1, 1),
        # (3, 1), 
        # (6, 1),
        # (6, 2),
        (6, 6)
    ]

    # 💡 优化点：自动读取 PBS 分配的 CPU 核心数，如果没有读取到则默认用 8 核
    num_workers = int(os.environ.get("PBS_NUM_PPN", 8))

    for kmer_size, stride in tokenizers:
        tokenizer = KmerTokenizer(kmer_size=kmer_size, stride=stride, add_reverse_complement=add_rc)
        out_dir = f"/home/wjiang34/LLM/TriEvo/dataset/refseq_viral/rep_ani_95/k{kmer_size}_s{stride}{'_rc' if add_rc else ''}"

        split_fasta_to_flat_numpy(
            fasta_path=fasta_path,
            out_dir=out_dir,
            tokenizer=tokenizer,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            num_workers=num_workers,
            batch_size=256,
            encode_params={"add_bos": False, "add_eos": True},
            keep_temp_fasta=True,
        )

if __name__ == "__main__":
    main()
    
