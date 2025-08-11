import os
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from tqdm import tqdm


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size 
                        for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer 
    # than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(boundary: tuple[str, str], path=None, special_tokens=None) -> Counter:
    start, end = boundary
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(path, "rb") as f:
        f.seek(start)
        doc = f.read(end - start).decode("utf-8", errors="ignore")
    # Run pre-tokenization on your chunk and store the counts 
    # for each pre-tokens
    chunk_strs = re.split("|".join(special_tokens), doc)
    c = Counter()
    for chunk_str in chunk_strs:
        pre_tokens = re.findall(PAT, chunk_str)
        c.update(pre_tokens)
    return c


def pre_tokenize(path: str, special_tokens: list, multiprocess: bool = True) -> Counter[str, int]:
    print("[pretokenization_example] Reading from:", path)
    num_processes = os.cpu_count()
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # Process
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    func = partial(process_chunk, path=path, special_tokens=special_tokens)
    if multiprocess:
        with Pool(processes=num_processes) as p:
            pretoken_counts = list(tqdm(
                p.imap_unordered(func, chunks), total=len(chunks),
                desc="[pretokenization_example] Pre-tokenizing", ncols=100
            ))
    else:
        pretoken_counts = []
        for chunk in tqdm(chunks, desc="[pretokenization_example] Pre-tokenizing", ncols=100): 
            pretoken_counts.append(func(chunk))
    total_counts = Counter()
    for c in pretoken_counts:
        total_counts.update(c)
    return total_counts


if __name__ == "__main__":
    ## Usage
    import sys
    path = sys.argv[1]
    special_tokens = ["<|endoftext|>"]

    total_counts = pre_tokenize(path, special_tokens, multiprocess=False)
    sorted = total_counts.most_common()
    print("Count of pre-tokens: ", len(sorted))
    print("10 most common: ", sorted[:10])
    print("10 least common: ", sorted[-10:])
    print("10 random: ", sorted[23453:23463])
