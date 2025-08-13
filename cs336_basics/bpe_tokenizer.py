import os
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

from cs336_basics.pretokenization_example import pre_tokenize
from tqdm import tqdm


def get_new_word_representation(
    word_representation: tuple[bytes], 
    merged: bytes
) -> tuple[bytes]:
    # form new word representation
    new_word_representation, skip_next = [], False
    N = len(word_representation)
    for i in range(N):
        if skip_next:
            skip_next = False
            continue
        if (i < N - 1) and word_representation[i]+word_representation[i+1] == merged:
            # new token used for word
            new_word_representation.append(merged) 
            skip_next = True
        else:
            # old token used for word
            new_word_representation.append(word_representation[i]) 
    return tuple(new_word_representation)


def process_word_with_merge(
    word: str, 
    new_token: bytes, 
    word_to_tokens: dict[str, tuple[bytes]],
    word_count: dict[str, int]
) -> tuple:
    old_word_representation = word_to_tokens[word]
    new_word_representation = get_new_word_representation(
        old_word_representation, new_token
    )
    # print(f"Before: {old_word_representation}\nAfter: {new_word_representation}\nReplacing: {other_pairs_to_replace}\n\n")

    # new word representation brings new pairs
    # 1. update pair_frequency
    # 2. update pair_to_words
    this_word_count = word_count[word]
    this_pair_frequency = Counter()
    for (b1, b2) in zip(new_word_representation[:-1], new_word_representation[1:]):
        this_pair_frequency[(b1, b2)] += this_word_count
    for (b1, b2) in zip(old_word_representation[:-1], old_word_representation[1:]):
        this_pair_frequency[(b1, b2)] -= this_word_count

    # send everything needed to apply update at global level
    return (word, new_word_representation, this_pair_frequency)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args:
        input_path: str 
            Path to a text file with BPE tokenizer training data.
        vocab_size: int 
            A positive integer that defines the maximum final vocabulary size 
            (including the initial byte vocabulary, vocabulary items produced from 
            merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. 
            These special tokens do not otherwise affect BPE training.
    
    Returns the resulting vocabulary and merges:
        vocab: dict[int, bytes] 
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) 
            to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] 
            A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that 
            <token1> was merged with <token2>. The merges should be ordered by 
            order of creation.
    """
    # initialize vocab
    vocab = {x: bytes((x,)) for x in range(256)}
    merges = []
    num_merges = vocab_size - (len(vocab) + len(special_tokens))

    # pre-tokenize
    word_count = pre_tokenize(input_path, special_tokens)
    # initialize representation of word with byte tokens
    word_tokens = {
        k: tuple(bytes([x]) for x in k.encode("utf-8"))
        for k in word_count
    }
    # start merging
    # construct pair count table and pair to words cache
    pair_to_words: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    pair_frequency: Counter[tuple[bytes, bytes], int] = Counter()
    for word, v in word_count.items():
        word_representation = word_tokens[word]
        for (b1, b2) in zip(word_representation[:-1], word_representation[1:]):
            pair_to_words[(b1, b2)].add(word)
            pair_frequency[(b1, b2)] += v
    
    for i in tqdm(range(num_merges), desc="[bpe_tokenizer] Training BPE", ncols=100):
        ####
        # Algorithm:
        # 1. find the most frequent pair and add to vocab
        # 2. for all words that contained this pair:
        #  a. form a new word representation
        #  b. form new pairs and count their frequency in pair_frequency using word_count
        #  c. update word_count to use new word representation
        #  d. update pair_to_words to use new word representation
        # 3. clean-up everything related to the pair
        ####
        # print(f"iteration {i}: {len(pair_frequency)}")
        # step 1
        # find most common pair
        sorted_pairs = sorted(
            pair_frequency.items(),
            key=lambda x: (x[1], x[0]), # (count, lexicographical order)
            reverse=True
        )
        this_merge: tuple[bytes, bytes] = sorted_pairs[0][0]
        # merge and form new token
        merges.append(this_merge)
        new_token, new_token_idx = this_merge[0] + this_merge[1], len(vocab)
        vocab[new_token_idx] = new_token
        
        # step 2
        # update representation and stats for all words containing this_merge pair
        target_words = pair_to_words[this_merge]
        # next for loop can be parallelized
        # results = []
        # for word in target_words:
        #     results.append(process_word_with_merge(
        #          word, new_token, word_tokens, word_count
        #      ))
        func = partial(
            process_word_with_merge,
            new_token=new_token, 
            word_to_tokens=word_tokens,
            word_count=word_count
        )
        with Pool(processes=os.cpu_count()) as p:
            results = p.map(func, target_words)
        
        # reduce results globally
        for res in results:
            (word, new_word_representation, 
             this_pair_frequency) = res
            old_word_representation = word_tokens[word]
            # perform updates
            word_tokens[word] = new_word_representation
            pair_frequency.update(this_pair_frequency)
            for (b1, b2) in zip(old_word_representation[:-1], old_word_representation[1:]):
                if word in pair_to_words[(b1, b2)]:
                    # need to check above because (b1, b2) may have been 
                    # removed in this iteration itself
                    # e.g., 't','e','s','t','e','d' and we remove 't','e'
                    # so it gets removed once already
                    pair_to_words[(b1, b2)].remove(word)
            for (b1, b2) in zip(new_word_representation[:-1], new_word_representation[1:]):
                pair_to_words[(b1, b2)].add(word)

        # step 3
        # forget this token as a pair
        del pair_to_words[this_merge]
        del pair_frequency[this_merge]
        current_pairs = list(pair_frequency.keys())
        for p in current_pairs:
            if pair_frequency[p] == 0:
                del pair_frequency[p]

    # add special tokens to vocab
    for i, token in enumerate(special_tokens):
        vocab[i + len(vocab)] = token.encode("utf-8")
    
    return vocab, merges


if __name__ == "__main__":
    import sys
    vocab, merges = train_bpe(sys.argv[1], int(sys.argv[2]), ["<|endoftext|>"])
    print("Vocab size:", len(vocab))
    print("Merges:", len(merges))
    import ipdb; ipdb.set_trace()
