# Assignment 1

## Problem (unicode1)

(a) `chr(0)` returns the null character `\x00`.

(b) `chr(0).__repr__()` gives `"\\x00"`. `print(chr(0))` outputs nothing, i.e., a null string.

(c) wherever this string occurs in text, when printing it shows up as a null string.


## Problem (unicode2)

(a) UTF-16 and UTF-32 have a minimum of two and four bytes per character, making the encoded sequence length very long. In comparison, UTF-8 uses a minimum of one byte per character. UTF-16 and UTF-32 are beneficial if using UTF-8 instead leads to two or more bytes per character on average.

(b) The function is incorrect because it assumes each charcter is a single byte long. In practice, however, if we use a string like "こんにちは", we will decode it as another string or will throw an error.

(c) 11011000 11011000 or [216, 216]. This does not decode to any Unicode character because it does not conform to the Unicode standard.


## Problem (`train_bpe_tinystories`)

(a) 5 mins 20 seconds. `b' accomplishment'` is the longest token.

(b) Sorting of frequencies of pairs takes the most amount of time. This operation occurs after every merge step and sorts over O(N) items each time where N is the number of pre-tokens identified in the pre-tokenization step.


## Problem (`train_bpe_expts_owt`)

(a) The longest token is `"ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ"` in terms of the byte-length of this string in `utf-8`. This is some random piece of text from the web.

(b) The tinystories tokenizer is less noisy with legible English words such as 
```
[' accomplishment',
 ' disappointment',
 ' responsibility',
 ' uncomfortable',
 ' compassionate',
 ' understanding',
 ' neighbourhood',
 ' Unfortunately',
 ' determination',
 ' encouragement',
 ' unfortunately',
 ' congratulated',
 ' extraordinary',
 ' granddaughter',
 ' disappointed',
 ' enthusiastic',
 ' accidentally',
 ' refrigerator',
 ' veterinarian',
 ' strawberries']
```
but the OWT vocab contains illegible strings in vocab as tokens such as 
```
['ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ',
 '----------------------------------------------------------------',
 '————————————————',
 '--------------------------------',
 '________________________________',
 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ',
 '================================',
 '................................',
 '********************************',
 '————————',
]
```