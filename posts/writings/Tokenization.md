# LLM Intro Class: Tokenization

Author: Webster Bei Yijie

## What is tokenization and why is it needed
Large language models ingest texts and output texts, which are strings or array of characters. However, neural networks that we use to build large language models are not able to handle strings directly. Instead, they handle floating point numbers. Therefore, we need a way to convert strings or array of characters into floating point numbers.  
Array of characters are discrete but floating numbers are continuous. In machine learning, a very common apporach to convert something discrete to something continuous is through embedding. Embedding is essentially a lookup table, typically represented by a tensor. The lookup table would contain `N` rows and `M` columns of float point numbers. The row indices `0` to `N-1` represents the lookup key, where as the whole row of floating point numbers (a vector) at a row index `X` represents the lookup value. Keys (integer) are discrete and values (vector of dimension `M`) are continuous. 
In PyTorch code, it is done through:
```Python
import torch

embedding_table = torch.nn.Embedding(N, M)
key = torch.tensor(3) # just an example, it could be any integer in [0, N-1]
value = embedding_table(key) # a tensor of shape (M,)
```

Therefore, if we are able to convert texts into integers within a bounded range (`0` to `N-1`) for some `N`, then we have a way to convert texts into something continuous that neural networks would be able to handle. Since there are infinitely many different strings, it is impossible to assign a single integer to each string input. Instead, we need to break the text down into smaller pieces, and assign each piece an integer index, so a piece of text gets converted to an array of integers.
This process is exactly `Tokenization`, it is a mapping from the space of strings to the space of integer arrays, and each integer or piece of text that the integer maps from is what we called `Token`.

## What properties should the tokenization mapping have?
My middle school math teacher says that a mapping can be one of: 1 to 1, 1 to many, many to 1, and many to many. In the case of tokenization function, it needs to be 1 to 1 mapping because we want it to be:
- deterministic, i.e. the same text input should give the same tokenized representation as output
- reversible (though not strictly required at this point, but nice to have for observability of tokenization process at least), i.e. if I'm given a tokenized representation of a text piece, I should be able to deterministically recover the text

In addition to that, the output from tokenization should be in a bounded range that is not too big, because otherwise the embedding tensor would be gigantic and taking too much storage space.

## What are some choices?
Let's only consider english text for now, assuming there are only 52 alphabets (including lower and upper case) and 10 digits, plus a handful of punctuation marks and white space. A simple tokenization strategy is to map each possible character to an integer. We can line up the list of possible characters and number them, and there would only be possibly less than 100 distinct characters in total.  
Say `abc...z` gets mapped to `1,2,3,....,26` and white space gets mapped to `0`, then `tokenize('my name is capybara')` would become `[13, 25, 0, 14, 1, 13, 5, 0, 9, 19, 0, 3, 1, 16, 25, 2, 1, 18, 1]`. This mapping is clearly invertible as well.

Alternatively, if we want to get fancier, we can take the whole Oxford English Dictionary and assign each individual word an index according to their order in the dictionary. Let's say `my` `name` `is` `capybara` as separate english words get assigned indices of `56` `80` `31` `17` (I'm just making things up), and white space is still assigned to `0`, then `tokenize('my name is capybara')` is now `[56, 0, 80, 0, 31, 0, 17]`. 

The above apporaches are what's called "character-level" and "word-level" tokenizations. 