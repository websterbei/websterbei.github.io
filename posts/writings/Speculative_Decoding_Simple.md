# Speculative Decoding: A Gentle Introduction

Author: Webster Bei Yijie

Large language models generate text one token at a time. Each token requires a full pass through the model, and most of the GPU sits idle during each pass because the bottleneck is moving data around in memory, not doing math. This makes generation slow, especially when low latency matters.

Speculative decoding is the most popular family of techniques to speed this up. The core idea: use something small and fast to *guess* what the big model will say next, then have the big model *check* those guesses all at once. When the guesses are right, you get multiple tokens for roughly the cost of one. When they're wrong, you just fall back to normal generation — no harm done.

The speedups are real: 2-3x faster token generation in production, with no loss in output quality. For tasks where the output is more predictable (like code), speedups can be even higher.

This post walks through the major approaches:

1. **Speculative decoding** — the original idea
2. **Medusa** — bolt-on prediction heads, no second model needed
3. **EAGLE1 / EAGLE2 / EAGLE3** — stronger drafting through better architecture
4. **N-gram speculation** — pattern matching, no neural network at all
5. **Parallel drafting** (DFlash, FastEagle) — newer approaches that draft multiple tokens at once

---

## The Basic Setup

Normal generation is straightforward: run the model, get one token, repeat. One pass = one token.

With speculative decoding, we add a cheap "proposer" alongside the main model:

1. The proposer quickly guesses the next several tokens.
2. The main model checks all those guesses in a single pass.
3. We keep every correct guess and fix the first wrong one.

The key insight is that *checking* multiple tokens costs about the same as *generating* one token (the memory bandwidth bottleneck doesn't change much whether you're processing 1 token or 5). So when the proposer guesses right, you're getting multiple tokens for the price of one.

Here is an animation to illustrate:
<iframe src="/animations/speculative_decoding.html" width="100%" height="420" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Why does this speed things up?

Every verification cycle, you get at least one token — even if all the guesses are wrong, the main model produces a correction token. When guesses are right, you get bonus tokens on top.

So the tokens per cycle is: **1 + (number of correct guesses)**

The catch is that acceptance rates drop at deeper positions. If the proposer has an 80% chance of getting position 1 right, and an 80% chance at position 2 *given* position 1 was right, the probability of getting both right is 64%. By position 4 or 5, the odds of an unbroken streak get slim. Every method below is, in one way or another, trying to keep those deeper-position acceptance rates high.

### How do we measure success?

- **Hit rate**: what fraction of the final output tokens came from the proposer (rather than from correction). Higher is better.
- **Per-position acceptance**: how often position 1 is accepted, position 2, position 3, etc. This is the most informative metric — it tells you where your proposer is strong and where it breaks down.
- **Draft token efficiency**: of all the tokens the proposer guessed, how many were actually accepted? This tells you when you're wasting effort by drafting too many tokens.

---

## A Short History

- **Xia et al. (2022/2023)** and **Leviathan et al. (2023)**: two independent teams published the core idea around the same time. Leviathan et al. formalized the draft-verify-correct loop with a mathematical guarantee that the output is identical to what the big model would produce on its own.
- **Medusa (2024)**: added lightweight prediction heads directly on the base model.
- **EAGLE series (2024+)**: improved draft quality through better architecture and richer inputs.
- **Recent work**: parallel drafting methods like DFlash and FastEagle.

---

## 1) Speculative Decoding (The Original)

This is the canonical version — what most people mean when they say "spec decode."

### How it works

You run two models side by side: the expensive target model and a much smaller draft model. Each cycle:

1. The draft model quickly generates K candidate tokens (say, 3-5 tokens ahead)
2. The target model checks all K candidates in one forward pass
3. Find the first position where the draft disagrees with the target
4. Keep everything before the disagreement
5. Use the target model's own prediction at the disagreement point as a correction
6. Continue from there

```python
while not done:
    draft_tokens = draft.generate(prefix, K)
    target_tokens = target.predict_positions(prefix, draft_tokens)
    j = first_mismatch(draft_tokens, target_tokens)
    accepted = draft_tokens[:j]
    correction = target_tokens[j]
    prefix = prefix + accepted + [correction]
```

### What you need to serve

- **Target model**: the full LLM whose output quality you want to preserve.
- **Draft model**: a much smaller model (often 10-50x fewer parameters) that uses the same vocabulary.
- A lightweight **accept/reject controller** that compares outputs.

### Why it's good

The beauty of this approach is its quality guarantee. The acceptance scheme is designed so that the final output distribution is mathematically identical to what the target model would produce on its own. A bad proposer just means fewer accepted tokens (slower), but *never worse quality*.

### The practical tension

- More draft tokens per cycle = more potential upside
- More draft tokens per cycle = more wasted work when guesses go wrong at early positions
- You need to run two models simultaneously, which complicates memory management and scheduling

In practice, K = 3 to 5 usually hits the sweet spot.

---

## 2) Medusa

Medusa takes a different approach: instead of running a separate draft model, it bolts extra "prediction heads" directly onto the target model.

### How it works

When the target model processes a token, it produces a hidden state — a rich internal representation of what it's about to say. Medusa adds small prediction heads on top of this hidden state. Each head is a lightweight network (one or two layers) that predicts a future token at a different offset:

- Head 1 predicts the token at position +1
- Head 2 predicts the token at position +2
- Head 3 predicts the token at position +3
- ...and so on

Since each head can propose multiple candidates (its top few guesses), Medusa organizes the combinations into a tree structure and verifies the most promising paths in one pass.

### Architecture

- The target model itself (unchanged)
- M small prediction heads (the only new parameters)
- Tree construction and verification logic

<iframe src="/animations/medusa.html" width="100%" height="500" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Why it's good

Deployment is simple:

1. **Single model**: no second model to load — just a few small heads added on top. Minimal extra memory.
2. **Shared resources**: everything uses one set of weights and one KV cache. No cross-model coordination.
3. **Cheap**: the heads are tiny and fast.

### Where it struggles

Each head only sees the *current* hidden state. It can't condition on what the other heads predicted — each offset is essentially an independent guess from the same information. This limits quality at deeper positions. Sometimes the predictions are correct in content but placed at the wrong positions.

---

## 3) EAGLE Family

The EAGLE family builds on a key observation: the target model's hidden states already encode a lot of information about what's coming next. The next token is essentially "one step away" from the last hidden state (just multiply by the output layer and pick the top prediction). So why not give the drafter direct access to these hidden states?

EAGLE goes further: it also feeds the *actual next token* back into the drafter, making it even easier to predict the token after that. This is the family's core innovation.

The EAGLE draft model itself is also much larger than Medusa's heads — a proper decoder architecture, not just a linear layer.

---

## 3.1 EAGLE1 / EAGLE2

### The core idea

EAGLE takes the target model's last hidden state and combines it with the token embedding to create a rich input for the draft model. The draft model shares the target model's embedding table and output layer (both frozen), so it operates in the same token space. Only the middle layers are trained.

### Architecture

At each position, the data flows like this:

1. Look up the token embedding from the target model's (frozen) embedding table
2. Get the target model's last hidden state at this position
3. Concatenate them and project down to the right size (this is the trainable part)
4. Pass through a stack of decoder layers (self-attention + MLP)
5. Project to token predictions using the target model's (frozen) output layer

Sharing the frozen embedding and output layers is important: it locks the draft model into the target model's vocabulary, so there are never tokenizer mismatch issues.

<iframe src="/animations/eagle_architecture.html" width="100%" height="560" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### How generation works

At inference time, the draft model generates tokens one by one:

1. Initialize a KV cache
2. Run a forward pass, pick the most likely next token
3. Record how confident the prediction is
4. Feed that token back in and repeat for K steps
5. Return the list of tokens and their confidence scores

### Why it's better

Because the draft model sees the target model's hidden state directly, it doesn't have to independently figure out what the target model "thinks" — it gets told. This significantly improves acceptance rates, especially at the first position.

EAGLE2 uses the same architecture but adds beam search based on confidence scores, proposing multiple candidate tokens per position for a higher chance of at least one being accepted.

<iframe src="/animations/eagle2_beam.html" width="100%" height="480" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

Since EAGLE drafting is autoregressive (each predicted token feeds into the next prediction), if an early prediction is wrong, later ones will probably be wrong too. But this doesn't matter — verification stops at the first mismatch anyway.

---

## 3.2 EAGLE3

EAGLE3 is where most of the architectural innovation happens. The goal is to improve acceptance rates at deeper positions (positions 3, 4, 5...), which is where EAGLE1/2 starts to lose accuracy.

### Multi-layer feature fusion

EAGLE1/2 only uses the *last* hidden state from the target model. EAGLE3 pulls hidden states from *multiple* layers — early, middle, and late. The intuition: different layers capture different kinds of information. Early layers have more local/syntactic features, late layers have more global/semantic features. Combining them gives the draft model a richer signal to work with.

<iframe src="/animations/eagle3_architecture.html" width="100%" height="540" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Training-time test (TTT)

This is probably the most interesting idea in EAGLE3. Here's the problem it solves:

During drafting, the draft model generates tokens one by one, feeding each prediction as input to the next step. If a prediction is wrong, the next step receives incorrect input — and its prediction will likely be wrong too. The draft model is essentially operating on its own (potentially noisy) context.

But during standard training, the model always sees *correct* inputs. This creates a mismatch between what the model trains on and what it actually encounters at inference time.

EAGLE3's TTT addresses this by simulating the same conditions during training: when predicting tokens at later positions, the draft model receives its *own previous output* (not the ground truth) as input. This trains the model to be robust to its own mistakes, keeping acceptance rates higher at deeper positions.

<iframe src="/animations/eagle3_ttt.html" width="100%" height="480" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

---

## 3.3 EAGLE1 vs EAGLE2 vs EAGLE3

The progression:

- **EAGLE1/2**: a clean, effective baseline. Combine token embedding + hidden state, project, run through decoder layers, predict. EAGLE2 adds beam search for better candidate selection.
- **EAGLE3**: pulls features from multiple layers of the base model, and trains the drafter to handle its own mistakes through TTT.

In practice, EAGLE3's deeper-position improvements compound: each extra percentage point of acceptance at position 3 or 4 directly translates into more tokens per cycle and better end-to-end speedup.

---

## 4) N-gram Speculation

N-gram speculation is the simplest form of speculative decoding: no neural network at all. The proposer is just a lookup table.

### How it works

As the model generates text, you build a cache of patterns: "every time I've seen tokens A-B-C, the next token was D." Given the most recent few tokens, you look up what followed them before and propose that as the draft.

For multi-token drafts, you chain lookups: "A-B-C was followed by D, and B-C-D was followed by E," so you propose D-E.

Verification works exactly the same as any other speculative method — the target model checks the proposals.

### Architecture

- A lookup structure (hash map or trie) built on-the-fly from the prompt and generated text
- A candidate builder that chains lookups
- The standard target model verifier

<iframe src="/animations/ngram.html" width="100%" height="520" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Why it's good

The proposer cost is essentially zero — a hash lookup takes nanoseconds, not milliseconds. No GPU memory, no extra model weights. It can be added to any target model with zero infrastructure changes.

### Where it falls short

N-gram matching only works when the text contains repetitive patterns. For code with repeated function signatures, structured JSON, or template-heavy text, it can be surprisingly effective. For creative or novel text, acceptance rates are very low.

---

## 5) Parallel Drafting Methods

The methods above all draft tokens sequentially — each draft token depends on the previous one. Newer approaches like FastEagle and DFlash ask: what if we draft multiple tokens *in parallel*?

FastEagle essentially unrolls K sequential EAGLE steps into a single K-layer model where each layer produces one token position simultaneously. It gives up the benefit of later positions seeing earlier decoded tokens in exchange for parallel speed.

DFlash is another concrete approach. Here's how it works:

### How DFlash works

1. Start with the last known token and fill in placeholder "MASK" tokens for future positions
2. Embed this block using the target model's (frozen) embedding table
3. Feed the target model's hidden context plus this masked block into the DFlash model
4. The DFlash model predicts all positions in one forward pass
5. Use those predictions as speculative proposals

The DFlash model uses bidirectional attention within the block (each position can see all other positions), unlike the causal (left-to-right only) attention used in standard autoregressive generation.

### Why it's appealing

Draft generation is no longer sequential: you produce K candidates in one pass instead of K sequential passes. This reduces proposer latency, especially for larger K. And since DFlash generates all tokens at once, you can afford a deeper (more capable) draft model while keeping the total cost similar to a shallower autoregressive drafter.

### The trade-off

Because each position can't condition on the actual decoded tokens before it (only on masks), per-position accuracy tends to be lower than autoregressive drafters like EAGLE. Training is also less efficient — autoregressive training can use every token as a learning signal, while DFlash needs to construct masked blocks explicitly. This is an active area of research.

---

## 6) Putting It All Together

Every speculative decoding method is playing the same game: maximize the number of tokens you get per verification cycle while minimizing the cost of proposing those tokens. The three levers are:

1. **Better proposals** — especially at deeper positions (positions 3, 4, 5...)
2. **Cheaper proposer** — less time and memory spent on drafting
3. **Lower overhead** — simpler verification, fewer cache operations

Here's how the methods compare:

| Method | Proposal quality | Proposer cost | Serving complexity | Best for |
|--------|-----------------|---------------|-------------------|----------|
| Medusa | Moderate, decays with depth | Very low (heads only) | Cheap addon heads | Easy deployment |
| EAGLE1/2 | Good | Moderate | Two models (shared vocab), separate KV | Better acceptance than Medusa |
| EAGLE3 | Best at deeper positions | Moderate-high | Two models (shared vocab), separate KV | Maximum acceptance depth |
| N-gram | Variable (depends on repetition) | Nearly zero | Low | Repetitive/structured text |
| DFlash-like | Active research | Low (parallel block) | Two models, separate KV | Latency-sensitive with large K |

---

## 7) What to Monitor in Practice

If you're evaluating a speculative decoding system, the single most important diagnostic is **per-position acceptance rate**. Aggregate hit rate is useful but it hides where your proposer is strong and where it breaks down.

For example, a system with acceptance rates `[75%, 55%, 30%]` at positions 1/2/3 tells a very different story than `[60%, 58%, 55%]`. The first system is great at position 1 but wastes most of its budget on positions 2-3. The second is weaker at position 1 but maintains quality deeper — and will likely give better end-to-end speedup when drafting 3 tokens.

Per-position acceptance is the right primary signal for deciding how many draft tokens to use, and for diagnosing whether architectural changes are actually helping.

---

## Disclaimer

For simplicity, this post focuses on the algorithmic and architectural differences between methods. Production performance also depends heavily on serving-system details like kernel fusion, batching strategy, KV cache management, and communication overhead between draft and target models.

---

## References

1. Xia, Ge, Wang, Chen, Wei, Sui. Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation (2022/2023). https://arxiv.org/abs/2203.16487
2. Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding (2023). https://arxiv.org/abs/2211.17192
3. Fu, Bailis, Stoica, Zhang. Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (2024). https://arxiv.org/abs/2402.02057
4. Cai, Li, Geng, Peng, Lee, Chen, Dao. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (2024). https://arxiv.org/abs/2401.10774
5. Li, Wei, Zhang, Zhang. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (2024). https://arxiv.org/abs/2401.15077
6. Li, Wei, Zhang, Zhang. EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (2024). https://arxiv.org/abs/2406.16858
7. Li, Wei, Zhang, Zhang. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test (2025). https://arxiv.org/abs/2503.01840
8. Chen, Liang, Liu. DFlash: Block Diffusion for Flash Speculative Decoding (2025). https://arxiv.org/abs/2602.06036
9. Zhang, Chen, Chen, He, Yuan, Zheng, Yang. FastEagle: A Fast and Efficient Multi-Token Speculative Decoding Framework (2025). https://arxiv.org/abs/2509.20416
