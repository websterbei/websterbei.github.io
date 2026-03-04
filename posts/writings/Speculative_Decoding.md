# Speculative Decoding: What it is, Why it Matters, and the Major Variants

Author: Webster Bei Yijie

Autoregressive LLM decoding is slow. Each token requires a full forward pass through the target model, and these passes are heavily memory-bandwidth bound, meaning you are paying a lot of wall-clock time per token even though most of the GPU compute is idle. This is especially painful for latency-sensitive applications.

Speculative decoding is the most widely adopted family of techniques to address this. The idea is deceptively simple: use something cheap and fast to guess what the expensive model is going to say, then verify those guesses in bulk. When the guesses are right, you get multiple tokens for the cost of slightly over 1 base model forward pass on the single token generated from the previous iteration (which is what you would have to pay regardless). When they are wrong, you pay the small penalty and fall back to normal decoding.

The benefit is real and significant: in production, speculative decoding routinely delivers 2-3x tokens-per-second speedup without changing model quality (more if the output is easier to guess, such as coding use cases). That is a rare "free lunch" in inference optimization.

This post covers the major variants of speculative decoding, how they work, and where they differ:

1. Speculative decoding (where it all started)
2. Medusa (first model-based solution that works quite well)
3. EAGLE1 / EAGLE2 / EAGLE3 (IMO still the current SOTA)
4. N-gram speculation (symbolic, not model-based)
5. Newer diffusion-based / parallel drafting approaches (e.g. DFlash, FastEagle)

---

## The Basic Setup

Without speculation, autoregressive generation with target model $p_\theta$ is:
$$
x_{t+1} \sim p_\theta(\cdot \mid x_{1:t})
$$
One forward pass, one token. Repeat until done.

With speculation, we add a cheap proposer $q_\phi$:

1. Proposer drafts $K$ future tokens cheaply.
2. Target model verifies those $K$ tokens in one batched pass.
3. We accept the longest correct prefix and emit a correction token at the first mismatch.

The key insight is that verifying $K$ tokens with the target model costs roughly the same as generating 1 token (since the bottleneck is memory bandwidth, not compute, and the KV cache operations parallelize well). So when the proposer guesses right, you get multiple tokens for the price of one.

Here is an animation to illustrate:
<iframe src="/animations/speculative_decoding.html" width="100%" height="420" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Why this actually speeds things up

Let $A$ be the number of accepted draft tokens per cycle.
$$
\mathbb{E}[\text{tokens per target pass}] \approx 1 + \mathbb{E}[A]
$$
The $+1$ is the correction token from the target (you always get at least one token per cycle, even if all drafts miss).

If you break down acceptance by position:
$$
a_j = \Pr(\text{position }j\text{ accepted} \mid 1,\dots,j-1\text{ accepted})
$$
$$
\mathbb{E}[A] = \sum_{j=1}^{K}\prod_{i=1}^{j} a_i
$$
This formula is central to everything that follows. It tells you that deep-position hit rate matters a lot: even if your first-position acceptance is 80%, the marginal value of each subsequent position decays multiplicatively. All the methods below are, in one way or another, trying to push those $a_j$ values higher.

### Evaluation metrics

Regardless of which speculative method you use, the metrics that matter are:
$$
\text{Hit Rate}=\frac{N_{\text{draft accepted}}}{N_{\text{total generated}}}
$$
$$
\text{Acc}_j = \frac{H_j}{T_j} \quad\text{(per-position acceptance at draft position }j\text{)}
$$
Hit rate is intuitive to understand: out of all the generated tokens per request, how many came from the draft model. Hit rate will be below 100% even in the ideal scenario. Suppose drafters propose $K$ tokens every turn and all $K$ get accepted — the hit rate will be $\frac{K}{1+K}$ since each verification step gives you at least one token regardless of draft token acceptance.  
Per-position acceptance measures the number of accepted tokens out of all the tokens proposed for each position (position 1 refers to the next token, position 2 refers to the token after that, and so on). The acceptance rate decreases with position since the values are implicitly conditioned: a proposed token for position $i+1$ is accepted only if the proposed token for position $i$ is also accepted.  
At Fireworks, we also measure draft token efficiency, which is defined as
$$
\text{Draft Token Efficiency}=\frac{N_{num\ tokens\ accepted}}{N_{num\ tokens\ proposed}}
$$
You can derive token efficiency from other metrics, but it is a convenient one to track directly.  
All of these metrics matter because each additional proposed token adds to verification cost (even more so for MLA-based models where compute can be a bottleneck), and these numbers tell us where to stop drafting.

---

## A Short History

- **Xia et al. (2022/2023)** and **Leviathan et al. (2023)**: two papers released around the same time that independently proposed the core idea. Xia et al. originally called their method "Aggressive Decoding" and later renamed it to "Speculative Decoding" — they claim the first public use of the term. Leviathan et al. formalized the draft-verify-correct loop with the rejection-sampling-based acceptance scheme that preserves exact target distribution. Both papers deserve credit for establishing the foundation.
- **Medusa (2024)**: train multiple "heads" on one backbone for token guesses.
- **EAGLE series (2024+)**: stronger drafter quality from architectural improvements and augmented inputs.
- **Recent attempts**: diffusion-based / parallel drafting methods.

### Bonus:
We will not go into these papers further, but some earlier papers that predate the term "speculative decoding" are relevant. They build on the idea that solving a system with sequential dependencies in parallel can be done via fixed-point Jacobi iteration methods. They come out from a pretty different perspective as the standard "propose and verify" paradigm that people do today. The notable papers I would recommend reading are:

- Song, Meng, Liao, Ermon. Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving (ICML 2021).
- Santilli, Severino, Postolache, Maiorca, Mancusi, Marin, Rodola. Accelerating Transformer Inference for Translation via Parallel Decoding (ACL 2023).
- Fu, Bailis, Stoica, Zhang. Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (2024).

---

## 1) Speculative Decoding

This is what most people mean when they say "spec decode." It is the canonical formulation from Leviathan et al. (2023) and the reference point for all later work.

### How it works

You maintain two models at serving time: the expensive target model and a smaller draft model. At each cycle, given prefix $x_{1:t}$:

1. Draft model autoregressively generates $K$ candidate tokens: $\hat{x}_{t+1:t+K}$
2. Target model evaluates all $K$ positions in one forward pass
3. Find first mismatch index $j$
4. Accept $\hat{x}_{t+1:t+j-1}$
5. Emit target model's own token at position $t+j$ as correction
6. Continue from the new prefix

In pseudo-code:

```python
while not done:
    draft_tokens = draft.generate(prefix, K)
    target_tokens = target.predict_positions(prefix, draft_tokens)
    j = first_mismatch(draft_tokens, target_tokens)
    accepted = draft_tokens[:j]
    correction = target_tokens[j]
    prefix = prefix + accepted + [correction]
```

### Architecture

You need to serve two models:

- **Target model**: full LLM with KV cache, the model whose output quality you want to preserve.
- **Draft model**: a smaller LM (typically same tokenizer, compatible vocabulary). Often 10-50x fewer parameters.
- A lightweight **accept/reject controller** that compares outputs.

### Benefits

The big advantage of this approach is conceptual simplicity and provable quality guarantees. In the original formulation, the acceptance/rejection scheme is designed so that the final output distribution is exactly $p_\theta$, regardless of how bad $q_\phi$ is. A bad proposer just means fewer accepted tokens per cycle (slower), but never worse quality.

### Training the draft model

A standard way to train $q_\phi$ is distillation from $p_\theta$:
$$
\mathcal{L}_{\text{draft}} = \sum_t \mathrm{KL}\left(p_\theta(\cdot\mid x_{\le t}) \,\|\, q_\phi(\cdot\mid x_{\le t})\right)
$$
In practice, many production draft models are just smaller versions of the target architecture trained on the same data, without explicit distillation. For example, using Qwen3 0.6B as the drafter of Qwen3 32B.

### Trade-offs

The main practical tension is:

- Larger $K$ = more potential upside per cycle
- Larger $K$ = more draft compute and lower late-position acceptance
- You also need to serve two models simultaneously, which makes inference scheduling and memory management more challenging.

You want to pick $K$ where:
$$
\text{speedup}(K) \approx \frac{1+\mathbb{E}[A(K)]}{1+\text{draft overhead}(K)}
$$
is maximized. In practice this is usually $K=2$ to $K=5$.

---

## 2) Medusa

Medusa takes a different approach: instead of a separate draft model, it adds extra prediction heads directly on the target model's backbone (you could also argue it's adding a small model that takes the base model's last-layer activation as input). Since the input is now last-layer hidden states, this simplifies the drafter's job — the base model already encodes its "intent" of what token to output next into the hidden states. This is evident if you run $R^2$ or linear predictability analysis between the hidden states of consecutively generated tokens. Typically there is already enough correlation to make drafter prediction possible via small linear layers.

### How it works

The target model computes hidden state $h_t$ as usual. Medusa adds $M$ extra heads on top of this:
$$
\ell_t^{(m)} = W^{(m)} h_t,\quad m=1,\dots,M
$$
Head $m$ predicts a candidate for the token at offset $t+m$. Each head is a lightweight MLP (typically one or two layers), so the overhead is small.

Since multiple heads can each produce multiple candidates (e.g. top-k per head), Medusa organizes the candidate combinations into a tree structure. The target model then verifies a tree of candidate sequences in one batched pass, and accepts the longest valid path through the tree.

### Architecture

- Shared transformer trunk (the target model itself, unchanged)
- $M$ lightweight prediction heads (additional parameters)
- Tree construction and verification logic (somewhat orthogonal; we can discuss this later)

<iframe src="/animations/medusa.html" width="100%" height="500" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Benefits

The big win is deployment simplicity:

1. **Single model serving**: no second model to load, only small additional memory to the super small drafting heads.
2. **Cache locality**: everything shares one set of KV caches and one set of weights. No cross-model coordination.
3. **Parameter efficiency**: heads are small and therefore cheap

### Weaknesses

The heads only see the *current* hidden state $h_t$. They do not get to condition on their own previous predictions, so each offset is essentially an independent prediction from the same features. This limits how much quality you can get at deeper offsets. Sometimes the predictions may be correct in content but placed at the wrong positions.

---

## 3) EAGLE Family

The EAGLE family embraces the idea of "letting the base model do the work" since it already does such a good job of encoding tokens into a high-dimensional representation where the next token is somewhat linearly predictable.   
Furthermore, when you are already at the last layer, the true next token is just one step away (multiply by `lm_head` and take the argmax). So it is beneficial to augment the drafter input with that true next token, making the task of predicting the token after that easier. This is a key innovation that the EAGLE family introduced.   
The EAGLE model itself — a one- or multi-layer decoder architecture — is also much larger than Medusa's heads.

---

## 3.1 EAGLE1 / EAGLE2

### The core idea

EAGLE1/2 takes the last hidden state from the target model and combines it with token embeddings to create a rich input for a lightweight autoregressive draft model. The draft model shares the target model's embedding table and LM head (both frozen), so it operates in the same token space. Only the middle layers are trainable.

### Architecture

At each token position $t$, the data flows like this:

1. Look up frozen token embedding from target model's embedding table:
   $$
   e_t = \text{Embed}(x_t) \quad\text{(frozen, from target model)}
   $$
2. Get target model's last hidden state $h_t$ at this position
3. Concatenate and project down to model dimension:
   $$
   z_t = W_f [e_t; h_t] \quad\text{(}\mathbb{R}^{2d}\to\mathbb{R}^d\text{, trainable)}
   $$
4. Pass $z_t$ through a stack of Llama-style decoder layers (self-attention + MLP)
5. Project to vocabulary logits via frozen LM head:
   $$
   \ell_t = W_{\text{lm}} z_t \quad\text{(frozen, from target model)}
   $$

The frozen embedding and LM head are important: they lock the draft model's input/output space to the target model's vocabulary, so you never have tokenizer mismatch issues.

<iframe src="/animations/eagle_architecture.html" width="100%" height="560" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### How generation works

At inference time, the draft model runs autoregressive generation:

1. Initialize a KV cache
2. Forward through the model, argmax the output logits to get the next token
3. Record confidence (max softmax probability)
4. Append predicted token, feed it back in, repeat for $K$ steps
5. Return list of tokens + list of confidences

### Benefits

The advantage of EAGLE1/2 over a standalone small LM is that the draft model sees the target model's hidden state directly. It doesn't have to independently figure out what the target model "thinks" — it gets told via $h_t$. This significantly improves first-position acceptance and usually helps deeper positions too. EAGLE1 and EAGLE2 essentially have the same architecture, but EAGLE2 adds beam search based on confidence to propose more tokens per position for a higher chance of acceptance.

<iframe src="/animations/eagle2_beam.html" width="100%" height="480" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>


Since EAGLE drafting is autoregressive, generating $K$ draft tokens requires $K$ sequential forward passes through the draft model. Each pass conditions on the previous prediction: if that prediction happens to match the target model's output, the next pass receives the correct input and has a good chance of predicting correctly too. If a prediction is wrong, all subsequent predictions are likely wrong as well — but this doesn't matter, because verification stops at the first mismatch anyway.

---

## 3.2 EAGLE3

EAGLE3 is where most of the architectural innovation happens. The high-level goal is to push acceptance rates at deeper draft positions (positions 2, 3, 4...), which is where EAGLE1/2 starts to lose accuracy. It does this through several complementary mechanisms.

### (A) Multi-layer feature fusion

EAGLE1/2 only uses the last hidden state from the target model. EAGLE3 can pull hidden states from multiple intermediate layers:
$$
f_t = \mathrm{concat}\left(h_t^{(\ell_1)},h_t^{(\ell_2)},\dots,h_t^{(\ell_m)}\right)
$$
$$
\tilde{h}_t = W_{\text{fc}} f_t
$$
The intuition is that different layers capture different kinds of information — early layers have more syntactic/local features, late layers have more semantic/global features. Giving the draft model access to multiple layers provides a richer signal for prediction.

<iframe src="/animations/eagle3_architecture.html" width="100%" height="540" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

If only one feature layer is used, the projection degenerates to identity (no extra parameters).

The original EAGLE3 paper uses layers like early, middle, and late (e.g. layer 2, layer $L/2$, layer $L-3$), though this is configurable.

### (B) Training-time test (TTT)

This is probably the most interesting idea in EAGLE3. The key observation is that during autoregressive draft generation, the draft model sees its own previous predictions as input for the next step. If those predictions are wrong, the model is operating on incorrect context (which is what happens in EAGLE1/2). Ideally, the model would be able to adapt on-the-fly to whatever context it finds itself in, even if that context includes its own mistakes and noise.

Concretely, in EAGLE3, when predicting tokens at later positions (positions 2, 3, ...), the draft model receives its own output from predicting the prior token, simulating what happens at actual inference time during multi-token drafting.

The practical benefit for speculative decoding is that TTT attention can adapt more gracefully as the draft model rolls out multiple steps. Even as the input context drifts (because it includes the model's own predictions rather than ground truth tokens), the TTT mechanism provides a way for the model to self-correct its internal representations. This helps maintain acceptance rates at deeper draft positions.

The following animation illustrates the layer-by-layer attention path during TTT training. Notice how each decoder layer at position $k$ attends to the corresponding layer at position $k-1$, and how the predicted (not ground truth) token is fed forward:

<iframe src="/animations/eagle3_ttt.html" width="100%" height="480" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

---

## 3.3 EAGLE1 vs EAGLE2 vs EAGLE3

The progression across versions is roughly:

- **EAGLE1/2**: a clean, effective baseline. Concat token embedding + hidden state, project, run through decoder layers, predict. The task is focused on next-1 token prediction

- **EAGLE3**: augments the input with features from intermediate layers of the base model, and addresses the deeper-position training-inference inconsistency through training-time test.

In practice, EAGLE3's deeper-position improvements compound: each extra percentage point of acceptance at position 3 or 4 translates directly into higher $\mathbb{E}[A]$ and better end-to-end speedup.

---

## 4) N-gram Speculation

N-gram speculation is the simplest form of speculative decoding: no neural draft model at all. The proposer is a lookup table.

### How it works

You build a cache of n-gram continuations from the prompt and/or previously generated text. Given the most recent $n-1$ tokens as suffix $s=x_{t-n+2:t}$:
$$
C(s,w) = \#\{w \text{ follows } s \text{ in the context so far}\}
$$
Propose:
$$
\hat{w}=\arg\max_w C(s,w)
$$
or take top-k candidates. Multi-token drafts can be chained by recursively looking up continuations.

The verification step is identical to any other speculative method: target model checks the proposed tokens.

### Architecture

- Suffix lookup structure (hash map or trie, built on-the-fly from context)
- Candidate builder
- Standard target verifier

<iframe src="/animations/ngram.html" width="100%" height="520" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Benefits

The proposer cost is essentially zero — a hash lookup is nanoseconds, not milliseconds. No GPU memory, no extra model weights. This makes it easy to deploy alongside any target model with no infrastructure changes.

### Limitations

The obvious problem: n-gram matching only works when the generation contains repetitive patterns. For code generation with repeated function signatures, structured JSON output, or template-heavy text, it can be surprisingly effective. For creative or novel text, acceptance rates are very low.

---

## 5) Parallel Drafting Methods

More recently, there have been attempts to move beyond strictly autoregressive draft generation. The idea is: instead of the draft model generating tokens one by one (which is itself sequential and slow), predict multiple tokens in parallel. FastEagle and DFlash are two flavors of this approach.

FastEagle essentially unrolls $K$-step EAGLE AR drafting into a single $K$-layer model where each layer produces a token at a position. In EAGLE, later positions enjoy the benefit of earlier tokens being explicitly decoded; FastEagle gives up this benefit (and the sequential data dependency) to trade for parallel drafting.

DFlash is one concrete implementation of this idea. It was very recently published and we will explore it further.

### How DFlash works

For a block of size $B$:

1. Build a "noise block": the last known token as anchor, followed by MASK tokens:
   $$
   [x_t,\text{MASK},\text{MASK},\dots,\text{MASK}]
   $$
2. Embed this block using the frozen target embedding table
3. Feed the target model's hidden context + noise block embedding into the DFlash model
4. The DFlash model predicts all $B$ positions in one forward pass
5. Positions $1..K$ are used as speculative proposals (position 0 is the anchor, which we already know)

### Architecture

A DFlash draft model typically has three components:

- **Trainable block predictor core**: sees both context hidden states and the noise block
- **Frozen embedding table** (from target model): used to embed the noise input, keeping token space aligned
- **Frozen LM head** (from target model): used to project outputs back to token probabilities

The attention mask allows each noise position to see the full context and all other noise positions (bidirectional within the block), which is different from causal autoregressive attention.

[Figure Placeholder: DFlash anchor-mask block parallel predictor]

### Benefits

The main appeal is that draft generation is no longer sequential: you produce $K$ candidates in one pass instead of $K$ sequential passes. This can significantly reduce proposer latency, especially for larger $K$.  
Moreover, since DFlash generates more tokens in one go, one can typically afford a multi-layer DFlash-style drafting model with effective cost similar to a single-layer EAGLE model with multiple AR drafting steps. More layers mean larger model capacity and greater ability to mimic base model behavior through prolonged training.

### Downside

DFlash training will have lower token efficiency than methods such as EAGLE due to its non-autoregressive nature. EAGLE training enjoys the same benefit as AR LLM pretraining — that is, you can use every token as a supervision signal to train the model. DFlash needs to rely on manual block construction and must therefore make a harder trade-off between training token utilization and cost.

### Connection to diffusion

DFlash is not continuous-time latent diffusion in the DDPM/score-matching sense. But the mechanism — "predict masked/noisy future positions conditioned on context" — shares the denoising intuition. Think of it as a discrete, one-step denoising model applied to a block of future tokens. This is an active area of research and the design space is still being explored. 

---

## 6) Putting It All Together

At the end of the day, every speculative decoding method is playing the same game. Per cycle:
$$
C = C_{\text{target}} + C_{\text{proposer}} + C_{\text{orchestration}}
$$
Throughput is roughly:
$$
\mathrm{TPS} \propto \frac{1+\mathbb{E}[A]}{C}
$$

So each method is trying to push one or more of:

1. Higher $\mathbb{E}[A]$ — better proposals, especially at deeper positions
2. Lower $C_{\text{proposer}}$ — cheaper draft generation
3. Lower orchestration overhead — simpler verification, fewer cache ops

Here is how the methods compare on these axes:

| Method | Proposal quality | Proposer cost | Serving complexity | Best for |
|--------|-----------------|---------------|-------------------|----------|
| Medusa | Moderate, decays with depth | Very low (heads only) | Cheap addon heads | Easy deployment |
| EAGLE1/2 | Good | Moderate | Two models (shared vocab), separate KV | Better acceptance than lookahead and Medusa |
| EAGLE3 | Best at deeper positions | Moderate-high | Two models (shared vocab), separate KV | Maximum acceptance depth, best alignment with AR verification |
| N-gram | Variable (repetition-dependent) | Nearly zero | Low | Repetitive/structured text |
| DFlash-like | TBD (active research) | Low (parallel block) | Two models, separate KV | Latency-sensitive with large K |

---

## 7) What to Monitor in Practice

If you are training or evaluating a speculative decoding system, the most important diagnostic is **per-position acceptance rate** $\text{Acc}_j$. Aggregate hit rate is useful but it hides where your proposer is strong and where it breaks down.

For example, a system with `[75%, 55%, 30%]` (per-position acceptance defined above) at positions 1/2/3 tells a very different story than `[60%, 58%, 55%]`. The first system is great at position 1 but wastes most of its budget on positions 2-3. The second is weaker at position 1 but maintains quality deeper — and will likely give better end-to-end speedup with $K=3$. (If you work out the math, in the case of `[75%, 55%, 30%]`, the expected accept length is simply `1 + 0.75 + 0.55 + 0.30 = 2.6`, that `1` is the free token from verification itself).

Per-position acceptance is exactly the right primary signal for deciding how many draft tokens to use, and for diagnosing whether architectural changes are actually helping.

---

## Disclaimer

For simplicity, I focused on the algorithmic and architectural differences between methods. I did not go into serving-system details like kernel fusion strategy, scheduler-level batching policy, KV cache management, or transport overhead between draft and target models. For actual production latency numbers, those low-level details matter a lot and can easily dominate the gains from algorithmic improvements.

---

## References

1. Xia, Ge, Wang, Chen, Wei, Sui. Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation (2022/2023). https://arxiv.org/abs/2203.16487
2. Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding (2023). https://arxiv.org/abs/2211.17192
3. Fu, Bailis, Stoica, Zhang. Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (2024). https://arxiv.org/abs/2402.02057
4. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (2024)
5. EAGLE / EAGLE2 / EAGLE3 papers and open implementations
6. Recent block-parallel / denoising-style speculative decoding work (including DFlash-style approaches)
