# Bloom filter, CDMA, HDC, linear attention and delta net.

Author: Webster Bei Yijie

This post is about four concepts that come from very different places but closely related:

1. Bloom filter, a probabilistic data structure
2. CDMA, a wireless communication protocol/technique
3. Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA), a brain-inspired computational framework
4. Linear attention and delta net, two transformer attention variants

Personally for me, I also learned about these concepts in the past couple years under different circumstances. I first learned about bloom filter in college from Duke CS 290 (special topic class, so the topics rotate every semester under the same code) Algorithmic Foundation in Data Science. The second was CDMA, learned in Duke CS 356 Computer Networking Architecture class. The third, HDC/VSA, I got introduced by Allen Pan now an electrical engineering PhD student at Stanford. At the time he was doing his masters at Berkeley and we collaborated on a research paper for using VSA/HDC to accelerate object detection (which unfortunately never got published). And the last, linear attention and delta net that I only learned after joining fireworks and started working on language models.

These concepts seem highly uncorrelated, yet share the same fundamental idea in Mathematics. So I figured, maybe introducing all of them in the same blogpost, threading through with this Mathematical idea, could ~~make me famous~~ be more inspirational.

---

## 1) Bloom Filter

Bloom filter is a data structure for set membership queries. Practically speaking, it is often used to keep track of whether something has appeared before, but only probabilistically. 

Suppose you have a large set:

$$
S = \{\text{apple}, \text{banana}, \text{capybara}, ...\}
$$

and you want to answer:

> Is `x` in the set?

The naive way is to store all elements in a hash table. That works, but sometimes you want something much more memory efficient, and you are ok with a small false positive rate. This is where Bloom filter comes in.

### How it works

A Bloom filter maintains a bit array of length $m$:

$$
B \in \{0,1\}^m
$$

Initially every bit is 0.

It also uses $k$ hash functions:

$$
h_1, h_2, ..., h_k
$$

Each hash function maps an item to an index in the bit array:

$$
h_i(x) \in \{0,1,...,m-1\}
$$

To insert an item $x$, set all its hashed positions to 1:

$$
B[h_1(x)] = 1,\quad B[h_2(x)] = 1,\quad ...,\quad B[h_k(x)] = 1
$$

To query whether $x$ is in the set, check those same positions:

$$
B[h_1(x)], B[h_2(x)], ..., B[h_k(x)]
$$

If any of them is 0, then $x$ is definitely not in the set. If all of them are 1, then $x$ is probably in the set. 

Here is an animation showing insertion and a false positive query:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/bloom_filter.html" width="100%" height="360" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### False positives

The "probably" is important.

Bloom filter has no false negatives:

- If an inserted item is queried later, all its bits must have been set to 1.

But it can have false positives:

- An item that was never inserted may happen to hash into positions that were already set by other items because the bit positions are shared.

For example, maybe `dog` was never inserted, but:

$$
h_1(\text{dog}), h_2(\text{dog}), h_3(\text{dog})
$$

all point to bits that were set by `apple`, `banana`, and `capybara`. Then the Bloom filter says "probably yes", even though the true answer is no.

The whole data structure is basically trading exactness for a compact memory. You don't store the items. You only store a compressed sketch of which locations have been touched.

### The practical knobs

The main knobs are:

1. Larger bit array $m$ means fewer collisions
2. More hash functions $k$ gives a more specific signature per item, but also fills the array faster
3. More inserted items means the bit array gets saturated and false positives go up

There is a sweet spot for $k$ given $m$ and number of inserted elements. Too few hashes and every item has a weak signature. Too many hashes and you turn too many bits on, which makes unrelated items look present. The knobs can be tuned based on the acceptable false positive rate and storage space limit. One very concrete application is in content recommendations. Imagine you own a product similar to instagram and you want to make sure posts already shown to a user is not gonna be shown to the same user again with high probability. One possible implementation is using a sliding window bloom filter (sliding over past X days let's say) to keep track of the posts already shown to the user within the sliding window period.

---

## 2) CDMA

CDMA stands for Code Division Multiple Access (and yes, it is the CDMA in mobile network). It is a way for multiple users to transmit over the same frequency band at the same time.

At first this sounds impossible becaues if Alice and Bob both speak into the same channel simultaneously, EM wave superposition/interference is gonna make two signals mingled together and get distorted into random electric and magnetic field fluctuations in the air. 

This is indeed true, because the receiver does observe a mixture of the signals:

$$
r(t) = s_{\text{Alice}}(t) + s_{\text{Bob}}(t) + \text{noise}
$$

However, there is a way to make sure that each of their original signals can still be recovered from the mixture. The trick is that Alice and Bob use different spreading codes.

### Spreading a bit

Imagine Alice wants to send one bit $b_A \in \{-1, +1\}$. Instead of sending the bit directly, she multiplies it by a longer code vector:

$$
c_A = [1, -1, 1, 1, -1, ...]
$$

So the transmitted signal for one bit becomes:

$$
x_A = b_A c_A
$$

Bob does the same thing with a different code:

$$
x_B = b_B c_B
$$

The receiver gets the sum:

$$
r = b_A c_A + b_B c_B + n
$$

where $n$ is noise.

### Recovering Alice

To recover Alice's bit, the receiver computes the dot product with Alice's code:

$$
\langle r, c_A \rangle
$$

Substitute in the mixture:

$$
\langle r, c_A \rangle
= b_A \langle c_A, c_A \rangle
+ b_B \langle c_B, c_A \rangle
+ \langle n, c_A \rangle
$$

If Alice and Bob's codes are chosen so that:

$$
\langle c_B, c_A \rangle \approx 0
$$

then Bob's signal mostly disappears when we correlate with Alice's code.

So the receiver sees:

$$
\langle r, c_A \rangle \approx b_A \lVert c_A \rVert^2 + \text{small noise}
$$

and can decode Alice's bit.

This is a very neat idea. Everybody can transmit at once, and the receiver separates them by correlating against the right code. It is almost like each user gets a different "direction" in signal space.

Here is the CDMA picture visually:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/cdma_codes.html" width="100%" height="360" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### The limitation

In the ideal world, all user codes are perfectly independent and don't interfere with each other.

In the real world:

- Codes have finite length
- Users may not be perfectly synchronized
- Noise exists
- Too many users means codes become harder to separate

So CDMA systems are really about managing interference. You want the intended signal to be large when matched against its code, and all other signals to look like small random background noise.

---

## 3) Hyperdimensional Computing / Vector Symbolic Architectures

Hyperdimensional Computing (HDC), also known as Vector Symbolic Architectures (VSA), might be the most alien concept among all four, but it conceptually it is actually pretty simple.

The basic idea is to represent symbols using very high-dimensional vectors. For example:

$$
\text{red},\ \text{circle},\ \text{color},\ \text{shape}
$$

each gets assigned a random vector:

$$
r_{\text{red}}, r_{\text{circle}}, r_{\text{color}}, r_{\text{shape}} \in \{-1,+1\}^D
$$

where $D$ might be 1,000 or 10,000.

The surprising thing is that in high dimensions, random vectors are almost orthogonal with high probability. So if you generate two unrelated random hypervectors $a$ and $b$:

$$
\frac{a^\top b}{D} \approx 0
$$

This gives you a huge address space "for free". You can create many random symbols and most of them will not interfere much with each other.

### Binding and bundling

HDC/VSA typically has two important operations:

1. **Binding**: combine two vectors into a new vector that represents an association
2. **Bundling**: add multiple vectors together into one memory vector

For bipolar vectors, binding can be elementwise multiplication:

$$
a \odot b
$$

This is useful because elementwise multiplication is approximately self-inverse:

$$
a \odot a = \mathbf{1}
$$

So if you bind a role and a value:

$$
\text{color} \odot \text{red}
$$

you can later unbind with `color` to recover `red` (just like `XOR` and LeetCode Problem 136):

$$
(\text{color} \odot \text{red}) \odot \text{color}
= \text{red}
$$

Now suppose we want to store a small object:

$$
\text{color} = \text{red}, \quad \text{shape} = \text{circle}
$$

We can store it as:

$$
M = \text{color} \odot \text{red}
+ \text{shape} \odot \text{circle}
$$

To ask "what is the color?", bind the memory with the color vector:

$$
M \odot \text{color}
= \text{red}
+ \text{shape} \odot \text{circle} \odot \text{color}
$$

The first term becomes the desired answer. The second term is not zero, but because `shape`, `circle`, and `color` are unrelated high-dimensional vectors, it behaves roughly like random noise.

Here is the binding / bundling / retrieval picture:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/hdc_vsa.html" width="100%" height="360" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

### Why this is interesting

HDC/VSA gives you something like symbolic data structures using vector operations.

You can represent:

- Role-value pairs
- Sets
- Sequences
- Trees / compositional structures

all as high-dimensional vectors, and retrieve pieces by applying the right inverse operation or similarity search.

Of course it is not magic. If you bundle too many things together, the noise grows. If symbols are not sufficiently independent, retrieval gets worse. But the core intuition is very clean: high-dimensional random vectors give you many nearly orthogonal directions, and you can use those directions as symbolic addresses.

---

## 4) Linear Attention and Delta Net

Regular softmax attention is:

$$
\text{Attn}(q_i, K, V)
= \sum_{j=1}^{i}
\frac{\exp(q_i^\top k_j)}
{\sum_{\ell=1}^{i}\exp(q_i^\top k_\ell)}
v_j
$$

For each query $q_i$, you compare it with all previous keys $k_j$, compute attention weights, and take a weighted sum of values.

This is powerful, but expensive. If sequence length is $N$, the attention matrix is $N \times N$. That is the famous quadratic cost.

Linear attention tries to rewrite attention so that we do not need to explicitly compare every query with every key. Delta Net starts from the same fast-weight / linear-attention view, but changes how the memory gets updated.

### Linear attention as a fast weight memory

The softmax similarity $\exp(q^\top k)$ can be viewed as a kernel:

$$
K(q,k) = \exp(q^\top k)
$$

Linear attention replaces or approximates this with a feature map:

$$
K(q,k) \approx \phi(q)^\top \phi(k)
$$

Then attention becomes:

$$
y_i =
\frac{
\sum_{j=1}^{i} \phi(q_i)^\top \phi(k_j) v_j
}{
\sum_{j=1}^{i} \phi(q_i)^\top \phi(k_j)
}
$$

The numerator can be rearranged:

$$
\sum_{j=1}^{i} \phi(q_i)^\top \phi(k_j) v_j
=
\phi(q_i)^\top
\left(
\sum_{j=1}^{i} \phi(k_j) v_j^\top
\right)
$$

Depending on row/column convention, people may write the memory matrix as:

$$
M_i = \sum_{j=1}^{i} v_j \phi(k_j)^\top
$$

Then retrieval is:

$$
y_i = M_i \phi(q_i)
$$

plus some normalization term.

### Why this is linear

The important part is that $M_i$ can be updated recurrently:

$$
M_i = M_{i-1} + v_i \phi(k_i)^\top
$$

So instead of storing all previous keys and values, we store a running matrix. Each new token writes one outer product into the matrix.

Here is a simplified animation of the running memory matrix:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/linear_attention_memory.html" width="100%" height="350" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

At inference time, this is very attractive:

- Regular attention cache grows with sequence length
- Linear attention state can be fixed size

So you can think of linear attention as replacing a list of key-value pairs with a compressed key-value memory matrix.

### The catch

Compression is not free.

If two keys are similar in feature space, their writes overlap in the memory matrix. When you query one key, you may accidentally retrieve part of the other value too.

In regular attention, you still have the original list of keys and values, so the model can decide token by token how much to attend to each past token. In linear attention, past tokens have already been added into a shared memory state. Once two writes collide in the same directions, the memory has no exact way to tell them apart.

This is one of the core reasons linear attention is hard. The recurrence is efficient, but the state is a lossy compressed memory.

### Delta Net as a smarter write rule

Delta Net keeps the same basic memory interpretation. For simplicity, I am dropping $\phi(\cdot)$ here and just writing keys as $k_i$. The additive linear attention update becomes:

$$
M_i = M_{i-1} + v_i k_i^\top
$$

This is an additive write. Every new association $(k_i, v_i)$ gets added into memory.

The problem is that additive writes are not very good when the same key appears multiple times with different values.

Suppose earlier we stored:

$$
k \rightarrow v_{\text{old}}
$$

and later we want:

$$
k \rightarrow v_{\text{new}}
$$

With the naive additive update, we just add another outer product:

$$
M_i = M_{i-1} + v_{\text{new}} k^\top
$$

Now querying $k$ retrieves something like:

$$
v_{\text{old}} + v_{\text{new}}
$$

which is usually not what we want.

Delta net changes the write rule. Before writing the new value, it asks:

> What does the current memory already predict for this key?

That prediction is:

$$
\hat{v}_i = M_{i-1} k_i
$$

Then instead of writing $v_i$ directly, it writes the residual:

$$
v_i - \hat{v}_i
$$

The update becomes:

$$
M_i = M_{i-1} + \beta_i (v_i - M_{i-1}k_i) k_i^\top
$$

where $\beta_i$ is a learned or computed gate that controls write strength.

Here is the Delta net update as predict, error, update:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/delta_net_update.html" width="100%" height="330" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

This is exactly the kind of update you would expect from online learning:

1. Predict with current weights
2. Compute error
3. Update weights in the direction that reduces the error

So Delta net can be viewed as a sequence model whose hidden state is a fast weight matrix, and each token performs a small learning update on that matrix.

### Why the delta rule helps

If the model sees the same key again, the delta update can overwrite or correct the previous association instead of blindly accumulating values forever.

If the current memory already predicts $v_i$ well, then:

$$
v_i - M_{i-1}k_i \approx 0
$$

and the update is small.

If the current memory predicts badly, the residual is large, and the memory changes more.

This makes Delta net more stable than pure additive linear attention, especially when the sequence contains repeated or evolving associations.

---

## The Hidden Common Shape

Now we can put the four things next to each other.

The common problem is:

> How do I store many pieces of information in one shared medium, and later retrieve the piece I care about?

Bloom filter stores set membership in one bit array.

CDMA stores multiple users' signals in one shared wireless channel.

HDC/VSA stores symbolic structures in high-dimensional vectors.

Linear attention and Delta Net store past token values / fast key-value associations in one recurrent weight matrix.

Different fields, different notation, but the shape is surprisingly similar:

1. Choose an address / code / key for each item
2. Write information into a shared storage using that address
3. Retrieve by probing the storage with a matching address
4. Hope unrelated addresses do not interfere too much

The best case is when the addresses are orthogonal.

The geometry looks like this:
<iframe src="/posts/writings/Bloom_Filter_CDMA_Linear_Attention_Delta_Net/animations/orthogonal_memory.html" width="100%" height="360" scrolling="no" style="border:none; border-radius:8px; overflow:hidden;" loading="lazy"></iframe>

---

## Orthogonality as the No-Interference Condition

Let's say we store values using outer products:

$$
M = \sum_{j} v_j k_j^\top
$$

Now query with key $k_i$:

$$
Mk_i = \sum_j v_j k_j^\top k_i
$$

Separate out the desired term:

$$
Mk_i = v_i k_i^\top k_i + \sum_{j \ne i} v_j k_j^\top k_i
$$

If keys are orthogonal:

$$
k_j^\top k_i = 0 \quad \text{for } j \ne i
$$

then all the cross terms disappear:

$$
Mk_i = v_i \lVert k_i \rVert^2
$$

After normalization, you recover $v_i$ cleanly.

If keys are not orthogonal, then the cross terms do not disappear:

$$
\sum_{j \ne i} v_j k_j^\top k_i
$$

This is interference.

That one equation basically explains a lot:

- In CDMA, other users vanish only if their spreading codes have low dot product with your code.
- In HDC/VSA, unrelated symbolic vectors act like noise only if they are nearly orthogonal.
- In linear attention, other tokens do not leak into your retrieved value only if their keys are sufficiently different from your query.
- In Delta net, correcting or overwriting an association is easy when keys occupy separate directions, and harder when many keys overlap.
- In Bloom filter, different items are represented by sparse bit positions; collisions happen when different items touch the same coordinates.

Same story, different costumes.

---

## Bloom Filter Through This Lens

Bloom filter may look slightly different because it uses bits instead of continuous vectors, but the intuition still fits.

Think of the bit array as a vector space with $m$ basis directions:

$$
e_1, e_2, ..., e_m
$$

Each hash output selects one basis direction. An item $x$ with $k$ hashes gets a sparse signature:

$$
s(x) = e_{h_1(x)} + e_{h_2(x)} + ... + e_{h_k(x)}
$$

Inserting $x$ turns on the coordinates in $s(x)$.

Querying $x$ checks whether all coordinates in $s(x)$ are already on.

If two items have disjoint hash positions, their signatures are orthogonal in the simple coordinate sense:

$$
s(x)^\top s(y) = 0
$$

They do not interfere.

If they share positions:

$$
s(x)^\top s(y) > 0
$$

then one item partially supports the query for another item. A false positive happens when the query signature is fully covered by the union of other inserted signatures.

So Bloom filter is a discrete, binary version of the same memory issue. The bit array is shared storage. Hash functions create addresses. False positives are address collisions.

---

## CDMA Through This Lens

CDMA is probably the cleanest example.

Each user gets a code vector $c_u$. The channel receives:

$$
r = \sum_u b_u c_u + n
$$

To retrieve user $i$, compute:

$$
\langle r, c_i \rangle
=
b_i \langle c_i, c_i \rangle
+
\sum_{u \ne i} b_u \langle c_u, c_i \rangle
+
\langle n, c_i \rangle
$$

The first term is the signal. The second term is multi-user interference. The third term is noise.

Good spreading codes make:

$$
\langle c_u, c_i \rangle \approx 0
$$

for $u \ne i$.

So retrieval is just matched filtering in a nearly orthogonal code space. This is exactly the "write many things into one medium, retrieve one by probing with its code" pattern.

---

## HDC/VSA Through This Lens

HDC/VSA fits almost too perfectly.

The memory vector:

$$
M = \sum_j r_j \odot v_j
$$

stores many role-value bindings in one vector. To retrieve the value for role $r_i$, bind again:

$$
M \odot r_i
= v_i + \sum_{j \ne i} r_j \odot v_j \odot r_i
$$

The desired value comes back because $r_i \odot r_i = \mathbf{1}$. The other terms become noise because unrelated high-dimensional vectors are nearly orthogonal / nearly random relative to the target value.

So HDC/VSA is basically saying: if the vector space is high-dimensional enough, we can tolerate the noise and still recover the nearest valid symbol.

This is very close in spirit to CDMA. In CDMA, users get spreading codes. In HDC/VSA, symbols get hypervectors. In both cases, retrieval works because the unwanted components have low correlation with the query.

---

## Linear Attention Through This Lens

Linear attention writes:

$$
M_i = \sum_{j \le i} v_j \phi(k_j)^\top
$$

and retrieves:

$$
y_i = M_i \phi(q_i)
$$

Expanding:

$$
y_i =
\sum_{j \le i} v_j \phi(k_j)^\top \phi(q_i)
$$

Each value $v_j$ contributes in proportion to how aligned its key is with the query.

If the query matches one key strongly and is orthogonal to all others, retrieval is clean. If many keys have non-trivial overlap with the query, retrieval becomes a mixture.

Softmax attention also uses similarity, but it keeps every past key/value around and computes normalized weights at read time. Linear attention pre-aggregates the memory. This makes it fast, but also means interference has already been baked into the state.

This is why feature map choice matters so much in linear attention. It is not just a mathematical trick to make attention linear. It defines the address space where tokens are stored and retrieved.

---

## Delta Net Through This Lens

Delta net starts from the same fast weight memory:

$$
M_i
$$

but changes the write from:

$$
M_i = M_{i-1} + v_i k_i^\top
$$

to:

$$
M_i = M_{i-1} + \beta_i (v_i - M_{i-1}k_i) k_i^\top
$$

This is still address-based memory. The key $k_i$ decides where the update lands. The value residual decides what gets written.

Orthogonality still matters. If $k_i$ is orthogonal to previous keys, the update modifies a new independent direction. If $k_i$ overlaps with previous keys, the update also changes what gets retrieved by other nearby keys.

The delta rule does not magically remove the need for good addresses. What it does is make the write smarter:

- Additive linear attention says: "write this value"
- Delta net says: "write the correction needed so this key retrieves this value"

That is a meaningful difference. It turns the memory update into an online error-correction step, which is much better behaved when associations repeat or change over time.

---

## Capacity, Collisions, and Interference

Once you view these systems as shared memories, the important question becomes capacity.

How many things can you store before retrieval becomes unreliable?

For Bloom filter, capacity is controlled by bit array size, number of hash functions, and number of inserted items. Too many inserts means too many bits are on, and false positives increase.

For CDMA, capacity is controlled by code length, code design, synchronization, power control, and noise. Too many users means cross-correlations accumulate and the receiver can no longer separate them cleanly.

For HDC/VSA, capacity is controlled by hypervector dimension, the number of bundled items, and how cleanly the system can do nearest-neighbor cleanup. Too many bindings in one vector means the noise term becomes too large.

For linear attention, capacity is controlled by feature dimension, feature map quality, key distribution, normalization, gating, and training. Too many overlapping keys means the recurrent state becomes a soup of partially conflicting values.

For Delta net, capacity is also controlled by key geometry and update dynamics. The delta rule helps with overwriting and correction, but if every key points in a similar direction, then every write still damages nearby memories.

This is why I think "orthogonality" is often the hidden resource. It is not just a nice linear algebra property. It is the thing that lets many memories coexist in the same substrate without stepping on each other.

---

## A Small Mental Model

The mental model I like is:

> Storage is easy. Retrieval is hard.

You can always add more stuff into a shared state. Set more bits. Add more signals. Add more outer products. Update more fast weights.

The hard part is retrieving the right thing later without dragging a bunch of unrelated stuff along with it.

Orthogonal or near-orthogonal addresses solve this by making the wrong things invisible to the read operation.

When addresses are not orthogonal, the system still works, but now you are living in the world of collisions and interference:

- Bloom filter gives false positives
- CDMA gets multi-user interference
- HDC/VSA retrieves the wrong nearest symbol when bundled noise gets too large
- Linear attention retrieves mixed values
- Delta net updates bleed across related keys

Same failure mode, just with different names.

---

## The End

Bloom filter, CDMA, HDC/VSA, and the linear attention / Delta Net family are obviously not the same thing. One is a data structure, one is a communication system, one is a symbolic vector computing framework, and the last is a neural sequence model memory mechanism.

But they share a very useful abstraction:

> Use high-dimensional addresses to write information into a shared medium, then retrieve by matching against the address.

If the addresses are orthogonal, retrieval is clean. If the addresses collide, retrieval becomes noisy.

This is a good lens for thinking about many ML architectures in general. A lot of "memory" mechanisms are not really about whether the model can store information. They are about whether the model can assign that information to separable directions, and later query the right direction without waking up all the neighboring memories.

Maybe human memory is not too different at the very high level. We confuse similar things all the time: two people with similar faces, two restaurants with similar names, two papers with similar ideas, two meetings that happened in the same room. On the other hand, we tend to remember distinct things much more easily. A weird event, a strange joke, a very different visual scene, a unique smell. Distinctiveness makes retrieval easier.

So perhaps memory is mostly associative. You do not retrieve a memory by looking up an exact address like a computer RAM. You retrieve it by giving the brain a cue, and that cue activates nearby things. If the cue is specific enough, one memory wins. If the cue overlaps with many memories, you get mixture, confusion, or the feeling that something is "on the tip of my tongue".

In math and engineering, we often measure this kind of associativeness using inner product, cosine similarity, correlation, Hamming overlap, etc. These are all ways of asking: "how much does this query match that stored thing?" For the human brain, I don't know what the right primitive is. Maybe it is overlap in neural firing patterns, maybe attractor basin dynamics, maybe some distributed biochemical thingy that I don't understand. But the question feels very natural:

> If inner product is the associativity operator for many engineered memory systems, what is the associativity operator for biological memory?

I don't have a clean answer, but I like this question. It suggests that forgetting and confusion are not necessarily bugs. They may simply be what happens when many memories live in one shared high-dimensional substrate and retrieval is based on similarity rather than exact addressing.

## References

1. Bloom, Burton H. Space/Time Trade-offs in Hash Coding with Allowable Errors (1970). https://dl.acm.org/doi/10.1145/362686.362692
2. Viterbi, Andrew J. CDMA: Principles of Spread Spectrum Communication (1995).
3. Kanerva, Pentti. Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors (2009). https://link.springer.com/article/10.1007/s12559-009-9009-8
4. Gayler, Ross. Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience (2003). https://www.researchgate.net/publication/242441313_Vector_Symbolic_Architectures_Answer_Jackendoff%27s_Challenges_for_Cognitive_Neuroscience
5. Katharopoulos, Vyas, Pappas, Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (2020). https://arxiv.org/abs/2006.16236
6. Schlag, Irie, Schmidhuber. Linear Transformers Are Secretly Fast Weight Programmers (2021). https://arxiv.org/abs/2102.11174
7. Yang, Zhang, McClelland, Sussillo. Gated Delta Networks: Improving Mamba2 with Delta Rule (2024). https://arxiv.org/abs/2412.06464
