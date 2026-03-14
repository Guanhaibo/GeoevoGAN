# GeoevoGAN: Geometric Evolution for Generative Adversarial Networks

**GeoevoGAN** is a generative adversarial network that infuses geometric (Clifford) algebra intuitions into the generator architecture. By combining **Star modules** (which mimic higher‑order blade interactions) and **Evo modules** (which evolve features via orthogonal transformations inspired by the Clifford group \(O(n)\)), GeoevoGAN achieves highly structured latent spaces, improved image quality, and stable training dynamics.

<p align="center">
  <em>Simplified overview: Grassmann seed → multi‑scale Star+Evo blocks → output.</em>
</p>

## Features

- **Geometric inductive bias** – leverages concepts from Clifford algebra (inner/outer products, orthogonal transformations) without enforcing a full algebraic framework.
- **Multi‑dimensional feature interaction** – Star and Evo modules operate on channel, height, and width dimensions to capture diverse correlations.
- **Disentangled conditioning** – a Grassmann seed produces an orthogonal matrix \(q\) that modulates the generator via QFiLM, enabling controllable synthesis.
- **Stable training** – built‑in spectral normalization, learnable residual scaling, and gradient penalties ensure robust adversarial learning.
- **High‑resolution ready** – demonstrated on 256×256 images; easily extendable to larger resolutions.

---

## Methodology

### Grassmann Seed
A random latent vector \(z\) is mapped to a matrix, QR‑decomposed to obtain an orthogonal matrix \(q \in \mathbb{R}^{B \times K \times r}\). This \(q\) serves as a disentangled condition vector. Optionally, the same decomposition can also generate an initial 4×4 feature map via learnable spatial atoms.

### Star Module
Inspired by the concept of **blades** in Clifford algebra, the Star module computes  
\[
y = (W_1 x) \odot (W_2 x)
\]  
which expands to a linear combination of all second‑order terms \(x_i x_j\). Stacking such blocks yields higher‑order interactions analogous to multi‑vector products.

### Evo Module Deep Dive

The Evo module is inspired by the inner and outer products in Clifford algebra, designed to explicitly model feature similarity and difference, thereby capturing rich local structures in images. Its core idea involves splitting the input feature into two branches, `det` and `ctx`, shifting them along a specified dimension, and computing their dot (inner) and wedge (outer) products. These are then fused via a gating mechanism and added back with a learnable residual connection.

#### 1. Branch Generation
The input feature $X$ is first normalized via GroupNorm, then split into two paths:
- **`det` branch**: A 1×1 convolution generating the main feature component, acting as a linear projection.
- **`ctx` branch**: A stack of depthwise convolutions (`dw_depth` layers), each followed by LeakyReLU, ending with a 1×1 convolution. This branch extracts local context, akin to smoothing or edge enhancement.

#### 2. Shift Operation
To establish cross-position (or cross-channel) relationships, `det` and `ctx` are shifted. The shift direction is controlled by `evo_dim`:
- **`evo_dim=1` (channel)**: Uses circular shift (`torch.roll`), as channel order is semantically arbitrary.
- **`evo_dim=2` (height)** or **`evo_dim=3` (width)**: Uses reflection padding + slicing to avoid boundary artifacts. Specifically, the feature is padded reflectively, then cropped to the original size, simulating translation without periodicity.

#### 3. Dot and Wedge Products
For each shift step $s$, shifted features `ctx_s` and `det_s` are computed, then:
- **Dot product (inner)**: $\mathrm{LeakyReLU}(\mathrm{det} \odot \mathrm{ctx}_s)$. This measures similarity, enhancing coherent textures.
- **Wedge product (outer)**: $(\mathrm{det} \odot \mathrm{ctx}_s) - (\mathrm{ctx} \odot \mathrm{det}_s)$. This antisymmetric combination is sensitive to differences, acting as a learnable edge detector. Its contribution is scaled by a learnable `wedge_scale` (initialized small, e.g., 0.1) to avoid early noise.

#### 4. Feature Fusion and Gating
All dot and wedge results across shift steps are concatenated along the channel dimension, then projected back to $D$ channels via a 1×1 convolution (`proj`), yielding fused feature $g$.  
Then, $x$ (normalized input) and $g$ are concatenated and fed into a two-layer 1×1 gating network, outputting a sigmoid weight $a$ (same shape as $g$). The updated feature is:
$$
h = \mathrm{det} + a \odot g
$$
ensuring `det` is preserved while $g$ is adaptively injected.

#### 5. Residual Connection
To avoid disrupting early training, a learnable residual scale `gamma` (initialized near zero, e.g., 1e-4) is applied:
$$
\mathrm{out} = X + \gamma \cdot h
$$
This makes the module act as an identity map initially, gradually activating as training progresses.

#### 6. Multi‑dimensional Synergy
By stacking Evo modules across different `evo_dim` (e.g., channel, height, width), the generator captures relationships from multiple angles: channel‑wise correlations and spatial structures. This multi‑view interaction enriches image details and global coherence.

### QFiLM Modulation
The orthogonal condition \(q\) is injected into intermediate layers (e.g., 32×32 and 128×128) through a lightweight Feature-wise Linear Modulation (FiLM) that learns channel‑wise affine transformations. The modulation is initialized near‑identity to avoid early training disruption.

For full details, please refer to the code and the [paper](link-to-paper).
