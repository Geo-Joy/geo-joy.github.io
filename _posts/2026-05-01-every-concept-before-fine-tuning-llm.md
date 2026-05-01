---
title: "Every Concept You Need Before Fine-Tuning an LLM"
date: 2026-05-01
permalink: /posts/every-concept-before-fine-tuning-llm/
seo_image: fine-tuning-guide/post.jpg
tags:
  - LLM
  - fine-tuning
  - LoRA
  - QLoRA
  - machine learning
---

*A practitioner's reference — LoRA, QLoRA, batch size, loss curves, and output formats explained. This is the concepts companion to **[I Fine-Tuned Gemma 4 to Detect Code Vulnerabilities — Here's What Happened](/posts/fine-tuned-gemma4-code-vulnerabilities/)**.*

*Most engineering teams use LLMs through APIs — prompt in, response out. The models themselves are a black box. Fine-tuning opens that box: instead of crafting better prompts, you adjust the model's weights directly. I recently ran my first fine-tuning experiment and spent more time understanding the concepts than writing the code. This post is the reference guide I wish existed when I started.*


<svg width="100%" viewBox="0 0 680 310" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="310" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <defs><marker id="arr3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
  <text x="340" y="24" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">The fine-tuning pipeline — where each concept fits</text>
  <!-- Stage 1: Setup -->
  <rect x="16" y="42" width="142" height="244" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="16" y="42" width="142" height="28" rx="10" fill="#EEEDFE"/>
  <rect x="16" y="56" width="142" height="14" fill="#EEEDFE"/>
  <text x="87" y="61" text-anchor="middle" font-size="11" font-weight="500" fill="#3C3489">1. Setup</text>
  <rect x="28" y="82" width="118" height="28" rx="4" fill="#EEEDFE" stroke="#B5ADF2" stroke-width="0.5"/>
  <text x="87" y="100" text-anchor="middle" font-size="10" fill="#3C3489">SFT paradigm</text>
  <rect x="28" y="116" width="118" height="28" rx="4" fill="#EEEDFE" stroke="#B5ADF2" stroke-width="0.5"/>
  <text x="87" y="134" text-anchor="middle" font-size="10" fill="#3C3489">Input/output pairs</text>
  <rect x="28" y="150" width="118" height="28" rx="4" fill="#EEEDFE" stroke="#B5ADF2" stroke-width="0.5"/>
  <text x="87" y="168" text-anchor="middle" font-size="10" fill="#3C3489">Chat template</text>
  <!-- Arrow 1→2 -->
  <line x1="158" y1="164" x2="194" y2="164" stroke="#b4b2a9" stroke-width="0.5" marker-end="url(#arr3)"/>
  <!-- Stage 2: Model Loading -->
  <rect x="198" y="42" width="160" height="244" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="198" y="42" width="160" height="28" rx="10" fill="#E6F1FB"/>
  <rect x="198" y="56" width="160" height="14" fill="#E6F1FB"/>
  <text x="278" y="61" text-anchor="middle" font-size="11" font-weight="500" fill="#0C447C">2. Load model</text>
  <rect x="210" y="82" width="136" height="28" rx="4" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="278" y="100" text-anchor="middle" font-size="10" fill="#0C447C">LoRA adapters</text>
  <rect x="210" y="116" width="136" height="28" rx="4" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="278" y="134" text-anchor="middle" font-size="10" fill="#0C447C">QLoRA (4-bit loading)</text>
  <rect x="210" y="150" width="136" height="28" rx="4" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="278" y="168" text-anchor="middle" font-size="10" fill="#0C447C">NF4 quantization</text>
  <rect x="210" y="184" width="136" height="28" rx="4" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="278" y="202" text-anchor="middle" font-size="10" fill="#0C447C">Gradient checkpointing</text>
  <!-- Arrow 2→3 -->
  <line x1="358" y1="164" x2="394" y2="164" stroke="#b4b2a9" stroke-width="0.5" marker-end="url(#arr3)"/>
  <!-- Stage 3: Training -->
  <rect x="398" y="42" width="142" height="244" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="398" y="42" width="142" height="28" rx="10" fill="#FAEEDA"/>
  <rect x="398" y="56" width="142" height="14" fill="#FAEEDA"/>
  <text x="469" y="61" text-anchor="middle" font-size="11" font-weight="500" fill="#633806">3. Train</text>
  <rect x="410" y="82" width="118" height="28" rx="4" fill="#FAEEDA" stroke="#DEA544" stroke-width="0.5"/>
  <text x="469" y="100" text-anchor="middle" font-size="10" fill="#633806">Batch size</text>
  <rect x="410" y="116" width="118" height="28" rx="4" fill="#FAEEDA" stroke="#DEA544" stroke-width="0.5"/>
  <text x="469" y="134" text-anchor="middle" font-size="10" fill="#633806">Grad accumulation</text>
  <rect x="410" y="150" width="118" height="28" rx="4" fill="#FAEEDA" stroke="#DEA544" stroke-width="0.5"/>
  <text x="469" y="168" text-anchor="middle" font-size="10" fill="#633806">Loss curves</text>
  <rect x="410" y="184" width="118" height="28" rx="4" fill="#FAEEDA" stroke="#DEA544" stroke-width="0.5"/>
  <text x="469" y="202" text-anchor="middle" font-size="10" fill="#633806">Epochs</text>
  <!-- Arrow 3→4 -->
  <line x1="540" y1="164" x2="576" y2="164" stroke="#b4b2a9" stroke-width="0.5" marker-end="url(#arr3)"/>
  <!-- Stage 4: Output -->
  <rect x="580" y="42" width="84" height="244" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="580" y="42" width="84" height="28" rx="10" fill="#E1F5EE"/>
  <rect x="580" y="56" width="84" height="14" fill="#E1F5EE"/>
  <text x="622" y="61" text-anchor="middle" font-size="11" font-weight="500" fill="#085041">4. Save</text>
  <rect x="590" y="82" width="64" height="28" rx="4" fill="#E1F5EE" stroke="#6BC8A8" stroke-width="0.5"/>
  <text x="622" y="100" text-anchor="middle" font-size="10" fill="#085041">Adapter</text>
  <rect x="590" y="116" width="64" height="28" rx="4" fill="#E1F5EE" stroke="#6BC8A8" stroke-width="0.5"/>
  <text x="622" y="134" text-anchor="middle" font-size="10" fill="#085041">Merged</text>
  <rect x="590" y="150" width="64" height="28" rx="4" fill="#E1F5EE" stroke="#6BC8A8" stroke-width="0.5"/>
  <text x="622" y="168" text-anchor="middle" font-size="10" fill="#085041">GGUF</text>
  <!-- Memory note at bottom -->
  <rect x="16" y="296" width="648" height="1" fill="none"/>
  <text x="340" y="306" text-anchor="middle" font-size="10" fill="#888780">Read left to right — each stage introduces the concepts explained in detail below.</text>
</svg>

---

## What is fine-tuning, and why not just prompt better?

Zero-shot prompting means giving an LLM instructions and hoping it follows them. It works surprisingly well for general tasks. But when you need a model to perform one specific task consistently — same format, same decision boundary, every time — fine-tuning has an edge.

You show the model thousands of input/output examples, and it adjusts its internal weights to reproduce that pattern. The result is a smaller, specialized model that does one thing reliably, versus a large general model that needs careful prompting and still varies.

---

## What's SFT?

**SFT** stands for **Supervised Fine-Tuning**. "Supervised" means you provide the right answers — input/output pairs. The model sees a code snippet (input) and the correct verdict like "VULNERABLE — CWE-120" (output), repeated thousands of times. It adjusts its weights to predict similar outputs for similar inputs.

This is different from **RLHF** (reinforcement learning from human feedback), where the model gets a score for how good its answer was, or unsupervised pre-training where the model just reads text with no labels. The **SFTTrainer** from the **TRL** (Transformer Reinforcement Learning) library — HuggingFace's toolkit for fine-tuning and aligning LLMs — handles the mechanics: tokenization, masking user messages so the model only learns to predict assistant responses, and running the training loop.

**When to use SFT vs alternatives:** SFT is the right choice when you have labeled data (input/output pairs) and want the model to produce structured, explainable responses. If you only needed a binary score without explanations, a **classification head** on top of the model would be simpler — though in my experiment I chose SFT because I wanted the model to also produce reasoning and CWE classifications alongside the verdict, not just a bare label. If you wanted to refine response *quality* after SFT, **DPO** (Direct Preference Optimization) takes pairs of good/bad responses and teaches the model to prefer the better one — that's the SFT → DPO pipeline most production models use. **RLHF** goes further with a full reward model and reinforcement learning, but that's overkill unless "good" is subjective and hard to label. For most fine-tuning projects, SFT is where you start.

---

## What are LoRA and QLoRA?

Full fine-tuning updates all of a model's parameters. For a model like Gemma 4 E4B, that's 8 billion numbers — you'd need 80–100GB of GPU memory. The breakdown: 16GB for weights in 16-bit. 16GB for **gradients** — a value per weight that tells the optimizer *which direction and how steeply* the loss changes with respect to that weight. The optimizer then decides how far to actually move. 64GB for Adam optimizer states (it tracks momentum and variance for every weight, both in 32-bit). Plus activations. Even with memory-efficient optimizers, you're looking at 60GB minimum. Not practical on most hardware.


<svg width="100%" viewBox="0 0 680 310" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="310" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="24" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">Three ways to fine-tune — Full FT vs LoRA vs QLoRA</text>
  <!-- === Full Fine-Tuning === -->
  <rect x="22" y="40" width="200" height="250" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="22" y="40" width="200" height="30" rx="10" fill="#FCEBEB"/>
  <rect x="22" y="56" width="200" height="14" fill="#FCEBEB"/>
  <text x="122" y="60" text-anchor="middle" font-size="12" font-weight="500" fill="#791F1F">Full Fine-Tuning</text>
  <!-- Model blocks - all trainable -->
  <rect x="46" y="82" width="152" height="22" rx="4" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="122" y="97" text-anchor="middle" font-size="10" fill="#A32D2D">Attention × 42</text>
  <rect x="46" y="110" width="152" height="22" rx="4" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="122" y="125" text-anchor="middle" font-size="10" fill="#A32D2D">MLP × 42</text>
  <rect x="46" y="138" width="152" height="22" rx="4" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="122" y="153" text-anchor="middle" font-size="10" fill="#A32D2D">Embeddings</text>
  <!-- Memory bar -->
  <rect x="46" y="178" width="152" height="44" rx="6" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.8"/>
  <text x="122" y="196" text-anchor="middle" font-size="10" font-weight="500" fill="#791F1F">8B params trainable</text>
  <text x="122" y="212" text-anchor="middle" font-size="9" fill="#A32D2D">Weights + grads + optim</text>
  <text x="122" y="240" text-anchor="middle" font-size="12" font-weight="500" fill="#A32D2D">~100 GB</text>
  <!-- Arrow -->
  <text x="233" y="165" text-anchor="middle" font-size="18" fill="#b4b2a9">→</text>
  <!-- === LoRA === -->
  <rect x="244" y="40" width="200" height="250" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="244" y="40" width="200" height="30" rx="10" fill="#E6F1FB"/>
  <rect x="244" y="56" width="200" height="14" fill="#E6F1FB"/>
  <text x="344" y="60" text-anchor="middle" font-size="12" font-weight="500" fill="#0C447C">LoRA</text>
  <!-- Base model blocks - frozen -->
  <rect x="266" y="82" width="110" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="321" y="97" text-anchor="middle" font-size="10" fill="#5F5E5A">Attn (frozen)</text>
  <!-- LoRA adapter blocks -->
  <rect x="382" y="82" width="42" height="22" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="403" y="97" text-anchor="middle" font-size="9" fill="#085041">+A</text>
  <rect x="266" y="110" width="110" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="321" y="125" text-anchor="middle" font-size="10" fill="#5F5E5A">MLP (frozen)</text>
  <rect x="382" y="110" width="42" height="22" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="403" y="125" text-anchor="middle" font-size="9" fill="#085041">+A</text>
  <rect x="266" y="138" width="110" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="321" y="153" text-anchor="middle" font-size="10" fill="#5F5E5A">Emb (frozen)</text>
  <!-- Memory bar -->
  <rect x="266" y="178" width="158" height="44" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="345" y="196" text-anchor="middle" font-size="10" font-weight="500" fill="#085041">42M params trainable</text>
  <text x="345" y="212" text-anchor="middle" font-size="9" fill="#0F6E56">Base frozen in 16-bit</text>
  <text x="345" y="240" text-anchor="middle" font-size="12" font-weight="500" fill="#0F6E56">~18 GB</text>
  <!-- Arrow -->
  <text x="455" y="165" text-anchor="middle" font-size="18" fill="#b4b2a9">→</text>
  <!-- === QLoRA === -->
  <rect x="466" y="40" width="192" height="250" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <rect x="466" y="40" width="192" height="30" rx="10" fill="#E1F5EE"/>
  <rect x="466" y="56" width="192" height="14" fill="#E1F5EE"/>
  <text x="562" y="60" text-anchor="middle" font-size="12" font-weight="500" fill="#085041">QLoRA</text>
  <!-- Base model blocks - frozen + compressed -->
  <rect x="486" y="82" width="94" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 2"/>
  <text x="533" y="97" text-anchor="middle" font-size="10" fill="#888780">Attn (4-bit)</text>
  <rect x="586" y="82" width="52" height="22" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="612" y="97" text-anchor="middle" font-size="9" fill="#085041">+A</text>
  <rect x="486" y="110" width="94" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 2"/>
  <text x="533" y="125" text-anchor="middle" font-size="10" fill="#888780">MLP (4-bit)</text>
  <rect x="586" y="110" width="52" height="22" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="612" y="125" text-anchor="middle" font-size="9" fill="#085041">+A</text>
  <rect x="486" y="138" width="94" height="22" rx="4" fill="#fff" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 2"/>
  <text x="533" y="153" text-anchor="middle" font-size="10" fill="#888780">Emb (4-bit)</text>
  <!-- Memory bar -->
  <rect x="486" y="178" width="152" height="44" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.8"/>
  <text x="562" y="196" text-anchor="middle" font-size="10" font-weight="500" fill="#085041">42M params trainable</text>
  <text x="562" y="212" text-anchor="middle" font-size="9" fill="#0F6E56">Base compressed to 4-bit NF4</text>
  <text x="562" y="240" text-anchor="middle" font-size="12" font-weight="500" fill="#0F6E56">~10 GB</text>
  <!-- Connecting note at bottom -->
  <text x="340" y="275" text-anchor="middle" font-size="10" fill="#5F5E5A">LoRA → QLoRA: the only change is compressing the frozen base from 16-bit to 4-bit.</text>
  <line x1="200" y1="290" x2="480" y2="290" stroke="#b4b2a9" stroke-width="0.5"/>
  <text x="340" y="302" text-anchor="middle" font-size="10" fill="#888780">Same adapters, same training math, same gradients. Only the base model's storage format differs.</text>
</svg>

**LoRA** (Low-Rank Adaptation) takes a different approach. You freeze the entire base model and inject small trainable matrices into specific layers. These are called **adapters**. Think of it as: you have a textbook (the base model). Instead of rewriting every page, you add sticky notes to the pages that matter. The textbook stays the same; the sticky notes customize it for your task.

In my experiment, I trained 42 million parameters out of 8 billion — just 0.53% of the model. Where does that number come from? For each target layer, LoRA adds two small matrices instead of updating the full weight matrix. Say an attention layer has a weight matrix of size 3072 × 3072 (~9.4 million parameters). LoRA replaces that with two tiny matrices:

```
Original weight:  3072 × 3072 = 9,437,184 parameters
LoRA adapter A:   16 × 3072   = 49,152 parameters
LoRA adapter B:   3072 × 16   = 49,152 parameters
Total per module: 98,304 parameters (vs 9.4 million)
```

The `16` is the LoRA rank — our chosen adapter size. Multiply across 7 target modules (q, k, v, o, gate, up, down) per layer, across all 42 transformer layers, and you get ~42 million trainable parameters. Increase the rank to 32 and it doubles to ~84M. Drop to rank 8 and it halves to ~21M. The rank is your dial between "learn more" and "use less memory."


<svg width="100%" viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="270" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="30" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">LoRA rank decomposition — Gemma 4 E4B attention layer</text>
  <rect x="40" y="54" width="140" height="140" rx="4" fill="#FCEBEB" stroke="#A32D2D" stroke-width="0.5"/>
  <text x="110" y="119" text-anchor="middle" font-size="13" font-weight="500" fill="#791F1F">Original W</text>
  <text x="110" y="139" text-anchor="middle" font-size="11" fill="#A32D2D">3072 × 3072</text>
  <text x="110" y="214" text-anchor="middle" font-size="11" fill="#791F1F">9.4M params</text>
  <text x="110" y="230" text-anchor="middle" font-size="11" fill="#A32D2D">All trainable</text>
  <text x="225" y="129" text-anchor="middle" font-size="18" fill="#888">→</text>
  <rect x="270" y="54" width="80" height="140" rx="4" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="310" y="119" text-anchor="middle" font-size="13" font-weight="500" fill="#085041">B</text>
  <text x="310" y="139" text-anchor="middle" font-size="11" fill="#0F6E56">3072 × 16</text>
  <text x="375" y="129" text-anchor="middle" font-size="14" fill="#888">×</text>
  <rect x="400" y="99" width="140" height="44" rx="4" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="470" y="117" text-anchor="middle" font-size="13" font-weight="500" fill="#085041">A</text>
  <text x="470" y="135" text-anchor="middle" font-size="11" fill="#0F6E56">16 × 3072</text>
  <text x="390" y="214" text-anchor="middle" font-size="11" fill="#085041">98K params (96× smaller)</text>
  <text x="390" y="230" text-anchor="middle" font-size="11" fill="#0F6E56">Only these train</text>
  <text x="340" y="258" text-anchor="middle" font-size="11" fill="#5F5E5A">× 7 modules × 42 layers = ~42M trainable (0.53% of Gemma 4 E4B)</text>
</svg>

Because the base model is frozen, gradients and optimizer states are only computed for the adapter — 42 million parameters, not 8 billion. That's why the memory drops dramatically.

The frozen base model still sits in GPU memory though. With standard LoRA, it stays at full 16-bit precision — that's ~8GB just for weights you're not even changing. That's where QLoRA comes in.

**QLoRA** is LoRA with exactly one change: compress the frozen base model to 4-bit when loading it into memory. The adapters, the training loop, the gradients, the optimizer — all identical to LoRA. The only difference is how much space the frozen base occupies in VRAM. In code, it's a single flag: `load_in_4bit=True`. Set it to `False` and you're doing standard LoRA. Set it to `True` and you're doing QLoRA.

But that one flag triggers more than simple compression. Under the hood, the **bitsandbytes** library applies several innovations from the [QLoRA paper](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023). The key one: it uses a smart compression method called **NF4** that's specifically designed for neural network weights — instead of rounding numbers uniformly (which loses a lot), it places the 4-bit quantization levels where the weight values are most dense. This preserves 95–98% of model quality despite the 4x compression.

To be explicit: the adapters you're training still run in full 16-bit precision — only the frozen base gets compressed. The base weights shrink from ~8GB to ~2.5GB, and the total setup fits comfortably on a single GPU. Everything else — LoRA rank, target modules, learning rate, batch size, gradient flow — stays the same.

Here's the memory contrast:

| | Full fine-tuning | LoRA | QLoRA |
|--|-----------------|------|-------|
| Base weights | 16GB (8B × 16-bit) | 16GB (8B × 16-bit, frozen) | **2.5GB** (8B × 4-bit, frozen) |
| Gradients | 16GB (8B params) | 84MB (42M adapter params) | **84MB** (42M adapter params) |
| Optimizer states | 64GB (8B × 2 × 32-bit) | 336MB (42M × 2 × 32-bit) | **336MB** (42M × 2 × 32-bit) |
| Total (+ activations) | **~100GB** | **~18GB** | **~10GB** |

Notice that LoRA and QLoRA have identical adapter sizes, gradient sizes, and optimizer sizes. The only row that changes is base weights — 16GB vs 2.5GB. That's the entire difference.

Naive 4-bit compression would lose meaningful quality. NF4 is what makes QLoRA work — it's the reason that one flag doesn't tank your results.


<svg width="100%" viewBox="0 0 680 410" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="410" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="170" y="28" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">Full fine-tuning</text>
  <text x="510" y="28" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">QLoRA (what we used)</text>
  <!-- Left: full fine-tuning -->
  <rect x="60" y="48" width="220" height="290" rx="14" fill="#F1EFE8" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="170" y="74" text-anchor="middle" font-size="12" font-weight="500" fill="#444441">Gemma 4 E4B — 8B params</text>
  <text x="170" y="90" text-anchor="middle" font-size="11" fill="#888780">All in 16-bit, all trainable</text>
  <rect x="80" y="106" width="180" height="36" rx="6" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="170" y="128" text-anchor="middle" font-size="11" font-weight="500" fill="#791F1F">Attention × 42 layers</text>
  <rect x="80" y="152" width="180" height="36" rx="6" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="170" y="174" text-anchor="middle" font-size="11" font-weight="500" fill="#791F1F">MLP × 42 layers</text>
  <rect x="80" y="198" width="180" height="36" rx="6" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="170" y="220" text-anchor="middle" font-size="11" font-weight="500" fill="#791F1F">Embeddings (PLE)</text>
  <text x="170" y="264" text-anchor="middle" font-size="11" fill="#791F1F">Every weight updated</text>
  <text x="170" y="282" text-anchor="middle" font-size="12" font-weight="500" fill="#A32D2D">~100 GB needed</text>
  <text x="170" y="354" text-anchor="middle" font-size="11" fill="#888780">Weights: 16 GB</text>
  <text x="170" y="370" text-anchor="middle" font-size="11" fill="#888780">Gradients: 16 GB</text>
  <text x="170" y="386" text-anchor="middle" font-size="11" fill="#888780">Optimizer: 64 GB</text>
  <!-- Right: QLoRA -->
  <rect x="400" y="48" width="220" height="290" rx="14" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="510" y="74" text-anchor="middle" font-size="12" font-weight="500" fill="#0C447C">Gemma 4 E4B — 8B params</text>
  <text x="510" y="90" text-anchor="middle" font-size="11" fill="#378ADD">Frozen in 4-bit NF4</text>
  <rect x="420" y="106" width="128" height="36" rx="6" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="484" y="128" text-anchor="middle" font-size="10" fill="#5F5E5A">Attention (frozen)</text>
  <rect x="554" y="106" width="52" height="36" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="580" y="128" text-anchor="middle" font-size="10" font-weight="500" fill="#085041">LoRA</text>
  <rect x="420" y="152" width="128" height="36" rx="6" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="484" y="174" text-anchor="middle" font-size="10" fill="#5F5E5A">MLP (frozen)</text>
  <rect x="554" y="152" width="52" height="36" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="580" y="174" text-anchor="middle" font-size="10" font-weight="500" fill="#085041">LoRA</text>
  <rect x="420" y="198" width="128" height="36" rx="6" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="484" y="220" text-anchor="middle" font-size="10" fill="#5F5E5A">Embeddings (frozen)</text>
  <text x="510" y="264" text-anchor="middle" font-size="11" fill="#085041">Only LoRA adapters train (0.53%)</text>
  <text x="510" y="282" text-anchor="middle" font-size="12" font-weight="500" fill="#0F6E56">~10 GB needed</text>
  <text x="510" y="354" text-anchor="middle" font-size="11" fill="#888780">Base weights: 2.5 GB (4-bit)</text>
  <text x="510" y="370" text-anchor="middle" font-size="11" fill="#888780">Adapter + grads: 168 MB</text>
  <text x="510" y="386" text-anchor="middle" font-size="11" fill="#888780">Optimizer: 336 MB</text>
  <!-- vs -->
  <text x="340" y="189" text-anchor="middle" font-size="13" fill="#888780">vs</text>
</svg>

---

## Why load the model in 4-bit? Won't that hurt accuracy?

A model's weights are just numbers — millions of them. Each number can be stored at different precisions:

- **16-bit:** high precision, like `3.141592653589793`. Takes more space.
- **4-bit:** lower precision, like `3.1`. Takes 4x less space.

The intuition says 4-bit should be much worse. And with naive rounding, it would be. But NF4 (the smart compression method used by QLoRA) places quantization levels where the weight values actually cluster rather than spacing them evenly. That's why the research shows 95–98% quality retention.


<svg width="100%" viewBox="0 0 680 420" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="420" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="26" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">Why NF4 beats naive 4-bit compression</text>
  <text x="340" y="42" text-anchor="middle" font-size="11" fill="#888780">Neural network weights follow a bell curve — most values cluster near zero.</text>
  <!-- === Top panel: Uniform 4-bit === -->
  <rect x="20" y="58" width="640" height="158" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="340" y="80" text-anchor="middle" font-size="12" font-weight="500" fill="#A32D2D">Uniform 4-bit (naive rounding)</text>
  <text x="340" y="96" text-anchor="middle" font-size="10" fill="#888780">16 evenly spaced levels across the full range</text>
  <!-- Bell curve -->
  <path d="M 90,186 C 140,186 186,140 230,92 C 260,60 296,44 340,40 C 384,44 420,60 450,92 C 494,140 540,186 590,186" fill="#FCEBEB" fill-opacity="0.5" stroke="#E24B4A" stroke-width="0.8"/>
  <!-- X-axis line -->
  <line x1="90" y1="186" x2="590" y2="186" stroke="#d4d3cf" stroke-width="0.5"/>
  <!-- Uniform tick marks (16 evenly spaced) -->
  <line x1="90" y1="186" x2="90" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="106" y1="186" x2="106" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="139" y1="186" x2="139" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="172" y1="186" x2="172" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="206" y1="186" x2="206" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="239" y1="186" x2="239" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="272" y1="186" x2="272" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="306" y1="186" x2="306" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="340" y1="186" x2="340" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="374" y1="186" x2="374" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="408" y1="186" x2="408" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="441" y1="186" x2="441" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="474" y1="186" x2="474" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="508" y1="186" x2="508" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="541" y1="186" x2="541" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <line x1="574" y1="186" x2="574" y2="196" stroke="#A32D2D" stroke-width="0.8"/>
  <!-- Wasted annotation brackets -->
  <line x1="90" y1="200" x2="90" y2="208" stroke="#A32D2D" stroke-width="0.5"/>
  <line x1="206" y1="200" x2="206" y2="208" stroke="#A32D2D" stroke-width="0.5"/>
  <line x1="90" y1="204" x2="206" y2="204" stroke="#A32D2D" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text x="148" y="218" text-anchor="middle" font-size="9" fill="#A32D2D">~6 levels wasted on near-empty tails</text>
  <!-- === Bottom panel: NF4 === -->
  <rect x="20" y="236" width="640" height="158" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="340" y="258" text-anchor="middle" font-size="12" font-weight="500" fill="#0F6E56">NF4 (QLoRA's method)</text>
  <text x="340" y="274" text-anchor="middle" font-size="10" fill="#888780">16 levels placed at quantiles of the normal distribution — dense where weights are dense</text>
  <!-- Same bell curve -->
  <path d="M 90,364 C 140,364 186,318 230,270 C 260,238 296,222 340,218 C 384,222 420,238 450,270 C 494,318 540,364 590,364" fill="#E1F5EE" fill-opacity="0.5" stroke="#1D9E75" stroke-width="0.8"/>
  <line x1="90" y1="364" x2="590" y2="364" stroke="#d4d3cf" stroke-width="0.5"/>
  <!-- NF4 tick marks (clustered near center) -->
  <line x1="194" y1="364" x2="194" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="218" y1="364" x2="218" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="238" y1="364" x2="238" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="254" y1="364" x2="254" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="270" y1="364" x2="270" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="284" y1="364" x2="284" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="298" y1="364" x2="298" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="312" y1="364" x2="312" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="326" y1="364" x2="326" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="340" y1="364" x2="340" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="354" y1="364" x2="354" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="368" y1="364" x2="368" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="382" y1="364" x2="382" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="396" y1="364" x2="396" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="412" y1="364" x2="412" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="430" y1="364" x2="430" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="452" y1="364" x2="452" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <line x1="478" y1="364" x2="478" y2="374" stroke="#0F6E56" stroke-width="0.8"/>
  <!-- Good coverage annotation -->
  <line x1="270" y1="378" x2="410" y2="378" stroke="#0F6E56" stroke-width="0.5"/>
  <line x1="270" y1="378" x2="270" y2="386" stroke="#0F6E56" stroke-width="0.5"/>
  <line x1="410" y1="378" x2="410" y2="386" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="340" y="396" text-anchor="middle" font-size="9" fill="#0F6E56">Dense where it matters — 95–98% quality retention</text>
</svg>

The other key insight: we're not training those compressed weights. They're frozen. The LoRA adapters running on top are in full 16-bit precision and can actually compensate for the small precision loss in the base. So by the time training is done, the fine-tuned model often performs nearly identically to one trained from a 16-bit base.

---

## If the base model is in 4-bit, how does training happen in 16-bit?

This is the most common confusion about QLoRA. The answer: the 4-bit is a **storage** format, not a **compute** format. The math always happens in 16-bit.

Here's what happens in a single forward pass through one layer:

```
Input → [Base layer weights: stored in 4-bit, dequantized to 16-bit on the fly]
       → output_base (16-bit)

Input → [LoRA adapter weights: stored and computed in 16-bit]
       → output_adapter (16-bit)

Final output = output_base + output_adapter
```

The base weights sit in GPU memory compressed to 4-bit. But when the model needs to do actual matrix multiplication, bitsandbytes **dequantizes them to 16-bit temporarily** for that one computation, then discards the 16-bit version. The 4-bit copy stays in memory as the permanent stored format — the 16-bit version only exists for a split second during the calculation.

The LoRA adapter is a separate small matrix that runs entirely in 16-bit. Its output gets **added** to the base layer's output. During backpropagation, gradients only flow through the adapter (because the base is frozen), so 16-bit precision is maintained end-to-end for everything that's actually learning.

So it's not "training 4-bit weights in 16-bit." It's:

- **Storing** base weights in 4-bit (saves memory)
- **Computing** with them in 16-bit (dequantize on the fly, preserves quality)
- **Training** only the adapter, which was always 16-bit

The 4-bit is purely a storage compression. The math always happens in 16-bit. That's why NF4 is designed the way it is — optimized for dequantizing back to 16-bit with minimal information loss.


<svg width="100%" viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="290" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
  <text x="340" y="26" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">QLoRA forward pass — one Gemma 4 E4B layer</text>
  <!-- Top path: base weights -->
  <rect x="40" y="52" width="88" height="40" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
  <text x="84" y="76" text-anchor="middle" font-size="12" font-weight="500" fill="#3C3489">Input</text>
  <line x1="128" y1="72" x2="168" y2="72" stroke="#888" stroke-width="0.5" marker-end="url(#arr)"/>
  <rect x="172" y="52" width="140" height="40" rx="8" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="242" y="68" text-anchor="middle" font-size="11" font-weight="500" fill="#0C447C">Base weights</text>
  <text x="242" y="84" text-anchor="middle" font-size="10" fill="#378ADD">Stored in 4-bit NF4</text>
  <line x1="312" y1="72" x2="352" y2="72" stroke="#888" stroke-width="0.5" marker-end="url(#arr)"/>
  <rect x="356" y="52" width="130" height="40" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="421" y="68" text-anchor="middle" font-size="11" font-weight="500" fill="#633806">Dequantize</text>
  <text x="421" y="84" text-anchor="middle" font-size="10" fill="#BA7517">4-bit → 16-bit</text>
  <line x1="486" y1="72" x2="526" y2="72" stroke="#888" stroke-width="0.5" marker-end="url(#arr)"/>
  <rect x="530" y="52" width="110" height="40" rx="8" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="585" y="76" text-anchor="middle" font-size="11" fill="#5F5E5A">Base output</text>
  <text x="421" y="110" text-anchor="middle" font-size="10" fill="#BA7517" font-style="italic">Temporary — discarded after computation</text>
  <!-- Bottom path: LoRA adapter -->
  <line x1="84" y1="92" x2="84" y2="154" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 3"/>
  <line x1="84" y1="154" x2="168" y2="154" stroke="#888" stroke-width="0.5" marker-end="url(#arr)"/>
  <rect x="172" y="134" width="140" height="40" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="242" y="150" text-anchor="middle" font-size="11" font-weight="500" fill="#085041">LoRA adapter</text>
  <text x="242" y="166" text-anchor="middle" font-size="10" fill="#0F6E56">Always 16-bit (42M params)</text>
  <line x1="312" y1="154" x2="585" y2="154" stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="4 3"/>
  <line x1="585" y1="154" x2="585" y2="112" stroke="#888" stroke-width="0.5" marker-end="url(#arr)"/>
  <!-- Add box -->
  <rect x="555" y="96" width="60" height="22" rx="4" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
  <text x="585" y="111" text-anchor="middle" font-size="10" fill="#3C3489">Add</text>
  <!-- Backward pass note -->
  <rect x="40" y="206" width="290" height="56" rx="10" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="185" y="228" text-anchor="middle" font-size="11" font-weight="500" fill="#085041">Backward pass</text>
  <text x="185" y="246" text-anchor="middle" font-size="10" fill="#0F6E56">Gradients flow only through LoRA adapter</text>
  <rect x="350" y="206" width="290" height="56" rx="10" fill="#fff" stroke="#B4B2A9" stroke-width="0.5"/>
  <text x="495" y="228" text-anchor="middle" font-size="11" font-weight="500" fill="#5F5E5A">Base weights: no gradients</text>
  <text x="495" y="246" text-anchor="middle" font-size="10" fill="#888780">Frozen — never updated, stays in 4-bit</text>
</svg>

---

## What does gradient checkpointing do?

During training, the GPU remembers the output of every layer so it can calculate gradients during backpropagation (the "learning" pass). For a model with dozens of layers, that eats a ton of VRAM — often more than the model weights themselves.

**Gradient checkpointing** says: "Don't remember everything. Throw away most intermediate outputs, and recompute them when needed during backpropagation." You trade compute time (recalculating) for memory savings (not storing it all).

Libraries like Unsloth offer custom implementations (`use_gradient_checkpointing="unsloth"`) that are smarter about which layers to save versus recompute, saving more memory with less speed penalty than PyTorch's default.

The three memory tricks work together:

- **4-bit loading** — shrinks model weights (8GB → 2.5GB)
- **Gradient checkpointing** — shrinks stored activations
- **LoRA** — only trains ~1% of parameters, so optimizer states are tiny

All three combined make it possible to fine-tune a multi-billion parameter model on a single GPU.

---

## What is learning rate?

The **learning rate** controls how big a step the optimizer takes on each weight update. After the model processes a batch and computes gradients, the learning rate determines how far the weights actually move in the direction those gradients suggest.

Too high and the model overshoots — loss jumps around erratically instead of decreasing. Too low and the model barely moves — loss flatlines even though the model hasn't converged. A common default for LoRA fine-tuning is `2e-4` (0.0002), which works well as a starting point. If your loss is oscillating wildly, try halving it. If your loss isn't moving, try doubling it.

---

## What are batch size and gradient accumulation?

**Batch size** = how many samples the GPU processes at once. Each sample sits in VRAM simultaneously. Bigger batch = more VRAM usage but faster training.

**Gradient accumulation** = how many batches to stack up before updating the weights. With `grad_accum=8`, the GPU processes 8 mini-batches one at a time, adds up the gradients, then makes one combined weight update.

The math: `batch_size × grad_accum = effective batch size`

Both of these give an effective batch of 8, but use memory differently:

- `batch_size=8, grad_accum=1` — fast (8 samples in parallel) but needs more VRAM
- `batch_size=1, grad_accum=8` — slow (1 sample at a time, 8 sequential passes) but uses minimal VRAM

The model learns the same thing either way — the weight updates are mathematically identical. You're trading speed for memory.


<svg width="100%" viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="300" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <defs><marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
  <text x="340" y="26" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">Two ways to get effective batch size = 8</text>
  <text x="340" y="42" text-anchor="middle" font-size="11" fill="#888780">Same result — mathematically identical weight update. Trade speed for memory.</text>
  <!-- Left column: bs=8, ga=1 -->
  <rect x="28" y="60" width="290" height="218" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="173" y="82" text-anchor="middle" font-size="12" font-weight="500" fill="#0C447C">batch_size=8, grad_accum=1</text>
  <text x="173" y="98" text-anchor="middle" font-size="10" fill="#888780">Fast ✦ more VRAM</text>
  <!-- 8 sample blocks in a row -->
  <rect x="44" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="58" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₁</text>
  <rect x="76" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="90" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₂</text>
  <rect x="108" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="122" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₃</text>
  <rect x="140" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="154" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₄</text>
  <rect x="172" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="186" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₅</text>
  <rect x="204" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="218" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₆</text>
  <rect x="236" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="250" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₇</text>
  <rect x="268" y="112" width="28" height="28" rx="4" fill="#E6F1FB" stroke="#378ADD" stroke-width="0.5"/>
  <text x="282" y="131" text-anchor="middle" font-size="9" fill="#0C447C">s₈</text>
  <!-- Bracket -->
  <line x1="44" y1="140" x2="44" y2="148" stroke="#b4b2a9" stroke-width="0.5"/>
  <line x1="296" y1="140" x2="296" y2="148" stroke="#b4b2a9" stroke-width="0.5"/>
  <line x1="44" y1="144" x2="296" y2="144" stroke="#b4b2a9" stroke-width="0.5"/>
  <text x="173" y="160" text-anchor="middle" font-size="10" fill="#378ADD" font-style="italic">All 8 loaded into GPU at once</text>
  <!-- Arrow down -->
  <line x1="173" y1="172" x2="173" y2="198" stroke="#888" stroke-width="0.5" marker-end="url(#arr2)"/>
  <!-- GPU box -->
  <rect x="112" y="202" width="122" height="28" rx="6" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="173" y="220" text-anchor="middle" font-size="11" font-weight="500" fill="#633806">1 forward pass</text>
  <text x="173" y="242" text-anchor="middle" font-size="10" fill="#5F5E5A">gradients averaged → 1 update</text>
  <!-- Right column: bs=1, ga=8 -->
  <rect x="342" y="60" width="310" height="218" rx="10" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="497" y="82" text-anchor="middle" font-size="12" font-weight="500" fill="#0F6E56">batch_size=1, grad_accum=8</text>
  <text x="497" y="98" text-anchor="middle" font-size="10" fill="#888780">Slow ✦ minimal VRAM</text>
  <!-- Sequential single samples -->
  <rect x="370" y="112" width="28" height="28" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="384" y="131" text-anchor="middle" font-size="9" fill="#085041">s₁</text>
  <text x="410" y="131" font-size="12" fill="#b4b2a9">→</text>
  <rect x="424" y="112" width="28" height="28" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="438" y="131" text-anchor="middle" font-size="9" fill="#085041">s₂</text>
  <text x="464" y="131" font-size="12" fill="#b4b2a9">→</text>
  <rect x="478" y="112" width="28" height="28" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="492" y="131" text-anchor="middle" font-size="9" fill="#085041">s₃</text>
  <text x="518" y="131" font-size="12" fill="#b4b2a9">→</text>
  <text x="540" y="131" font-size="12" fill="#b4b2a9">...</text>
  <text x="560" y="131" font-size="12" fill="#b4b2a9">→</text>
  <rect x="572" y="112" width="28" height="28" rx="4" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="586" y="131" text-anchor="middle" font-size="9" fill="#085041">s₈</text>
  <text x="497" y="158" text-anchor="middle" font-size="10" fill="#0F6E56" font-style="italic">1 sample at a time × 8 sequential passes</text>
  <!-- Arrow to accumulator -->
  <line x1="497" y1="168" x2="497" y2="178" stroke="#888" stroke-width="0.5" marker-end="url(#arr2)"/>
  <!-- GPU / accumulator -->
  <rect x="400" y="182" width="194" height="28" rx="6" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
  <text x="497" y="200" text-anchor="middle" font-size="11" font-weight="500" fill="#3C3489">8 forward passes (one at a time)</text>
  <!-- Accumulator -->
  <rect x="440" y="218" width="114" height="22" rx="4" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="497" y="234" text-anchor="middle" font-size="10" fill="#633806">sum all 8 gradients</text>
  <text x="497" y="250" text-anchor="middle" font-size="10" fill="#5F5E5A">then → 1 combined update</text>
</svg>

---

## What do training loss and validation loss mean?

Both measure how surprised the model is by the correct answer. The model reads an input, predicts the next token in the expected response, and the loss reflects how wrong those predictions are. Lower = better.

- **Training loss:** measured on data the model is learning from. It will always keep going down — the model is memorizing these examples.
- **Validation loss:** measured on data the model has never trained on. This is the reality check.

The relationship matters:

- **Both going down** — the model is learning and generalizing. Good.
- **Training going down, validation going up** — overfitting. The model is memorizing rather than learning patterns.
- **Both stuck** — the model isn't learning. Learning rate may be too low.

Always watch the validation loss to decide when to stop training. Don't trust epoch count defaults from tutorials — your data and model will tell you the right answer.

**Why does the loss oscillate step-to-step?** If you look at the raw (unsmoothed) loss curve, it won't decrease in a clean line — it zigzags. This is normal, and it correlates directly with noise in your dataset. Each training batch samples a different mix of correctly and incorrectly labeled data. A batch that happens to contain mostly clean, correctly labeled examples gives the model a consistent gradient signal — loss drops. The next batch might contain several mislabeled samples, producing contradictory gradients — loss spikes. With a dataset like DiverseVul (~60% label accuracy for the vulnerable class), these contradictions happen frequently, and the zigzag is pronounced.

Three things control how spiky the curve looks. **Batch size:** smaller batches sample fewer examples per step, so the label noise ratio varies more between batches — more oscillation. **Learning rate:** higher values amplify the effect of noisy gradients, making each spike bigger. **Data quality:** the noisier the labels, the more batches disagree with each other on what the model should learn. Increasing batch size smooths the curve cosmetically, but doesn't fix the underlying problem — the model is still receiving contradictory supervision from mislabeled data.

The validation loss plateau is the real signal here. When it flatlines while training loss keeps dropping, the model has learned everything the clean labels can teach. Further training just memorizes the noise — which is why the growing gap between training and validation loss is the clearest sign to stop.

<svg width="100%" viewBox="0 0 680 290" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="290" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="26" text-anchor="middle" font-size="13" font-weight="500" fill="#3d3d3a">Reading loss curves during training</text>
  <!-- Panel 1: Good fit -->
  <rect x="24" y="44" width="200" height="222" rx="8" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="124" y="66" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Good fit</text>
  <!-- Axes -->
  <line x1="48" y1="80" x2="48" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <line x1="48" y1="240" x2="208" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <text x="32" y="164" text-anchor="middle" font-size="9" fill="#b4b2a9" transform="rotate(-90 32 164)">loss</text>
  <text x="128" y="254" text-anchor="middle" font-size="9" fill="#b4b2a9">epochs</text>
  <!-- Training line -->
  <polyline points="52,94 68,110 84,118 100,122 116,124 132,125 148,126 164,127 180,128 196,128 204,129" fill="none" stroke="#378ADD" stroke-width="1.8" stroke-linecap="round"/>
  <!-- Validation line -->
  <polyline points="52,100 68,116 84,126 100,132 116,136 132,138 148,140 164,141 180,142 196,142 204,143" fill="none" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3" stroke-linecap="round"/>
  <!-- Legend -->
  <line x1="60" y1="190" x2="80" y2="190" stroke="#378ADD" stroke-width="1.8"/>
  <text x="86" y="194" font-size="10" fill="#5F5E5A">training</text>
  <line x1="60" y1="208" x2="80" y2="208" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3"/>
  <text x="86" y="212" font-size="10" fill="#5F5E5A">validation</text>
  <text x="124" y="232" text-anchor="middle" font-size="10" fill="#0F6E56">Both decreasing ✦ still learning</text>
  <!-- Panel 2: Overfitting -->
  <rect x="240" y="44" width="200" height="222" rx="8" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="340" y="66" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Overfitting</text>
  <line x1="264" y1="80" x2="264" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <line x1="264" y1="240" x2="424" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <text x="248" y="164" text-anchor="middle" font-size="9" fill="#b4b2a9" transform="rotate(-90 248 164)">loss</text>
  <text x="344" y="254" text-anchor="middle" font-size="9" fill="#b4b2a9">epochs</text>
  <!-- Training line -->
  <polyline points="268,96 284,116 300,128 316,134 332,136 348,137 364,137 380,137 396,136 412,136 420,135" fill="none" stroke="#378ADD" stroke-width="1.8" stroke-linecap="round"/>
  <!-- Validation line (down then back up = overfitting) -->
  <polyline points="268,102 284,118 300,126 316,130 332,132 348,130 364,124 380,116 396,106 412,98 420,92" fill="none" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3" stroke-linecap="round"/>
  <line x1="276" y1="190" x2="296" y2="190" stroke="#378ADD" stroke-width="1.8"/>
  <text x="302" y="194" font-size="10" fill="#5F5E5A">training</text>
  <line x1="276" y1="208" x2="296" y2="208" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3"/>
  <text x="302" y="212" font-size="10" fill="#5F5E5A">validation</text>
  <text x="340" y="232" text-anchor="middle" font-size="10" fill="#A32D2D">Validation rising ✦ stop now</text>
  <!-- Panel 3: Stuck -->
  <rect x="456" y="44" width="200" height="222" rx="8" fill="#fff" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="556" y="66" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Not learning</text>
  <line x1="480" y1="80" x2="480" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <line x1="480" y1="240" x2="640" y2="240" stroke="#d4d3cf" stroke-width="0.5"/>
  <text x="464" y="164" text-anchor="middle" font-size="9" fill="#b4b2a9" transform="rotate(-90 464 164)">loss</text>
  <text x="560" y="254" text-anchor="middle" font-size="9" fill="#b4b2a9">epochs</text>
  <!-- Training line (flat) -->
  <polyline points="484,144 496,142 512,142 528,141 544,141 560,140 576,140 592,140 608,139 624,139 632,139" fill="none" stroke="#378ADD" stroke-width="1.8" stroke-linecap="round"/>
  <!-- Validation line (flat) -->
  <polyline points="484,150 496,148 512,147 528,147 544,146 560,146 576,146 592,146 608,145 624,145 632,145" fill="none" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3" stroke-linecap="round"/>
  <line x1="492" y1="190" x2="512" y2="190" stroke="#378ADD" stroke-width="1.8"/>
  <text x="518" y="194" font-size="10" fill="#5F5E5A">training</text>
  <line x1="492" y1="208" x2="512" y2="208" stroke="#E24B4A" stroke-width="1.8" stroke-dasharray="5 3"/>
  <text x="518" y="212" font-size="10" fill="#5F5E5A">validation</text>
  <text x="556" y="232" text-anchor="middle" font-size="10" fill="#BA7517">Both flat ✦ check learning rate</text>
</svg>

---

## What are epochs?

One epoch = the model sees every training sample once. Multiple epochs mean the model sees the same data repeatedly — each pass reinforces what it learned and helps it pick up patterns it missed the first time.

Whether you need multiple epochs depends on the dataset. A small, clean dataset might benefit from 5–10 epochs. A large or noisy dataset — one pass is often enough.

---

## What formats does a fine-tuned model produce?

**LoRA adapter (~80–160MB)** — just the trained adapter weights. The size depends on save precision: ~84MB at 16-bit, ~168MB at 32-bit. To use this, you load the base model and attach the adapter on top. You can swap adapters at runtime — train one for vulnerability detection, another for code review, another for documentation. Same base model, different skills. One 8GB base + three small adapters is much cheaper than three separate full models.

```python
model = load("google/gemma-4-E4B-it")
model.load_adapter("my-vuln-detector-lora")
```

**Merged model (~8GB)** — base model + adapter baked together into one set of files. You need this as a clean starting point for converting to other formats. Why save in 16-bit when you loaded in 4-bit? Because the 4-bit was a temporary memory trick for training. The original model exists in 16-bit on HuggingFace — the merge retrieves those original full-precision weights and combines them with your 16-bit adapter. You're not upscaling 4-bit back to 16-bit; you're going back to the source and folding in what the adapter learned.

**GGUF (~2.5GB quantized)** — a single-file format created by the llama.cpp project, used by Ollama, LM Studio, and llama.cpp for running models locally without Python or PyTorch.

---

## Can you keep the adapter separate or must you merge?

For Python/HuggingFace use: keep them separate. You get adapter swapping, smaller files, and flexibility. Only merge when the next step requires it — specifically GGUF conversion, which needs a complete model.

Think of it as two ecosystems:

| | SafeTensors (HuggingFace) | GGUF (llama.cpp) |
|--|--------------------------|-------------------|
| Swap LoRA adapters at runtime | Yes | No — baked in |
| Run in Ollama / LM Studio | No | Yes |
| Run without Python | No | Yes |
| Multiple skills, one base model | Yes | Need separate GGUF per skill |

---

## Practical tip: save to Google Drive

If you're training on Google Colab, mount Drive at the start and write outputs there. Colab sessions die without warning — free tier disconnects after 30–90 minutes of inactivity, and even paid tiers have session limits. I lost a full training run before learning this.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

*This is the concepts reference for "The Security Engineer's Practical Guide to LLMs." Read the experiment: **[I Fine-Tuned Gemma 4 to Detect Code Vulnerabilities — Here's What Happened](/posts/fine-tuned-gemma4-code-vulnerabilities/)**.*