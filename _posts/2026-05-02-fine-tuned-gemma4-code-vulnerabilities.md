---
title: "I Fine-Tuned Gemma 4 to Detect Code Vulnerabilities — Here's What Happened"
date: 2026-05-XX
permalink: /posts/fine-tuned-gemma4-code-vulnerabilities/
tags:
  - LLM
  - fine-tuning
  - Gemma 4
  - code security
  - Unsloth
---

*One GPU, one epoch, three evaluation surprises, and recall that jumped from 4% to 51%. If you want the concepts behind the decisions (LoRA, QLoRA, NF4, batch size, loss curves), read the companion reference: **[Every Concept You Need Before Fine-Tuning an LLM](/posts/every-concept-before-fine-tuning-llm/)**.*

*I work with LLMs daily through APIs and orchestration pipelines. But there's a difference between using models and understanding what happens inside them. I wanted to get hands-on with the training process itself — so I picked a domain I know well (code security), grabbed a public dataset, and fine-tuned Google's Gemma 4 E4B on a Colab A100 over a weekend. Code vulnerability detection is the vehicle here, not the destination — every technique applies to any domain. That said, there's a practical angle: a fine-tuned local model can analyze code without sending it to a cloud API. For teams working on proprietary codebases, air-gapped environments, or regulated industries where code cannot leave the network, a local model — even a modest one — fills a niche that commercial cloud scanners can't.*

---

## The setup

**Model:** Google's Gemma 4 E4B — a dense model with 8 billion total parameters and ~4.5 billion effective parameters during inference. The "E" stands for "Effective" — the model uses **Per-Layer Embeddings (PLE)**, where large embedding lookup tables add to the total parameter count but aren't used in the forward computation, so the effective compute footprint is much smaller than the total ([source: Google model card](https://huggingface.co/google/gemma-4-E4B-it)). Instruction-tuned and multimodal (text, vision, audio).

**Dataset:** [DiverseVul](https://huggingface.co/datasets/bstee615/diversevul) — ~330,000 C/C++ functions labeled as vulnerable or safe, spanning 150 CWE categories.

**Tool:** [Unsloth](https://unsloth.ai) — handles QLoRA loading, optimized training, and GGUF export.

**Hardware:** Google Colab with an A100 GPU (40GB VRAM). I initially tried a free-tier T4 (16GB) but hit out-of-memory errors during training even with QLoRA and batch size of 1. The A100's 40GB gives comfortable headroom for QLoRA fine-tuning and supports bf16 precision (more numerically stable than the T4's fp16).

<svg width="100%" viewBox="0 0 680 130" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="130" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="24" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Experiment pipeline</text>
  <rect x="20" y="44" width="110" height="52" rx="8" fill="#E6F1FB" stroke="#85B7EB" stroke-width="0.5"/>
  <text x="75" y="66" text-anchor="middle" font-size="11" font-weight="500" fill="#0C447C">DiverseVul</text>
  <text x="75" y="82" text-anchor="middle" font-size="10" fill="#378ADD">330k functions</text>
  <text x="147" y="72" text-anchor="middle" font-size="14" fill="#888">→</text>
  <rect x="163" y="44" width="100" height="52" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="213" y="66" text-anchor="middle" font-size="11" font-weight="500" fill="#633806">Balance</text>
  <text x="213" y="82" text-anchor="middle" font-size="10" fill="#BA7517">3k + 3k = 6k</text>
  <text x="280" y="72" text-anchor="middle" font-size="14" fill="#888">→</text>
  <rect x="296" y="44" width="100" height="52" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="346" y="66" text-anchor="middle" font-size="11" font-weight="500" fill="#085041">QLoRA</text>
  <text x="346" y="82" text-anchor="middle" font-size="10" fill="#0F6E56">1 epoch, ~2 hrs</text>
  <text x="413" y="72" text-anchor="middle" font-size="14" fill="#888">→</text>
  <rect x="429" y="44" width="100" height="52" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
  <text x="479" y="66" text-anchor="middle" font-size="11" font-weight="500" fill="#3C3489">Evaluate</text>
  <text x="479" y="82" text-anchor="middle" font-size="10" fill="#7F77DD">3 iterations</text>
  <text x="546" y="72" text-anchor="middle" font-size="14" fill="#888">→</text>
  <rect x="562" y="44" width="100" height="52" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="612" y="66" text-anchor="middle" font-size="11" font-weight="500" fill="#085041">Save</text>
  <text x="612" y="82" text-anchor="middle" font-size="10" fill="#0F6E56">LoRA + GGUF</text>
  <text x="340" y="118" text-anchor="middle" font-size="10" fill="#888780">Gemma 4 E4B · A100 GPU · Unsloth · DiverseVul (C/C++)</text>
</svg>

---

## The dataset

DiverseVul is extracted from vulnerability-fixing commits on GitHub — projects like the Linux kernel, OpenSSL, FFmpeg, and ImageMagick. Each function is labeled vulnerable (1) or safe (0). Note: the dataset is C/C++ only — a different profile from the JavaScript/Python/TypeScript vibe-coded apps mentioned above, but the fine-tuning process is identical regardless of language.

Two properties matter:

**It's heavily imbalanced.** ~95% safe, ~5% vulnerable. Training on this raw teaches the model to always say "SAFE" and achieve 95% accuracy while catching nothing. Fix: balanced sampling — I took 3,000 vulnerable and 3,000 safe functions for training, 500 + 500 for validation.

```python
raw = load_dataset("bstee615/diversevul")
vuln = [r for r in raw["train"] if r["target"] == 1 and 30 < len(r["func"]) < 3200]
safe = [r for r in raw["train"] if r["target"] == 0 and 30 < len(r["func"]) < 3200]
train_balanced = random.sample(vuln, 3000) + random.sample(safe, 3000)
```

**The labels are noisy.** The DiverseVul authors themselves report **60% label accuracy** for vulnerable functions, measured by manually verifying a random sample of 50 ([Table 8, DiverseVul paper, RAID 2023](https://surrealyz.github.io/files/pubs/raid23-diversevul.pdf)). The main sources of error: vulnerabilities spread across multiple functions, and non-vulnerable functions changed in the same commit as the fix. This puts a hard ceiling on achievable performance. For a learning experiment, this is acceptable. For production, you'd invest heavily in label quality first.

Each sample is formatted as a Gemma 4 chat conversation for SFT:

```python
text = (
    f"<start_of_turn>system\n{SYSTEM}<end_of_turn>\n"
    f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
    f"<start_of_turn>model\n{reply}<end_of_turn>\n"
)
```

---

## Training

```python
CONFIG = dict(
    model       = "google/gemma-4-E4B-it",
    max_seq_len = 512,
    lora_rank   = 16,
    epochs      = 1,
    batch_size  = 8,
    grad_accum  = 1,           # effective batch = 8
    lr          = 2e-4,
    samples_per_class = 3000,  # 3k vuln + 3k safe = 6k total
)
```

LoRA adapters targeted all attention and MLP layers (`q/k/v/o` projections, `gate/up/down` projections). After loading:

```
GPU: NVIDIA A100-SXM4-40GB
VRAM after model load: ~3.2 / 40.0 GB
Trainable: 42,401,792 / 8,038,558,240 (0.53%)
```

Training completed in approximately 1 hour 45 minutes on the A100 for one epoch.

![Training loss curve](/images/gemma4-vuln/curves.png)

![Unsloth training output](/images/gemma4-vuln-experiment/fine-tune-progress.png)

**Training loss** dropped sharply from ~9.5 to ~1.3 in the first 100 steps. (A starting loss of ~9.5 is higher than typical text models — this is normal for Gemma 4's multimodal architecture with its large vocabulary. The model hasn't seen our task format before, so early predictions are essentially random across the full token space.) It continued declining gradually after that.

**Validation loss** dropped to ~2.3 and plateaued completely. Additional training steps reduced training loss but didn't improve generalization. I had originally configured 3 epochs, but the validation curve made the decision clear: stop at 1 epoch. The model absorbed the clean, obvious patterns quickly. Further training was fitting the noisy labels, not learning new patterns.

![Finetune outputs](/images/gemma4-vuln-experiment/after-fine-tune.jpeg)
---

## Evaluation: three iterations to honest numbers

Evaluating this model correctly turned out to be harder than training it.

<svg width="100%" viewBox="0 0 680 160" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="160" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="24" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Three iterations to honest numbers</text>
  <!-- Iteration 1 -->
  <rect x="30" y="44" width="180" height="72" rx="8" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="120" y="66" text-anchor="middle" font-size="12" font-weight="600" fill="#791F1F">94.5% accuracy</text>
  <text x="120" y="82" text-anchor="middle" font-size="10" fill="#A32D2D">Imbalanced test set</text>
  <text x="120" y="96" text-anchor="middle" font-size="10" fill="#A32D2D">195 safe, 5 vulnerable</text>
  <text x="120" y="110" text-anchor="middle" font-size="9" font-style="italic" fill="#791F1F">Misleading ✗</text>
  <!-- Arrow -->
  <text x="232" y="82" text-anchor="middle" font-size="14" fill="#888">→</text>
  <!-- Iteration 2 -->
  <rect x="250" y="44" width="180" height="72" rx="8" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="340" y="66" text-anchor="middle" font-size="12" font-weight="600" fill="#633806">52.5% / 7% recall</text>
  <text x="340" y="82" text-anchor="middle" font-size="10" fill="#BA7517">Balanced, but prompt echo</text>
  <text x="340" y="96" text-anchor="middle" font-size="10" fill="#BA7517">Model parroting template</text>
  <text x="340" y="110" text-anchor="middle" font-size="9" font-style="italic" fill="#633806">Prompt bug ✗</text>
  <!-- Arrow -->
  <text x="452" y="82" text-anchor="middle" font-size="14" fill="#888">→</text>
  <!-- Iteration 3 -->
  <rect x="470" y="44" width="180" height="72" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="560" y="66" text-anchor="middle" font-size="12" font-weight="600" fill="#085041">61.0% / F1 0.567</text>
  <text x="560" y="82" text-anchor="middle" font-size="10" fill="#0F6E56">Balanced + fixed prompt</text>
  <text x="560" y="96" text-anchor="middle" font-size="10" fill="#0F6E56">51 of 100 vulns caught</text>
  <text x="560" y="110" text-anchor="middle" font-size="9" font-style="italic" fill="#085041">Real result ✓</text>
  <text x="340" y="146" text-anchor="middle" font-size="10" fill="#888780">Each iteration fixed the measurement, not the model — the weights never changed</text>
</svg>

---

### The accuracy trap

First run on 200 random test samples: **94.5% accuracy**. Impressive — until you check the distribution. 195 safe, 5 vulnerable. The raw test set mirrors the original dataset's 95/5 imbalance. The model said "SAFE" almost every time and scored well by default.

**Lesson:** always evaluate on a balanced test set. Accuracy on imbalanced data is meaningless.

---

### The prompt echo

Balanced evaluation (100 vulnerable + 100 safe): **52.5% accuracy, 7% recall**. Something was clearly wrong. I looked at the actual model outputs:

```
CWE: CWE-416
Model said: SAFE and a brief reason.

CWE: CWE-20, CWE-787
Model said: SAFE and a brief reason.
```

The model wasn't analyzing code — it was **echoing the prompt**. The training data used the phrase "Reply with VULNERABLE or SAFE and a brief reason." At inference time, the model encountered this substring and completed the most probable next tokens — which were the rest of the training template. This is a generation artifact: the model had learned the task, but the decoding followed a memorized path instead of producing new analysis.

The fix was simple: change the prompt wording at inference so it couldn't trigger the memorized completion. Same model, same weights, different question:

```python
# Triggered memorized template completion
"Reply with VULNERABLE or SAFE and a brief reason."

# Fixed — new wording, model produces actual analysis
"Is it VULNERABLE or SAFE? Explain your reasoning."
```

The model immediately started producing real analysis:

```
CWE: unknown
Model: This function is VULNERABLE. The function uses fork() to
execute a command in a child process...

CWE: CWE-190
Model: VULNERABLE. The function TIFFReadRawStrip1 is vulnerable
to a buffer overflow when reading raw data from a TIFF file...
```

**Lesson:** fine-tuning teaches a conversational pattern, not just a task. The inference prompt must align with — but not exactly match — the training format. If the prompt contains a substring from training targets, the model may complete the template rather than reason about the input.

---

### The real numbers

Balanced evaluation, 200 samples (100 vulnerable + 100 safe), corrected prompt, with `random.seed(42)` for reproducibility. Both the fine-tuned and zero-shot models were evaluated with the identical prompt and the same 200 samples for a fair comparison:

|  | Fine-tuned | Zero-shot (no training) | Delta |
|--|-----------|------------------------|-------|
| Accuracy | 61.0% | 45.5% | +15.5% |
| Precision | 63.7% | 23.5% | +40.2% |
| Recall | 51.0% | 4.0% | +47.0% |
| F1 | 0.567 | 0.068 | +0.499 |

The base Gemma 4 E4B caught 4 out of 100 vulnerabilities zero-shot — essentially guessing. The fine-tuned version caught 51, bringing recall from near-zero to about half. Not perfect, but a clear signal that the fine-tuning worked, especially given the noisy labels in the training data.

<svg width="100%" viewBox="0 0 680 170" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="170" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="24" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Recall comparison — how many vulnerabilities were caught out of 100</text>
  <!-- Zero-shot -->
  <text x="145" y="60" text-anchor="end" font-size="11" fill="#5F5E5A">Zero-shot (base)</text>
  <rect x="155" y="46" width="20" height="22" rx="3" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="185" y="62" font-size="11" font-weight="500" fill="#791F1F">4 / 100</text>
  <!-- Fine-tuned -->
  <text x="145" y="100" text-anchor="end" font-size="11" fill="#5F5E5A">Fine-tuned (ours)</text>
  <rect x="155" y="86" width="255" height="22" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="420" y="102" font-size="11" font-weight="500" fill="#085041">51 / 100</text>
  <!-- Scale -->
  <line x1="155" y1="122" x2="655" y2="122" stroke="#e5e4e0" stroke-width="0.5"/>
  <text x="155" y="140" font-size="9" fill="#888780">0</text>
  <text x="405" y="140" text-anchor="middle" font-size="9" fill="#888780">50</text>
  <text x="655" y="140" text-anchor="end" font-size="9" fill="#888780">100</text>
  <text x="340" y="160" text-anchor="middle" font-size="10" fill="#888780">Same 200 test samples · same prompt · random.seed(42)</text>
</svg>

---

## What did fine-tuning actually change?

Here's what's counterintuitive: we didn't teach Gemma 4 about vulnerabilities. It already knew. The model was pre-trained on code, security advisories, CWE descriptions, and countless discussions about buffer overflows and injection attacks. The zero-shot baseline proved this — it sometimes gave detailed, correct explanations of why code was dangerous.

But it only caught 4 out of 100 vulnerabilities in our eval. Why?

Because our eval looked for the word "VULNERABLE" in the response. The base model would say things like "this code has potential security implications that warrant further review" — technically correct analysis, but our parser reads that as SAFE because it doesn't contain the keyword. A smarter parser that also caught phrases like "security flaw" or "dangerous" would have narrowed the gap — but the inconsistency and lack of structured verdicts would remain. The model knew the answer but expressed it in a way our system couldn't reliably use.

Fine-tuning was essentially **response format alignment** — teaching the model to package what it already knew into the structured output we needed:

1. **Lead with a verdict** — always say VULNERABLE or SAFE first, not a hedged paragraph
2. **Be consistent** — same format every time, not sometimes three paragraphs and sometimes one word
3. **Commit to a decision** — no "this could potentially be problematic" — yes or no

Think of it as a senior security consultant who knows everything about vulnerabilities but has never used your team's reporting template. They can write a brilliant analysis, but they can't fill in the "Severity: HIGH/MEDIUM/LOW" field consistently. Fine-tuning taught the consultant to use the template.

This is an important insight for anyone considering fine-tuning: if the base model already understands your domain, you may not need thousands of examples to teach it new knowledge. You need enough examples to teach it your expected response structure. In our case, one epoch was sufficient — the model learned the format fast, because the underlying knowledge was already there.

---

## What it catches and what it misses

Running the fine-tuned model against 200 vulnerable test samples grouped by CWE reveals a clear pattern. A caveat: sample sizes per CWE are small (some have only 4 samples), so these recall numbers are indicative of trends, not statistically robust benchmarks.

**Strong performers (>60% recall):**

| CWE | Description | Caught | Total | Recall |
|-----|-------------|--------|-------|--------|
| CWE-310 | Cryptographic issues | 3 | 4 | 75.0% |
| CWE-20 | Input validation | 12 | 17 | 70.6% |
| CWE-200 | Information exposure | 4 | 6 | 66.7% |
| CWE-787 | Out-of-bounds write | 16 | 25 | 64.0% |

**Weak spots (<35% recall):**

| CWE | Description | Caught | Total | Recall |
|-----|-------------|--------|-------|--------|
| CWE-415 | Double free | 0 | 4 | 0.0% |
| CWE-401 | Memory leak | 1 | 4 | 25.0% |
| CWE-399 | Resource management | 1 | 4 | 25.0% |
| CWE-416 | Use after free | 4 | 12 | 33.3% |

<svg width="100%" viewBox="0 0 680 310" xmlns="http://www.w3.org/2000/svg" style="max-width:680px;margin:1.5em auto;display:block">
  <style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>
  <rect x="0" y="0" width="680" height="310" rx="12" fill="#f9f9f7" stroke="#e5e4e0" stroke-width="1"/>
  <text x="340" y="24" text-anchor="middle" font-size="12" font-weight="500" fill="#3d3d3a">Recall by CWE — what the model catches vs misses</text>
  <!-- CWE-310 75% -->
  <text x="155" y="52" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-310 Crypto</text>
  <rect x="165" y="40" width="375" height="18" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="548" y="54" font-size="10" fill="#085041">75%</text>
  <!-- CWE-20 70.6% -->
  <text x="155" y="78" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-20 Input val.</text>
  <rect x="165" y="66" width="353" height="18" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="526" y="80" font-size="10" fill="#085041">70.6%</text>
  <!-- CWE-200 66.7% -->
  <text x="155" y="104" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-200 Info exp.</text>
  <rect x="165" y="92" width="333" height="18" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="506" y="106" font-size="10" fill="#085041">66.7%</text>
  <!-- CWE-787 64% -->
  <text x="155" y="130" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-787 OOB write</text>
  <rect x="165" y="118" width="320" height="18" rx="3" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="493" y="132" font-size="10" fill="#085041">64%</text>
  <!-- Divider -->
  <line x1="30" y1="150" x2="650" y2="150" stroke="#e5e4e0" stroke-width="0.5" stroke-dasharray="4 3"/>
  <!-- CWE-476 55.6% -->
  <text x="155" y="174" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-476 NULL deref</text>
  <rect x="165" y="162" width="278" height="18" rx="3" fill="#FAEEDA" stroke="#BA7517" stroke-width="0.5"/>
  <text x="451" y="176" font-size="10" fill="#633806">55.6%</text>
  <!-- CWE-416 33.3% -->
  <text x="155" y="200" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-416 Use after free</text>
  <rect x="165" y="188" width="167" height="18" rx="3" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="340" y="202" font-size="10" fill="#791F1F">33.3%</text>
  <!-- CWE-401 25% -->
  <text x="155" y="226" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-401 Mem leak</text>
  <rect x="165" y="214" width="125" height="18" rx="3" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="298" y="228" font-size="10" fill="#791F1F">25%</text>
  <!-- CWE-415 0% -->
  <text x="155" y="252" text-anchor="end" font-size="10" fill="#5F5E5A">CWE-415 Double free</text>
  <rect x="165" y="240" width="3" height="18" rx="1" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="178" y="254" font-size="10" fill="#791F1F">0%</text>
  <!-- Legend -->
  <rect x="165" y="278" width="12" height="12" rx="2" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
  <text x="183" y="289" font-size="9" fill="#5F5E5A">Pattern-based (localized signatures)</text>
  <rect x="380" y="278" width="12" height="12" rx="2" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
  <text x="398" y="289" font-size="9" fill="#5F5E5A">State-tracking (execution flow)</text>
</svg>

The model catches vulnerabilities with obvious, localized code signatures — unchecked inputs, buffer writes without bounds checking, weak crypto usage. These are patterns where a single line or function call is the red flag.

Where it struggles is with **state-tracking bugs** — double frees, use-after-free, memory leaks. These vulnerabilities require understanding execution flow across multiple lines: memory was allocated here, freed there, and then accessed again somewhere else. A model looking at a single function in isolation has limited ability to track that kind of stateful reasoning.

Fine-tuning taught the model to recognize vulnerability *signatures*, not to perform deep program analysis. True flow-sensitive analysis would likely require either a much larger model, a multi-file context approach, or combining the LLM with static analysis tools — for example, using Semgrep or CodeQL to identify candidate functions, then the LLM to classify and explain. That hybrid approach is worth exploring in a future post.

---

## Key takeaways

**Watch the validation loss, not the training loss.** Training loss always keeps dropping — that's memorization. Validation loss tells you when to stop. Mine plateaued halfway through epoch 1.

**Evaluation is harder than training.** My reported accuracy changed from 94.5% to 52.5% to 61% across three iterations. Each time, the problem was measurement, not the model.

**Prompt alignment matters more than you'd expect.** The model learned fine — but the inference prompt triggered a memorized template completion instead of actual analysis. Changing the prompt wording fixed it instantly, with no retraining.

**Data quality is the ceiling.** With ~60% label accuracy ([DiverseVul, RAID 2023](https://surrealyz.github.io/files/pubs/raid23-diversevul.pdf)), no training configuration will produce great results. For production, invest in labels first. For learning, noisy data teaches you the process just as well.

**Practical note:** if you're training on Google Colab, save to Google Drive early and often. I lost a full training run when the session disconnected. Mount Drive at the start and set your output directory there.

---

## The outputs
![Outputs](/images/gemma4-vuln-experiment/other-folders.jpeg)
The fine-tuned model is saved in three formats: a **LoRA adapter** (~160MB), a **merged 16-bit SafeTensors** model (~8GB), and a **GGUF Q4_K_M** file (~2.5GB). The evaluation in this post was done on the SafeTensors LoRA checkpoint. The GGUF version hasn't been evaluated yet — that's the focus of the next post.

---

## What's next

In the next post, I'll take the GGUF file and benchmark different quantization levels — Q4 vs Q5 vs Q8 — measuring what you lose when you shrink a model from 8GB to 2.5GB. Does Q4 still catch the buffer overflows that Q8 catches? Where exactly is the quality cliff?

The code for the full experiment: [https://github.com/Geo-Joy/llm-vuln-detector](https://github.com/Geo-Joy/llm-vuln-detector)

---

*This is Part 1 of "The Security Engineer's Practical Guide to LLMs." Concepts reference: **[Every Concept You Need Before Fine-Tuning an LLM](/posts/every-concept-before-fine-tuning-llm/)**. Next: What you lose when you shrink a model 4x.*