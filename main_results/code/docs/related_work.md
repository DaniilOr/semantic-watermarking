# Related work found by web search (18 September 2026)

A handful of targeted searches, not a systematic novelty review. Grouped by how each line of work relates to this method.

**Same diagnosis for code and other low-entropy text.** "Who Wrote this Code? Watermarking for Code Generation"
(SWEET) applies a token-level watermark only at positions whose entropy exceeds a threshold, and detects only on those
positions; later entropy-aware code watermarks modulate strength by local entropy, and "Invisible Entropy" and
"Watermarking Low-entropy Generation" target the same regime. This is the uncertainty gate of this project, arrived
at independently, applied to a vocabulary-partition watermark instead of a likelihood tilt. Their remedy of also
restricting *detection* to high-entropy positions is not implemented here and is the obvious next step for code.

**Secrets placed in the prompt.** "Watermarking Language Models through Language Models" uses one model to write a
watermarking instruction, a second to generate under it, and a third, trained, to detect. "In-Context Watermarks"
embeds rules (initial letters, acrostics, lexical choices, invisible characters) purely by instruction and detects with
rule-specific tests. Both steer through the prompt. This project keeps the generating prompt clean, steers at the
sampling step using the same model conditioned on a secret phrase as a reference distribution, and detects with a
likelihood contrast that needs no trained detector. The spec's own baseline of putting the secret in the system prompt
was weak (development: 5–10% detection).

**Paraphrase-robust semantic watermarks.** SemStamp, k-SemStamp, SimMark and SAMark work at sentence level: candidate
sentences are embedded with a paraphrase-robust encoder and accepted only if they fall in key-determined regions.
They are built for rewriting attacks and need an external encoder; here robustness is a by-product (47–54% of
detections survive a strong same-model paraphrase).

**Detection versus identification with many keys.** "Multi-use LLM Watermarking and the False Detection Problem" shows
that using one signal both to detect and to identify the user inflates false detections as users grow, and proposes two
separate watermarks. The same split emerged here empirically: the within-text statistic answers presence, the
same-construction decoy rank test answers which key. Multi-bit and personalised schemes (SAEMark, StealthInk, WASA,
provably robust multi-bit watermarks) address attribution by encoding identifiers rather than by ranking keys.

**Detection statistics.** "Improving Detection of Watermarked Language Models" and "Predictive Likelihood Ratios for
Language Model Watermark Detection" study likelihood-ratio tests for key-based watermarks; the entropy-based detection
method (EWD) weights tokens by entropy at detection time.

**Different goals.** The original green-list watermark is the standard token-level baseline (no model needed at
detection, not run here by design). "Undetectable Watermarks for Language Models" gives cryptographic schemes whose
presence cannot be noticed without the key; this project's presence setting is the opposite, since any background
phrase reveals that a text was steered. "LLM Fingerprinting via Semantically Conditioned Watermarks" conditions a
trained-in watermark on a prompt domain to prove model ownership.

**Not found:** a published scheme that uses the generating model conditioned on a secret phrase as the reference
distribution, tilts sampling by the log-ratio to a neutral or decoy context, and detects by the within-text contrast.
The nearest technical relatives are contrastive decoding and guidance-style sampling, which are not watermarks.
