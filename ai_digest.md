# AI Trust & Safety Digest

A curated current snapshot of AI trust, safety, and governance, including as many materially important fresh items as the day warrants.

Today’s signal is concentrated in evaluation reliability, agent control, privacy, and training-time robustness rather than broad policy movement. The strongest fresh items propose new ways to audit multimodal inconsistency, distinguish misalignment from mere failure, harden agents outside their own runtime, and defend models against poisoning and inference-time privacy leakage.

## Worth Opening

- [Auditing order sensitivity in multimodal LLMs](https://arxiv.org/abs/2606.26079v1) - Shows a basic reliability failure mode that emerging evaluation standards increasingly expect developers to measure. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [Model Forensics investigates whether concerning behavior actually reflects misalignment](https://arxiv.org/abs/2606.26071v1) - Separating misalignment from confusion could improve incident diagnosis, evaluations, and escalation decisions. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [The Unfireable Safety Kernel proposes execution-time alignment outside the agent runtime](https://arxiv.org/abs/2606.26057v1) - Externalizing controls targets a core weakness of in-agent guardrails that prompts or tools may bypass. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [Defending text summarization models against data poisoning with detect-unlearn-restore](https://arxiv.org/abs/2606.26036v1) - Addresses a practical training-time attack path for fine-tuned models deployed on small datasets. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [Privacy vulnerabilities of attention layers in tabular foundation models](https://arxiv.org/abs/2606.26021v1) - Highlights inference-time leakage risks when sensitive records are supplied as in-context examples. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [Why multi-step tool-use reinforcement learning collapses and how supervisory signals fix it](https://arxiv.org/abs/2606.26027v1) - Relevant to safer agent training because unstable RL can degrade behavior in long-horizon tool use. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [How robust is OCR reasoning under visual perturbations?](https://arxiv.org/abs/2606.26041v1) - Adds evidence on multimodal robustness where small visual degradations can trigger downstream reasoning errors. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [Helping build shared standards for advanced AI](https://openai.com/index/helping-build-shared-standards-for-advanced-ai) - Standards work can shape common expectations for evaluations, safeguards, and cross-border governance. Source: OpenAI News. Published: 2026-06-23T13:00:00Z.
- [The Tatoxa system for text detoxification in Tatar](https://arxiv.org/abs/2606.26015v1) - Extends harmful-content mitigation research to a low-resource language often missed by safety tooling. Source: arXiv. Published: 2026-06-25T13:08:37Z.
- [RevengeBench studies reverse engineering of code-space policies from behavior](https://arxiv.org/abs/2606.26094v1) - Could inform audits of whether hidden policies or internal rules can be inferred from observed behavior. Source: arXiv. Published: 2026-06-25T13:08:37Z.

Selected from recent trust-and-safety relevant posts and papers by labs, evaluators, standards bodies, policy organizations, and arXiv, then summarized with GPT-5.4. Includes as many materially important fresh items as the day warrants.

Generated at: 2026-06-25T13:09:56Z
