# ellamind-base-eval

Curated [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task configurations for English and German base model evaluation, with a focus on providing a comprehensive, validated German benchmark suite. Prompt formatting closely follows [OLMES](https://arxiv.org/abs/2406.08446).

Every task is verified by running it against multiple reference models to check that scores are plausible and that metrics correlate with model quality. Easy suite tasks are additionally validated during early-stage pretraining to ensure they produce useful signal from the start.

## Setup

Install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and point `lm_eval` at this repository via `--include_path`:

```bash
lm_eval --tasks deu_base_main --include_path /path/to/ellamind/base-eval --model ...
```

## Suites

We provide three tiers of evaluation suites per language:

| Suite | Language | Purpose |
|-------|----------|---------|
| `eng_base_easy` | English | Small-scale experiments and early-stage pretraining |
| `eng_base_main` | English | In-loop evaluation |
| `eng_base_full` | English | All English tasks |
| `deu_base_easy` | German | Small-scale experiments and early-stage pretraining |
| `deu_base_main` | German | In-loop evaluation |
| `deu_base_full` | German | All German tasks |

The English suites are based on the OLMo 3 evaluation suites[^1].

[^1]: OLMo 3 Tech Report, Section A.4.1, p. 95 — https://arxiv.org/abs/2512.13961

### Suite Overview

**Easy** suites use cheap, likelihood-based evaluation formats (rank classification and bits-per-byte) suitable for tracking progress during early-stage pretraining. For small-scale experiments and ablations.

**Main** suites use generation-based formats (multiple choice, free-form generation, chain-of-thought, code execution) for more meaningful evaluation at checkpoints.

**Full** suites include every task defined in this repository.

#### English Main (`eng_base_main`)

| Group | # Benchmarks | Benchmarks |
|-------|--------------|------------|
| stem_qa_mc | 5 | ARC, MMLU STEM, MedMCQA, MedQA, SciQ |
| nonstem_qa_mc | 9 | MMLU Humanities/Social/Other, CSQA, PiQA, SocialIQA, CoQA, DROP, Jeopardy, NaturalQs, SQuAD |
| agieval_mc | 5 | AQUA-RAT, GaoKao, LogiQA, LSAT, SAT |
| gen | 9 | HellaSwag, WinoGrande, Lambada, Basic Skills, DROP, Jeopardy, NaturalQs, SQuAD, CoQA |
| math | 4 | GSM8K, GSM-Symbolic, Minerva Math, MATH-500 |
| code | 2 | HumanEval, MBPP |
| heldout | 5 | LBPP, BBH (27), MMLU Pro (14), DeepMind Math (56), GPQA |

#### German Main (`deu_base_main`)

| Group | # Benchmarks | Benchmarks |
|-------|--------------|------------|
| deu_stem_qa_mc | 2 | ARC, MMMLU filtered STEM (19 subjects) |
| deu_nonstem_qa_mc | 4 | MMMLU filtered non-STEM (13 subjects), CSQA, PiQA, SocialIQA |
| deu_agieval_mc | 5 | AQUA-RAT, GaoKao, LogiQA, LSAT, SAT (German) |
| deu_gen | 3 | HellaSwag, WinoGrande, CoQA |
| deu_math | 2 | Minerva Math, GSM8K Platinum |
| deu_code | 2 | HumanEval, MBPP (German docstrings) |
| deu_glp / deu_glp_mc | 1 | German Language Proficiency (22 topics × RC + MC) |
| deu_heldout | 5 | MMLU Pro, GPQA, HLE, INCLUDE, SimpleQA |

For detailed task listings, format variants, and per-benchmark documentation, see [`tasks/README.md`](tasks/README.md).

## Hints & Caveats

- **Context length**: MMLU 5-shot contexts may get truncated at 2048 tokens (the `lm-eval` default). Use `max_length=4096` if the model supports it.
- **Prefix-caching**: Enabling prefix-caching speeds up evaluation significantly due to few-shot examples being cached.
- **Code execution**: `--confirm_run_unsafe_code` and `HF_ALLOW_CODE_EVAL=1` are required for HumanEval, MBPP, LBPP, and DeepSeek LeetCode.

## Repository Structure

```
ellamind-base-eval/
├── suites/              # Evaluation suite definitions
│   ├── eng/             # English group configs
│   └── deu/             # German group configs
├── tasks/               # Task configurations (see tasks/README.md)
│   ├── eng/             # 28 English benchmarks, ~470 task configs
│   └── deu/             # 19 German benchmarks, ~260 task configs
└── README.md
```

## Citation

```bibtex
@software{ellamind_base_eval,
  author       = {ellamind},
  title        = {ellamind-base-eval: English and German Base Model Evaluation Suites},
  year         = {2025},
  url          = {https://github.com/ellamind/base-eval}
}
```

