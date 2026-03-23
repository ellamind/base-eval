# Evaluation Tasks

## Structure

Tasks are organized by language, with a flat benchmark-based structure:

```
tasks/
├── deu/                       # German tasks (11 benchmarks)
│   ├── arc/
│   │   ├── _arc_rc_template.yaml    # Template (underscore prefix)
│   │   ├── _arc_bpb_template.yaml   # BPB template
│   │   ├── arc_challenge_rc.yaml    # Task config (no prefix)
│   │   ├── arc_challenge_bpb.yaml
│   │   └── utils.py                 # Per-benchmark utilities
│   └── ...
└── eng/                       # English tasks (28 benchmarks)
    └── ...
```

Suite definitions live in `suites/` and compose tasks into evaluation groups.

## Naming Conventions

- **`_` prefix**: Templates, group definitions (not standalone tasks)
- **No prefix**: Runnable task configs
- **Language prefix**: `deu_` for German tasks, none for English
- **Suffixes** denote the evaluation format (see Task Variants)

## Task Variants

| Suffix | Format | Output Type | Metrics |
|--------|--------|-------------|---------|
| `_rc` | Reading comprehension (cloze/likelihood) | `multiple_choice` | acc, acc_norm |
| `_mc` | Multiple choice (generation) | `multiple_choice` | acc, acc_norm |
| `_bpb` | Bits-per-byte | `multiple_choice` | bits_per_byte |
| `_gen` | Generative QA | `generate_until` | exact_match, f1 |
| `_cot` | Chain-of-thought | `generate_until` | exact_match |
| `_code` | Code generation | `generate_until` | pass@k |

## Template Pattern

Tasks use YAML template inheritance to reduce duplication:

```yaml
# _template.yaml - shared config (lean: only what's common)
dataset_path: ellamind/arc-german-preview
test_split: test
fewshot_config:
  sampler: first_n

# task.yaml - includes template, adds task-specific config
include: _template.yaml
task: deu_arc_challenge_rc
output_type: multiple_choice
metric_list:
  - metric: acc
  - metric: acc_norm
```

---

## Suites Overview

Suites compose individual tasks into evaluation groups. See `suites/README.md` for full details.

### Parent Suites

| Suite | Language | Type | Groups |
|-------|----------|------|--------|
| `eng_base_main` | English | Generation-based | `stem_qa_mc`, `nonstem_qa_mc`, `agieval_mc`, `gen`, `math`, `code`, `heldout` |
| `eng_base_easy` | English | BPB + RC | `math_bpb`, `code_bpb`, `qa_rc`, `qa_bpb` |
| `deu_base_main` | German | Generation-based | `deu_stem_qa_mc`, `deu_nonstem_qa_mc`, `deu_agieval_mc`, `deu_gen`, `deu_math`, `deu_code`, `deu_glp`, `deu_glp_mc`, `deu_heldout` |
| `deu_base_easy` | German | BPB + RC | `deu_math_bpb`, `deu_code_bpb`, `deu_qa_rc`, `deu_qa_bpb`, `deu_glp` |

---

## All Tasks at a Glance

*Format key: MC = multiple choice (A/B/C/D), RC = rank classification (pick best completion), BPB = bits per byte (log-likelihood), Gen = free-form generation, CoT = chain-of-thought generation.*
*Rows with wildcard task names aggregate all available variants for that benchmark family.*

### English

| Task                                | Formats          | Few-shot | Metric                             | Source                            | Description                                                                      |
|-------------------------------------|------------------|----------|------------------------------------|-----------------------------------|----------------------------------------------------------------------------------|
| `agieval_*` (×9)                    | MC, RC, BPB      |        5 | acc_norm / bpb                     | `ellamind/agieval`                | Academic aptitude tests — AQUA-RAT, GaoKao, LogiQA, LSAT, SAT (9 subsets)       |
| `arc_challenge_*`                   | MC, RC, BPB      |        5 | acc_norm / bpb                     | `allenai/ai2_arc`                 | Grade-school science questions (challenge split)                                 |
| `arc_easy_*`                        | MC, RC, BPB      |        5 | acc_norm / bpb                     | `allenai/ai2_arc`                 | Grade-school science questions (easy split)                                      |
| `basic_skills_*` (×7)               | RC, BPB          |        5 | acc_norm / bpb                     | `ellamind/basic-skills`           | Fundamental capability checks across 7 skill areas                               |
| `bbh_*_cot` (×27)                   | CoT              |        3 | exact_match                        | `lukaemon/bbh`                    | BigBench Hard — 27 challenging reasoning tasks with chain-of-thought             |
| `coqa_*`                            | Gen, MC, RC, BPB |    0 / 5 | exact_match / f1 / acc_norm / bpb  | `EleutherAI/coqa`                 | Conversational question answering                                                |
| `csqa_*`                            | MC, RC, BPB      |        5 | acc_norm / bpb                     | `tau/commonsense_qa`              | 5-way commonsense reasoning                                                      |
| `deepmind_math_*_gen` (×56)         | Gen              |        5 | exact_match                        | `ellamind/deepmind-math-sample`   | Math problems across 56 categories (100 per category, SymPy checking)            |
| `deepseek_leetcode_code`            | Gen              |        0 | pass@1, pass@16 (n=32)             | `davidheineman/deepseek-leetcode` | LeetCode code generation (180 problems, code execution)                          |
| `drop_*`                            | Gen, MC, RC, BPB |        5 | exact_match / f1 / acc_norm / bpb  | `EleutherAI/drop`                 | Discrete reasoning over paragraphs                                               |
| `gpqa_*`                            | MC, CoT          |    0 / 5 | acc_norm / exact_match             | `Idavidrein/gpqa`                 | Graduate-level STEM questions (main: 448, diamond: 198 items)                    |
| `gsm8k_gen`                         | CoT              |        8 | pass@1, pass@4 (n=8)               | `openai/gsm8k`                    | Grade-school math word problems with chain-of-thought sampling                   |
| `gsm_symbolic_*`                    | Gen, CoT         |        8 | pass@1, pass@4 (n=8)               | `apple/GSM-Symbolic`              | Symbolic math word problems (main, p1, p2 subsets)                               |
| `hellaswag_*`                       | RC, BPB          |        5 | acc_norm / bpb                     | `allenai/hellaswag`               | Sentence completion; includes an easy RC prompt variant                          |
| `humaneval_*`                       | Gen, BPB         |        3 | pass@1, pass@16 (n=32) / bpb       | `openai/openai_humaneval`         | Python function generation from docstrings; BPB uses OLMo3-style fenced prompts  |
| `jeopardy_*`                        | Gen, MC, RC, BPB |        5 | exact_match / f1 / acc_norm / bpb  | `soldni/jeopardy`                 | Trivia question answering                                                        |
| `lab_bench_*_bpb` (×2)              | BPB              |        3 | bpb                                | `futurehouse/lab-bench`           | Biology lab QA BPB tasks (`dbqa`, `protocolqa`)                                  |
| `lambada_*`                         | Gen, RC, BPB     |        0 | acc / acc_norm / bpb               | `EleutherAI/lambada_openai`       | Word prediction from broad context                                               |
| `lbpp_code`                         | Gen              |        0 | pass@1, pass@16 (n=32)             | `ellamind/lbpp`                   | Less Basic Python Programming (162 problems, code execution)                     |
| `mbpp_*`                            | Gen, BPB         |        3 | pass@1, pass@16 (n=32) / bpb       | `google-research-datasets/mbpp`   | Python programming problems                                                      |
| `medmcqa_*`                         | MC, RC, BPB      |        5 | acc_norm / bpb                     | `openlifescienceai/medmcqa`       | Medical entrance exam questions                                                  |
| `medqa_*`                           | MC, RC, BPB      |        5 | acc_norm / bpb                     | `davidheineman/medqa-en`          | Medical licensing exam questions                                                 |
| `minerva_math_*` (×7)               | Gen, BPB         |        4 | pass@1, pass@4 (n=4) / bpb         | `EleutherAI/hendrycks_math`       | Competition math benchmark across 7 categories                                   |
| `minerva_math_500_gen`              | Gen              |        4 | pass@1, pass@16 (n=32)             | `HuggingFaceH4/MATH-500`          | MATH-500 subset (500 problems, mirrors OLMES config)                             |
| `mmlu_*` (×57)                      | MC, RC, BPB      |        5 | acc_norm / bpb                     | `cais/mmlu`                       | Massive multitask knowledge benchmark across 57 subjects                         |
| `mmlu_pro_*` (×14)                  | MC, RC           |        5 | acc_norm                           | `TIGER-Lab/MMLU-Pro`              | Harder MMLU variant — up to 10-way questions across 14 subjects                  |
| `mt_mbpp_*_bpb` (×17)               | BPB              |        3 | bpb                                | `allenai/multilingual_mbpp`       | Code-solution perplexity across 17 programming languages                         |
| `naturalqs_*`                       | Gen, MC, RC, BPB |        5 | exact_match / f1 / acc_norm / bpb  | `google-research-datasets/nq_open` | Open-domain question answering                                                  |
| `piqa_*`                            | MC, RC, BPB      |        5 | acc_norm / bpb                     | `ybisk/piqa`                      | Physical commonsense reasoning                                                   |
| `qasper_yesno_*`                    | RC, BPB          |        5 | acc_norm / bpb                     | `allenai/qasper-yesno`            | Scientific paper yes/no question answering                                       |
| `sciq_*`                            | MC, RC, BPB      |        5 | acc_norm / bpb                     | `allenai/sciq`                    | Science exam questions                                                           |
| `sciriff_yesno_*`                   | RC, BPB          |        5 | acc_norm / bpb                     | `allenai/sciriff-yesno`           | Scientific yes/no question answering                                             |
| `socialiqa_*`                       | MC, RC, BPB      |        5 | acc_norm / bpb                     | `allenai/social_i_qa`             | Social intelligence and reaction prediction                                      |
| `squad_*`                           | Gen, MC, RC, BPB |        5 | exact_match / f1 / acc_norm / bpb  | `allenai/squad`                   | Reading comprehension                                                            |
| `winogrande_*`                      | MC, RC, BPB      |        5 | acc_norm / bpb                     | `allenai/winogrande`              | Pronoun coreference resolution; includes an easy RC prompt variant               |

### German

| Task                                | Formats          | Few-shot | Metric                       | Source                                        | Description                                                                      |
|-------------------------------------|------------------|----------|------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------|
| `deu_agieval_*` (×9)               | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/agieval-multilingual`               | Academic aptitude tests — 9 subsets translated to German                         |
| `deu_arc_challenge_*`               | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/arc-multilingual`                   | Grade-school science questions (challenge split, 1172 items)                     |
| `deu_arc_easy_*`                    | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/arc-multilingual`                   | Grade-school science questions (easy split, 2376 items)                          |
| `deu_coqa_*`                        | Gen, BPB         |    0 / 5 | exact_match / f1 / bpb       | `ellamind/coqa-multilingual`                  | Conversational question answering                                                |
| `deu_csqa_*`                        | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/csqa-multilingual`                  | 5-way commonsense reasoning from ConceptNet                                      |
| `deu_glp_*` (×22)                   | MC, RC           |    5 / 0 | acc_norm                     | `ellamind/german-language-proficiency`        | German Language Proficiency — minimal-pair judgments across 22 phenomena         |
| `deu_gpqa_*`                        | MC, CoT          |    0 / 5 | acc_norm / exact_match       | `ellamind/gpqa-multilingual`                  | Graduate-level STEM questions (main: 448, diamond: 198 items)                    |
| `deu_gsm8k_platinum_cot`            | CoT              |        8 | exact_match                  | `ellamind/gsm8k-platinum-multilingual`        | Grade-school math word problems (platinum-verified subset)                       |
| `deu_hellaswag_*`                   | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/hellaswag-multilingual`             | Sentence completion — pick the most plausible continuation                       |
| `deu_hle_*`                         | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/hle-multilingual`                   | Expert-level questions across many domains (626 items)                           |
| `deu_humaneval_*`                   | Gen, BPB         |        3 | pass@1 / bpb                 | `ellamind/humaneval-multilingual`             | Python function generation from German docstrings; BPB uses OLMo3-style fenced prompts |
| `deu_include_*` (×3)                | MC, RC, BPB      |        5 | acc_norm / bpb               | `CohereForAI/include-base-44`                 | Germany-specific regional knowledge (STEM, social science, driving license)      |
| `deu_mbpp_*`                        | Gen, BPB         |        3 | pass@1 / bpb                 | `ellamind/mbpp-multilingual`                  | Python function generation from German descriptions                              |
| `deu_minerva_math_*` (×7)           | Gen, BPB         |        4 | pass@1 / bpb                 | `ellamind/hendrycks-math-multilingual`        | MATH benchmark — 7 categories, gen sampled 4× for pass@k                         |
| `deu_mmlu_pro_*` (×14)              | MC, RC, CoT      |        5 | acc_norm / exact_match       | `li-lab/MMLU-ProX`                            | Harder MMLU variant — up to 10-way questions (14 subjects)                       |
| `deu_mmmlu_*` (×57)                 | MC, RC, BPB      |        5 | acc_norm / bpb               | `openai/MMMLU`                                | Multilingual MMLU — 57 subjects across STEM, humanities, social sciences, other  |
| `deu_piqa_*`                        | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/piqa-multilingual`                  | Physical intuition — pick the plausible action                                   |
| `deu_siqa_*`                        | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/siqa-multilingual`                  | Social intelligence — predict reactions and emotions                             |
| `deu_simpleqa_*`                    | Gen, MC, RC, BPB |        5 | exact_match / acc_norm / bpb | `ellamind/simpleqa-verified-multilingual`     | Short-answer factual questions (verified subset, 979 items)                      |
| `deu_winogrande_*`                  | MC, RC, BPB      |        5 | acc_norm / bpb               | `ellamind/winogrande-multilingual`            | Pronoun coreference resolution                                                   |

---

## German Tasks (deu)

13 benchmarks, ~261 task configs.

### deu_base_main

| Task | Dataset | Type | n-shot | Metrics |
|------|---------|------|--------|---------|
| `deu_arc_challenge_mc` | `ellamind/arc-german-preview` | MC | 5 | acc, acc_norm |
| `deu_arc_easy_mc` | `ellamind/arc-german-preview` | MC | 5 | acc, acc_norm |
| `deu_csqa_mc` | `ellamind/csqa-multilingual` (deu) | MC | 5 | acc, acc_norm |
| `deu_hellaswag_mc` | `ellamind/hellaswag-multilingual` (deu) | MC | 0 | acc, acc_norm |
| `deu_piqa_mc` | `ellamind/piqa-multilingual` (deu) | MC | 0 | acc, acc_norm |
| `deu_siqa_mc` | `ellamind/siqa-multilingual` (deu) | MC | 0 | acc, acc_norm |
| `deu_winogrande_mc` | `ellamind/winogrande-multilingual` (deu) | MC | 0 | acc, acc_norm |
| `deu_mmmlu_filtered_stem` (19 subj.) | `openai/MMMLU` (DE_DE) | MC | 0 | acc, acc_norm |
| `deu_mmmlu_filtered_nonstem` (13 subj.) | `openai/MMMLU` (DE_DE) | MC | 0 | acc, acc_norm |
| `deu_minerva_math_gen` (7 cat.) | `ellamind/hendrycks-math-multilingual` (deu) | Gen | 4 | exact_match, math_verify |
| `deu_humaneval_code` | `ellamind/humaneval-multilingual` (deu) | Code | 3 | pass@1 |
| `deu_mbpp_code` | `ellamind/mbpp-multilingual` (deu) | Code | 3 | pass@1 |
| `deu_glp` (22 subtasks) | `ellamind/german-language-proficiency-preview` | RC | 0 | acc, acc_norm |
| `deu_simpleqa_gen` | `ellamind/simpleqa-verified-multilingual` | Gen | 5 | exact_match |
| `deu_simpleqa_mc` | `ellamind/simpleqa-verified-multilingual` | MC | 5 | acc, acc_norm |
| `deu_hle_mc` | `ellamind/hle-multilingual` (deu) | MC | 5 | acc, acc_norm |

### deu_base_easy

| Group | Tasks | n-shot | Metric |
|-------|-------|--------|--------|
| `deu_qa_rc` | ARC, CSQA, SiQA, PiQA, WinoGrande, HellaSwag, MMMLU (32 subj.), SimpleQA + easy variants (13 tasks) | varies | acc, acc_norm |
| `deu_qa_bpb` | ARC, CSQA, SiQA, PiQA, WinoGrande, HellaSwag, MMMLU (32 subj.), SimpleQA (9 tasks) | 5 | bits_per_byte |
| `deu_math_bpb` | Minerva Math (7 categories) | 4 | bits_per_byte |
| `deu_code_bpb` | HumanEval, MBPP | 3 | bits_per_byte |
| `deu_glp` | 22 grammar subtasks | 0 | acc, acc_norm |
| `deu_hle_rc` | HLE (626 questions) | 5 | acc, acc_norm |
| `deu_hle_bpb` | HLE (626 questions) | 5 | bits_per_byte |

German Language Proficiency (GLP) uses RC in the easy suite (BPB not suitable due to short targets).

### All German Tasks by Benchmark

#### ARC (`tasks/deu/arc/`)
Dataset: `ellamind/arc-german-preview`

| Task | Type | n-shot |
|------|------|--------|
| `deu_arc_challenge_rc` | RC | 5 |
| `deu_arc_easy_rc` | RC | 5 |
| `deu_arc_challenge_mc` | MC | 5 |
| `deu_arc_easy_mc` | MC | 5 |
| `deu_arc_challenge_bpb` | BPB | 5 |
| `deu_arc_easy_bpb` | BPB | 5 |

#### CSQA (`tasks/deu/csqa/`)
Dataset: `ellamind/csqa-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_csqa_rc` | RC | 5 |
| `deu_csqa_easy_rc` | RC | 5 |
| `deu_csqa_mc` | MC | 5 |
| `deu_csqa_easy_mc` | MC | 5 |
| `deu_csqa_bpb` | BPB | 5 |

#### HellaSwag / HellaSwag (`tasks/deu/hellaswag/`)
Dataset: `ellamind/hellaswag-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_hellaswag_rc` | RC | 0 |
| `deu_hellaswag_easy_rc` | RC | 0 |
| `deu_hellaswag_mc` | MC | 0 |
| `deu_hellaswag_bpb` | BPB | 0 |

#### WinoGrande / WinoGrande (`tasks/deu/winogrande/`)
Dataset: `ellamind/winogrande-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_winogrande_rc` | RC | 0 |
| `deu_winogrande_mc` | MC | 0 |
| `deu_winogrande_bpb` | BPB | 0 |

#### PiQA (`tasks/deu/piqa/`)
Dataset: `ellamind/piqa-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_piqa_rc` | RC | 0 |
| `deu_piqa_easy_rc` | RC | 0 |
| `deu_piqa_mc` | MC | 0 |
| `deu_piqa_bpb` | BPB | 0 |

#### SiQA (`tasks/deu/siqa/`)
Dataset: `ellamind/siqa-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_siqa_rc` | RC | 0 |
| `deu_siqa_easy_rc` | RC | 0 |
| `deu_siqa_mc` | MC | 0 |
| `deu_siqa_bpb` | BPB | 0 |

#### MMMLU (`tasks/deu/mmmlu/`)
Dataset: `openai/MMMLU` (DE_DE subset)

171 task configs: 57 subjects x 3 variants (mc, rc, bpb).

Task name pattern: `deu_mmmlu_{subject}_{mc,rc,bpb}`

Group definitions define filtered (32 subjects) and full (57 subjects) variants:
- `deu_mmmlu_filtered` (MC), `deu_mmmlu_filtered_rc`, `deu_mmmlu_filtered_bpb`
- `deu_mmmlu` (full 57 subjects), with STEM/humanities/social_sciences/other subgroups

#### Minerva Math (`tasks/deu/minerva_math/`)
Dataset: `ellamind/hendrycks-math-multilingual` (config: `deu`)

14 task configs: 7 categories x 2 variants (gen, bpb).

| Category | Gen Task | BPB Task |
|----------|----------|----------|
| Algebra (1177) | `deu_minerva_math_algebra_gen` | `deu_minerva_math_algebra_bpb` |
| Counting & Probability (471) | `deu_minerva_math_counting_and_probability_gen` | `deu_minerva_math_counting_and_probability_bpb` |
| Geometry (472) | `deu_minerva_math_geometry_gen` | `deu_minerva_math_geometry_bpb` |
| Intermediate Algebra (880) | `deu_minerva_math_intermediate_algebra_gen` | `deu_minerva_math_intermediate_algebra_bpb` |
| Number Theory (536) | `deu_minerva_math_number_theory_gen` | `deu_minerva_math_number_theory_bpb` |
| Prealgebra (867) | `deu_minerva_math_prealgebra_gen` | `deu_minerva_math_prealgebra_bpb` |
| Precalculus (539) | `deu_minerva_math_precalculus_gen` | `deu_minerva_math_precalculus_bpb` |

Group tasks: `deu_minerva_math_gen` (all 7 Gen), `deu_minerva_math_bpb` (all 7 BPB).

#### HumanEval (`tasks/deu/humaneval/`)
Dataset: `ellamind/humaneval-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_humaneval_code` | Code | 3 |
| `deu_humaneval_bpb` | BPB | 3 |

#### MBPP (`tasks/deu/mbpp/`)
Dataset: `ellamind/mbpp-multilingual` (config: `deu`)

| Task | Type | n-shot |
|------|------|--------|
| `deu_mbpp_code` | Code | 3 |
| `deu_mbpp_bpb` | BPB | 3 |

#### German Language Proficiency (GLP) (`tasks/deu/glp/`)
Dataset: `ellamind/german-language-proficiency-preview`

22 task configs, all RC format, 0-shot:

adjective_ending, als_wie, article_case, article_gender, capitalization, collocation,
connector, dass_das, discourse, konjunktiv, n_deklination, naturalness, perfect_aux,
pronoun_case, prose_quality, punctuation, register, seit_seid, ss_eszett,
verb_conjugation, word_order_nebensatz, word_order_v2

Task name pattern: `deu_glp_{subtask}_rc`

#### SimpleQA Verified (`tasks/deu/simpleqa/`)
Dataset: `ellamind/simpleqa-verified-multilingual` (deu subset)

1,000 short-form factual QA prompts testing parametric knowledge. 5 examples reserved for fewshot, ~16 flagged for review = ~979 evaluation examples.

| Task | Type | n-shot |
|------|------|--------|
| `deu_simpleqa_gen` | Gen | 5 |
| `deu_simpleqa_mc` | MC | 5 |
| `deu_simpleqa_rc` | RC | 5 |
| `deu_simpleqa_bpb` | BPB | 5 |

Gen variant uses alias-aware exact_match (checks against `answer + answer_aliases`). MC/RC use answer + 4 hard distractors (5 choices, shuffled deterministically).

#### HLE (`tasks/deu/hle/`)
Dataset: `ellamind/hle-multilingual` (deu subset)

800 expert-level questions from Humanity's Last Exam across Math, Physics, CS/AI, Biology, Humanities, Chemistry, Engineering, and Other. 169 flagged for review + 5 reserved for fewshot = ~626 evaluation examples. Variable choice count: multipleChoice questions have 4-31 choices, exactMatch questions have 4 choices (1 correct + 3 generated distractors).

| Task | Type | n-shot |
|------|------|--------|
| `deu_hle_mc` | MC | 5 |
| `deu_hle_rc` | RC | 5 |
| `deu_hle_bpb` | BPB | 5 |

MC/RC use correct_answer + incorrect_answers as choices (shuffled deterministically). BPB scores only the gold correct_answer.

### German-Specific Notes

#### MMMLU Filtering

German suites use `deu_mmmlu_filtered` (32 translation-safe subjects):

- **Math & formal reasoning (7):** abstract_algebra, college_mathematics, elementary_mathematics, high_school_mathematics, high_school_statistics, formal_logic, econometrics
- **Core STEM (9):** college/high_school physics, chemistry, biology, conceptual_physics, astronomy, electrical_engineering
- **Computer science (4):** college/high_school CS, computer_security, machine_learning
- **Medicine & health (6):** anatomy, clinical_knowledge, college/professional_medicine, medical_genetics, virology
- **Accounting / quant econ (3):** professional_accounting, high_school macro/microeconomics
- **Borderline safe (3):** logical_fallacies, nutrition, human_aging

The full 57-subject version is available as `deu_mmmlu`.

#### German Language Proficiency (GLP)

22 subtasks covering German-specific grammar and fluency phenomena. Each subtask filters the dataset by error type. German Language Proficiency (GLP) is RC-only (no BPB variant) because targets are too short for meaningful BPB evaluation.

See `tasks/deu/glp/README.md` for detailed descriptions of each error type.

#### Easy Variants

Several German QA tasks have `_easy` variants (e.g. `deu_csqa_easy_rc`, `deu_piqa_easy_rc`) that use simpler distractors. These are included in the `deu_qa_rc` group but not in the BPB groups.

#### Cross-Language Comparison

`eng_base_main_reduced` provides English equivalents for all German `deu_base_main` tasks (excluding glp):

| German | English |
|--------|---------|
| `deu_arc_challenge` | `arc_challenge_rc` |
| `deu_arc_easy` | `arc_easy_rc` |
| `deu_csqa` | `csqa_rc` |
| `deu_siqa` | `socialiqa_rc` |
| `deu_piqa` | `piqa_rc` |
| `deu_winogrande` | `winogrande_rc` |
| `deu_hellaswag` | `hellaswag_rc` |
| `deu_mmmlu` | `mmlu_rc` |
| `deu_minerva_math` | `minerva_math_gen` |
| `deu_humaneval` | `humaneval_code` |
| `deu_mbpp` | `mbpp_code` |

---

## English Tasks (eng)

28 benchmarks, ~471 task configs.

### eng_base_main

#### STEM QA MC (`stem_qa_mc`)

| Task | Dataset | n-shot | Metrics |
|------|---------|--------|---------|
| `arc_easy_mc` | `allenai/ai2_arc` (ARC-Easy) | 5 | acc, acc_norm |
| `arc_challenge_mc` | `allenai/ai2_arc` (ARC-Challenge) | 5 | acc, acc_norm |
| `mmlu_stem_mc` (18 subj.) | MMLU | 5 | acc, acc_norm |
| `medmcqa_mc` | MedMCQA | 5 | acc, acc_norm |
| `medqa_mc` | MedQA (USMLE) | 5 | acc, acc_norm |
| `sciq_mc` | SciQ | 5 | acc, acc_norm |

#### Non-STEM QA MC (`nonstem_qa_mc`)

| Task | Dataset | n-shot | Metrics |
|------|---------|--------|---------|
| `mmlu_humanities_mc` (13 subj.) | MMLU | 5 | acc, acc_norm |
| `mmlu_social_sciences_mc` (11 subj.) | MMLU | 5 | acc, acc_norm |
| `mmlu_other_mc` (14 subj.) | MMLU | 5 | acc, acc_norm |
| `csqa_mc` | CommonsenseQA | 5 | acc, acc_norm |
| `piqa_mc` | PiQA | 5 | acc, acc_norm |
| `socialiqa_mc` | Social IQA | 5 | acc, acc_norm |
| `drop_mc` | DROP | 5 | acc, acc_norm |
| `jeopardy_mc` | Jeopardy | 5 | acc, acc_norm |
| `naturalqs_mc` | Natural Questions | 5 | acc, acc_norm |
| `coqa_mc` | CoQA | 5 | acc, acc_norm |

#### Generative Tasks (`gen`)

| Task | Dataset | Type | n-shot | Metrics |
|------|---------|------|--------|---------|
| `hellaswag_rc` | HellaSwag | RC | 5 | acc, acc_norm |
| `winogrande_rc` | WinoGrande | RC | 5 | acc, acc_norm |
| `lambada_gen_olmes` | LAMBADA | Gen | 0 | acc |
| `drop_gen` | DROP | Gen | 5 | exact_match, f1 |
| `jeopardy_gen` | Jeopardy | Gen | 5 | exact_match, f1 |
| `naturalqs_gen` | Natural Questions | Gen | 5 | exact_match, f1 |
| `squad_gen` | SQuAD | Gen | 5 | exact_match, f1 |
| `coqa_gen` | CoQA | Gen | 0 | exact_match, f1 |

#### Math (`math`)

| Task | Dataset | n-shot | Metrics |
|------|---------|--------|---------|
| `gsm8k_gen` | GSM8K | 8 | pass@k, maj@k, exact_match |
| `gsm_symbolic_all` (3 variants) | GSM-Symbolic | 8 | pass@1 |
| `minerva_math_gen` (7 cat.) | MATH (hendrycks) | 4 | pass@1 |

#### Code (`code`)

| Task | Dataset | n-shot | Metrics |
|------|---------|--------|---------|
| `humaneval_code` | HumanEval | 3 | pass@1, pass@10, pass@100 |
| `mbpp_code` | MBPP | 3 | pass@1, pass@10, pass@100 |

### eng_base_easy

#### QA RC (`qa_rc`)

| Task | Dataset | n-shot |
|------|---------|--------|
| `arc_easy_rc` | ARC-Easy | 5 |
| `arc_challenge_rc` | ARC-Challenge | 5 |
| `mmlu_rc` (57 subj.) | MMLU | 5 |
| `csqa_rc` | CommonsenseQA | 5 |
| `hellaswag_rc` | HellaSwag | 5 |
| `winogrande_rc` | WinoGrande | 5 |
| `socialiqa_rc` | Social IQA | 5 |
| `piqa_rc` | PiQA | 5 |
| `coqa_rc` | CoQA | 0 |
| `drop_rc` | DROP | 5 |
| `jeopardy_rc` | Jeopardy | 5 |
| `naturalqs_rc` | Natural Questions | 5 |
| `squad_rc` | SQuAD | 5 |
| `sciq_rc` | SciQ | 5 |
| `qasper_yesno_rc` | QASPER | 0 |
| `lambada_rc` | LAMBADA | 0 |
| `medmcqa_rc` | MedMCQA | 5 |
| `medqa_rc` | MedQA | 5 |

All tasks report acc and acc_norm.

#### QA BPB (`qa_bpb`)

Same benchmarks as QA RC (with `_bpb` suffix), plus:
- `lab_bench_dbqa_bpb` (LAB-Bench database QA)
- `lab_bench_protocolqa_bpb` (LAB-Bench protocol QA)

All tasks report bits_per_byte.

#### Math BPB (`math_bpb`)

Minerva Math 7 categories with `_bpb` suffix.

#### Code BPB (`code_bpb`)

| Task | Description |
|------|-------------|
| `humaneval_bpb` | HumanEval BPB (OLMo3-style fenced prompt/completion) |
| `mbpp_bpb` | MBPP BPB |
| `mt_mbpp_{lang}_bpb` (17 langs) | MT-MBPP: bash, c, cpp, csharp, go, haskell, java, javascript, matlab, php, python, r, ruby, rust, scala, swift, typescript |

### All English Benchmarks

| Benchmark | Directory | Variants | Dataset |
|-----------|-----------|----------|---------|
| ARC | `arc/` | mc, rc, bpb | `allenai/ai2_arc` |
| Basic Skills | `basic_skills/` | rc, bpb | `allenai/basic-skills` |
| CoQA | `coqa/` | gen, mc, rc, bpb | CoQA |
| CSQA | `csqa/` | mc, rc, bpb | CommonsenseQA |
| DROP | `drop/` | gen, mc, rc, bpb | DROP |
| GSM8K | `gsm8k/` | gen | GSM8K |
| GSM-Symbolic | `gsm_symbolic/` | gen, cot | GSM-Symbolic (+p1, +p2 variants) |
| HellaSwag | `hellaswag/` | rc, bpb | HellaSwag |
| HumanEval | `humaneval/` | code, bpb | HumanEval |
| Jeopardy | `jeopardy/` | gen, mc, rc, bpb | Jeopardy |
| LAB-Bench | `lab_bench/` | bpb | LAB-Bench (dbqa, protocolqa) |
| LAMBADA | `lambada/` | gen, rc, bpb | LAMBADA |
| MBPP | `mbpp/` | code, bpb | MBPP |
| MedMCQA | `medmcqa/` | mc, rc, bpb | MedMCQA |
| MedQA | `medqa/` | mc, rc, bpb | MedQA (USMLE) |
| Minerva Math | `minerva_math/` | gen, bpb | MATH (hendrycks), 7 categories |
| MMLU | `mmlu/` | mc, rc, bpb | MMLU (57 subjects) |
| MT-MBPP | `mt_mbpp/` | bpb | MT-MBPP (17 programming languages) |
| Natural Questions | `naturalqs/` | gen, mc, rc, bpb | Natural Questions |
| PiQA | `piqa/` | mc, rc, bpb | PiQA |
| QASPER | `qasper/` | rc, bpb | QASPER (yes/no) |
| SciQ | `sciq/` | mc, rc, bpb | SciQ |
| SciRIFF | `sciriff/` | rc, bpb | SciRIFF (yes/no) |
| Social IQA | `socialiqa/` | mc, rc, bpb | Social IQA |
| SQuAD | `squad/` | gen, mc, rc, bpb | SQuAD |
| WinoGrande | `winogrande/` | rc, bpb | WinoGrande |

### English-Specific Notes

#### MMLU Organization

MMLU is split into 4 subject groups for MC evaluation:
- `mmlu_stem_mc` (18 subjects)
- `mmlu_humanities_mc` (13 subjects)
- `mmlu_social_sciences_mc` (11 subjects)
- `mmlu_other_mc` (14 subjects)

RC and BPB variants use flat groups: `mmlu_rc` (57 subjects), `mmlu_bpb` (57 subjects).

#### GSM-Symbolic Variants

`gsm_symbolic_all` aggregates three difficulty levels:
- `gsm_symbolic_gen` - base
- `gsm_symbolic_p1_gen` - perturbation level 1
- `gsm_symbolic_p2_gen` - perturbation level 2

#### Basic Skills

Note: `basic_skills_rc` is not included in the `gen` suite because `allenai/basic-skills` uses deprecated dataset scripts. Tasks exist but may require manual dataset setup.

Subtasks: arithmetic, coding, common_knowledge, logical_reasoning, pattern, string_operations.

#### MT-MBPP Languages

17 programming languages for multilingual code BPB: bash, c, cpp, csharp, go, haskell, java, javascript, matlab, php, python, r, ruby, rust, scala, swift, typescript.

---

## Adding New Tasks

### For German

1. Create a new directory under `tasks/deu/<benchmark>/`
2. Create a lean template `_<name>_template.yaml` with shared config (dataset, fewshot)
3. Create task-specific configs for each variant (RC, BPB, CoT, code)
4. Add `utils.py` with task-specific functions (process_docs, fewshot, BPB)
5. Add the task to the relevant group in `suites/deu/` and/or `suites/deu_base_main.yaml`

### For English

1. Create a new directory under `tasks/eng/<benchmark>/`
2. Create a lean template `_<name>_template.yaml` with shared config
3. Create task-specific configs for each variant
4. Add `utils.py` with task-specific functions
5. Add the task to the relevant group in `suites/eng/` and/or `suites/eng_base_main.yaml`

### Template Best Practices

- Use `_` prefix for template and group files
- Keep templates lean: only shared config (dataset, fewshot, splits)
- Move output-type-specific config to task files (generation_kwargs, metric_list)
- Separate templates for different output types if needed
- Use `target_delimiter: ""` for BPB tasks where `doc_to_target` starts with a space
- Use `fewshot_config: sampler: first_n` for deterministic fewshot
