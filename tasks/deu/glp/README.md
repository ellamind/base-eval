# German German Language Proficiency (GLP)

## Paper

Title: `TBD`

Abstract: `TBD`

The German German Language Proficiency (GLP) benchmark evaluates language models on their ability to identify correct German text among variants containing grammatical errors or fluency issues. It uses a binary-choice cloze format where models must select the grammatically correct continuation given a context prefix.

The benchmark covers 22 distinct error types spanning grammar, orthography, punctuation, and fluency dimensions. Performance is reported as a **macro average** across all error types, ensuring equal weight regardless of sample count per category.

Homepage: `TBD`

### Citation

```text
TBD
```

### Groups, Tags, and Tasks

#### Groups

* `deu_glp`: Macro-averaged accuracy across all 22 German grammar and fluency subtasks

#### Tags

* `deu_glp`: All subtasks are tagged for grouped evaluation

#### Tasks

| Task | Error Type | Samples | Description |
|------|------------|---------|-------------|
| `deu_glp_adjective_ending` | ADJECTIVE_ENDING_ERROR | 321 | German adjective declension/ending errors |
| `deu_glp_als_wie` | ALS_WIE_ERROR | 274 | Confusion between "als" and "wie" in comparisons |
| `deu_glp_article_case` | ARTICLE_CASE_ERROR | 329 | Article case errors (nominative/accusative/dative/genitive) |
| `deu_glp_article_gender` | ARTICLE_GENDER_ERROR | 333 | Article gender errors (der/die/das) |
| `deu_glp_capitalization` | CAPITALIZATION_ERROR | 310 | German noun capitalization errors |
| `deu_glp_collocation` | COLLOCATION_FLUENCY | 342 | Unnatural word combinations and collocations |
| `deu_glp_connector` | CONNECTOR_FLUENCY | 342 | Discourse connector and conjunction errors |
| `deu_glp_dass_das` | DASS_DAS_ERROR | 339 | Confusion between "dass" (conjunction) and "das" (article/pronoun) |
| `deu_glp_discourse` | DISCOURSE_FLUENCY | 342 | Discourse-level fluency and coherence issues |
| `deu_glp_konjunktiv` | KONJUNKTIV_ERROR | 278 | Subjunctive mood (Konjunktiv I/II) errors |
| `deu_glp_n_deklination` | N_DEKLINATION_ERROR | 360 | N-declension (weak masculine noun) errors |
| `deu_glp_naturalness` | NATURALNESS_FLUENCY | 341 | Unnatural phrasing and expression |
| `deu_glp_perfect_aux` | PERFECT_AUX_PARTICIPLE_ERROR | 342 | Perfect tense auxiliary (haben/sein) and participle errors |
| `deu_glp_pronoun_case` | PRONOUN_CASE_ERROR | 327 | Pronoun case errors |
| `deu_glp_prose_quality` | PROSE_QUALITY_FLUENCY | 342 | Overall prose quality and style issues |
| `deu_glp_punctuation` | GERMAN_PUNCTUATION_ERROR | 340 | German-specific punctuation errors (comma rules, etc.) |
| `deu_glp_register` | REGISTER_ERROR | 342 | Register/formality level inconsistencies |
| `deu_glp_seit_seid` | SEIT_SEID_ERROR | 274 | Confusion between "seit" (since) and "seid" (are) |
| `deu_glp_ss_eszett` | SS_ESZETT_ERROR | 336 | Confusion between "ss" and "ß" (Eszett) |
| `deu_glp_verb_conjugation` | VERB_CONJUGATION_ERROR | 326 | Verb conjugation errors |
| `deu_glp_word_order_nebensatz` | WORD_ORDER_NEBENSATZ_ERROR | 342 | Subordinate clause word order (verb-final) errors |
| `deu_glp_word_order_v2` | WORD_ORDER_V2_ERROR | 342 | Main clause V2 (verb-second) word order errors |

**Total samples:** 7,283

### Dataset

* **Source:** `ellamind/german-language-proficiency-preview` (HuggingFace)
* **Format:** Binary multiple choice (cloze completion)
* **Split:** train (used as validation)

#### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `ctx` | string | Context prefix for cloze completion |
| `endings` | list[str] | Two options: [option_a, option_b] |
| `label` | int | Index of correct option (0 or 1) |
| `error_type` | string | Error category (e.g., "CAPITALIZATION_ERROR") |
| `language` | string | Language code ("de") |

#### Example

```json
{
  "id": "cloze-abc123",
  "ctx": "Der Vertrag regelt die technische ",
  "endings": ["Gebäudeausrüstung.", "gebäudeausrüstung."],
  "label": 0,
  "error_type": "CAPITALIZATION_ERROR",
  "language": "de"
}
```

### Metrics

| Metric | Description |
|--------|-------------|
| `acc` | Standard accuracy - choice with highest log-likelihood |
| `acc_norm` | Length-normalized accuracy - adjusts for option length bias |

The group-level metrics are **macro-averaged** across all 22 subtasks (equal weight per error type, regardless of sample count).

### Evaluation

```bash
# Run full benchmark (macro average over 22 error types)
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks deu_glp \
    --include_path /path/to/deu_glp

# Run single subtask
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks deu_glp_capitalization \
    --include_path /path/to/deu_glp
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
  * The main task is `deu_glp` which computes macro average over all subtasks
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
  * See task table above
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* **v0.1** (2026-01-26): Initial preview release with 22 error types and 7,283 samples
