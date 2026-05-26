# MMDocRAG Redundancy Study — Results Report

---

## 1. Setup

| Parameter | Value |
|-----------|-------|
| Dataset | MMDocRAG `evaluation_20.jsonl`, n=2000 |
| Variants | original, gold\_redundant, negative\_redundant, mixed\_redundant |
| Pipeline A retriever | ColPali v1.3-hf (multi-vector image) + BGE-large-en-v1.5 (text) |
| Pipeline A generator | Qwen2.5-VL-7B-Instruct, `do_sample=False` (greedy) |
| Pipeline B retriever | VisRAG-Ret (openbmb) |
| Pipeline B generator | MiniCPM-V 2.6, `sampling=False`, `num_beams=1` (greedy) |
| Top-k | 10 |
| Answer target | `parse_answer(answer_short)` — list literals joined with space |
| Bootstrap CIs | 95%, 1000 resamples |

---

## 2. Track 1 — Answer Quality

Format: `mean [ci_low, ci_high]`

| Pipeline | Variant | EM | F1 | Precision | Recall | Jaccard | ref_tok_cov |
|---|---||---||---||---||---||---||---|
| A | original | 0.0130 [0.0080, 0.0185] | 0.3051 [0.2960, 0.3149] | 0.2382 [0.2289, 0.2475] | 0.5442 [0.5316, 0.5574] | 0.2025 [0.1948, 0.2107] | 0.0880 [0.0755, 0.1000] |
| A | gold_redundant | 0.0190 [0.0125, 0.0245] | 0.3548 [0.3451, 0.3657] | 0.2780 [0.2682, 0.2882] | 0.6240 [0.6107, 0.6376] | 0.2433 [0.2348, 0.2528] | 0.1580 [0.1420, 0.1740] |
| A | negative_redundant | 0.0090 [0.0050, 0.0135] | 0.2979 [0.2890, 0.3073] | 0.2268 [0.2187, 0.2360] | 0.5554 [0.5425, 0.5690] | 0.1958 [0.1886, 0.2044] | 0.0910 [0.0775, 0.1040] |
| A | mixed_redundant | 0.0150 [0.0100, 0.0205] | 0.3461 [0.3363, 0.3560] | 0.2669 [0.2573, 0.2763] | 0.6263 [0.6124, 0.6391] | 0.2355 [0.2269, 0.2444] | 0.1620 [0.1460, 0.1785] |
| B | original | 0.0255 [0.0180, 0.0325] | 0.3166 [0.3069, 0.3270] | 0.2933 [0.2821, 0.3043] | 0.4392 [0.4267, 0.4515] | 0.2147 [0.2058, 0.2235] | 0.0615 [0.0500, 0.0725] |
| B | gold_redundant | 0.0285 [0.0205, 0.0360] | 0.3403 [0.3296, 0.3511] | 0.3032 [0.2921, 0.3140] | 0.5040 [0.4909, 0.5178] | 0.2348 [0.2251, 0.2449] | 0.1015 [0.0875, 0.1150] |
| B | negative_redundant | 0.0230 [0.0165, 0.0295] | 0.3076 [0.2978, 0.3166] | 0.2764 [0.2654, 0.2864] | 0.4474 [0.4344, 0.4598] | 0.2072 [0.1984, 0.2156] | 0.0615 [0.0505, 0.0720] |
| B | mixed_redundant | 0.0275 [0.0200, 0.0350] | 0.3340 [0.3235, 0.3443] | 0.2939 [0.2832, 0.3047] | 0.5037 [0.4902, 0.5167] | 0.2286 [0.2191, 0.2381] | 0.0980 [0.0850, 0.1105] |

---

## 3. Track 2 — Evidence Selection

Format: `mean [ci_low, ci_high]`

| Pipeline | Variant | P@10 | R@10 | F1@10 | Jaccard@10 | mean\_gold\_set\_size | n\_skipped |
|---|---||---||---||---||---||---||---|
| A | original | 0.1711 [0.1647, 0.1775] | 0.5606 [0.5435, 0.5767] | 0.2465 [0.2385, 0.2545] | 0.1540 [0.1482, 0.1597] | 2.91 | 0 |
| A | gold_redundant | 0.2806 [0.2740, 0.2873] | 0.6505 [0.6403, 0.6604] | 0.3701 [0.3634, 0.3770] | 0.2397 [0.2343, 0.2451] | 4.52 | 0 |
| A | negative_redundant | 0.1600 [0.1538, 0.1660] | 0.5251 [0.5082, 0.5405] | 0.2304 [0.2225, 0.2382] | 0.1428 [0.1374, 0.1481] | 2.91 | 0 |
| A | mixed_redundant | 0.2639 [0.2575, 0.2702] | 0.6164 [0.6056, 0.6268] | 0.3488 [0.3418, 0.3556] | 0.2230 [0.2176, 0.2284] | 4.52 | 0 |
| B | original | 0.1545 [0.1490, 0.1600] | 0.5254 [0.5096, 0.5413] | 0.2246 [0.2176, 0.2316] | 0.1366 [0.1319, 0.1415] | 2.91 | 0 |
| B | gold_redundant | 0.2542 [0.2486, 0.2603] | 0.6039 [0.5939, 0.6141] | 0.3379 [0.3319, 0.3441] | 0.2125 [0.2080, 0.2173] | 4.52 | 0 |
| B | negative_redundant | 0.1427 [0.1376, 0.1480] | 0.4921 [0.4766, 0.5076] | 0.2081 [0.2016, 0.2149] | 0.1254 [0.1208, 0.1300] | 2.91 | 0 |
| B | mixed_redundant | 0.2346 [0.2292, 0.2401] | 0.5615 [0.5515, 0.5716] | 0.3123 [0.3066, 0.3180] | 0.1930 [0.1888, 0.1974] | 4.52 | 0 |

---

## 4. CI Overlap Significance (95% Bootstrap)

`***` = no overlap (significant)  ·  `*` = partial overlap  ·  `ns` = heavy overlap

### Pipeline A

**EM**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | ns | ns | ns |
| gold\_r |   | — | * | ns |
| neg\_r |   |   | — | * |
| mix\_r |   |   |   | — |

**F1**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | ns | *** |
| gold\_r |   | — | *** | ns |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

**R@10**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | *** | *** |
| gold\_r |   | — | *** | *** |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

**F1@10**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | *** | *** |
| gold\_r |   | — | *** | *** |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

### Pipeline B

**EM**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | ns | ns | ns |
| gold\_r |   | — | ns | ns |
| neg\_r |   |   | — | ns |
| mix\_r |   |   |   | — |

**F1**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | ns | * |
| gold\_r |   | — | *** | ns |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

**R@10**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | *** | *** |
| gold\_r |   | — | *** | *** |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

**F1@10**

| | orig | gold\_r | neg\_r | mix\_r |
|---|---|---|---|---|
| orig | — | *** | *** | *** |
| gold\_r |   | — | *** | *** |
| neg\_r |   |   | — | *** |
| mix\_r |   |   |   | — |

---

## 5. Modality Mix in Top-10 (mean slots per question)

| Pipeline | Variant | text\_orig | image\_orig | text\_twin | total |
|---|---|---|---|---|---|
| A | original | 7.878 | 2.115 | 0.000 | 9.993 |
| A | gold_redundant | 6.770 | 2.059 | 1.167 | 9.996 |
| A | negative_redundant | 6.340 | 1.655 | 2.005 | 10.000 |
| A | mixed_redundant | 5.513 | 1.655 | 2.832 | 10.000 |
| B | original | 6.795 | 3.197 | 0.000 | 9.993 |
| B | gold_redundant | 5.793 | 3.135 | 1.069 | 9.997 |
| B | negative_redundant | 4.966 | 3.119 | 1.915 | 10.000 |
| B | mixed_redundant | 4.212 | 3.075 | 2.713 | 10.000 |

---

## 6. Prediction-Length Statistics

| Pipeline | Variant | Mean chars | Median chars | Mean tokens | RefCov% |
|---|---|---|---|---|---|
| A | original | 366.1 | 380 | 56.13 | 8.80% |
| A | gold_redundant | 347.6 | 347 | 53.41 | 15.80% |
| A | negative_redundant | 382.1 | 412 | 58.67 | 9.10% |
| A | mixed_redundant | 358.9 | 375 | 55.20 | 16.20% |
| B | original | 245.2 | 173 | 37.25 | 6.15% |
| B | gold_redundant | 270.1 | 195 | 41.03 | 10.15% |
| B | negative_redundant | 266.9 | 194 | 40.41 | 6.15% |
| B | mixed_redundant | 280.0 | 202 | 42.38 | 9.80% |

---

## 7. Smoke-Test Diagnostic (Pipeline B, dev split, n=50, first 10 per variant)

Buckets: **V+** verbose-correct · **V-** verbose-wrong · **C+** concise-correct · **C-** concise-wrong · **R** refusal/empty

| Variant | Condition | V+ | V- | C+ | C- | R |
|---|---|---|---|---|---|---|
| original | beam (num\_beams=3) | 0 | 5 | 0 | 5 | 0 |
| original | greedy (num\_beams=1) | 0 | 3 | 0 | 7 | 0 |
| gold_redundant | beam (num\_beams=3) | 0 | 5 | 0 | 5 | 0 |
| gold_redundant | greedy (num\_beams=1) | 0 | 3 | 3 | 4 | 0 |
| negative_redundant | beam (num\_beams=3) | 0 | 6 | 0 | 4 | 0 |
| negative_redundant | greedy (num\_beams=1) | 0 | 3 | 1 | 6 | 0 |
| mixed_redundant | beam (num\_beams=3) | 0 | 6 | 0 | 4 | 0 |
| mixed_redundant | greedy (num\_beams=1) | 0 | 5 | 2 | 3 | 0 |

---

## 8. Engineering Notes

### Beam-search-to-greedy validity correction (Pipeline B)

MiniCPM-V 2.6's `chat()` with `sampling=False` silently uses `num_beams=3, repetition_penalty=1.2`
rather than greedy decoding. Pipeline A uses `do_sample=False` (true greedy). The asymmetry was
identified as an experiment-validity issue and corrected by passing `num_beams=1` as a kwarg
(which overrides via the kwargs-intersection update in the model's `chat()` implementation).
Effect on dev-split smoke test (n=50): 80–86% of predictions changed across all 4 variants;
EM improved +0.080 to +0.140, F1 improved +0.078 to +0.118. Fix applied before full eval run.

### ColPali weight-key mismatch (Pipeline A)

A `transformers` version mismatch caused a weight-key loading error in ColPali v1.3-hf.
Resolved via version pinning before the main evaluation runs.

### Wall-clock timing

| Phase | Pipeline | Duration |
|---|---|---|
| n=200 timing trial (beam, eval) | B | 181.6 min total (154.9 min generation) |
| n=20 timing trial (greedy, eval) | B | 6.6 min total (0.7 min generation, 2.10 s/item) |
| n=20 timing trial (greedy, eval) | A | 3.0 min total (1.2 min generation, 3.60 s/item) |
| Full eval (parallel, GPUs 1 & 3) | A | 7h 20min |
| Full eval (parallel, GPUs 1 & 3) | B | 6h 50min |

### Frozen dataset

Pre-built variant JSONL files shared with team as `mmdocrag_redundancy_v1/`.
Contains 8 files (4 eval + 4 dev variants), `build_variant.py` frozen copy,
`checksums.txt` (md5), and `README.md`. All 8 checksums verified clean.