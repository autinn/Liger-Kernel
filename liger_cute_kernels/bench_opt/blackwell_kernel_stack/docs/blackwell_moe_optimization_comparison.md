# Blackwell LigerCute MoE Optimization Comparison

## Scope

| Label | Revision | Role |
|---|---|---|
| A | `db3ed8` | Single-B200 pre-pipeline baseline (`AccStages=1`) |
| W0 | `448cbbe` | Control before warp-3 migration (`AccStages=2`) |
| W1 | `a52b5f` | Warp-role specialization package |
| B | `69cfed` | Accumulator/backward pipeline with single- and multi-GPU SM100 tables |
| C | `761d687` | Paired-CTA 2SM MLP3/MLP4 plus tuned mainloop depth |

A/W0/W1/B/C are compared on one B200. B/C are compared on eight B200s.
A is intentionally excluded from the 8-GPU comparison because it has no
8-GPU tuning table; retuning it would create a different baseline.

Combined latency is `forward_ms + backward_ms` per shape. Combined speedup
is `(before_fwd + before_bwd) / (after_fwd + after_bwd)`, not an average
of the two pass speedups.

## Reproduction status

```text
status=partial
finished_at=2026-08-12T08:45:31Z
summary=/home/jobuser/.copilot/session-state/c92f3156-8fbd-4bcd-8c77-8a452227c705/files/blackwell-detached-matrix/SUMMARY.md
results=/home/jobuser/.copilot/session-state/c92f3156-8fbd-4bcd-8c77-8a452227c705/files/blackwell-detached-matrix/results
```

## Aggregate results

| Comparison | Cases | Geomean | Median | Min | Max | Wins | Parity | Regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_A_to_W0_fwd | 35 | 0.977x | 0.973x | 0.861x | 1.214x | 11 | 2 | 22 |
| single_W0_to_W1_warp_specialization_fwd | 35 | 1.008x | 1.005x | 0.975x | 1.120x | 16 | 18 | 1 |
| single_A_to_W1_total_mlp1_fwd | 35 | 0.985x | 0.979x | 0.870x | 1.216x | 12 | 1 | 22 |
| single_W1_to_B_remaining_pipeline_fwd | 35 | 1.021x | 1.023x | 0.902x | 1.080x | 28 | 4 | 3 |
| single_A_to_B_fwd | 35 | 1.006x | 1.017x | 0.888x | 1.227x | 22 | 0 | 13 |
| single_A_to_B_bwd | 35 | 1.493x | 1.567x | 1.089x | 1.997x | 35 | 0 | 0 |
| single_A_to_B_fwd_plus_bwd | 35 | 1.401x | 1.443x | 1.081x | 1.845x | 35 | 0 | 0 |
| single_B_to_C_2sm_stage_tuning_fwd | 35 | 1.004x | 1.002x | 0.917x | 1.126x | 12 | 21 | 2 |
| single_B_to_C_2sm_stage_tuning_bwd | 35 | 1.062x | 1.044x | 0.997x | 1.183x | 33 | 2 | 0 |
| single_B_to_C_2sm_stage_tuning_fwd_plus_bwd | 35 | 1.049x | 1.035x | 0.997x | 1.140x | 33 | 2 | 0 |
| single_A_to_C_overall_fwd | 35 | 1.011x | 1.015x | 0.893x | 1.221x | 20 | 1 | 14 |
| single_A_to_C_overall_bwd | 35 | 1.586x | 1.605x | 1.243x | 2.129x | 35 | 0 | 0 |
| single_A_to_C_overall_fwd_plus_bwd | 35 | 1.470x | 1.468x | 1.174x | 1.945x | 35 | 0 | 0 |
| eight_B_to_C_2sm_stage_tuning_fwd | 35 | 1.002x | 1.000x | 0.954x | 1.158x | 8 | 19 | 8 |
| eight_B_to_C_2sm_stage_tuning_bwd | 35 | 1.127x | 1.139x | 1.013x | 1.282x | 35 | 0 | 0 |
| eight_B_to_C_2sm_stage_tuning_fwd_plus_bwd | 35 | 1.096x | 1.103x | 1.007x | 1.211x | 35 | 0 | 0 |

The three-way A/B/C comparison is single-GPU. The 8-GPU comparison is
B/C only because db3ed8 has no 8-GPU tuning table; retuning it would
change the historical baseline.

## Per-shape results

The complete machine-readable table is preserved at
`/home/jobuser/.copilot/session-state/c92f3156-8fbd-4bcd-8c77-8a452227c705/files/blackwell-detached-matrix/comparison_rows.csv`.

### eight_B_to_C_2sm_stage_tuning_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 4.9019 | 3.8587 | 1.270x |
| llama4_scout_t16k | 15.7622 | 12.2987 | 1.282x |
| llama4_scout_t1k | 2.4367 | 2.1856 | 1.115x |
| llama4_scout_t2k | 3.6765 | 3.0811 | 1.193x |
| llama4_scout_t8k | 9.2493 | 7.4759 | 1.237x |
| mixtral_8x22b | 18.4122 | 15.5205 | 1.186x |
| mixtral_8x22b_t16k | 67.2777 | 56.8105 | 1.184x |
| mixtral_8x22b_t1k | 5.3626 | 4.3584 | 1.230x |
| mixtral_8x22b_t2k | 9.7605 | 8.1502 | 1.198x |
| mixtral_8x22b_t8k | 34.8413 | 30.1974 | 1.154x |
| mixtral_8x7b | 11.2667 | 9.4169 | 1.196x |
| mixtral_8x7b_t16k | 38.0837 | 33.3069 | 1.143x |
| mixtral_8x7b_t1k | 3.4123 | 2.8210 | 1.210x |
| mixtral_8x7b_t2k | 6.3910 | 5.3522 | 1.194x |
| mixtral_8x7b_t8k | 20.6775 | 17.5598 | 1.178x |
| qwen3_235b_a22b | 6.5583 | 5.6330 | 1.164x |
| qwen3_235b_a22b_t16k | 22.4626 | 19.7280 | 1.139x |
| qwen3_235b_a22b_t1k | 3.4355 | 3.0985 | 1.109x |
| qwen3_235b_a22b_t2k | 4.5022 | 3.9483 | 1.140x |
| qwen3_235b_a22b_t8k | 12.0009 | 10.3673 | 1.158x |
| qwen3_30b_a3b | 3.3517 | 3.1300 | 1.071x |
| qwen3_30b_a3b_t16k | 9.3221 | 8.3242 | 1.120x |
| qwen3_30b_a3b_t1k | 1.5826 | 1.5194 | 1.042x |
| qwen3_30b_a3b_t2k | 2.2135 | 2.0700 | 1.069x |
| qwen3_30b_a3b_t8k | 4.5702 | 4.1814 | 1.093x |
| qwen3_5_122b_a10b | 4.4666 | 4.1312 | 1.081x |
| qwen3_5_122b_a10b_t16k | 12.0079 | 11.0182 | 1.090x |
| qwen3_5_122b_a10b_t1k | 3.1776 | 3.0327 | 1.048x |
| qwen3_5_122b_a10b_t2k | 3.2403 | 3.0966 | 1.046x |
| qwen3_5_122b_a10b_t8k | 6.7227 | 6.2548 | 1.075x |
| qwen3_5_35b_a3b | 2.8050 | 2.7135 | 1.034x |
| qwen3_5_35b_a3b_t16k | 6.9483 | 6.7197 | 1.034x |
| qwen3_5_35b_a3b_t1k | 2.0081 | 1.9803 | 1.014x |
| qwen3_5_35b_a3b_t2k | 2.0520 | 2.0247 | 1.013x |
| qwen3_5_35b_a3b_t8k | 4.2021 | 4.0576 | 1.036x |

### eight_B_to_C_2sm_stage_tuning_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 1.5040 | 1.5011 | 1.002x |
| llama4_scout_t16k | 4.0194 | 4.0301 | 0.997x |
| llama4_scout_t1k | 0.6743 | 0.6555 | 1.029x |
| llama4_scout_t2k | 0.9718 | 0.9686 | 1.003x |
| llama4_scout_t8k | 2.4681 | 2.4674 | 1.000x |
| mixtral_8x22b | 5.4121 | 5.3923 | 1.004x |
| mixtral_8x22b_t16k | 20.1632 | 20.1154 | 1.002x |
| mixtral_8x22b_t1k | 1.6620 | 1.6629 | 0.999x |
| mixtral_8x22b_t2k | 3.1211 | 3.1052 | 1.005x |
| mixtral_8x22b_t8k | 10.0140 | 10.4531 | 0.958x |
| mixtral_8x7b | 2.9025 | 2.9038 | 1.000x |
| mixtral_8x7b_t16k | 9.9688 | 9.9862 | 0.998x |
| mixtral_8x7b_t1k | 1.0930 | 1.1351 | 0.963x |
| mixtral_8x7b_t2k | 1.7697 | 1.7421 | 1.016x |
| mixtral_8x7b_t8k | 5.4765 | 5.4054 | 1.013x |
| qwen3_235b_a22b | 2.1191 | 2.1245 | 0.997x |
| qwen3_235b_a22b_t16k | 6.7085 | 6.6951 | 1.002x |
| qwen3_235b_a22b_t1k | 1.0370 | 1.0378 | 0.999x |
| qwen3_235b_a22b_t2k | 1.3777 | 1.3786 | 0.999x |
| qwen3_235b_a22b_t8k | 3.5523 | 3.5210 | 1.009x |
| qwen3_30b_a3b | 0.9782 | 0.9883 | 0.990x |
| qwen3_30b_a3b_t16k | 2.4989 | 2.5150 | 0.994x |
| qwen3_30b_a3b_t1k | 0.6532 | 0.6850 | 0.954x |
| qwen3_30b_a3b_t2k | 0.7059 | 0.7000 | 1.008x |
| qwen3_30b_a3b_t8k | 1.3573 | 1.3555 | 1.001x |
| qwen3_5_122b_a10b | 1.6390 | 1.6476 | 0.995x |
| qwen3_5_122b_a10b_t16k | 4.2100 | 4.2203 | 0.998x |
| qwen3_5_122b_a10b_t1k | 1.1471 | 1.1482 | 0.999x |
| qwen3_5_122b_a10b_t2k | 1.1807 | 1.1809 | 1.000x |
| qwen3_5_122b_a10b_t8k | 2.4072 | 2.4510 | 0.982x |
| qwen3_5_35b_a3b | 1.1045 | 1.1000 | 1.004x |
| qwen3_5_35b_a3b_t16k | 2.6373 | 2.6490 | 0.996x |
| qwen3_5_35b_a3b_t1k | 1.0097 | 0.8722 | 1.158x |
| qwen3_5_35b_a3b_t2k | 0.8170 | 0.8239 | 0.992x |
| qwen3_5_35b_a3b_t8k | 1.6081 | 1.5970 | 1.007x |

### eight_B_to_C_2sm_stage_tuning_fwd_plus_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 6.4059 | 5.3598 | 1.195x |
| llama4_scout_t16k | 19.7816 | 16.3288 | 1.211x |
| llama4_scout_t1k | 3.1110 | 2.8411 | 1.095x |
| llama4_scout_t2k | 4.6483 | 4.0497 | 1.148x |
| llama4_scout_t8k | 11.7174 | 9.9433 | 1.178x |
| mixtral_8x22b | 23.8243 | 20.9128 | 1.139x |
| mixtral_8x22b_t16k | 87.4409 | 76.9259 | 1.137x |
| mixtral_8x22b_t1k | 7.0246 | 6.0213 | 1.167x |
| mixtral_8x22b_t2k | 12.8816 | 11.2554 | 1.144x |
| mixtral_8x22b_t8k | 44.8553 | 40.6505 | 1.103x |
| mixtral_8x7b | 14.1692 | 12.3207 | 1.150x |
| mixtral_8x7b_t16k | 48.0525 | 43.2931 | 1.110x |
| mixtral_8x7b_t1k | 4.5053 | 3.9561 | 1.139x |
| mixtral_8x7b_t2k | 8.1607 | 7.0943 | 1.150x |
| mixtral_8x7b_t8k | 26.1540 | 22.9652 | 1.139x |
| qwen3_235b_a22b | 8.6774 | 7.7575 | 1.119x |
| qwen3_235b_a22b_t16k | 29.1711 | 26.4231 | 1.104x |
| qwen3_235b_a22b_t1k | 4.4725 | 4.1363 | 1.081x |
| qwen3_235b_a22b_t2k | 5.8799 | 5.3269 | 1.104x |
| qwen3_235b_a22b_t8k | 15.5532 | 13.8883 | 1.120x |
| qwen3_30b_a3b | 4.3299 | 4.1183 | 1.051x |
| qwen3_30b_a3b_t16k | 11.8210 | 10.8392 | 1.091x |
| qwen3_30b_a3b_t1k | 2.2358 | 2.2044 | 1.014x |
| qwen3_30b_a3b_t2k | 2.9194 | 2.7700 | 1.054x |
| qwen3_30b_a3b_t8k | 5.9275 | 5.5369 | 1.071x |
| qwen3_5_122b_a10b | 6.1056 | 5.7788 | 1.057x |
| qwen3_5_122b_a10b_t16k | 16.2179 | 15.2385 | 1.064x |
| qwen3_5_122b_a10b_t1k | 4.3247 | 4.1809 | 1.034x |
| qwen3_5_122b_a10b_t2k | 4.4210 | 4.2775 | 1.034x |
| qwen3_5_122b_a10b_t8k | 9.1299 | 8.7058 | 1.049x |
| qwen3_5_35b_a3b | 3.9095 | 3.8135 | 1.025x |
| qwen3_5_35b_a3b_t16k | 9.5856 | 9.3687 | 1.023x |
| qwen3_5_35b_a3b_t1k | 3.0178 | 2.8525 | 1.058x |
| qwen3_5_35b_a3b_t2k | 2.8690 | 2.8486 | 1.007x |
| qwen3_5_35b_a3b_t8k | 5.8102 | 5.6546 | 1.028x |

### single_A_to_B_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 9.3069 | 6.5181 | 1.428x |
| llama4_scout_t16k | 18.9602 | 15.7668 | 1.203x |
| llama4_scout_t1k | 8.0804 | 4.8723 | 1.658x |
| llama4_scout_t2k | 8.1565 | 5.2042 | 1.567x |
| llama4_scout_t8k | 11.0901 | 8.9342 | 1.241x |
| mixtral_8x22b | 20.5326 | 16.8873 | 1.216x |
| mixtral_8x22b_t16k | 66.8486 | 59.6560 | 1.121x |
| mixtral_8x22b_t1k | 13.9139 | 7.3936 | 1.882x |
| mixtral_8x22b_t2k | 15.8553 | 10.0194 | 1.582x |
| mixtral_8x22b_t8k | 36.2367 | 31.4093 | 1.154x |
| mixtral_8x7b | 12.6824 | 10.0083 | 1.267x |
| mixtral_8x7b_t16k | 40.6529 | 37.3351 | 1.089x |
| mixtral_8x7b_t1k | 8.5571 | 4.2855 | 1.997x |
| mixtral_8x7b_t2k | 9.7304 | 5.9282 | 1.641x |
| mixtral_8x7b_t8k | 22.2522 | 20.0209 | 1.111x |
| qwen3_235b_a22b | 13.2792 | 8.5099 | 1.560x |
| qwen3_235b_a22b_t16k | 32.4921 | 19.9098 | 1.632x |
| qwen3_235b_a22b_t1k | 8.1781 | 6.2938 | 1.299x |
| qwen3_235b_a22b_t2k | 9.7046 | 6.7770 | 1.432x |
| qwen3_235b_a22b_t8k | 19.3369 | 12.3377 | 1.567x |
| qwen3_30b_a3b | 5.4342 | 3.3372 | 1.628x |
| qwen3_30b_a3b_t16k | 13.7327 | 7.8774 | 1.743x |
| qwen3_30b_a3b_t1k | 3.3983 | 2.2446 | 1.514x |
| qwen3_30b_a3b_t2k | 4.0971 | 2.8200 | 1.453x |
| qwen3_30b_a3b_t8k | 8.1039 | 4.6973 | 1.725x |
| qwen3_5_122b_a10b | 11.3161 | 7.3093 | 1.548x |
| qwen3_5_122b_a10b_t16k | 24.0585 | 13.6278 | 1.765x |
| qwen3_5_122b_a10b_t1k | 9.0971 | 6.4422 | 1.412x |
| qwen3_5_122b_a10b_t2k | 9.1602 | 6.4959 | 1.410x |
| qwen3_5_122b_a10b_t8k | 15.5947 | 9.3696 | 1.664x |
| qwen3_5_35b_a3b | 6.1943 | 3.7300 | 1.661x |
| qwen3_5_35b_a3b_t16k | 13.8015 | 7.5141 | 1.837x |
| qwen3_5_35b_a3b_t1k | 4.8052 | 3.0333 | 1.584x |
| qwen3_5_35b_a3b_t2k | 4.8360 | 3.0674 | 1.577x |
| qwen3_5_35b_a3b_t8k | 8.7722 | 5.0618 | 1.733x |

### single_A_to_B_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 2.0261 | 1.6512 | 1.227x |
| llama4_scout_t16k | 4.1284 | 4.1660 | 0.991x |
| llama4_scout_t1k | 0.8775 | 0.9033 | 0.971x |
| llama4_scout_t2k | 1.1198 | 1.0149 | 1.103x |
| llama4_scout_t8k | 2.2280 | 2.3078 | 0.965x |
| mixtral_8x22b | 4.7161 | 4.4458 | 1.061x |
| mixtral_8x22b_t16k | 19.4566 | 17.9336 | 1.085x |
| mixtral_8x22b_t1k | 1.9665 | 1.9085 | 1.030x |
| mixtral_8x22b_t2k | 2.4621 | 2.4417 | 1.008x |
| mixtral_8x22b_t8k | 8.7084 | 9.1148 | 0.955x |
| mixtral_8x7b | 2.7105 | 2.6362 | 1.028x |
| mixtral_8x7b_t16k | 10.4889 | 9.9610 | 1.053x |
| mixtral_8x7b_t1k | 1.1092 | 0.9523 | 1.165x |
| mixtral_8x7b_t2k | 1.4761 | 1.4180 | 1.041x |
| mixtral_8x7b_t8k | 4.9893 | 4.9302 | 1.012x |
| qwen3_235b_a22b | 2.0436 | 1.9827 | 1.031x |
| qwen3_235b_a22b_t16k | 5.8934 | 5.7369 | 1.027x |
| qwen3_235b_a22b_t1k | 1.2253 | 1.3729 | 0.892x |
| qwen3_235b_a22b_t2k | 1.4616 | 1.4203 | 1.029x |
| qwen3_235b_a22b_t8k | 3.2686 | 3.2445 | 1.007x |
| qwen3_30b_a3b | 0.8334 | 0.8124 | 1.026x |
| qwen3_30b_a3b_t16k | 2.0625 | 1.9143 | 1.077x |
| qwen3_30b_a3b_t1k | 0.4996 | 0.5437 | 0.919x |
| qwen3_30b_a3b_t2k | 0.6157 | 0.6882 | 0.895x |
| qwen3_30b_a3b_t8k | 1.1977 | 1.1450 | 1.046x |
| qwen3_5_122b_a10b | 1.5893 | 1.6729 | 0.950x |
| qwen3_5_122b_a10b_t16k | 3.7168 | 3.6476 | 1.019x |
| qwen3_5_122b_a10b_t1k | 1.2684 | 1.3397 | 0.947x |
| qwen3_5_122b_a10b_t2k | 1.3118 | 1.3713 | 0.957x |
| qwen3_5_122b_a10b_t8k | 2.2904 | 2.2577 | 1.014x |
| qwen3_5_35b_a3b | 0.9034 | 0.9778 | 0.924x |
| qwen3_5_35b_a3b_t16k | 1.9372 | 1.8717 | 1.035x |
| qwen3_5_35b_a3b_t1k | 0.6851 | 0.7715 | 0.888x |
| qwen3_5_35b_a3b_t2k | 0.7215 | 0.7968 | 0.905x |
| qwen3_5_35b_a3b_t8k | 1.2375 | 1.2165 | 1.017x |

### single_A_to_B_fwd_plus_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 11.3330 | 8.1693 | 1.387x |
| llama4_scout_t16k | 23.0886 | 19.9328 | 1.158x |
| llama4_scout_t1k | 8.9579 | 5.7756 | 1.551x |
| llama4_scout_t2k | 9.2763 | 6.2191 | 1.492x |
| llama4_scout_t8k | 13.3181 | 11.2420 | 1.185x |
| mixtral_8x22b | 25.2487 | 21.3331 | 1.184x |
| mixtral_8x22b_t16k | 86.3052 | 77.5896 | 1.112x |
| mixtral_8x22b_t1k | 15.8804 | 9.3021 | 1.707x |
| mixtral_8x22b_t2k | 18.3174 | 12.4611 | 1.470x |
| mixtral_8x22b_t8k | 44.9451 | 40.5241 | 1.109x |
| mixtral_8x7b | 15.3929 | 12.6445 | 1.217x |
| mixtral_8x7b_t16k | 51.1418 | 47.2961 | 1.081x |
| mixtral_8x7b_t1k | 9.6663 | 5.2378 | 1.845x |
| mixtral_8x7b_t2k | 11.2065 | 7.3462 | 1.525x |
| mixtral_8x7b_t8k | 27.2415 | 24.9511 | 1.092x |
| qwen3_235b_a22b | 15.3228 | 10.4926 | 1.460x |
| qwen3_235b_a22b_t16k | 38.3855 | 25.6467 | 1.497x |
| qwen3_235b_a22b_t1k | 9.4034 | 7.6667 | 1.227x |
| qwen3_235b_a22b_t2k | 11.1662 | 8.1973 | 1.362x |
| qwen3_235b_a22b_t8k | 22.6055 | 15.5822 | 1.451x |
| qwen3_30b_a3b | 6.2676 | 4.1496 | 1.510x |
| qwen3_30b_a3b_t16k | 15.7952 | 9.7917 | 1.613x |
| qwen3_30b_a3b_t1k | 3.8979 | 2.7883 | 1.398x |
| qwen3_30b_a3b_t2k | 4.7128 | 3.5082 | 1.343x |
| qwen3_30b_a3b_t8k | 9.3016 | 5.8423 | 1.592x |
| qwen3_5_122b_a10b | 12.9054 | 8.9822 | 1.437x |
| qwen3_5_122b_a10b_t16k | 27.7753 | 17.2754 | 1.608x |
| qwen3_5_122b_a10b_t1k | 10.3655 | 7.7819 | 1.332x |
| qwen3_5_122b_a10b_t2k | 10.4720 | 7.8672 | 1.331x |
| qwen3_5_122b_a10b_t8k | 17.8851 | 11.6273 | 1.538x |
| qwen3_5_35b_a3b | 7.0977 | 4.7078 | 1.508x |
| qwen3_5_35b_a3b_t16k | 15.7387 | 9.3858 | 1.677x |
| qwen3_5_35b_a3b_t1k | 5.4903 | 3.8048 | 1.443x |
| qwen3_5_35b_a3b_t2k | 5.5575 | 3.8642 | 1.438x |
| qwen3_5_35b_a3b_t8k | 10.0097 | 6.2783 | 1.594x |

### single_A_to_C_overall_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 9.3069 | 6.2430 | 1.491x |
| llama4_scout_t16k | 18.9602 | 13.3283 | 1.423x |
| llama4_scout_t1k | 8.0804 | 4.8156 | 1.678x |
| llama4_scout_t2k | 8.1565 | 5.0816 | 1.605x |
| llama4_scout_t8k | 11.0901 | 8.0718 | 1.374x |
| mixtral_8x22b | 20.5326 | 14.4252 | 1.423x |
| mixtral_8x22b_t16k | 66.8486 | 50.5258 | 1.323x |
| mixtral_8x22b_t1k | 13.9139 | 7.0614 | 1.970x |
| mixtral_8x22b_t2k | 15.8553 | 8.9514 | 1.771x |
| mixtral_8x22b_t8k | 36.2367 | 27.3252 | 1.326x |
| mixtral_8x7b | 12.6824 | 8.5750 | 1.479x |
| mixtral_8x7b_t16k | 40.6529 | 32.7125 | 1.243x |
| mixtral_8x7b_t1k | 8.5571 | 4.0190 | 2.129x |
| mixtral_8x7b_t2k | 9.7304 | 5.5159 | 1.764x |
| mixtral_8x7b_t8k | 22.2522 | 17.1703 | 1.296x |
| qwen3_235b_a22b | 13.2792 | 8.1303 | 1.633x |
| qwen3_235b_a22b_t16k | 32.4921 | 18.6793 | 1.739x |
| qwen3_235b_a22b_t1k | 8.1781 | 6.3137 | 1.295x |
| qwen3_235b_a22b_t2k | 9.7046 | 6.7049 | 1.447x |
| qwen3_235b_a22b_t8k | 19.3369 | 11.5693 | 1.671x |
| qwen3_30b_a3b | 5.4342 | 3.1990 | 1.699x |
| qwen3_30b_a3b_t16k | 13.7327 | 7.4571 | 1.842x |
| qwen3_30b_a3b_t1k | 3.3983 | 2.2264 | 1.526x |
| qwen3_30b_a3b_t2k | 4.0971 | 2.7568 | 1.486x |
| qwen3_30b_a3b_t8k | 8.1039 | 4.4183 | 1.834x |
| qwen3_5_122b_a10b | 11.3161 | 7.1244 | 1.588x |
| qwen3_5_122b_a10b_t16k | 24.0585 | 13.0690 | 1.841x |
| qwen3_5_122b_a10b_t1k | 9.0971 | 6.4092 | 1.419x |
| qwen3_5_122b_a10b_t2k | 9.1602 | 6.4661 | 1.417x |
| qwen3_5_122b_a10b_t8k | 15.5947 | 9.1572 | 1.703x |
| qwen3_5_35b_a3b | 6.1943 | 3.6391 | 1.702x |
| qwen3_5_35b_a3b_t16k | 13.8015 | 7.4054 | 1.864x |
| qwen3_5_35b_a3b_t1k | 4.8052 | 2.9865 | 1.609x |
| qwen3_5_35b_a3b_t2k | 4.8360 | 3.0308 | 1.596x |
| qwen3_5_35b_a3b_t8k | 8.7722 | 4.9564 | 1.770x |

### single_A_to_C_overall_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 2.0261 | 1.6592 | 1.221x |
| llama4_scout_t16k | 4.1284 | 4.1511 | 0.995x |
| llama4_scout_t1k | 0.8775 | 0.9054 | 0.969x |
| llama4_scout_t2k | 1.1198 | 1.0151 | 1.103x |
| llama4_scout_t8k | 2.2280 | 2.3165 | 0.962x |
| mixtral_8x22b | 4.7161 | 4.4430 | 1.061x |
| mixtral_8x22b_t16k | 19.4566 | 17.6032 | 1.105x |
| mixtral_8x22b_t1k | 1.9665 | 1.6943 | 1.161x |
| mixtral_8x22b_t2k | 2.4621 | 2.4663 | 0.998x |
| mixtral_8x22b_t8k | 8.7084 | 8.9820 | 0.970x |
| mixtral_8x7b | 2.7105 | 2.6418 | 1.026x |
| mixtral_8x7b_t16k | 10.4889 | 10.8623 | 0.966x |
| mixtral_8x7b_t1k | 1.1092 | 0.9502 | 1.167x |
| mixtral_8x7b_t2k | 1.4761 | 1.4213 | 1.039x |
| mixtral_8x7b_t8k | 4.9893 | 4.9144 | 1.015x |
| qwen3_235b_a22b | 2.0436 | 1.9839 | 1.030x |
| qwen3_235b_a22b_t16k | 5.8934 | 5.7024 | 1.033x |
| qwen3_235b_a22b_t1k | 1.2253 | 1.3724 | 0.893x |
| qwen3_235b_a22b_t2k | 1.4616 | 1.4152 | 1.033x |
| qwen3_235b_a22b_t8k | 3.2686 | 3.2098 | 1.018x |
| qwen3_30b_a3b | 0.8334 | 0.8104 | 1.028x |
| qwen3_30b_a3b_t16k | 2.0625 | 1.9184 | 1.075x |
| qwen3_30b_a3b_t1k | 0.4996 | 0.5454 | 0.916x |
| qwen3_30b_a3b_t2k | 0.6157 | 0.6632 | 0.928x |
| qwen3_30b_a3b_t8k | 1.1977 | 1.1460 | 1.045x |
| qwen3_5_122b_a10b | 1.5893 | 1.6650 | 0.955x |
| qwen3_5_122b_a10b_t16k | 3.7168 | 3.6622 | 1.015x |
| qwen3_5_122b_a10b_t1k | 1.2684 | 1.3296 | 0.954x |
| qwen3_5_122b_a10b_t2k | 1.3118 | 1.3625 | 0.963x |
| qwen3_5_122b_a10b_t8k | 2.2904 | 2.2619 | 1.013x |
| qwen3_5_35b_a3b | 0.9034 | 0.9697 | 0.932x |
| qwen3_5_35b_a3b_t16k | 1.9372 | 1.8697 | 1.036x |
| qwen3_5_35b_a3b_t1k | 0.6851 | 0.7664 | 0.894x |
| qwen3_5_35b_a3b_t2k | 0.7215 | 0.7858 | 0.918x |
| qwen3_5_35b_a3b_t8k | 1.2375 | 1.2043 | 1.028x |

### single_A_to_C_overall_fwd_plus_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 11.3330 | 7.9022 | 1.434x |
| llama4_scout_t16k | 23.0886 | 17.4794 | 1.321x |
| llama4_scout_t1k | 8.9579 | 5.7210 | 1.566x |
| llama4_scout_t2k | 9.2763 | 6.0967 | 1.522x |
| llama4_scout_t8k | 13.3181 | 10.3883 | 1.282x |
| mixtral_8x22b | 25.2487 | 18.8682 | 1.338x |
| mixtral_8x22b_t16k | 86.3052 | 68.1290 | 1.267x |
| mixtral_8x22b_t1k | 15.8804 | 8.7557 | 1.814x |
| mixtral_8x22b_t2k | 18.3174 | 11.4177 | 1.604x |
| mixtral_8x22b_t8k | 44.9451 | 36.3072 | 1.238x |
| mixtral_8x7b | 15.3929 | 11.2168 | 1.372x |
| mixtral_8x7b_t16k | 51.1418 | 43.5748 | 1.174x |
| mixtral_8x7b_t1k | 9.6663 | 4.9692 | 1.945x |
| mixtral_8x7b_t2k | 11.2065 | 6.9372 | 1.615x |
| mixtral_8x7b_t8k | 27.2415 | 22.0847 | 1.234x |
| qwen3_235b_a22b | 15.3228 | 10.1142 | 1.515x |
| qwen3_235b_a22b_t16k | 38.3855 | 24.3817 | 1.574x |
| qwen3_235b_a22b_t1k | 9.4034 | 7.6861 | 1.223x |
| qwen3_235b_a22b_t2k | 11.1662 | 8.1201 | 1.375x |
| qwen3_235b_a22b_t8k | 22.6055 | 14.7791 | 1.530x |
| qwen3_30b_a3b | 6.2676 | 4.0094 | 1.563x |
| qwen3_30b_a3b_t16k | 15.7952 | 9.3755 | 1.685x |
| qwen3_30b_a3b_t1k | 3.8979 | 2.7718 | 1.406x |
| qwen3_30b_a3b_t2k | 4.7128 | 3.4200 | 1.378x |
| qwen3_30b_a3b_t8k | 9.3016 | 5.5643 | 1.672x |
| qwen3_5_122b_a10b | 12.9054 | 8.7894 | 1.468x |
| qwen3_5_122b_a10b_t16k | 27.7753 | 16.7312 | 1.660x |
| qwen3_5_122b_a10b_t1k | 10.3655 | 7.7388 | 1.339x |
| qwen3_5_122b_a10b_t2k | 10.4720 | 7.8286 | 1.338x |
| qwen3_5_122b_a10b_t8k | 17.8851 | 11.4191 | 1.566x |
| qwen3_5_35b_a3b | 7.0977 | 4.6088 | 1.540x |
| qwen3_5_35b_a3b_t16k | 15.7387 | 9.2751 | 1.697x |
| qwen3_5_35b_a3b_t1k | 5.4903 | 3.7529 | 1.463x |
| qwen3_5_35b_a3b_t2k | 5.5575 | 3.8166 | 1.456x |
| qwen3_5_35b_a3b_t8k | 10.0097 | 6.1607 | 1.625x |

### single_A_to_W0_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 2.0261 | 1.6692 | 1.214x |
| llama4_scout_t16k | 4.1284 | 4.2066 | 0.981x |
| llama4_scout_t1k | 0.8775 | 0.9065 | 0.968x |
| llama4_scout_t2k | 1.1198 | 1.0233 | 1.094x |
| llama4_scout_t8k | 2.2280 | 2.4336 | 0.916x |
| mixtral_8x22b | 4.7161 | 4.5096 | 1.046x |
| mixtral_8x22b_t16k | 19.4566 | 18.2087 | 1.069x |
| mixtral_8x22b_t1k | 1.9665 | 1.9272 | 1.020x |
| mixtral_8x22b_t2k | 2.4621 | 2.4673 | 0.998x |
| mixtral_8x22b_t8k | 8.7084 | 9.6018 | 0.907x |
| mixtral_8x7b | 2.7105 | 2.6446 | 1.025x |
| mixtral_8x7b_t16k | 10.4889 | 10.2615 | 1.022x |
| mixtral_8x7b_t1k | 1.1092 | 0.9626 | 1.152x |
| mixtral_8x7b_t2k | 1.4761 | 1.4272 | 1.034x |
| mixtral_8x7b_t8k | 4.9893 | 4.8910 | 1.020x |
| qwen3_235b_a22b | 2.0436 | 2.0843 | 0.980x |
| qwen3_235b_a22b_t16k | 5.8934 | 6.1701 | 0.955x |
| qwen3_235b_a22b_t1k | 1.2253 | 1.3817 | 0.887x |
| qwen3_235b_a22b_t2k | 1.4616 | 1.4550 | 1.005x |
| qwen3_235b_a22b_t8k | 3.2686 | 3.4315 | 0.953x |
| qwen3_30b_a3b | 0.8334 | 0.8577 | 0.972x |
| qwen3_30b_a3b_t16k | 2.0625 | 2.0373 | 1.012x |
| qwen3_30b_a3b_t1k | 0.4996 | 0.5598 | 0.892x |
| qwen3_30b_a3b_t2k | 0.6157 | 0.7050 | 0.873x |
| qwen3_30b_a3b_t8k | 1.1977 | 1.2145 | 0.986x |
| qwen3_5_122b_a10b | 1.5893 | 1.7106 | 0.929x |
| qwen3_5_122b_a10b_t16k | 3.7168 | 3.8180 | 0.973x |
| qwen3_5_122b_a10b_t1k | 1.2684 | 1.3689 | 0.927x |
| qwen3_5_122b_a10b_t2k | 1.3118 | 1.4019 | 0.936x |
| qwen3_5_122b_a10b_t8k | 2.2904 | 2.3871 | 0.960x |
| qwen3_5_35b_a3b | 0.9034 | 1.0043 | 0.900x |
| qwen3_5_35b_a3b_t16k | 1.9372 | 1.9814 | 0.978x |
| qwen3_5_35b_a3b_t1k | 0.6851 | 0.7961 | 0.861x |
| qwen3_5_35b_a3b_t2k | 0.7215 | 0.8183 | 0.882x |
| qwen3_5_35b_a3b_t8k | 1.2375 | 1.2820 | 0.965x |

### single_A_to_W1_total_mlp1_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 2.0261 | 1.6663 | 1.216x |
| llama4_scout_t16k | 4.1284 | 4.1826 | 0.987x |
| llama4_scout_t1k | 0.8775 | 0.9064 | 0.968x |
| llama4_scout_t2k | 1.1198 | 1.0230 | 1.095x |
| llama4_scout_t8k | 2.2280 | 2.3411 | 0.952x |
| mixtral_8x22b | 4.7161 | 4.4915 | 1.050x |
| mixtral_8x22b_t16k | 19.4566 | 18.6831 | 1.041x |
| mixtral_8x22b_t1k | 1.9665 | 1.7212 | 1.143x |
| mixtral_8x22b_t2k | 2.4621 | 2.4543 | 1.003x |
| mixtral_8x22b_t8k | 8.7084 | 9.3613 | 0.930x |
| mixtral_8x7b | 2.7105 | 2.6315 | 1.030x |
| mixtral_8x7b_t16k | 10.4889 | 10.2082 | 1.027x |
| mixtral_8x7b_t1k | 1.1092 | 0.9375 | 1.183x |
| mixtral_8x7b_t2k | 1.4761 | 1.4249 | 1.036x |
| mixtral_8x7b_t8k | 4.9893 | 4.8453 | 1.030x |
| qwen3_235b_a22b | 2.0436 | 2.0796 | 0.983x |
| qwen3_235b_a22b_t16k | 5.8934 | 6.1973 | 0.951x |
| qwen3_235b_a22b_t1k | 1.2253 | 1.3838 | 0.885x |
| qwen3_235b_a22b_t2k | 1.4616 | 1.4538 | 1.005x |
| qwen3_235b_a22b_t8k | 3.2686 | 3.4130 | 0.958x |
| qwen3_30b_a3b | 0.8334 | 0.8558 | 0.974x |
| qwen3_30b_a3b_t16k | 2.0625 | 2.0269 | 1.018x |
| qwen3_30b_a3b_t1k | 0.4996 | 0.5597 | 0.893x |
| qwen3_30b_a3b_t2k | 0.6157 | 0.6988 | 0.881x |
| qwen3_30b_a3b_t8k | 1.1977 | 1.2108 | 0.989x |
| qwen3_5_122b_a10b | 1.5893 | 1.7119 | 0.928x |
| qwen3_5_122b_a10b_t16k | 3.7168 | 3.7969 | 0.979x |
| qwen3_5_122b_a10b_t1k | 1.2684 | 1.3733 | 0.924x |
| qwen3_5_122b_a10b_t2k | 1.3118 | 1.4026 | 0.935x |
| qwen3_5_122b_a10b_t8k | 2.2904 | 2.3559 | 0.972x |
| qwen3_5_35b_a3b | 0.9034 | 0.9971 | 0.906x |
| qwen3_5_35b_a3b_t16k | 1.9372 | 1.9721 | 0.982x |
| qwen3_5_35b_a3b_t1k | 0.6851 | 0.7871 | 0.870x |
| qwen3_5_35b_a3b_t2k | 0.7215 | 0.8074 | 0.894x |
| qwen3_5_35b_a3b_t8k | 1.2375 | 1.2769 | 0.969x |

### single_B_to_C_2sm_stage_tuning_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 6.5181 | 6.2430 | 1.044x |
| llama4_scout_t16k | 15.7668 | 13.3283 | 1.183x |
| llama4_scout_t1k | 4.8723 | 4.8156 | 1.012x |
| llama4_scout_t2k | 5.2042 | 5.0816 | 1.024x |
| llama4_scout_t8k | 8.9342 | 8.0718 | 1.107x |
| mixtral_8x22b | 16.8873 | 14.4252 | 1.171x |
| mixtral_8x22b_t16k | 59.6560 | 50.5258 | 1.181x |
| mixtral_8x22b_t1k | 7.3936 | 7.0614 | 1.047x |
| mixtral_8x22b_t2k | 10.0194 | 8.9514 | 1.119x |
| mixtral_8x22b_t8k | 31.4093 | 27.3252 | 1.149x |
| mixtral_8x7b | 10.0083 | 8.5750 | 1.167x |
| mixtral_8x7b_t16k | 37.3351 | 32.7125 | 1.141x |
| mixtral_8x7b_t1k | 4.2855 | 4.0190 | 1.066x |
| mixtral_8x7b_t2k | 5.9282 | 5.5159 | 1.075x |
| mixtral_8x7b_t8k | 20.0209 | 17.1703 | 1.166x |
| qwen3_235b_a22b | 8.5099 | 8.1303 | 1.047x |
| qwen3_235b_a22b_t16k | 19.9098 | 18.6793 | 1.066x |
| qwen3_235b_a22b_t1k | 6.2938 | 6.3137 | 0.997x |
| qwen3_235b_a22b_t2k | 6.7770 | 6.7049 | 1.011x |
| qwen3_235b_a22b_t8k | 12.3377 | 11.5693 | 1.066x |
| qwen3_30b_a3b | 3.3372 | 3.1990 | 1.043x |
| qwen3_30b_a3b_t16k | 7.8774 | 7.4571 | 1.056x |
| qwen3_30b_a3b_t1k | 2.2446 | 2.2264 | 1.008x |
| qwen3_30b_a3b_t2k | 2.8200 | 2.7568 | 1.023x |
| qwen3_30b_a3b_t8k | 4.6973 | 4.4183 | 1.063x |
| qwen3_5_122b_a10b | 7.3093 | 7.1244 | 1.026x |
| qwen3_5_122b_a10b_t16k | 13.6278 | 13.0690 | 1.043x |
| qwen3_5_122b_a10b_t1k | 6.4422 | 6.4092 | 1.005x |
| qwen3_5_122b_a10b_t2k | 6.4959 | 6.4661 | 1.005x |
| qwen3_5_122b_a10b_t8k | 9.3696 | 9.1572 | 1.023x |
| qwen3_5_35b_a3b | 3.7300 | 3.6391 | 1.025x |
| qwen3_5_35b_a3b_t16k | 7.5141 | 7.4054 | 1.015x |
| qwen3_5_35b_a3b_t1k | 3.0333 | 2.9865 | 1.016x |
| qwen3_5_35b_a3b_t2k | 3.0674 | 3.0308 | 1.012x |
| qwen3_5_35b_a3b_t8k | 5.0618 | 4.9564 | 1.021x |

### single_B_to_C_2sm_stage_tuning_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 1.6512 | 1.6592 | 0.995x |
| llama4_scout_t16k | 4.1660 | 4.1511 | 1.004x |
| llama4_scout_t1k | 0.9033 | 0.9054 | 0.998x |
| llama4_scout_t2k | 1.0149 | 1.0151 | 1.000x |
| llama4_scout_t8k | 2.3078 | 2.3165 | 0.996x |
| mixtral_8x22b | 4.4458 | 4.4430 | 1.001x |
| mixtral_8x22b_t16k | 17.9336 | 17.6032 | 1.019x |
| mixtral_8x22b_t1k | 1.9085 | 1.6943 | 1.126x |
| mixtral_8x22b_t2k | 2.4417 | 2.4663 | 0.990x |
| mixtral_8x22b_t8k | 9.1148 | 8.9820 | 1.015x |
| mixtral_8x7b | 2.6362 | 2.6418 | 0.998x |
| mixtral_8x7b_t16k | 9.9610 | 10.8623 | 0.917x |
| mixtral_8x7b_t1k | 0.9523 | 0.9502 | 1.002x |
| mixtral_8x7b_t2k | 1.4180 | 1.4213 | 0.998x |
| mixtral_8x7b_t8k | 4.9302 | 4.9144 | 1.003x |
| qwen3_235b_a22b | 1.9827 | 1.9839 | 0.999x |
| qwen3_235b_a22b_t16k | 5.7369 | 5.7024 | 1.006x |
| qwen3_235b_a22b_t1k | 1.3729 | 1.3724 | 1.000x |
| qwen3_235b_a22b_t2k | 1.4203 | 1.4152 | 1.004x |
| qwen3_235b_a22b_t8k | 3.2445 | 3.2098 | 1.011x |
| qwen3_30b_a3b | 0.8124 | 0.8104 | 1.002x |
| qwen3_30b_a3b_t16k | 1.9143 | 1.9184 | 0.998x |
| qwen3_30b_a3b_t1k | 0.5437 | 0.5454 | 0.997x |
| qwen3_30b_a3b_t2k | 0.6882 | 0.6632 | 1.038x |
| qwen3_30b_a3b_t8k | 1.1450 | 1.1460 | 0.999x |
| qwen3_5_122b_a10b | 1.6729 | 1.6650 | 1.005x |
| qwen3_5_122b_a10b_t16k | 3.6476 | 3.6622 | 0.996x |
| qwen3_5_122b_a10b_t1k | 1.3397 | 1.3296 | 1.008x |
| qwen3_5_122b_a10b_t2k | 1.3713 | 1.3625 | 1.006x |
| qwen3_5_122b_a10b_t8k | 2.2577 | 2.2619 | 0.998x |
| qwen3_5_35b_a3b | 0.9778 | 0.9697 | 1.008x |
| qwen3_5_35b_a3b_t16k | 1.8717 | 1.8697 | 1.001x |
| qwen3_5_35b_a3b_t1k | 0.7715 | 0.7664 | 1.007x |
| qwen3_5_35b_a3b_t2k | 0.7968 | 0.7858 | 1.014x |
| qwen3_5_35b_a3b_t8k | 1.2165 | 1.2043 | 1.010x |

### single_B_to_C_2sm_stage_tuning_fwd_plus_bwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 8.1693 | 7.9022 | 1.034x |
| llama4_scout_t16k | 19.9328 | 17.4794 | 1.140x |
| llama4_scout_t1k | 5.7756 | 5.7210 | 1.010x |
| llama4_scout_t2k | 6.2191 | 6.0967 | 1.020x |
| llama4_scout_t8k | 11.2420 | 10.3883 | 1.082x |
| mixtral_8x22b | 21.3331 | 18.8682 | 1.131x |
| mixtral_8x22b_t16k | 77.5896 | 68.1290 | 1.139x |
| mixtral_8x22b_t1k | 9.3021 | 8.7557 | 1.062x |
| mixtral_8x22b_t2k | 12.4611 | 11.4177 | 1.091x |
| mixtral_8x22b_t8k | 40.5241 | 36.3072 | 1.116x |
| mixtral_8x7b | 12.6445 | 11.2168 | 1.127x |
| mixtral_8x7b_t16k | 47.2961 | 43.5748 | 1.085x |
| mixtral_8x7b_t1k | 5.2378 | 4.9692 | 1.054x |
| mixtral_8x7b_t2k | 7.3462 | 6.9372 | 1.059x |
| mixtral_8x7b_t8k | 24.9511 | 22.0847 | 1.130x |
| qwen3_235b_a22b | 10.4926 | 10.1142 | 1.037x |
| qwen3_235b_a22b_t16k | 25.6467 | 24.3817 | 1.052x |
| qwen3_235b_a22b_t1k | 7.6667 | 7.6861 | 0.997x |
| qwen3_235b_a22b_t2k | 8.1973 | 8.1201 | 1.010x |
| qwen3_235b_a22b_t8k | 15.5822 | 14.7791 | 1.054x |
| qwen3_30b_a3b | 4.1496 | 4.0094 | 1.035x |
| qwen3_30b_a3b_t16k | 9.7917 | 9.3755 | 1.044x |
| qwen3_30b_a3b_t1k | 2.7883 | 2.7718 | 1.006x |
| qwen3_30b_a3b_t2k | 3.5082 | 3.4200 | 1.026x |
| qwen3_30b_a3b_t8k | 5.8423 | 5.5643 | 1.050x |
| qwen3_5_122b_a10b | 8.9822 | 8.7894 | 1.022x |
| qwen3_5_122b_a10b_t16k | 17.2754 | 16.7312 | 1.033x |
| qwen3_5_122b_a10b_t1k | 7.7819 | 7.7388 | 1.006x |
| qwen3_5_122b_a10b_t2k | 7.8672 | 7.8286 | 1.005x |
| qwen3_5_122b_a10b_t8k | 11.6273 | 11.4191 | 1.018x |
| qwen3_5_35b_a3b | 4.7078 | 4.6088 | 1.021x |
| qwen3_5_35b_a3b_t16k | 9.3858 | 9.2751 | 1.012x |
| qwen3_5_35b_a3b_t1k | 3.8048 | 3.7529 | 1.014x |
| qwen3_5_35b_a3b_t2k | 3.8642 | 3.8166 | 1.012x |
| qwen3_5_35b_a3b_t8k | 6.2783 | 6.1607 | 1.019x |

### single_W0_to_W1_warp_specialization_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 1.6692 | 1.6663 | 1.002x |
| llama4_scout_t16k | 4.2066 | 4.1826 | 1.006x |
| llama4_scout_t1k | 0.9065 | 0.9064 | 1.000x |
| llama4_scout_t2k | 1.0233 | 1.0230 | 1.000x |
| llama4_scout_t8k | 2.4336 | 2.3411 | 1.039x |
| mixtral_8x22b | 4.5096 | 4.4915 | 1.004x |
| mixtral_8x22b_t16k | 18.2087 | 18.6831 | 0.975x |
| mixtral_8x22b_t1k | 1.9272 | 1.7212 | 1.120x |
| mixtral_8x22b_t2k | 2.4673 | 2.4543 | 1.005x |
| mixtral_8x22b_t8k | 9.6018 | 9.3613 | 1.026x |
| mixtral_8x7b | 2.6446 | 2.6315 | 1.005x |
| mixtral_8x7b_t16k | 10.2615 | 10.2082 | 1.005x |
| mixtral_8x7b_t1k | 0.9626 | 0.9375 | 1.027x |
| mixtral_8x7b_t2k | 1.4272 | 1.4249 | 1.002x |
| mixtral_8x7b_t8k | 4.8910 | 4.8453 | 1.009x |
| qwen3_235b_a22b | 2.0843 | 2.0796 | 1.002x |
| qwen3_235b_a22b_t16k | 6.1701 | 6.1973 | 0.996x |
| qwen3_235b_a22b_t1k | 1.3817 | 1.3838 | 0.999x |
| qwen3_235b_a22b_t2k | 1.4550 | 1.4538 | 1.001x |
| qwen3_235b_a22b_t8k | 3.4315 | 3.4130 | 1.005x |
| qwen3_30b_a3b | 0.8577 | 0.8558 | 1.002x |
| qwen3_30b_a3b_t16k | 2.0373 | 2.0269 | 1.005x |
| qwen3_30b_a3b_t1k | 0.5598 | 0.5597 | 1.000x |
| qwen3_30b_a3b_t2k | 0.7050 | 0.6988 | 1.009x |
| qwen3_30b_a3b_t8k | 1.2145 | 1.2108 | 1.003x |
| qwen3_5_122b_a10b | 1.7106 | 1.7119 | 0.999x |
| qwen3_5_122b_a10b_t16k | 3.8180 | 3.7969 | 1.006x |
| qwen3_5_122b_a10b_t1k | 1.3689 | 1.3733 | 0.997x |
| qwen3_5_122b_a10b_t2k | 1.4019 | 1.4026 | 0.999x |
| qwen3_5_122b_a10b_t8k | 2.3871 | 2.3559 | 1.013x |
| qwen3_5_35b_a3b | 1.0043 | 0.9971 | 1.007x |
| qwen3_5_35b_a3b_t16k | 1.9814 | 1.9721 | 1.005x |
| qwen3_5_35b_a3b_t1k | 0.7961 | 0.7871 | 1.011x |
| qwen3_5_35b_a3b_t2k | 0.8183 | 0.8074 | 1.014x |
| qwen3_5_35b_a3b_t8k | 1.2820 | 1.2769 | 1.004x |

### single_W1_to_B_remaining_pipeline_fwd

| Model | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| llama4_scout | 1.6663 | 1.6512 | 1.009x |
| llama4_scout_t16k | 4.1826 | 4.1660 | 1.004x |
| llama4_scout_t1k | 0.9064 | 0.9033 | 1.003x |
| llama4_scout_t2k | 1.0230 | 1.0149 | 1.008x |
| llama4_scout_t8k | 2.3411 | 2.3078 | 1.014x |
| mixtral_8x22b | 4.4915 | 4.4458 | 1.010x |
| mixtral_8x22b_t16k | 18.6831 | 17.9336 | 1.042x |
| mixtral_8x22b_t1k | 1.7212 | 1.9085 | 0.902x |
| mixtral_8x22b_t2k | 2.4543 | 2.4417 | 1.005x |
| mixtral_8x22b_t8k | 9.3613 | 9.1148 | 1.027x |
| mixtral_8x7b | 2.6315 | 2.6362 | 0.998x |
| mixtral_8x7b_t16k | 10.2082 | 9.9610 | 1.025x |
| mixtral_8x7b_t1k | 0.9375 | 0.9523 | 0.984x |
| mixtral_8x7b_t2k | 1.4249 | 1.4180 | 1.005x |
| mixtral_8x7b_t8k | 4.8453 | 4.9302 | 0.983x |
| qwen3_235b_a22b | 2.0796 | 1.9827 | 1.049x |
| qwen3_235b_a22b_t16k | 6.1973 | 5.7369 | 1.080x |
| qwen3_235b_a22b_t1k | 1.3838 | 1.3729 | 1.008x |
| qwen3_235b_a22b_t2k | 1.4538 | 1.4203 | 1.024x |
| qwen3_235b_a22b_t8k | 3.4130 | 3.2445 | 1.052x |
| qwen3_30b_a3b | 0.8558 | 0.8124 | 1.053x |
| qwen3_30b_a3b_t16k | 2.0269 | 1.9143 | 1.059x |
| qwen3_30b_a3b_t1k | 0.5597 | 0.5437 | 1.030x |
| qwen3_30b_a3b_t2k | 0.6988 | 0.6882 | 1.015x |
| qwen3_30b_a3b_t8k | 1.2108 | 1.1450 | 1.057x |
| qwen3_5_122b_a10b | 1.7119 | 1.6729 | 1.023x |
| qwen3_5_122b_a10b_t16k | 3.7969 | 3.6476 | 1.041x |
| qwen3_5_122b_a10b_t1k | 1.3733 | 1.3397 | 1.025x |
| qwen3_5_122b_a10b_t2k | 1.4026 | 1.3713 | 1.023x |
| qwen3_5_122b_a10b_t8k | 2.3559 | 2.2577 | 1.043x |
| qwen3_5_35b_a3b | 0.9971 | 0.9778 | 1.020x |
| qwen3_5_35b_a3b_t16k | 1.9721 | 1.8717 | 1.054x |
| qwen3_5_35b_a3b_t1k | 0.7871 | 0.7715 | 1.020x |
| qwen3_5_35b_a3b_t2k | 0.8074 | 0.7968 | 1.013x |
| qwen3_5_35b_a3b_t8k | 1.2769 | 1.2165 | 1.050x |

## Interpretation constraints

- `448cbbe -> a52b5f` is the controlled warp-role package comparison:
  communication consolidation, warp-3 UMMA migration, dual epilogue
  warpgroups, and barrier updates with `AccStages=2` held constant.
- B-to-C backward speedup combines paired-CTA 2SM execution and the
  shape-selected SMEM mainloop stage depth; it is not cluster-launch
  overhead in isolation.
- Any missing or failed sweep remains absent rather than being imputed.
