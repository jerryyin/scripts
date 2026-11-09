# agentic_dataset.json characterisation (DeepSeek V4.1 tokenizer + chat_template.jinja)

Source: lightseekorg/agentic-dataset @ 2779acd48a05c211f2f82eee71bdbdf688d984ca, agentic_dataset.json, sha256 5ff62c32b10de6dc2d90e4e93ebc6ec7abe76f807a9fcdd5c0421192edc41923, 15954535 bytes. Metadata: model_path moonshotai/Kimi-K2.5, SWE-bench/SWE-smith-trajectories split tool, first_turn_length 50000, subsequent_turn_length 800, min_turns 10, max_turns 15, 71 conversations.

Token counts: `tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)` with the model dir tokenizer (default thinking_mode="thinking", drop_thinking=true), exactly what evalscope uses for client-side counts. Each prior assistant reply is budgeted as a FULL 500-token placeholder (worst case; real replies are <=500 tokens and with the deepseek_v31 reasoning parser the content part sent back as history is usually shorter because reasoning is dropped). The salt probe "[session c64-01699-ffffffff]\n" is included in turn 1. Max request = prompt + 500 output; limit = max_model_len 65536 - margin 1000 => prompt budget 64036.

## Summary

- turns per conversation: min 10, median 11, mean 12, max 15
- turn-1 tokens, Kimi (dataset prompt_tokens): min 50016, median 50112, mean 50119, max 50320
- turn-1 tokens, DeepSeek, UNTRUNCATED (CI file as-is): min 52353, median 58036, mean 57988, max 63944
- per-turn user increment, DeepSeek bare tokens (mean per conv): min 833.0909090909091, median 904.7692307692307, mean 901, max 997.2222222222222; overall min 11, max 1089
- chat-template overhead per subsequent turn (user+assistant wrappers): ~4.4 tokens
- CI file as-is with DeepSeek tokenizer: max request (last turn prompt + 500) would be computed as untruncated_first + sum(increments) + (T-1)*(500+overhead) + 500 ~= 65279..82422 tokens => ALL 71 conversations exceed 65536 at their last turns (the first turn alone is up to 64K).
- AFTER turn-1 truncation to ~42000 DeepSeek tokens: max request min 54630, median 56843, mean 57361, max 62234; all 71/71 conversations fit with ALL their turns (t_fit == src_turns for all).

## Per conversation

| idx | turns | kimi turn1 | DS turn1 untrunc | DS turn1 (trunc) | turn1 msgs before/after | DS incr/turn mean (min-max) | tmpl ovh/turn | DS max request (all turns, trunc) | fits |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 50084 | 59080 | 41997 | 20/13 | 889 (847-996) | 4.1 | 57819 | True |
| 1 | 13 | 50112 | 59759 | 41999 | 27/21 | 903 (806-947) | 4.0 | 59382 | True |
| 2 | 13 | 50131 | 62363 | 41990 | 32/7 | 935 (911-970) | 4.2 | 59756 | True |
| 3 | 12 | 50160 | 58036 | 41998 | 39/31 | 853 (202-974) | 4.4 | 57430 | True |
| 4 | 13 | 50108 | 53737 | 42000 | 26/24 | 918 (886-1019) | 4.8 | 59573 | True |
| 5 | 11 | 50084 | 59080 | 41997 | 20/13 | 891 (847-996) | 4.1 | 56451 | True |
| 6 | 10 | 50104 | 58855 | 42000 | 25/25 | 913 (886-927) | 4.7 | 55262 | True |
| 7 | 11 | 50212 | 56779 | 41995 | 52/33 | 910 (887-953) | 4.3 | 56633 | True |
| 8 | 10 | 50100 | 58940 | 42000 | 24/24 | 921 (887-953) | 4.4 | 55325 | True |
| 9 | 13 | 50100 | 56936 | 42000 | 24/23 | 910 (789-990) | 4.0 | 59473 | True |
| 10 | 12 | 50124 | 57878 | 41999 | 30/26 | 911 (787-959) | 4.0 | 58062 | True |
| 11 | 11 | 50128 | 56487 | 41997 | 31/24 | 907 (889-922) | 4.0 | 56607 | True |
| 12 | 10 | 50136 | 52438 | 41998 | 33/26 | 867 (805-974) | 4.3 | 54839 | True |
| 13 | 10 | 50164 | 58533 | 41998 | 40/29 | 939 (894-992) | 4.0 | 55485 | True |
| 14 | 10 | 50172 | 56094 | 41994 | 42/19 | 916 (886-982) | 5.0 | 55287 | True |
| 15 | 11 | 50128 | 56052 | 41999 | 31/24 | 875 (824-903) | 4.6 | 56293 | True |
| 16 | 12 | 50124 | 60039 | 42000 | 30/19 | 911 (890-1051) | 4.0 | 58062 | True |
| 17 | 13 | 50016 | 62873 | 42000 | 3/3 | 841 (11-951) | 4.5 | 58649 | True |
| 18 | 14 | 50080 | 62551 | 41993 | 19/5 | 891 (580-1046) | 4.3 | 60634 | True |
| 19 | 12 | 50080 | 57991 | 41999 | 19/16 | 895 (835-930) | 4.5 | 57892 | True |
| 20 | 15 | 50060 | 62479 | 41995 | 14/4 | 884 (387-949) | 4.4 | 61938 | True |
| 21 | 12 | 50116 | 55981 | 42000 | 28/25 | 899 (898-910) | 4.0 | 57934 | True |
| 22 | 11 | 50080 | 53946 | 42000 | 19/19 | 896 (823-937) | 4.5 | 56507 | True |
| 23 | 12 | 50068 | 56548 | 42000 | 16/15 | 833 (11-960) | 4.3 | 57211 | True |
| 24 | 11 | 50203 | 58178 | 41997 | 50/39 | 934 (881-969) | 4.0 | 56881 | True |
| 25 | 11 | 50052 | 63944 | 41997 | 12/6 | 890 (870-930) | 4.8 | 56449 | True |
| 26 | 13 | 50116 | 60511 | 42000 | 28/28 | 859 (11-1024) | 4.0 | 58853 | True |
| 27 | 12 | 50120 | 58246 | 41998 | 29/25 | 890 (705-975) | 4.3 | 57839 | True |
| 28 | 10 | 50116 | 59042 | 41998 | 28/16 | 891 (750-929) | 4.7 | 55061 | True |
| 29 | 10 | 50220 | 56301 | 41996 | 54/48 | 889 (873-927) | 4.4 | 55035 | True |
| 30 | 11 | 50092 | 54686 | 42000 | 22/21 | 913 (874-946) | 4.6 | 56676 | True |
| 31 | 10 | 50196 | 57160 | 42000 | 48/38 | 873 (853-898) | 4.0 | 54889 | True |
| 32 | 10 | 50188 | 55607 | 41993 | 46/26 | 914 (891-937) | 4.3 | 55259 | True |
| 33 | 11 | 50320 | 56205 | 41994 | 79/61 | 891 (866-911) | 4.2 | 56446 | True |
| 34 | 10 | 50084 | 60148 | 42000 | 20/17 | 997 (990-1006) | 4.0 | 56011 | True |
| 35 | 12 | 50100 | 57133 | 41998 | 24/19 | 903 (864-912) | 4.7 | 57980 | True |
| 36 | 10 | 50080 | 56398 | 42000 | 19/19 | 909 (873-934) | 4.8 | 55228 | True |
| 37 | 11 | 50080 | 62551 | 41993 | 19/5 | 904 (876-988) | 4.2 | 56570 | True |
| 38 | 14 | 50128 | 59085 | 41999 | 31/21 | 923 (889-962) | 4.1 | 61045 | True |
| 39 | 15 | 50160 | 56497 | 41996 | 39/28 | 906 (864-941) | 4.2 | 62234 | True |
| 40 | 11 | 50056 | 59549 | 42000 | 13/13 | 878 (724-951) | 4.7 | 56330 | True |
| 41 | 10 | 50064 | 62422 | 41991 | 15/4 | 908 (859-954) | 4.4 | 55204 | True |
| 42 | 14 | 50096 | 57550 | 41998 | 23/15 | 929 (825-1089) | 4.5 | 61135 | True |
| 43 | 11 | 50112 | 62149 | 41994 | 27/6 | 958 (908-1044) | 4.6 | 57120 | True |
| 44 | 11 | 50180 | 58213 | 41996 | 44/36 | 917 (862-950) | 4.5 | 56711 | True |
| 45 | 12 | 50068 | 63374 | 41995 | 16/4 | 901 (857-930) | 4.5 | 57957 | True |
| 46 | 10 | 50080 | 56398 | 42000 | 19/19 | 909 (873-934) | 4.8 | 55228 | True |
| 47 | 11 | 50104 | 59009 | 41996 | 25/17 | 928 (896-1054) | 4.1 | 56819 | True |
| 48 | 14 | 50104 | 54990 | 41997 | 25/21 | 893 (383-989) | 4.2 | 60656 | True |
| 49 | 10 | 50028 | 59148 | 42000 | 6/6 | 925 (878-965) | 4.9 | 55368 | True |
| 50 | 14 | 50144 | 58561 | 41995 | 35/29 | 946 (923-969) | 4.3 | 61343 | True |
| 51 | 10 | 50172 | 56094 | 41994 | 42/19 | 916 (886-982) | 5.0 | 55287 | True |
| 52 | 10 | 50148 | 53351 | 41996 | 36/29 | 880 (845-887) | 4.0 | 54951 | True |
| 53 | 10 | 50148 | 53266 | 42000 | 36/36 | 843 (820-915) | 4.7 | 54630 | True |
| 54 | 11 | 50084 | 61885 | 41993 | 20/5 | 834 (118-971) | 5.0 | 55887 | True |
| 55 | 13 | 50100 | 58092 | 42000 | 24/23 | 844 (11-964) | 4.7 | 58679 | True |
| 56 | 12 | 50108 | 58307 | 42000 | 26/23 | 881 (724-999) | 4.0 | 57740 | True |
| 57 | 12 | 50084 | 56118 | 42000 | 20/18 | 913 (870-948) | 4.5 | 58091 | True |
| 58 | 13 | 50060 | 52353 | 41999 | 14/12 | 875 (808-938) | 4.3 | 59056 | True |
| 59 | 13 | 50096 | 55284 | 41999 | 23/17 | 891 (856-911) | 4.2 | 59247 | True |
| 60 | 10 | 50244 | 58023 | 41986 | 60/35 | 912 (899-932) | 4.6 | 55238 | True |
| 61 | 14 | 50092 | 56312 | 41999 | 22/18 | 905 (869-957) | 4.2 | 60816 | True |
| 62 | 11 | 50120 | 62725 | 41988 | 29/4 | 888 (732-968) | 4.9 | 56414 | True |
| 63 | 12 | 50148 | 58624 | 41995 | 36/22 | 923 (882-964) | 4.5 | 58196 | True |
| 64 | 12 | 50112 | 54095 | 41998 | 27/23 | 870 (822-906) | 4.2 | 57611 | True |
| 65 | 10 | 50188 | 55607 | 41993 | 46/26 | 914 (891-937) | 4.3 | 55259 | True |
| 66 | 11 | 50184 | 59568 | 41995 | 45/30 | 931 (918-945) | 4.0 | 56843 | True |
| 67 | 10 | 50120 | 58607 | 41998 | 29/23 | 887 (880-894) | 4.0 | 55013 | True |
| 68 | 13 | 50104 | 54990 | 41997 | 25/21 | 935 (896-989) | 4.2 | 59769 | True |
| 69 | 13 | 50068 | 54987 | 42000 | 16/16 | 927 (857-1050) | 4.2 | 59674 | True |
| 70 | 10 | 50131 | 62363 | 41990 | 32/7 | 932 (911-970) | 4.2 | 55413 | True |

Per-turn context for request k (1-based) ~= DS turn1 + sum_{j=2..k}(incr_j + 500 + ovh); exact values verified at the last turn.

## Derived workload files

| file | instances (measured+warmup) | turns total (measured) | turns mean/min/max | max request tokens | bytes | sha256 |
|---|---|---|---|---|---|---|
| data/workload-c32.json | 1200+2 | 13895 | 11.58/10/15 | 62234 | 202775593 | 0cc97d7abaa530e02bbdd414fc09d8ab4b914d2b3cdf817fbd46b82c69e1df35 |
| data/workload-c64.json | 1800+2 | 20845 | 11.58/10/15 | 62234 | 303960082 | be764854a88211c46ef4285c3b0ac2a7f9adb5abb3912e562359261ebe110352 |

## How evalscope 1.11.1 replays `--dataset swe_smith --dataset-path X --multi-turn` (source: the installed package, `site-packages/evalscope/perf/`)

1. **Loading** (`plugin/datasets/swe_smith.py` `_load_from_json`): reads the whole JSON, takes `conversations`,
   rotates the list by `--dataset-offset` (`conversations[offset:] + conversations[:offset]`), yields every
   conversation as `List[Turn]`; each turn holds only the DELTA user messages. `prompt_tokens` and
   `--min-turns` are NOT used in pre-built mode (min/max_turns only drive live construction / random_multi_turn).
2. **Preload cap** (`multi_turn_benchmark.py`): loads at most `args.total_count = number + warmup_count`
   conversations from the rotated list.
3. **History** (`core/strategies/multi_turn.py` `_worker`): the context starts empty; for each turn the delta user
   messages are appended, the whole context is sent, and after success the MODEL'S OWN reply is appended as
   `{"role":"assistant","content": benchmark_data.generated_text}`. Dataset assistant messages are never sent
   (the builder already dropped them). generated_text is the streamed `delta.content` only, so with the server's
   deepseek_v31 reasoning parser the reasoning part is not sent back (and the template's drop_thinking would drop it anyway).
   A failed turn abandons the rest of that conversation.
4. **--max-turns** (pre-built mode): truncates each conversation to the first max_turns turns; not used here.
5. **--parallel** = number of concurrently active conversations (asyncio workers); each worker runs one
   conversation's turns strictly sequentially (turn N+1 is sent only after turn N's full response). So at most
   `parallel` requests are in flight.
6. **--number** = total conversations claimed in the phase (not requests). Workers claim from
   `_all_conversations[conv_index % len]` — i.e. the pool is CYCLED (repeated) if number > loaded conversations.
   `conv_index` is continuous across the warmup and benchmark phases.
7. **--duration** (`_compute_deadline`, `_run_phase`): soft exit — after the deadline workers stop CLAIMING new
   conversations, but a claimed conversation runs ALL its remaining turns. Whichever of number/duration is hit first
   ends the phase. Hence a 900 s run actually lasts 900 s + up to one conversation duration (drain tail).
8. **Warmup** (`--warmup-num`) is a separate phase that ignores duration; we instead run a separate 2-conversation
   evalscope invocation, as TokenSpeed CI does.
9. **Token counts**: with `--stream` the openai plugin sets `stream_options.include_usage` and uses server usage;
   it tokenizes client-side with `--tokenizer-path` only if usage is missing. analyze_sustained.py computes per-request
   TPOT = (latency - TTFT)/(completion_tokens - 1) from the db (on the mock it equals evalscope's Avg TPOT, 5.98 ms);
   db start/completed times are `time.perf_counter()` (not epoch).
10. **Request body**: `{model, messages, max_tokens 500, stream true, stream_options, temperature 0.0, ignore_eos true}`
    (verified on the mock: dryrun/salt-check.json).

Implications: replaying the CI file as-is with --duration 900 at c32/c64 would (a) exceed 65536 for every conversation
with the DeepSeek tokenizer, and (b) cycle the 71 conversations, re-sending identical ~42-64K-token prefixes that would hit
the prefix cache. Hence the derived salted, truncated workload files.
