# Archived tensor trace validation

This validates saved tensor values only. The saved tensor .bin/.json files are present, but the original trace-capture executable and hook source are not, so this is historical evidence rather than a current-HEAD reproduction.

| ubatch | query | physical KV span | seq token counts | mask mismatches | causal-count mismatches | hidden logical shape | K current | K cache view |
|---:|---:|---:|---|---:|---:|---|---|---|
| 0 | 512 | 512 | 0:128,1:256,2:128 | 0 | 0 | `[512, 1024]` | `[8, 512, 128]` | `[512, 8, 128]` |
| 1 | 512 | 1024 | 2:384,3:128 | 0 | 0 | `[512, 1024]` | `[8, 512, 128]` | `[1024, 8, 128]` |
| 2 | 512 | 1536 | 3:512 | 0 | 0 | `[512, 1024]` | `[8, 512, 128]` | `[1536, 8, 128]` |
| 3 | 384 | 2048 | 3:384 | 0 | 0 | `[384, 1024]` | `[8, 384, 128]` | `[2048, 8, 128]` |

- Query tokens checked: 1920
- Physical cells after the last ubatch: 1920
- Mask value/ownership/causal mismatches: 0
- Per-query visible-count mismatches: 0
- Layer-0 hidden equals embedding bytes for every ubatch: True
