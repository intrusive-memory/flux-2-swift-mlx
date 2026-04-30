# CDN Ship Log — OPERATION FAREWELL EMBRACE
# mission/farewell-embrace/01

---

## Sortie 2 — lmstudio-community/Qwen3-4B-MLX-4bit

### Preflight df check

```
Filesystem   1024-blocks       Used Available Capacity iused      ifree %iused  Mounted on
/dev/disk3s5  1948455240 1475728944 449093144    77% 6283353 4490931440    0%   /System/Volumes/Data
```

Free space: ~449 GB available. Threshold: 6 GB. PASS.

---

### Sortie 2 — attempt 2

**Timestamp**: 2026-04-30 06:47:52 local

#### Preflight df check (re-verified)

```
Filesystem   1024-blocks       Used Available Capacity iused      ifree %iused  Mounted on
/dev/disk3s5  1948455240 1476561512 448260576    77% 6280302 4482605760    0%   /System/Volumes/Data
```

Free space: ~448 GB available. Threshold: 6 GB. PASS.

#### acervo ship invocation

```
acervo ship lmstudio-community/Qwen3-4B-MLX-4bit --no-verify
```

Environment loader prefix applied (HF_TOKEN and R2_PUBLIC_URL existence verified; values not echoed).

**Selected output** (progress bars and lock-file uploads omitted for brevity):

```
Fetching 11 files: 100%|██████████| 11/11 [00:00<00:00, 3830.10it/s]
Downloaded lmstudio-community/Qwen3-4B-MLX-4bit to /tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit (verification skipped).
manifest written to /tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/manifest.json
CHECK 4 passed: all staged files match the manifest.
[... upload: .lock files + model files + manifest.json to s3://intrusive-memory-audio/models/lmstudio-community_Qwen3-4B-MLX-4bit/ ...]
Completed 2.1 GiB/2.1 GiB (2.2 MiB/s) with 1 file(s) remaining
manifest.json uploaded to CDN.
CHECK 5 passed: CDN manifest verified.
CHECK 6 passed: config.json spot-check succeeded.
Ship complete for lmstudio-community/Qwen3-4B-MLX-4bit.
```

**Exit code**: 0

**Local manifest**: `/tmp/acervo-staging/lmstudio-community_Qwen3-4B-MLX-4bit/manifest.json`
- Size: 1876 bytes
- SHA-256: `8f0dcb90c7fc4e789d880b07a0a8b9da2e98a0cff082acb34a779d2e5befe759`

#### Manifest HTTP status

CDN URL pattern note: `acervo` stages and uploads to the underscore-slug form (`lmstudio-community_Qwen3-4B-MLX-4bit`). The dispatch prompt's `/models/<slug>/` path uses that same form.

- `$R2_PUBLIC_URL/models/lmstudio-community_Qwen3-4B-MLX-4bit/manifest.json` → HTTP **200**
- `$R2_PUBLIC_URL/models/lmstudio-community/Qwen3-4B-MLX-4bit/manifest.json` → HTTP 404 (slash form — not how acervo uploads)

#### Tokenizer-artifact grep

```
      "path" : "added_tokens.json",
      "path" : "special_tokens_map.json",
      "path" : "tokenizer.json",
      "path" : "tokenizer_config.json",
```

All four required tokenizer artifacts confirmed present in the CDN manifest.

CHECK 6 passed (per acervo output). Sortie 2 COMPLETE.

---

## Sortie 3 — lmstudio-community/Qwen3-8B-MLX-4bit

**Timestamp**: 2026-04-30 (local)

### Preflight df check

```
/dev/disk3s5  1948455240 1476577708 448244380    77% 6280413 4482443800    0%   /System/Volumes/Data
```

Free space: ~448 GB available. Threshold: 10 GB. PASS.

### acervo ship invocation

```
acervo ship lmstudio-community/Qwen3-8B-MLX-4bit --no-verify
```

Environment loader prefix applied (HF_TOKEN and R2_PUBLIC_URL existence verified; values not echoed).

**Selected output** (progress bars and lock-file uploads omitted for brevity):

```
Fetching files: 100%|██████████| [00:00<00:00, ...it/s]
Downloaded lmstudio-community/Qwen3-8B-MLX-4bit to /tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit (verification skipped).
manifest written to /tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/manifest.json
CHECK 4 passed: all staged files match the manifest.
[... upload: .lock files + model files + tokenizer files to s3://intrusive-memory-audio/models/lmstudio-community_Qwen3-8B-MLX-4bit/ ...]
Completed 4.3 GiB/4.3 GiB (2.2 MiB/s) with 1 file(s) remaining
upload: .../manifest.json to s3://intrusive-memory-audio/models/lmstudio-community_Qwen3-8B-MLX-4bit/manifest.json
manifest.json uploaded to CDN.
CHECK 5 passed: CDN manifest verified.
CHECK 6 passed: config.json spot-check succeeded.
Ship complete for lmstudio-community/Qwen3-8B-MLX-4bit.
```

**Exit code**: 0

**Local manifest**: `/tmp/acervo-staging/lmstudio-community_Qwen3-8B-MLX-4bit/manifest.json`
- Size: 1876 bytes
- SHA-256: `9cdc1cbee73c69bd93812e604de14204c8d4079aa1819923d6a4339e13d4f188`

### Manifest HTTP status

CDN slug: `lmstudio-community_Qwen3-8B-MLX-4bit` (underscore form — same pattern as Sortie 2).

- `$R2_PUBLIC_URL/models/lmstudio-community_Qwen3-8B-MLX-4bit/manifest.json` → HTTP **200**

### Tokenizer-artifact grep

```
      "path" : "added_tokens.json",
      "path" : "special_tokens_map.json",
      "path" : "tokenizer.json",
      "path" : "tokenizer_config.json",
```

All four tokenizer artifacts confirmed present in the CDN manifest. Required artifact `tokenizer.json` is present.

CHECK 6 passed (per acervo output). Sortie 3 COMPLETE.

