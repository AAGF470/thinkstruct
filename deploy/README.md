# Deploying the demo to Raya (Part 2 proof of concept)

**What runs:** two containers.
- `app` is the FastAPI search API plus the **Patent Explorer** UI. The UI is pre-built into `app/static/`, so the image needs no Node. It runs on CPU only and serves a read-only index bundle. Don't set `ALLOW_DEBUG_DELAY` on a shared server: it exists only for demonstrating the stale-result guard.
- `nginx` sits in front: basic-auth login, an optional IP allowlist, per-IP rate limits, and `noindex` headers.

**The split of work:**
- **The laptop does the heavy computing** (parsing and embedding, `build_index.py`) and ships a **10.7 MB** bundle.
- **Raya only embeds one query at a time:** about 100–320 ms on CPU (measure, step 5).

```
laptop: build_index.py → export_index.py → qwen3.tar.gz (10.7 MB) ──scp──▶ Raya
Raya:   nginx :8080 (auth, rate limit) → app :8000 (uvicorn, 1 worker) → /index (read-only)
                                           └─ /hf volume: model weights, downloaded once (~1.2 GB)
```

**Resources on Raya:**

| item | size |
|---|---|
| Docker image | ≈ 2 GB (Python 3.12 slim + CPU-only PyTorch + deps; no CUDA) |
| model cache volume | ≈ 1.2 GB (Qwen3-Embedding-0.6B) |
| index | 16 MB unpacked |
| RAM | ≈ 3 GB (container limit 6 GB) |
| CPU | limited to 3 of 4 vCPU |

Idle CPU is about zero: nothing runs between requests.

---

## 1. On the laptop: build the bundle

```bash
.\.venv\Scripts\python.exe export_index.py
```

This writes `deploy/index/qwen3/` and `deploy/index/qwen3.tar.gz`, and prints a sha256. The vectors are truncated to 512 dimensions and stored as float16; [results/compression_qwen3.md](../results/compression_qwen3.md) shows no measurable quality loss.

## 2. Get the code and the bundle onto Raya

The code comes from the GitHub repo. Data, caches, the bundle and the password file are gitignored, so they never go through git.

```bash
git clone <your-repo-url> ~/thinkstruct
```

```bash
scp deploy/index/qwen3.tar.gz raya:~/thinkstruct/deploy/index/
```

On Raya, check that the sha256 matches what step 1 printed, then unpack:

```bash
cd ~/thinkstruct/deploy/index && sha256sum qwen3.tar.gz && tar xzf qwen3.tar.gz
```

## 3. Create the login (on Raya)

`-B` stores a bcrypt hash. `read -s` keeps the password out of your shell history.

```bash
cd ~/thinkstruct/deploy && read -s -p "password for reviewer: " P && echo && docker run --rm httpd:2.4-alpine htpasswd -nbB reviewer "$P" > htpasswd && unset P
```

Run it again with `>>` instead of `>` to add more users. For an IP allowlist as well, uncomment the `allow` / `deny` lines in [nginx.conf](nginx.conf).

## 4. Start it

```bash
cd ~/thinkstruct && docker compose -f deploy/docker-compose.yml up -d --build
```

The first start downloads the model, so nginx waits until the app's health check passes (allow up to about 10 minutes). Watch it with:

```bash
docker compose -f deploy/docker-compose.yml logs -f app
```

Then check it:

```bash
curl -u reviewer http://localhost:8080/api/health
```

Open `http://<raya-host>:8080/` in a browser and sign in.

## 5. Measure query latency on Raya's CPU

Qwen3 loads in bfloat16. That was 2× faster than float32 on the laptop's Ryzen, but on CPUs without native bf16 support it can be much slower. Time both:

```bash
docker compose -f deploy/docker-compose.yml exec app python -c "import time; from search import SearchEngine; e=SearchEngine('/index/qwen3'); e.model(); e.embed_query('warm'); t=time.perf_counter(); [e.embed_query(f'tire pressure sensor {i}') for i in range(10)]; print(f'{(time.perf_counter()-t)*100:.0f} ms/query')"
```

If it's slow, set `EMBED_DTYPE: float32` in `docker-compose.yml`, run `up -d` again, and re-measure. Record the number in `ARCHITECTURE.md` §7.

## 5b. Check that it finds what it should

From the laptop, through nginx and the login. It prompts for the password; `--quick` tests about 40 patents instead of all 640.

```bash
.\.venv\Scripts\python.exe smoke_test.py --url http://<raya-host>:8080 --user reviewer --quick
```

It should print **PASS** and match the local run in [results/smoke_127_0_0_1.md](../results/smoke_127_0_0_1.md), with the same ranks; only latency differs. The run also exercises nginx's rate limit. If you see `429` errors, use `--quick` or raise the `burst` in `nginx.conf`.

## 6. Optional: TLS / a domain

- **If Raya already runs nginx or Caddy with certificates:** change the port mapping to `"127.0.0.1:8080:80"` and proxy your domain to it. The container's nginx keeps doing auth and rate limiting.
- **Otherwise:** anyone who knows `http://<ip>:8080` still has to log in, but the password travels unencrypted. Use TLS before sharing it widely.

## Operations

| task | command |
|---|---|
| update the code | `git pull && docker compose -f deploy/docker-compose.yml up -d --build` |
| ship a new index | scp the new tarball, `tar xzf`, then `docker compose -f deploy/docker-compose.yml restart app` |
| turn on reranking | `ENABLE_RERANK: "1"` (MiniLM, about 3 s/query on 4 vCPU; see ARCHITECTURE.md §7) |
| stop everything | `docker compose -f deploy/docker-compose.yml down`; add `-v` to also delete the model cache |
| logs | `docker compose -f deploy/docker-compose.yml logs --tail 100 app nginx` |

**Not verified yet:** this setup has only been run with uvicorn on the laptop (`INDEX_DIR=deploy/index/qwen3 uvicorn app.main:app`). Docker isn't installed there, so the image and compose file have not been built. Step 4 is their first real test.
