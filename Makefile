PY ?= uv run python
BIN := target/release/pacer
TRACES := traceA traceB coder thinking
THREADS ?= 14
REPORT := PYTHONPATH=python $(PY) -m pacer.report

.PHONY: all build data rows models validate select test summary docs figures reproduce \
	ablations transfer overhead extras lrb live demo check clean

all: build

build:
	cargo build --release -p pacer

data:
	./scripts/fetch_data.sh

# Cache-independent hazard rows from each trace's first hour (seconds).
rows: build
	mkdir -p data/labels/hazard
	for t in $(TRACES); do \
		$(BIN) hazard --trace data/raw/qwen_$${t}_blksz_16.jsonl --cutoff 3600 --rate 0.15 \
			--per-touch 8 --out data/labels/hazard/$${t}.npy; \
	done

# Compact models (fixed configuration) and larger models (horizon-tuned configuration).
models: rows
	mkdir -p models results/training
	for t in $(TRACES); do \
		PYTHONPATH=python $(PY) -m pacer.hazard data/labels/hazard/$${t}.npy \
			--num-leaves 31 --learning-rate 0.25 --wa-out models/wa_$${t}.json \
			--out models/fast_$${t}.json --report results/training/fast_$${t}.json; \
		PYTHONPATH=python $(PY) -m pacer.hazard data/labels/hazard/$${t}.npy \
			--out models/hazard_$${t}.json --report results/training/hazard_$${t}.json; \
	done

# Every choice is made on t = 45-60 min of hour 1.
validate: build
	$(BIN) sweep --spec configs/validation_heuristics.json \
		--out results/validation_heuristics.jsonl --threads $(THREADS)
	$(BIN) sweep --spec configs/validation_pacer.json --out results/validation_pacer.jsonl \
		--threads $(THREADS)

select:
	$(REPORT) select results/validation_heuristics.jsonl results/validation_pacer.jsonl \
		--out configs/test.json --pacer-out configs/test_pacer.json

# The held-out hour, once.
test: build
	$(BIN) sweep --spec configs/test.json --out results/test_baselines.jsonl --threads $(THREADS)
	$(BIN) sweep --spec configs/test_pacer.json --out results/test_pacer.jsonl --threads $(THREADS)
	cat results/test_baselines.jsonl results/test_pacer.jsonl > results/final_test.jsonl

summary:
	$(REPORT) summarize results/final_test.jsonl --out results/final_summary.json

figures:
	PYTHONPATH=python $(PY) -m pacer.figures

docs: summary figures
	$(REPORT) fill README.md docs/results.md

reproduce: data models validate select test docs

extras: ablations transfer overhead docs

# The LRB-style first attempt, kept as an ablation: imitate Belady on logged sampled-LRU states,
# then one DAgger round. Its labels look past the training hour (see the engineering log).
lrb: build
	mkdir -p data/labels/lrb
	for c in 0.01 0.02 0.05 0.1; do \
		$(BIN) label --trace data/raw/qwen_traceA_blksz_16.jsonl --driver age --cap $$c \
			--window 0,3600 --rate 0.3 --out data/labels/lrb/age_$$c.npy; \
	done
	PYTHONPATH=python $(PY) -m pacer.train data/labels/lrb/age_*.npy --out data/labels/lrb/v0.json
	for c in 0.01 0.02 0.05 0.1; do \
		$(BIN) label --trace data/raw/qwen_traceA_blksz_16.jsonl --driver model \
			--model data/labels/lrb/v0.json --cap $$c --window 0,3600 --rate 0.3 \
			--out data/labels/lrb/v0_$$c.npy; \
	done
	PYTHONPATH=python $(PY) -m pacer.train data/labels/lrb/age_*.npy data/labels/lrb/v0_*.npy \
		--out models/lrb_traceA.json

ablations: build
	for t in traceA coder; do \
		for v in nosession:"prev_gap mean_gap a_over_gap session_age turn" nohistory:"n_access fanout"; do \
			PYTHONPATH=python $(PY) -m pacer.hazard data/labels/hazard/$${t}.npy \
				--num-leaves 31 --learning-rate 0.25 --drop $${v#*:} \
				--out models/ablate_$${v%%:*}_$${t}.json \
				--report results/training/ablate_$${v%%:*}_$${t}.json; \
		done; \
	done
	$(BIN) sweep --spec configs/ablations.json --out results/ablations.jsonl --threads $(THREADS)

transfer: build
	$(BIN) sweep --spec configs/transfer.json --out results/transfer.jsonl --threads $(THREADS)

overhead: build
	PYTHONPATH=python $(PY) scripts/overhead.py

live:
	uv run --extra live python scripts/live_replay.py --limit 400 --start 3600 --tpot 0.002 \
		--cap 0.08 --repeats 2 --check 200
	uv run --extra live python scripts/calibrate_prefill.py

demo: build
	PYTHONPATH=python $(PY) scripts/demo.py

check:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test -p pacer
	uv run ruff check .
	uv run ruff format --check .
	PACER_BIN=$(BIN) uv run pytest -m "not slow and not live"

clean:
	cargo clean
	rm -rf data/labels
