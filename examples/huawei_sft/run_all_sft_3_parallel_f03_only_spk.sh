#!/usr/bin/env bash
# Backward-compatible alias: now launches all 4 F03-only jobs (incl. base 1e-6).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_all_sft_4_parallel_f03_only_spk.sh" "$@"
