# MoE gather-index / readfirstlane investigation (issue #1885)

Preserved notes, fixes, and reference data from the gfx1250 MoE gather-index `v_readfirstlane` / TDM descriptor-churn investigation and the related gather/scatter scalar-load (sload) wave-uniform proposal. Moved here from the loose container home during a state sync so they survive teardown; this is a grab-bag of prior-session artifacts, not a single linear write-up.

## Contents

- `GOLDEN-1885-readfirstlane.md` — definitive summary of the #1885 gather-index `v_readfirstlane` finding.
- `moe_readfirstlane_eval.md` — full evaluation of the in-loop `v_readfirstlane` / descriptor churn and why it's an LLVM bug.
- `uniformizeAddr-fix-IR-walkthrough.md` — before/after IR walkthrough of the `uniformizeAddr` + `!invariant.load` fix.
- `llvm_knob_exploration.md` — LLVM knob exploration for the TDM descriptor `v_readfirstlane` churn.
- `llvm_pass_fix_results.md` — results of the LLVM pass fix.
- `gather_sload_plan.md` — proposal: scalar loads for wave-uniform TDM gather/scatter indices (gfx1250).
- `overnight_plan.md` / `overnight_results.md` — overnight run plan and results (descriptor churn, eval, ticket repro).
- `investigation-ledger.md` — `/loop` demo ledger (patched-llc flag safe & effective).
- `sload_uniform_main.patch` — the gather/scatter sload wave-uniform patch.
- `llvm_readfirstlane_hoist.patch` — the LLVM `readfirstlane` hoist patch.
- `ticket_1885.json` — downloaded GitHub issue #1885 reference.
