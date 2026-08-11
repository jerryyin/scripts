# /loop demo: patched-llc flag safe & effective — ledger

## Goal & done-claims  (human writes CLAIM; loop derives the test)
- C1: with the fix, @bug (uniform) in-loop readfirstlane -> 0 | test: <loop derives> | status: unchecked
- C2: with the fix, @safe (divergent) unchanged vs stock       | test: <loop derives> | status: unchecked
DONE = both survived. On DONE: CronDelete 3286e26f and summarize.

## Variables in play
- llc {stock 56421f92, patched /root/llvm-project/install}, fn {@bug,@safe}, flag {on,off}, IR fixed=/root/scripts/triton/reproducers/amdgpu_readfirstlane_licm/ir/repro.ll

## Established facts (how verified)
- iter1 CONTROL: stock llc, same IR -> bug=4, safe=4. [ran llc, counted in-loop rfl]

## Hypotheses
- flag hoists only in uniform loops (safe & effective).

## Next decisive experiment
- iter2: PATCHED llc, same IR -> bug,safe. Schemas: Boundary(stock->patched)+Coverage(both fns). Expect bug 4->0, safe 4->4.

## Dead ends
- (none)
