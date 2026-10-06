# Resumable Groq capture operations

This evaluation-only runner captures the frozen 16-case, 48-execution vector/graph/hybrid experiment. It does not initialize a RAGAS judge. Application factories, model fallback behavior, retrieval, prompts, and the corpora are not changed by the runner.

## Current quota hold

The operator-reported Groq telemetry confirms both TPM and TPD exhaustion for `openai/gpt-oss-120b`. The latest TPD observation was limit 200,000, usage 199,554, request 716; a separate TPM observation was limit 8,000, usage 7,739, request 2,257. No provider reset time is inferred. The CLI refuses provider calls unless `--confirm-tpd-reset` is explicitly supplied after checking that Groq's daily quota has reset.

## Workflow

1. After quota reset, run the two-case smoke (vector-only, same frozen corpus and pinned Groq model):

   ```powershell
   python -m evals.resumable_groq_capture smoke --confirm-tpd-reset
   ```

   The smoke must complete route, KB grade, generation, and grounding for both frozen representative cases. It is checkpointed, and a matching successful smoke is not repeated.

2. Capture bounded batches (default two cases per batch; each case runs vector-only, graph-only, then hybrid):

   ```powershell
   python -m evals.resumable_groq_capture run-batch --confirm-tpd-reset --batch-size 2
   ```

   Reinvoke only after reviewing quota headroom. Completed executions are resumed only from the exact experiment-hash directory and only when the full persisted metadata matches. Each successful execution is atomically checkpointed. A 429 or local token-budget stop ends the batch; it is not retried. Incomplete cases/modes are absent from the completed records.

3. Once all cases are complete, validate and merge all batch files from that one experiment-hash directory:

   ```powershell
   python -m evals.resumable_groq_capture merge --batches evals/results/resumable-groq/<experiment-hash>/batches/batch-*.json --output evals/results/resumable-groq/<experiment-hash>/merged.json
   ```

   The merger rejects differing experiment metadata, duplicate successful rows, wrong execution order, missing cases/modes, and non-completed execution rows. Earlier quota-stopped batches remain explicitly marked and their completed rows can be reused; the merge succeeds only after the missing case/mode executions have later completed under identical metadata. Interruption records are retained in the merged artifact, not counted as retrieval failures. It does not calculate semantic judge metrics.

## Quota pacing and telemetry

The evaluation callback makes a conservative token reservation per Groq call using a local tokenizer estimate with 25% input margin, message framing, and the configured output-token ceiling. A SQLite rolling ledger is shared across batches in the output root. When necessary, the runner waits until a reservation ages out of the 60-second window; a request too large for the configured 8,000 TPM budget or the local 200,000 TPD budget fails closed. Prompt/evidence content is never written to that ledger. Each provider callback event retains stage, provider, model, outcome/status, latency, token usage when reported, quota headers/body excerpt when available, estimate/reservation, and pacing wait seconds.

Actual Groq accounting may differ from the local tokenizer estimate; the runner stops on provider 429 responses and does not treat a partial run as a comparison. A reset attestation is an explicit operator action, not an automatic retry policy.
