# Copyable kickoff prompt

Work in `/Users/zhengfeili/2026_Fall/10703/10703_Project` on our Bailando++ DRL
course project. Implement and carry out the bounded midpoint beat-objective plan
in `docs/MIDPOINT_BEAT_OBJECTIVE_PLAN.md`. Start by reading applicable AGENTS.md
instructions, README.md, that plan, `docs/BASIC_BEAT_IMPLEMENTATION_PLAN.md`, and
the existing pilot reports/scripts. Inspect actual code and artifacts; some named
modules in the documents are proposals rather than implemented functionality.

The research question is whether Gaussian beat-proximity reward improves RL
fine-tuning compared with the original bin rule under a common stochastic
actor–critic trainer. Required scope: frozen baseline, bin RL, BAS RL, two training
seeds, side-by-side BAS and tolerance-based precision/recall/F1, quality diagnostics,
and a midpoint report. F1 is evaluation-only initially. Do not expand into GRPO,
DPO, hierarchy, reward-model training, or schedule/regularizer sweeps.

Proceed in dependency order:
1. Verify coordinate/export conventions and implement/reuse a shared CPU evaluator
   with analytic and upstream differential checks. Preserve legacy reproduction.
2. Rescore existing cached outputs, accounting for every expected case; audit and
   freeze music-disjoint adaptation/validation/test splits without inventing data.
3. Implement the common on-policy stochastic trainer and checkpoint/update checks.
   Compare normalized bin reward and BAS on identical motion events. Keep frozen
   components fixed, actor/critic optimizers separate, and likelihood conditioning
   identical to generation. Preserve both upper/lower action probabilities.
4. Run development reward diagnostics and a representative smoke/cost benchmark.
   Freeze the common config and spending cap before the four short runs. Check
   existing budget authorization; do not treat this prompt as unlimited GPU spend.
   Continue all authorized work. If additional spending approval is necessary,
   finish the executable setup and provide measured cost and a bounded launch
   command so approval is the final step, not an early planning interruption.
5. Execute authorized bounded runs, evaluate the prespecified final checkpoints,
   and produce reproducible tables, learning curves, paired metric analysis, and
   an honest midpoint report. Never present pending runs as completed results.

Use the project's `10703` Conda environment locally:
`/Users/zhengfeili/opt/anaconda3/bin/conda run -n 10703 python`.
Remote GPU jobs must use the established project runtime. Preserve user changes
and historical artifacts. Reuse existing scripts/infrastructure where appropriate.
Do not assume the published paper and our pilot used identical protocols, and do
not compare historical greedy training with the new trainer as an objective-only
ablation. Do not call a motion speed minimum a physical strike, tracked beats
downbeats, or automatic timing scores human preferences.

Keep concise progress updates. Resolve routine implementation decisions yourself;
ask only for genuinely missing constraints or required authorization. At completion,
report what was implemented, verification evidence, exact training/evaluation
counts and spend, results and limitations, file links, and any concrete remaining
blocker. If work must pause for approval, leave a reproducible resume command and
complete all independent local work first.
