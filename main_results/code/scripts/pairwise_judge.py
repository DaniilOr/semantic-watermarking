"""Disabled: the local-model judge was removed at the user's request (no local 8B model is loaded for judging).

Quality judging is done by scripts/api_judge.py through an external API model.  This stub remains only so that
job scripts submitted before the change skip the step cleanly instead of failing.
"""
print("[pairwise_judge] local judge disabled -- use scripts/api_judge.py (external API model). Skipping.")
