# assistant v0.4 history-dependent corpus

Every family's correct option is a threshold function of a latent state that is
written into the history, so the item is only answerable by binding and reading the
history. All conditions carry history, including neutral (impersonal). Matched
supervision holds across neutral/self/other/spp. No external model provider is used.

Anchor instruction-following data is included identically in every condition's train
split to prevent repetitive answer-format collapse.

`training/<condition>/<split>.jsonl` is the handoff surface for GPU training.
