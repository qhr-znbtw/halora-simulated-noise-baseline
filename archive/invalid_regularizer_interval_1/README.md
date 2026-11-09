# Invalid regularizer-interval experiment

This directory preserves the first `regularized_lora` and `halora` runs for audit
purposes. They are scientifically invalid because the structural regularizer was
applied at every optimizer step (`interval: 1`) instead of only at the synchronized
alignment steps (`interval: 400`).

With 1,300 optimizer steps, the invalid runs applied the regularizer 1,300 times.
The corrected strict-paper schedule applies it four times: steps 0, 400, 800, and
1,200. Do not use the archived accuracy values to evaluate HaLoRA.
