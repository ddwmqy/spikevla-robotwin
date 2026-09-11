# SmoothSpike runtime provenance

Upstream: https://github.com/CayleyZ/SmoothSpike

Model: kailai1104/SmoothSpike/checkpoints/smoothspike-bert-base/model.safetensors

The runtime uses the upstream spikingbert_rot_inf.py model after the upstream
convertor.py folds each layer's H2/H3 matrices into adjacent linear weights.
The global H1 transform and all LIF neuron dynamics remain active. The frozen
TurboVLA text backbone runs with T=4 and eager spike-driven attention.

Local integration changes are intentionally limited to:

- package-relative imports;
- CPU allocation during model construction before DDP device placement;
- optional import of the CUDA-only fast Hadamard kernel, which is not used by
  the frozen fused runtime;
- support for TurboVLA/GroundingDINO [B, L, L] sub-sentence masks;
- caching the fixed H1 matrix-sign transform after the backbone is frozen.

The published checkpoint is loaded into the full upstream
BertForMaskedLM definition. All non-aliased checkpoint tensors are required;
only the safetensors-omitted tied decoder aliases may be absent. TurboVLA then
uses model.bert.last_hidden_state and discards the MLM head.

Citation: "SmoothSpike: Spiking Transformer with Learnable Hadamard Transformation".
See the upstream repository for bibliographic details and terms.
