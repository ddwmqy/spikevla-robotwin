# SDT-V3 runtime

Upstream: https://github.com/BICLab/Spike-Driven-Transformer-V3

Paper: [Scaling Spike-Driven Transformer With Efficient Spike Firing Approximation Training](https://arxiv.org/abs/2411.16061).

Run `python scripts/setup_third_party.py` from the project root to fetch
`SDT_V3/Classification/Model_Base/models.py` and apply the recorded integration
patch. The only change to this file is removal of an unused torchinfo import.

The adapter uses the official 19M `Efficient_Spiking_Transformer_l` backbone.
Weights must be downloaded separately and must match its actual tensor shapes.
The official SFA implementation is retained, including its multilevel firing
representation. The classification head is not used for VLA visual tokens.

See the parent directory's README for upstream terms and provenance.
