# Assignment 5: improve decoding in NMT

## 1. Experiment with Beam Search

The baseline model is used: `assignments/03/baseline/checkpoints/checkpoint_best.pt`

### translate with beam search for evaluation:

The following python script does translation using beam search for beam sizes `[1,3,5,10,15,20,25]` and generates two plots.

```
python assignments/05/beam_size_experiment.py
```

![Bleu Scores](01_beam_sizes/bleu_score_vs_beam_size.png)
![Brevity Penalty](01_beam_sizes/BP_values_vs_bleu_score.png)
