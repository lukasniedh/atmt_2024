# Assignment 3: Improving Low-Resource NMT

In this assignment we use fr-en data from the Tatoeba
corpus and investigate methods for improving low-resource NMT.

Your task is to experiment with techniques for improving
low-resource NMT systems.

## Baseline

The data used to train the baseline model was prepared using
the script `preprocess_data.sh`.
This may be useful if you choose to apply subword
segmentation or a data augmentation method.

## First strategy: Tune a hyper-parameter

The Baseline is using a learning rate of 0.0003 the firs thing we want to test if what effect the learning rate has on the bleu score. We use the same preprocessed data as in the baseline.

### train with 0.0005 lr:
```
python train.py \
    --data data/en-fr/prepared \
    --source-lang fr \
    --target-lang en \
    --save-dir assignments/03/learning-rate-a/checkpoints \
    --lr 0.0005
```
### translate for evaluation:
```
python translate.py \
    --data data/en-fr/prepared \
    --dicts data/en-fr/prepared \
    --checkpoint-path assignments/03/learning-rate-a/checkpoints/checkpoint_best.pt \
    --output assignments/03/learning-rate-a/translations.txt
```
### postprocessing:
```
bash scripts/postprocess.sh \
    assignments/03/learning-rate-a/translations.txt \
    assignments/03/learning-rate-a/translations.p.txt en
```
### calculate BLEU score
```
cat \
    assignments/03/learning-rate-a/translations.p.txt \
    | sacrebleu data/en-fr/raw/test.en
```
### Results:
```
{
 "name": "BLEU",
 "score": 16.6,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3",
 "verbose_score": "44.6/22.0/12.0/6.4 (BP = 1.000 ratio = 1.297 hyp_len = 5048 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.4.3"
}
```

### train with 0.001 lr:
```
python train.py \
    --data data/en-fr/prepared \
    --source-lang fr \
    --target-lang en \
    --log-file assignments/03/learning-rate-b/exp.log \
    --save-dir assignments/03/learning-rate-b/checkpoints \
    --lr 0.001
```

### translate for evaluation:
```
python translate.py \
--data data/en-fr/prepared \
--dicts data/en-fr/prepared \
--checkpoint-path assignments/03/learning-rate-b/checkpoints/checkpoint_best.pt \
--output assignments/03/learning-rate-b/translations.txt
```

### postprocessing:
```
bash scripts/postprocess.sh \
    assignments/03/learning-rate-b/translations.txt \
    assignments/03/learning-rate-b/translations.p.txt en
```

### calculate BLEU score
```
cat \
    assignments/03/learning-rate-b/translations.p.txt \
    | sacrebleu data/en-fr/raw/test.en
```

### Results
```
{
 "name": "BLEU",
 "score": 9.6,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3",
 "verbose_score": "34.4/13.8/6.5/2.8 (BP = 1.000 ratio = 1.410 hyp_len = 5489 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.4.3"
}
```

## Second strategy: word threshold for source and target language

### preprocess data for word limit = 5
```
bash assignments/03/preprocess_data_2.sh
```

### train
```
python train.py \
    --data data/en-fr/prepared_2 \
    --source-lang fr \
    --target-lang en \
    --log-file assignments/03/preprocess_a/exp.log \
    --save-dir assignments/03/preprocess_a/checkpoints \
```

### translate for evaluation:
```
python translate.py \
--data data/en-fr/prepared_2 \
--dicts data/en-fr/prepared_2 \
--checkpoint-path assignments/03/preprocess_a/checkpoints/checkpoint_best.pt \
--output assignments/03/preprocess_a/translations.txt
```

### postprocessing:
```
bash scripts/postprocess.sh \
    assignments/03/preprocess_a/translations.txt \
    assignments/03/preprocess_a/translations.p.txt en
```

### calculate BLEU score
```
cat \
    assignments/03/preprocess_a/translations.p.txt \
    | sacrebleu data/en-fr/raw/test.en
```

### Results
```
{
 "name": "BLEU",
 "score": 16.1,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3",
 "verbose_score": "44.2/21.6/11.8/6.0 (BP = 1.000 ratio = 1.322 hyp_len = 5147 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.4.3"
}
```

### preprocess data for word limit = 10
```
bash assignments/03/preprocess_data_3.sh
```

### train
```
python train.py \
    --data data/en-fr/prepared_3 \
    --source-lang fr \
    --target-lang en \
    --log-file assignments/03/preprocess_b/exp.log \
    --save-dir assignments/03/preprocess_b/checkpoints \
```

### translate for evaluation:
```
python translate.py \
--data data/en-fr/prepared_3 \
--dicts data/en-fr/prepared_3 \
--checkpoint-path assignments/03/preprocess_b/checkpoints/checkpoint_best.pt \
--output assignments/03/preprocess_b/translations.txt
```

### postprocessing:
```
bash scripts/postprocess.sh \
    assignments/03/preprocess_b/translations.txt \
    assignments/03/preprocess_b/translations.p.txt en
```

### calculate BLEU score
```
cat \
    assignments/03/preprocess_b/translations.p.txt \
    | sacrebleu data/en-fr/raw/test.en
```

### Results
```
{
 "name": "BLEU",
 "score": 11.8,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3",
 "verbose_score": "39.0/16.8/8.1/3.6 (BP = 1.000 ratio = 1.427 hyp_len = 5555 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.4.3"
}
```