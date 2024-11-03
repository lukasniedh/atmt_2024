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

Why? What data? how preprocessing? changes in the code? selected hyper-parameter? what training command? how were the models evaluated?

Previous work has argued for larger batch sizes in NMT (Morishita et al., 2017; Neishi et al., 2017), but we find that using smaller batches is beneficial in lowresource settings.

More aggressive dropout, including dropping whole words at random (Gal and Ghahramani, 2016), is also likely to be more important.

The Baseline is using a learning rate of 0.0003 the firs thing we want to test if what effect the learning rate has on the bleu score. in the paper they suggest with higher learning rate.. we will test it with 0.0005 and 0.001.
We use the same preprocessed data as in the baseline.
### train with 0.0005 lr:
```
python train.py \
    --data data/en-fr/prepared \
    --source-lang fr \
    --target-lang en \
    --save-dir assignments/03/learning-rate-a/checkpoints \
    --lr 0.0005
```
### evaluate:
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

### evaluate:
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

## Second strategy: ??

## compare to baseline

table, visualization, qualitative analysis..

## what did I learn?