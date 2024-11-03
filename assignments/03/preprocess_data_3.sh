#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed_3/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed_3/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed_3/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed_3/tm.$src --corpus $data/preprocessed_3/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed_3/tm.$tgt --corpus $data/preprocessed_3/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed_3/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed_3/tm.$src > $data/preprocessed_3/train.$src 
cat $data/preprocessed_3/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed_3/tm.$tgt > $data/preprocessed_3/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_3/tm.$src > $data/preprocessed_3/$split.$src
    cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed_3/tm.$tgt > $data/preprocessed_3/$split.$tgt
done

# remove tmp files
rm $data/preprocessed_3/train.$src.p
rm $data/preprocessed_3/train.$tgt.p

# preprocess all files for model training
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared_3/ --train-prefix $data/preprocessed_3/train --valid-prefix $data/preprocessed_3/valid --test-prefix $data/preprocessed_3/test --tiny-train-prefix $data/preprocessed_3/tiny_train --threshold-src 10 --threshold-tgt 10 --num-words-src 4000 --num-words-tgt 4000

echo "done!"