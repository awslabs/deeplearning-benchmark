#!/bin/bash
set -exuo pipefail

git clone https://github.com/Ishitori/gluon-nlp
cd gluon-nlp/
git checkout bidaf_assembled

pip3 install gluonnlp --pre --user
pip3 install nltk --user
python3 -m nltk.downloader punkt

python3 -m scripts.question_answering.train_question_answering --train --evaluate --preprocessed_dataset_path scripts/question_answering/preprocessed_dataset_q30_nltk.p --save_dir scripts/question_answering/output_1gpu/ --word_vocab_path scripts/question_answering/squad_word_vocab.p --char_vocab_path scripts/question_answering/squad_char_vocab.p --lr 4 --batch_size 240 --gpu 0,1,2,3 --early_stop 0 --log_interval 0

cd ..
rm -rf gluon-nlp/
