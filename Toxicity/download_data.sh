#!/bin/bash

kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification 
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d data

chmod 750 data/test.csv
chmod 750 data/train.csv
chmod 750 data/sample_submission.csv

rm jigsaw-unintended-bias-in-toxicity-classification.zip