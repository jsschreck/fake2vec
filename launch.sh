#!/bin/bash

mkdir data models results
mkdir models/classifier models/doc2vec

python utils/scrape_publisher.py
python utils/text_processor.py
python utils/train_doc2vec.py
python utils/train_classifier.py

cd app
python main.py
