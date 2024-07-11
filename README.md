# CAS
Master's thesis: Leverage Large Language Models for Implicit Constrained Data Augmentation with Semantic Enrichment in Biomedical Relation Extraction
## Biomedical Relation Extraction
### Fine-tuning RE model
```
python script.py train --mode biore --train_file dataset/train_test_aug2.csv --test_file test_transformers.csv
```
### Fine-tuning Novelty model
```
python script.py train --mode novelty --train_file dataset/train_original_large.csv --test_file dataset/test.csv
```
### Inference of both RE and Novelty
```
python script.py evaluate --eval_file dataset/bc8_biored_task1_test.csv --path_relation model_all --path_novel model_novelty_no_none_novel --output_file result/bc8_biored_task1_test-relation-no_none_novel-modelall.csv
```
