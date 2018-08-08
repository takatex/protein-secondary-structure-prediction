# protein-secondary-structure-prediction

PyTorch implementations of protein secondary structure prediction on CB513.

I used CB513 dataset of https://github.com/alrojo/CB513.

# Usage
```
usage: python main.py [-h] [-e EPOCHS] [-b BATCH_SIZE_TRAIN]
                      [-b_test BATCH_SIZE_TEST] [-k K_FOLD] [--save_dir SAVE_DIR]
                      [--no_cuda] [--seed S]

Protein Secondary Structure Prediction

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to run (default: 1000)
  -b BATCH_SIZE_TRAIN, --batch_size_train BATCH_SIZE_TRAIN
                        input batch size for training (default: 128)
  -b_test BATCH_SIZE_TEST, --batch_size_test BATCH_SIZE_TEST
                        input batch size for testing (default: 1024)
  -k K_FOLD, --k_fold K_FOLD
                        K-Folds cross-validator (default: 10)
  --save_dir SAVE_DIR   Result path (default: ../data/result)
  --no_cuda             disables CUDA training
  --seed S              random seed (default: 1)
```

# Reference
- https://arxiv.org/pdf/1604.07176.pdf
- https://arxiv.org/pdf/1604.07176.pdf
