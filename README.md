# Synth-AST

- For training all the layers of the model, use `run_train.py`.
- For training only the FC head while the AST backbone is kept frozen, use `run_train_2.py`.
- For training some of the AST layers, use `run_train_3.py`. A warm up for (default=2) epochs with AST frozen takes, then unfreeze up to (default=2) last layers. The default values can be changed in this script.
