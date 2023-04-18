# Cross-cultural communication using Machine Translation
A use of machine translation, an NLP technique, for language translation for efficient cross-cultural communication.


* Project Content
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.9 or above
* PyTorch >= 2.0 or above
* PytorchLightning >= 2.0 or above
* Gradio

For more details please refer `requirements.txt`

## Features
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `Trainer` handles partial model training (using checkpoint saving/resuming), training process logging, and more.
  * `DataLoader` handles batch generation, data shuffling, and training-validation-testing data splitting.
  * `BaseModel` provides basic model summary.
  * `T5` provides implementation of T5 architecture from the huggingface community

## Folder Structure
  ```
  NMT-project/
  │
  ├── README.md             - details of the complete project structure
  │
  ├── train.py              - main script to start training
  │
  ├── test.py               - evaluation of trained model
  │
  ├── gui.py                - inference of the model using GUI
  │
  ├── requirements.txt      - python package requirements
  │
  ├── config.json           - holds configuration for training
  │
  ├── dataloader/           - data pre-processing and data loading
  │   └── dataloader.py
  │
  ├── data/                 - default directory for storing input data (will be created during training)
  │
  ├── model/                - models, losses, and metrics
  │   │── T5.py	          - actual T5 model
  │   └── T5_sanity.py      - sanity test of the trained model
  │
  ├── trainer/              - training, validation and model optimization
  │   └── trainer.py
  │
  ├── slides/               - contains project slides and result sheet
  │  
  └── utils/                - small utility functions for printing model parameters
      └── util.py
  
  ```

## Usage
The code in this repository is an implementation of using T5 model for language translation using the techniques of Natural Language Processing in a Deep Learning Network.

Try `python train.py` to train.
Try `python test.py` to test.

### Config file format
Config files are in `.json` format:
```javascript
{
    "device": "cuda",                 // training device
    
    "dataset": {
        "ids": [0, 1, 2, 3, 4, 5, 6], // ids refer to the language mappings that should be used for training. language mapping is the indices of self.config_name at ./dataloader/dataloader.py 
        "cache_dir": "../dataset",    // dataset would be downloaded here
        "batch_size": 16,             // batch size for training
        "num_workers": 8              // number of parallel process to spawn to run data processing
    },

    "model": {                        
        "freeze_till": -1             // number of layers of model to freeze on training. -1 freeze any layers.
    },

    "optim_args": {
        "lr": 1e-4                    // optimizer learning rate
    },

    "trainer": {                        // pytorch-lightning training config
        "limit_train_batches": 0.25,    // ratio of training data to use in an epoch
        "max_epochs": 1,                // maximum number of epochs to train
        "deterministic": false,         // whether the model will run in deterministic mode
        "log_every_n_steps": 2048,      // number of steps after logs will be given
        "accelerator": "gpu",           // device to use for training the model
        "check_val_every_n_epoch": 1,   // number of epochs to check validation performance
        "precision": "16-mixed",        // model quantization
        "enable_progress_bar": true,    // progress bar 
        "default_root_dir": "./logs",   // directory to give the logs and model checkpoint
        "enable_checkpointing": true,   // whether to save the best model
        "benchmark": true,              // use the fastest algorithm while training
        "max_time": null                // any time-limit set to train the model
    },

    "weight": "T5.pth",                 // final T5 model weight used for inference

    "fit": {
        "ckpt_path": null               // path to the last checkpoint file (*.ckpt) used to resume training
    }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by adding the path of the `*.ckpt` file to the `ckpt_path` parameter in `config.json`. The `*.ckpt` file will be generated at the `default_root_dir` while training.


### Checkpoints and Weights
The last training checkpoint can be found in this link: [T5.ckpt](https://mega.nz/file/kjAEyIbY#wAT7NfvlumvXqphdBNarvd_5mX69_jQx_AJIA0FVD9o)
The best weight of the model can be found in this link: [T5.pth](https://mega.nz/file/tqRx0B7R#_ewI4E8ZGm-MYxHGdy0eED6ACczkgGgLiDt4xbFGGnE)

### Inference
To check the output of the model, first download the [T5.pth](https://mega.nz/file/tqRx0B7R#_ewI4E8ZGm-MYxHGdy0eED6ACczkgGgLiDt4xbFGGnE) model weights inside the directory configured in the config file under 'weight'. And then run the `gui.py`.

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is developed by [Sujesh Padhi](https://github.com/sujeshpadhi91) and [Quwsar Ohi](https://github.com/QuwsarOhi/) for course CPSC601 L05 - Natural
Language Processing.
