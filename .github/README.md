# Natural Question Answering (trainer code)
Library to easily finetune a transformer on the NaturalQuestions task

# Usage
You can also check out the code in [this](https://colab.research.google.com/drive/1jOibWneDd3Bu7aAJqxk0PUFKYOM9ExoP?usp=sharing) colab file

### Set up environment
```
!git clone https://github.com/mariusjohan/NaturalQuestions.git

# Use % to permanently change working directory
%cd NaturalQuestions

!pip install -r requirements.txt

# Use cpu if running on CPU hardware otherwise cu111 if you have the newest cuda version
!pip install torch==1.7.1+{{hardware}} -f https://download.pytorch.org/whl/torch_stable.html
```

```python
MODEL_NAME = 'bert-base-uncased'

import config
config.MODEL_DIR = 'your/path/to/model-folder'
config.create_env()
```

### Setup Kaggle and download dataset
```python
if 'kaggle.json' not in os.listdir():
    # Set up Kaggle
    from google.colab import files
    files.upload()
    !mkdir ~/.kaggle 
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

    import kaggle

if 'simplified-nq-train.jsonl' not in os.listdir(config.DATA_DIR):
    # Downlad the files
    !kaggle competitions download -c tensorflow2-question-answering

    # Unzip the files
    !unzip simplified-nq-train.jsonl.zip -d $DATA_DIR
    !unzip simplified-nq-test.jsonl.zip -d $DATA_DIR
```

### Run model
```python
from main import create_args, train

# Create_args is only for easy testing. Not recommended when finetuning since the hyperparameters is "fixed"
args = create_args(MODEL_NAME)
train(**args)
```