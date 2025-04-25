
# Starting environment or clean VM with e.g.:

sudo docker run -it --name fake_news --gpus=1 -p 8521:8888 nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

Prepare Python:

apt-get update
apt-get install git

https://github.com/GrimmV/fake_news_copilot.git

apt install python3
python3 -m venv .venv
cd fake_news_copilot
source .venv/bin/activate

Dependencies (Or do try `pip install -r requirements.txt`):

```
pip install pandas
pip install torch
pip install transformers
pip install scikit-learn
pip install textstat
pip install spacy
pip install nltk
python -m spacy download en_core_web_sm
pip install datasets
```

For sentence similarity: 

`pip install sentence-transformers`

`pip install notebook`

# How to work with this

Preprocess training data and train the model:

```python model_training_rf.py```

Store data with model predictions:

```python utils/store_training_data.py```

Run python REPL to store all XAI and other relevant Data:

`python`

`from utils.retrieve_xai import XAIRetriever`
`retriever = XAIRetriever()`
`retriever.retrieve_confusion()`
`for all methods in XAIRetriever:`
`retriever. ... `
