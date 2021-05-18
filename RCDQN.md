# Multi-Passage Reading Comprehension Deep Q-Network
https://github.com/XiaoxiaoGuo/rcdqn

## Dependencies
- Python
- Jericho https://github.com/microsoft/jericho
- PyTorch
- spaCy

## Setup
- create new conda environment python=3.7
- pip install jericho==2.4.2
- pip install spaCy==3.0.4
- pip install torch torchvision
- python -m spacy download en_core_web_sm
- fix jericho/util.py:

    replace line 42
    ``` python 
    spacy_nlp = spacy.load('en')
    ```
    with
    ``` python
    import en_core_web_sm
    spacy_nlp = en_core_web_sm.load()
    ```

- download http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip
- and copy glove.6B.100d.txt into rcdqn/agents/glove/
- run ``` python glove_utils.py ```
- run ``` python train.py --batch_size 64 --env_id "zork1.z5 "```

## Results
- default training steps 100.000
- estimated time 5h
- full gpu memory usage 8g, ram 5g, 10%cpu load
- killed because of gpu out of memory, memory leak???
