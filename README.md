# TeBaG-RL
# Text-Based Games with Reinfocement Learning
Project for the seminar course "Applied Deep Learning for NLP" at Technical University Munich.
Work done by Florian Donhauser, Noah Dormann, and Max Lamparth.

# Structure
- Agents: Holds the different types of agents, one with an NNLM embedding, and one with a BERT model
- Environments: Holds the different types of enviroments, currently a tf_environment which works as a large wrapper around TextWorld games
- Resources: .txt files with lists of verbs and objects which are used by the agent to interact with the TextWorld environment
- Scripts: These scripts are used to create the TextWorld games we use for training
- Tests: Some tests we wrote during developmen to test the agent and the environment
- ``tf_TextWorld_RL.ipynb``: The jupyter notebook file we encourage you to try out yourself. It has support for Google Colab using the link at the top of the file.
