# contextual-word-analogies
Application of word analogy task on contextual embeddings.

We provide two main scripts for evaluation of ELMo embeddings on word analogy task: one for Embeddia (and original English ELMo) ELMo models (`elmo_analogies.py`), one for models by ELMoForManyLangs (`efml_analogies.py`). The reason for two scripts is that the format of embeddings differs.
The whole process is a bit complicated and simplified for ELMo (not efml) via a bash script (`elmo_analogies.sh`), which first makes sure all the words we're looking for appear in the vocabulary, then calculates the non-contextual layer of ELMo embeddings (via `elmo_layer0/get_layer0_embs.py`). Finally it evaluates the contextual embeddings with `elmo_analogies.py`, utilizing just calculated non-contextual layer in the process (to speed up already lenghty evaluation ~500-1000 minutes).
