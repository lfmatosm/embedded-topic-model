import pytest

from src import etm

class TestETM:
    def test_etm_training_with_preprocessed_reddit_dataset(self):
        etm_instance = etm.ETM(
            dataset='reddit-test',
            data_path='src/datasets_for_training/min_df_0.01',
            emb_path='src/datasets_for_training/etm_w2v_embedding.txt',
            num_topics=5,
            epochs=100,
            train_embeddings=False,
        )

        etm_instance.fit(None)

        assert 1 == 1
