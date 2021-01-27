import joblib
from embedded_topic_model import etm


class TestETM:
    def test_etm_training_with_preprocessed_dummy_dataset(self):
        vocabulary, embeddings, train_dataset, _, _ = joblib.load('test/resources/train_resources.test')

        etm_instance = etm.ETM(
            vocabulary,
            embeddings,
            num_topics=3,
            epochs=50,
            train_embeddings=False,
        )

        etm_instance.fit(train_dataset)

        topics = etm_instance.get_topics()

        assert isinstance(topics, list) and len(topics) == 3
