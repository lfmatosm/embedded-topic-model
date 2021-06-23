from embedded_topic_model.models import etm
import joblib
import torch


class TestETM:
    def test_etm_training_with_preprocessed_dummy_dataset(self):
        vocabulary, embeddings, train_dataset, _ = joblib.load(
            'tests/resources/train_resources.test')

        etm_instance = etm.ETM(
            vocabulary,
            embeddings,
            num_topics=3,
            epochs=50,
            train_embeddings=False,
        )

        expected_no_topics = 3
        expected_no_documents = 10
        expected_t_w_dist_sums = torch.ones(expected_no_topics)
        expected_d_t_dist_sums = torch.ones(expected_no_documents)

        etm_instance.fit(train_dataset)

        topics = etm_instance.get_topics()

        assert len(topics) == expected_no_topics, \
            "no. of topics error: exp = {}, result = {}".format(expected_no_topics, len(topics))

        coherence = etm_instance.get_topic_coherence()

        assert coherence != 0.0, \
            'zero coherence returned'

        diversity = etm_instance.get_topic_diversity()

        assert diversity != 0.0, \
            'zero diversity returned'

        t_w_mtx = etm_instance.get_topic_word_matrix()

        assert len(t_w_mtx) == expected_no_topics, \
            "no. of topics in topic-word matrix error: exp = {}, result = {}".format(expected_no_topics, len(t_w_mtx))

        t_w_dist = etm_instance.get_topic_word_dist()
        assert len(t_w_dist) == expected_no_topics, \
            "topic-word distribution error: exp = {}, result = {}".format(expected_no_topics, len(t_w_dist))

        similar_words = etm_instance.get_most_similar_words(["boat", "day", "sun"], 5)
        assert len(similar_words) == 3, \
            "get_most_similar_words error: expected {} keys but {} were returned for method call".format(3, len(similar_words))
        for key in similar_words.keys():
            assert 0 <= len(similar_words[key]) <= 5, \
                "get_most_similar_words error: expected <= {} elements but got {} for {} key".format(5, len(similar_words[key], key))


        t_w_dist_below_zero_elems = t_w_dist[t_w_dist < 0]
        assert len(t_w_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the topic-word distribution'

        t_w_dist_sums = torch.sum(t_w_dist, 1)
        assert torch.allclose(
            t_w_dist_sums, expected_t_w_dist_sums), "t_w_dist_sums error: exp = {}, result = {}".format(
            expected_t_w_dist_sums, t_w_dist_sums)

        d_t_dist = etm_instance.get_document_topic_dist()
        assert len(d_t_dist) == expected_no_documents, \
            "document-topics distribution error: exp = {}, result = {}".format(expected_no_documents, len(d_t_dist))

        d_t_dist_below_zero_elems = d_t_dist[d_t_dist < 0]
        assert len(d_t_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the document-topic distribution'

        d_t_dist_sums = torch.sum(d_t_dist, 1)
        assert torch.allclose(
            d_t_dist_sums, expected_d_t_dist_sums), "d_t_dist_sums error: exp = {}, result = {}".format(
            expected_d_t_dist_sums, d_t_dist_sums)

    def test_etm_training_with_preprocessed_dummy_dataset_and_embeddings_file(
            self):
        vocabulary, _, train_dataset, _ = joblib.load(
            'tests/resources/train_resources.test')

        etm_instance = etm.ETM(
            vocabulary,
            embeddings='tests/resources/train_w2v_embeddings.wordvectors',
            num_topics=3,
            epochs=50,
            train_embeddings=False,
        )

        expected_no_topics = 3
        expected_no_documents = 10
        expected_t_w_dist_sums = torch.ones(expected_no_topics)
        expected_d_t_dist_sums = torch.ones(expected_no_documents)

        etm_instance.fit(train_dataset)

        topics = etm_instance.get_topics()

        assert len(topics) == expected_no_topics, \
            "no. of topics error: exp = {}, result = {}".format(expected_no_topics, len(topics))

        coherence = etm_instance.get_topic_coherence()

        assert coherence != 0.0, \
            'zero coherence returned'

        diversity = etm_instance.get_topic_diversity()

        assert diversity != 0.0, \
            'zero diversity returned'

        t_w_mtx = etm_instance.get_topic_word_matrix()

        assert len(t_w_mtx) == expected_no_topics, \
            "no. of topics in topic-word matrix error: exp = {}, result = {}".format(expected_no_topics, len(t_w_mtx))

        t_w_dist = etm_instance.get_topic_word_dist()
        assert len(t_w_dist) == expected_no_topics, \
            "topic-word distribution error: exp = {}, result = {}".format(expected_no_topics, len(t_w_dist))

        t_w_dist_below_zero_elems = t_w_dist[t_w_dist < 0]
        assert len(t_w_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the topic-word distribution'

        t_w_dist_sums = torch.sum(t_w_dist, 1)
        assert torch.allclose(
            t_w_dist_sums, expected_t_w_dist_sums), "t_w_dist_sums error: exp = {}, result = {}".format(
            expected_t_w_dist_sums, t_w_dist_sums)

        d_t_dist = etm_instance.get_document_topic_dist()
        assert len(d_t_dist) == expected_no_documents, \
            "document-topics distribution error: exp = {}, result = {}".format(expected_no_documents, len(d_t_dist))

        d_t_dist_below_zero_elems = d_t_dist[d_t_dist < 0]
        assert len(d_t_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the document-topic distribution'

        d_t_dist_sums = torch.sum(d_t_dist, 1)
        assert torch.allclose(
            d_t_dist_sums, expected_d_t_dist_sums), "d_t_dist_sums error: exp = {}, result = {}".format(
            expected_d_t_dist_sums, d_t_dist_sums)

    def test_etm_training_with_preprocessed_dummy_dataset_and_c_wordvec_txt_embeddings_file(
            self):
        vocabulary, _, train_dataset, _ = joblib.load(
            'tests/resources/train_resources.test')

        etm_instance = etm.ETM(
            vocabulary,
            embeddings='tests/resources/train_w2v_embeddings.wordvectors.txt',
            num_topics=3,
            epochs=50,
            train_embeddings=False,
            use_c_format_w2vec=True,
        )

        expected_no_topics = 3
        expected_no_documents = 10
        expected_t_w_dist_sums = torch.ones(expected_no_topics)
        expected_d_t_dist_sums = torch.ones(expected_no_documents)

        etm_instance.fit(train_dataset)

        topics = etm_instance.get_topics()

        assert len(topics) == expected_no_topics, \
            "no. of topics error: exp = {}, result = {}".format(expected_no_topics, len(topics))

        coherence = etm_instance.get_topic_coherence()

        assert coherence != 0.0, \
            'zero coherence returned'

        diversity = etm_instance.get_topic_diversity()

        assert diversity != 0.0, \
            'zero diversity returned'

        t_w_mtx = etm_instance.get_topic_word_matrix()

        assert len(t_w_mtx) == expected_no_topics, \
            "no. of topics in topic-word matrix error: exp = {}, result = {}".format(expected_no_topics, len(t_w_mtx))

        t_w_dist = etm_instance.get_topic_word_dist()
        assert len(t_w_dist) == expected_no_topics, \
            "topic-word distribution error: exp = {}, result = {}".format(expected_no_topics, len(t_w_dist))

        t_w_dist_below_zero_elems = t_w_dist[t_w_dist < 0]
        assert len(t_w_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the topic-word distribution'

        t_w_dist_sums = torch.sum(t_w_dist, 1)
        assert torch.allclose(
            t_w_dist_sums, expected_t_w_dist_sums), "t_w_dist_sums error: exp = {}, result = {}".format(
            expected_t_w_dist_sums, t_w_dist_sums)

        d_t_dist = etm_instance.get_document_topic_dist()
        assert len(d_t_dist) == expected_no_documents, \
            "document-topics distribution error: exp = {}, result = {}".format(expected_no_documents, len(d_t_dist))

        d_t_dist_below_zero_elems = d_t_dist[d_t_dist < 0]
        assert len(d_t_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the document-topic distribution'

        d_t_dist_sums = torch.sum(d_t_dist, 1)
        assert torch.allclose(
            d_t_dist_sums, expected_d_t_dist_sums), "d_t_dist_sums error: exp = {}, result = {}".format(
            expected_d_t_dist_sums, d_t_dist_sums)

    def test_etm_training_with_preprocessed_dummy_dataset_and_c_wordvec_bin_embeddings_file(
            self):
        vocabulary, _, train_dataset, _ = joblib.load(
            'tests/resources/train_resources.test')

        etm_instance = etm.ETM(
            vocabulary,
            embeddings='tests/resources/train_w2v_embeddings.wordvectors.bin',
            num_topics=3,
            epochs=50,
            train_embeddings=False,
            use_c_format_w2vec=True,
        )

        expected_no_topics = 3
        expected_no_documents = 10
        expected_t_w_dist_sums = torch.ones(expected_no_topics)
        expected_d_t_dist_sums = torch.ones(expected_no_documents)

        etm_instance.fit(train_dataset)

        topics = etm_instance.get_topics()

        assert len(topics) == expected_no_topics, \
            "no. of topics error: exp = {}, result = {}".format(expected_no_topics, len(topics))

        coherence = etm_instance.get_topic_coherence()

        assert coherence != 0.0, \
            'zero coherence returned'

        diversity = etm_instance.get_topic_diversity()

        assert diversity != 0.0, \
            'zero diversity returned'

        t_w_mtx = etm_instance.get_topic_word_matrix()

        assert len(t_w_mtx) == expected_no_topics, \
            "no. of topics in topic-word matrix error: exp = {}, result = {}".format(expected_no_topics, len(t_w_mtx))

        t_w_dist = etm_instance.get_topic_word_dist()
        assert len(t_w_dist) == expected_no_topics, \
            "topic-word distribution error: exp = {}, result = {}".format(expected_no_topics, len(t_w_dist))

        t_w_dist_below_zero_elems = t_w_dist[t_w_dist < 0]
        assert len(t_w_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the topic-word distribution'

        t_w_dist_sums = torch.sum(t_w_dist, 1)
        assert torch.allclose(
            t_w_dist_sums, expected_t_w_dist_sums), "t_w_dist_sums error: exp = {}, result = {}".format(
            expected_t_w_dist_sums, t_w_dist_sums)

        d_t_dist = etm_instance.get_document_topic_dist()
        assert len(d_t_dist) == expected_no_documents, \
            "document-topics distribution error: exp = {}, result = {}".format(expected_no_documents, len(d_t_dist))

        d_t_dist_below_zero_elems = d_t_dist[d_t_dist < 0]
        assert len(d_t_dist_below_zero_elems) == 0, \
            'there are elements smaller than 0 in the document-topic distribution'

        d_t_dist_sums = torch.sum(d_t_dist, 1)
        assert torch.allclose(
            d_t_dist_sums, expected_d_t_dist_sums), "d_t_dist_sums error: exp = {}, result = {}".format(
            expected_d_t_dist_sums, d_t_dist_sums)
