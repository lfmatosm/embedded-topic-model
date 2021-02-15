from embedded_topic_model.utils import embedding
from gensim.models import KeyedVectors


def test_create_word2vec_embedding_from_dataset():
    documents = [
        "Peanut butter and jelly caused the elderly lady to think about her past.",
        "Toddlers feeding raccoons surprised even the seasoned park ranger.",
        "You realize you're not alone as you sit in your bedroom massaging your calves after a long day of playing tug-of-war with Grandpa Joe in the hospital.",
        "She wondered what his eyes were saying beneath his mirrored sunglasses.",
        "He was disappointed when he found the beach to be so sandy and the sun so sunny.",
        "Flesh-colored yoga pants were far worse than even he feared.",
        "The wake behind the boat told of the past while the open sea for told life in the unknown future.",
        "Improve your goldfish's physical fitness by getting him a bicycle.",
        "Harrold felt confident that nobody would ever suspect his spy pigeon.",
        "Nudist colonies shun fig-leaf couture.",
    ]

    dimensionality = 240
    embeddings = embedding.create_word2vec_embedding_from_dataset(
        documents, dim_rho=dimensionality)

    assert isinstance(
        embeddings, KeyedVectors), "embeddings isn't KeyedVectors instance"

    vector = embeddings['Peanut']

    assert len(vector) == dimensionality, "lenght of 'Peanut' vector doesn't match: exp = {}, result = {}".format(
        dimensionality, len(vector))
