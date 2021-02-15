from embedded_topic_model.utils import embedding, preprocessing
import os
import joblib

sentences = [
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
vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
    sentences, debug_mode=True)

embeddings = embedding.create_word2vec_embedding_from_dataset(
    sentences, 
    embedding_file_path='tests/resources/train_w2v_embeddings.wordvectors', 
    save_c_format_w2vec=True,
    debug_mode=True,
)

os.makedirs(os.path.dirname('tests/resources/train_resources.test'), exist_ok=True)
joblib.dump(
    (vocabulary,
     embeddings,
     train_dataset,
     test_dataset),
    './train_resources.test',
    compress=8)

print('the end')
