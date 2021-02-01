from embedded_topic_model.utils import preprocessing


def test_create_etm_datasets():
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

    no_documents_in_train = 7
    no_documents_in_test = 3

    vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(documents, train_size=0.7)

    assert isinstance(vocabulary, list), "vocabulary isn't list"

    assert len(train_dataset['tokens']) == no_documents_in_train and len(train_dataset['counts']) == no_documents_in_train, \
        "lengths of tokens and counts for training dataset doesn't match"

    assert len(test_dataset['test']['tokens']) == no_documents_in_test and len(test_dataset['test']['counts']) == no_documents_in_test, \
        "lengths of tokens and counts for testing dataset doesn't match"
