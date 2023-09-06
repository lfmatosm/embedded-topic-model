# Changelog

This changelog was inspired by the [keep-a-changelog](https://github.com/olivierlacan/keep-a-changelog) project and follows [semantic versioning](https://semver.org).

## [1.1.0] - 2023-09-05

### Added

- ([#3f27ee](https://github.com/lffloyd/embedded-topic-model/commit/3f27ee)) adds `transform` method
- ([#f98f3f](https://github.com/lffloyd/embedded-topic-model/commit/f98f3f)) adds example jupyter notebook
- ([#683bec](https://github.com/lffloyd/embedded-topic-model/commit/683bec)) adds contributing and conduct guidelines

### Changed

- ([#f98f3f](https://github.com/lffloyd/embedded-topic-model/commit/f98f3f), [#c918a4](https://github.com/lffloyd/embedded-topic-model/commit/c918a4)) updates documentation

## [1.0.2] - 2021-06-23

### Changed

- deactivates debug mode by default
- documents get_most_similar_words method

## [1.0.1] - 2021-02-15

### Changed

- optimizes original word2vec TXT file input for model training
- updates README.md

## [1.0.0] - 2021-02-15

### Added

- adds support for original word2vec pretrained embeddings files on both formats (BIN/TXT)

### Changed

- optimizes handling of gensim's word2vec mapping file for better memory usage

## [0.1.1] - 2021-02-01

### Added

- support for python 3.6

## [0.1.0] - 2021-02-01

### Added

- ETM training with partially tested support for original ETM features.
- ETM corpus preprocessing scripts - including word2vec embeddings training - adapted from the original code.
- adds methods to retrieve document-topic and topic-word probability distributions from the trained model.
- adds docstrings for tested API methods.
- adds unit and integration tests for ETM and preprocessing scripts.
