# Named Entity Recognition with Character-Level Models

## Main Goal

* Implementation of ["Named Entity Recognition with Character-Level Models"](https://nlp.stanford.edu/manning/papers/conll-ner.pdf) by Dan Klein, Joseph Smarr, Huy Nguyen and Christopher D. 

* Try to adopt the paper idea to Hebrew

## Models

### Character level HMM

This model characters are emitted one at a time and there is one state per character, each state's identity depends only on the previous state and each character's identity depends on both the current state and on the previous n-1 characters. 
In the figure below, we can see the HMM where C nodes are characters and S nodes are states.

![](https://user-images.githubusercontent.com/11094765/43415766-1ce71600-943f-11e8-9517-86143c496cf6.jpg)

### A Character-Based CMM (MEMM)

This model each state's identity depends on the it's corresponding character the previous state and additional features that can include additional previous states or characters, part of speech, or capitalization indicator (for corresponding character or previous), given this model we will use Viterbi algorithm to find the most likely labeling. 

## Dataset

**CoNLL 2003** (English) -  includes 1,393 English articles where 4 entity types are labeled - **LOC** (location), **ORG** (organization), **PER** (person) and **MISC** (miscellaneous). 
The dataset is divided into train set (946 articles), test set (231 articles) and development set (216 articles). 

**Sport5** (Hebrew) - includes blabla 
