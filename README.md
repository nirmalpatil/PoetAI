>Aditya Malani\
>Carnegie Mellon University\
>Pittsburgh, PA 15213\
amalani@andrew.cmu.edu

>Aditya Bindra\
>Carnegie Mellon University\
>Pittsburgh, PA 15213\
adityabi@andrew.cmu.edu

>Bandish Parikh\
>Carnegie Mellon University\
>Pittsburgh, PA 15213\
>bparikh@andrew.cmu.edu

>Nirmalsing Patil\
>Carnegie Mellon University\
>Pittsburgh, PA 15213\
>nirmalsp@andrew.cmu.edu

>Yashash Gaurav\
>Carnegie Mellon University\
>Pittsburgh, PA 15213\
ygaurav@andrew.cmu.edu

'''
├── LICENSE
├── README.md
├── data
│   ├── limericks_clean_with_@.csv
│   ├── limericks_clean_with_@and#.csv
│   ├── limericks_no_profanity.csv
│   ├── limericks_no_punc_digit.csv
│   └── limericks_original.csv
├── evaluations
│   └── rhyming_evaluation.ipynb
├── experiments
│   └── PoetAI_345M.ipynb
├── preprocessing
│   └── Limerick_Processing.ipynb
└── readmeassets
    └── images
        ├── context.jpeg
        ├── eval.png
        ├── gantt_chart.png
        ├── metrics_epoch_loss.png
        └── training_flow.jpeg
'''
# PoetAI
Generation of Poem by training and fine tuning GPT2
For the longest time, machine learning models have been deployed to perform
analytical tasks and identify patterns in data that are otherwise difficult to discern
by the human eye. These tasks are highly objective in nature and have a fixed
“true” value that we want the machine to be able to predict. However, what if we
could use ‘art’-ificial intelligence to create art? The expected output here wouldn’t
be a numerical value or a class, but rather an art piece that is creative, expressive,
and appealing to humans. For our project, we focus on using machine learning to
generate poetry. Specifically, PoetAI generates limericks (5-line contextual poems
with AABBA rhyme scheme) of high quality where quality is assessed using 2
metrics - rhyming and context. We are proposing a pipeline driven by transformer-
based models that generates limericks and then scores and filters the good quality
limericks score based on various aspects including rhyme and context. A final
language model tries to fix the rhyming to enhance limerick quality. PoetAI holds
numerous business applications, especially in the field of customized marketing
and messaging

# 1 Introduction
Poetry is an outcome of creation. It is a form of literary work that is often characterized by an
interplay of thought-provoking words meant to stimulate the human mind. A limerick is a short
humorous form of verse that has five lines and follows the AABBA rhyme scheme. What if we
could use ‘art’-ificial intelligence to create art? While a lot of research has been done in the field of
Natural Language Understanding, the area pertaining to generation and qualitative analysis of poetry
still remains to be explored. Even after much research, AI has been criticized for lacking creativity
and for occasionally producing texts that make no sense. The expected output in this case isn’t a
simple numerical value or a class label, but rather an art piece that is meant to be creative, expressive,
and appealing to humans. We derived that, in the context of an Artificial Intelligence algorithm,
creativity is just the development of clearly stated mathematical objective functions that a model must
be optimized on. The desired output of creativity can not be captured by conventional loss functions.
Our main objective is to respond to the question, "What makes this piece of poetry/limerick a good
one?" while giving objective functions to grade a specific piece of poetry/limerick. Will a large
language model be able to learn the art of poetry? In this project we used 4 models as described in the
flow section, GPT2 for limerick generation, LSTM based rhyme scorer for evaluating the rhyming
33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
scheme of the limerick, pretrained sentence transformer model MiniLM-L6-V2 for evaluating context
of the limericks and Masking BERT for fixing the rhyme of a limerick with a good context score. This
paper starts with Literature review section followed by methodology which contains the details about
the architecture of all four models used. All the experiments are listed in Training and Experiment
section.

# 2 Literature Review
There have been few attempts in poetry compared to other use cases where GPT-2 is used for, and each of them have employed varied techniques and approaches towards poetry generation. Some attempts with Sonnet and RNNs are also studied to understand the evolution of poetry generation over the time.

## 2.1 GPoet-2 : GPT2 based transformer with forward and reverse fine tuning
Another work that we referenced is GPoeT-2: A GPT-2 Based Poem Generator [6], where the authors
propose a two-stage free-form limerick generation. The proposed two-stage generation uses the
forward language model to generate a limerick’s first line with high quality and diversity, then uses
the reverse language model to generate the rest of the four lines given the first line generated by
forwarding LM. They also select and evaluate a few metrics that quantify the idea of “good poetry”
such as syntactical correctness, lexical diversity, and subject/topic consistency.

## 2.2 LimGen : GPT2 based limerick generator which uses Search Metrics to enforce rhyme
One of the recent works in this space is another GPT-2 implementation called LimGen by Jianyou
Wang et al.[8] where they use search metrics such as the Adaptive Multi-Templated Constraint
algorithm that constrains their search to the space of realistic poems, the Multi-Templated Beam
Search algorithm which searches efficiently through the space, and the probabilistic story-line
algorithm that aims to provide coherent story-lines related to a user-provided prompt word.

## 2.3 Deep-Speare : RNN based Sonnet generator with rhyming dictionary to enforce rhyming
Deep-speare[17] is a Sonnet based model used to capture language, rhyming and meter of poetry.
These models under-performed in generating human level poetry but served as good reference for
rhyme capture with models. Rhyme was enforced by a cosine similarity of the last words generated
by the model and a loss function was employed to penalize model when it was not rhyming. A
rhyming dictionary was maintained to pick words based on the context.



# 3 Dataset description

As ours is a unsupervised learning problem, our dataset does not have labels and consequently does not have validation and test sets. Except we need a corpus of training data to teach out GPT how to behave given prompts or without them. Our model is expected to produce limericks that conform to the AABBA rhyme scheme and so our dataset, for now, is solely based on Sam Ballas’ datasets of 90,000 limericks [2], which he collected by scraping the oedilf.com website. We are working with this dataset solely because of its size which also helps us teach our model about the varied ways that rhyming structures can work. Some samples from our dataset:

> Sample 1:\
> the ball was defended by cole\
> and over the goal-line did roll.\
> the cross to be borne: a\
> quick kick from the corner?\
> a header, a strike, it's a goal!

> Sample 2:\
> how dare you invade my sweet life,\
> you bringer of conflict and strife?\
> until you came along,\
> not a thing had gone wrong,\
> but now discord and friction are rife!

Following this, we want to further explore using Mad Kane’s repository of humourous limericks [2]. “A smile is a curve that sets everything straight.” – Phyllis Diller by teaching our model to be humourous (we hope we do) we want to help the academics who read our work laugh a little as they build on our work. Some of the sample we hope to incorporate as we go forward:


> Sample 1:\
> A strange silhouette in the sky;\
> A rustling of wings from on high.\
> Not angels divine,\
> But migrating swine –\
> Those pigs finally learned how to fly!\
> – Paul Haebig



> Sample 2:\
> There was an inventor named Knight\
> Who studied the science of flight.\
> He thought he’d be first,\
> But his efforts were cursed.\
> His designs never turned out quite Wright.\
> – Fred Bortz


We are hoping to come up with metrics that can help us understand the performance of our model. Tracking rhyme scheme, and coherence. And if possible add that as a loss that the model may try to minimize.





# 4 Conclusion
We introduced a transformers based pipeline for limerick generation, evaluation and correction. The
idea was to utilize deep learning models to generate limericks that appeal to humans. While our
approach sets up a good baseline for limerick generation and evaluation, there is scope to further
11
improve and build on it. Firstly, our current pipeline focuses on context and rhyming. However,
there are various other aspects of limericks that can be included like figure of speech, punchline,
etc. Secondly, we can improve on our correction model to further enhance its ability to improve a
limerick. Lastly, the current pipeline is generation-filtration. There are possibilities to somehow teach
the model to rhyme training by penalizing it for not adhering to rhyme scheme. The possibilities
with art generation using NLP are endless and hold immense business and real-world applications
including but not limited to customized marketing, education, etc. With our work, we were able to
generate decent quality limericks and successfully quantify two important aspects of rhyming and
context.

# References
[1] Alexander, J.A. & Mozer, M.C. (1995) Template-based algorithms for connectionist rule extraction. In
G. Tesauro, D.S. Touretzky and T.K. Leen (eds.), Advances in Neural Information Processing Systems 7, pp.
609–616. Cambridge, MA: MIT Press.
[2] Bower, J.M. & Beeman, D. (1995) The Book of GENESIS: Exploring Realistic Neural Models with the
GEneral NEural SImulation System. New York: TELOS/Springer–Verlag.
[3] Hasselmo, M.E., Schnell, E. & Barkai, E. (1995) Dynamics of learning and recall at excitatory recurrent
synapses and cholinergic modulation in rat hippocampal region CA3. Journal of Neuroscience 15(7):5249-5262.
[4] Previous PoetAI work : https://github.com/YashashGaurav/poetai-public
[5]T. Nguyen, P. Nguyen, H. Pham, T. Bui, T. Nguyen and D. Luong, "SP-GPT2: Semantics Improvement in
Vietnamese Poetry Generation," 2021 20th IEEE International Conference on Machine Learning and Applications
(ICMLA), 2021, pp. 1576-1581, doi: 10.1109/ICMLA52953.2021.00252.
[6] Kai-Ling Lo, Rami Ariss and Philipp Kurz (2022), “GPoeT-2: A GPT-2 Based Poem Generator”, https:
//arxiv.org/pdf/2205.08847.pdf
[7] S. Ballas, “PoetRNN,” Available at https://github.com/sballas8/PoetRNN/ (2015)
[8] Jianyou Wang, Xiaoxuan Zhang, Yuren Zhou, Christopher Suh, Cynthia Rudin There Once Was a Really Bad
Poet, It Was Automated but You Didn’t Know It. Transactions of the Association for Computational Linguistics
2021; 9 605–620. doi: https://doi.org/10.1162/tacl{_}a{_}00387
[9] N.A., https://developers.google.com/machine-learning/gan/discriminator
[10] L.S. Marvin (2018) Siamese LSTM, https://github.com/MarvinLSJ/LSTM-siamese
[11] cschaefer26 (2021), DeepPhonemizer, https://github.com/as-ideas/DeepPhonemizer
[12] Zenodo Dataset , https://zenodo.org/record/5722527#.Y0szkOzMK3I4
[13] Prosodic, a metrical-phonological parser written in Python, https://github.com/quadrismegistus/
prosodic
[14]Understanding the open pre-trained transformers (OPT) library. (n.d.).
Retrieved November 15, 2022, from https://towardsdatascience.com/
understanding-the-open-pre-trained-transformers-opt-library-193a29c14a15
[15]Decoder-only architecture used by GPT-2. - researchgate. (n.d.). Retrieved November 15, 2022, from
https://www.researchgate.net/figure/Decoder-Only-Architecture-used-by-GPT-2_fig1_
349521999
[16]The annotated GPT-2. Committed towards better future. (2020, February 18). Retrieved November 14, 2022,
from https://amaarora.github.io/2020/02/18/annotatedGPT2.html
[17] Jey Han Lau et al. “Deep-speare: A Joint Neural Model of Poetic Language, Meter and Rhyme”. In: (2018).
DOI: 10.48550/ARXIV.1807.03491. URL: https://arxiv.org/abs/1807. 03491.
[18] all-MiniLM-L6-V2, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[19]ApurbaSengupta. (n.d.). Apurbasengupta/Lyrics-generation-using-bert: Generating English rock lyrics
using bert. GitHub. Retrieved December 9, 2022, from https://github.com/ApurbaSengupta/
Lyrics-Generation-using-BERT
[20]Bert 101 - state of the art NLP model explained. BERT 101 - State Of The Art NLP Model Explained. (n.d.).
Retrieved December 9, 2022, from https://huggingface.co/blog/bert-101
[21]Oedilf. (n.d.). Retrieved December 14, 2022, from http://www.oedilf.com/db/Lim.php
[22]Abdibayev, A., Riddell, A., Igarashi, Y., amp; Rockmore, D. (2021, September 21).
Dataset of limericks for computational poetics. Zenodo. Retrieved December 14, 2022, from
https://zenodo.org/record/5722527.Y0szkOzMK3I
[23]Horev, R. (2018, November 17). Bert explained: State of the art language model for NLP. Medium. Retrieved
December 14, 2022, from https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-
nlp-f8b21a9b6270
13
