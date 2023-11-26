# Personalized Daily Arxiv Papers 11/26/2023
Total relevant papers: 10

Table of contents with paper titles:

0. [Visual In-Context Prompting](#link0)
**Authors:** Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, Lei Zhang, Jianfeng Gao

1. [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](#link1)
**Authors:** Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, Varun Jampani

2. [Covariance alignment: from maximum likelihood estimation to
  Gromov-Wasserstein](#link2)
**Authors:** Yanjun Han, Philippe Rigollet, George Stepaniants

3. [Labeling Neural Representations with Inverse Recognition](#link3)
**Authors:** Kirill Bykov, Laura Kopf, Shinichi Nakajima, Marius Kloft, Marina M. -C. Höhne

4. [Risk-sensitive Markov Decision Process and Learning under General
  Utility Functions](#link4)
**Authors:** Zhengqi Wu, Renyuan Xu

5. [A Survey of Serverless Machine Learning Model Inference](#link5)
**Authors:** Kamil Kojs

6. [On diffusion-based generative models and their error bounds: The
  log-concave case with full convergence estimates](#link6)
**Authors:** Stefano Bruno, Ying Zhang, Dong-Young Lim, Ömer Deniz Akyildiz, Sotirios Sabanis

7. [Adaptive Sampling for Deep Learning via Efficient Nonparametric Proxies](#link7)
**Authors:** Shabnam Daghaghi, Benjamin Coleman, Benito Geordie, Anshumali Shrivastava

8. [PaSS: Parallel Speculative Sampling](#link8)
**Authors:** Giovanni Monea, Armand Joulin, Edouard Grave

9. [$σ$-PCA: a unified neural model for linear and nonlinear principal
  component analysis](#link9)
**Authors:** Fahdi Kanavati, Lucy Katsnith, Masayuki Tsuneki

---
## 0. [Visual In-Context Prompting](https://arxiv.org/abs/2311.13601) <a id="link0"></a>
**ArXiv ID:** 2311.13601
**Authors:** Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, Lei Zhang, Jianfeng Gao

**Abstract:** In-context prompting in large language models (LLMs) has become a prevalent
approach to improve zero-shot capabilities, but this idea is less explored in
the vision domain. Existing visual prompting methods focus on referring
segmentation to segment the most relevant object, falling short of addressing
many generic vision tasks like open-set segmentation and detection. In this
paper, we introduce a universal visual in-context prompting framework for both
tasks. In particular, we build on top of an encoder-decoder architecture, and
develop a versatile prompt encoder to support a variety of prompts like
strokes, boxes, and points. We further enhance it to take an arbitrary number
of reference image segments as the context. Our extensive explorations show
that the proposed visual in-context prompting elicits extraordinary referring
and generic segmentation capabilities to refer and detect, yielding competitive
performance to close-set in-domain datasets and showing promising results on
many open-set segmentation datasets. By joint training on COCO and SA-1B, our
model achieves $57.7$ PQ on COCO and $23.2$ PQ on ADE20K. Code will be
available at https://github.com/UX-Decoder/DINOv.


---

## 1. [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](https://arxiv.org/abs/2311.13600) <a id="link1"></a>
**ArXiv ID:** 2311.13600
**Authors:** Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, Varun Jampani

**Abstract:** Methods for finetuning generative models for concept-driven personalization
generally achieve strong results for subject-driven or style-driven generation.
Recently, low-rank adaptations (LoRA) have been proposed as a
parameter-efficient way of achieving concept-driven personalization. While
recent work explores the combination of separate LoRAs to achieve joint
generation of learned styles and subjects, existing techniques do not reliably
address the problem; they often compromise either subject fidelity or style
fidelity. We propose ZipLoRA, a method to cheaply and effectively merge
independently trained style and subject LoRAs in order to achieve generation of
any user-provided subject in any user-provided style. Experiments on a wide
range of subject and style combinations show that ZipLoRA can generate
compelling results with meaningful improvements over baselines in subject and
style fidelity while preserving the ability to recontextualize. Project page:
https://ziplora.github.io


---

## 2. [Covariance alignment: from maximum likelihood estimation to
  Gromov-Wasserstein](https://arxiv.org/abs/2311.13595) <a id="link2"></a>
**ArXiv ID:** 2311.13595
**Authors:** Yanjun Han, Philippe Rigollet, George Stepaniants

**Abstract:** Feature alignment methods are used in many scientific disciplines for data
pooling, annotation, and comparison. As an instance of a permutation learning
problem, feature alignment presents significant statistical and computational
challenges. In this work, we propose the covariance alignment model to study
and compare various alignment methods and establish a minimax lower bound for
covariance alignment that has a non-standard dimension scaling because of the
presence of a nuisance parameter. This lower bound is in fact minimax optimal
and is achieved by a natural quasi MLE. However, this estimator involves a
search over all permutations which is computationally infeasible even when the
problem has moderate size. To overcome this limitation, we show that the
celebrated Gromov-Wasserstein algorithm from optimal transport which is more
amenable to fast implementation even on large-scale problems is also minimax
optimal. These results give the first statistical justification for the
deployment of the Gromov-Wasserstein algorithm in practice.


---

## 3. [Labeling Neural Representations with Inverse Recognition](https://arxiv.org/abs/2311.13594) <a id="link3"></a>
**ArXiv ID:** 2311.13594
**Authors:** Kirill Bykov, Laura Kopf, Shinichi Nakajima, Marius Kloft, Marina M. -C. Höhne

**Abstract:** Deep Neural Networks (DNNs) demonstrated remarkable capabilities in learning
complex hierarchical data representations, but the nature of these
representations remains largely unknown. Existing global explainability
methods, such as Network Dissection, face limitations such as reliance on
segmentation masks, lack of statistical significance testing, and high
computational demands. We propose Inverse Recognition (INVERT), a scalable
approach for connecting learned representations with human-understandable
concepts by leveraging their capacity to discriminate between these concepts.
In contrast to prior work, INVERT is capable of handling diverse types of
neurons, exhibits less computational complexity, and does not rely on the
availability of segmentation masks. Moreover, INVERT provides an interpretable
metric assessing the alignment between the representation and its corresponding
explanation and delivering a measure of statistical significance, emphasizing
its utility and credibility. We demonstrate the applicability of INVERT in
various scenarios, including the identification of representations affected by
spurious correlations, and the interpretation of the hierarchical structure of
decision-making within the models.


---

## 4. [Risk-sensitive Markov Decision Process and Learning under General
  Utility Functions](https://arxiv.org/abs/2311.13589) <a id="link4"></a>
**ArXiv ID:** 2311.13589
**Authors:** Zhengqi Wu, Renyuan Xu

**Abstract:** Reinforcement Learning (RL) has gained substantial attention across diverse
application domains and theoretical investigations. Existing literature on RL
theory largely focuses on risk-neutral settings where the decision-maker learns
to maximize the expected cumulative reward. However, in practical scenarios
such as portfolio management and e-commerce recommendations, decision-makers
often persist in heterogeneous risk preferences subject to outcome
uncertainties, which can not be well-captured by the risk-neural framework.
Incorporating these preferences can be approached through utility theory, yet
the development of risk-sensitive RL under general utility functions remains an
open question for theoretical exploration.
  In this paper, we consider a scenario where the decision-maker seeks to
optimize a general utility function of the cumulative reward in the framework
of a Markov decision process (MDP). To facilitate the Dynamic Programming
Principle and Bellman equation, we enlarge the state space with an additional
dimension that accounts for the cumulative reward. We propose a discretized
approximation scheme to the MDP under enlarged state space, which is tractable
and key for algorithmic design. We then propose a modified value iteration
algorithm that employs an epsilon-covering over the space of cumulative reward.
When a simulator is accessible, our algorithm efficiently learns a near-optimal
policy with guaranteed sample complexity. In the absence of a simulator, our
algorithm, designed with an upper-confidence-bound exploration approach,
identifies a near-optimal policy while ensuring a guaranteed regret bound. For
both algorithms, we match the theoretical lower bounds for the risk-neutral
setting.


---

## 5. [A Survey of Serverless Machine Learning Model Inference](https://arxiv.org/abs/2311.13587) <a id="link5"></a>
**ArXiv ID:** 2311.13587
**Authors:** Kamil Kojs

**Abstract:** Recent developments in Generative AI, Computer Vision, and Natural Language
Processing have led to an increased integration of AI models into various
products. This widespread adoption of AI requires significant efforts in
deploying these models in production environments. When hosting machine
learning models for real-time predictions, it is important to meet defined
Service Level Objectives (SLOs), ensuring reliability, minimal downtime, and
optimizing operational costs of the underlying infrastructure. Large machine
learning models often demand GPU resources for efficient inference to meet
SLOs. In the context of these trends, there is growing interest in hosting AI
models in a serverless architecture while still providing GPU access for
inference tasks. This survey aims to summarize and categorize the emerging
challenges and optimization opportunities for large-scale deep learning serving
systems. By providing a novel taxonomy and summarizing recent trends, we hope
that this survey could shed light on new optimization perspectives and motivate
novel works in large-scale deep learning serving systems.


---

## 6. [On diffusion-based generative models and their error bounds: The
  log-concave case with full convergence estimates](https://arxiv.org/abs/2311.13584) <a id="link6"></a>
**ArXiv ID:** 2311.13584
**Authors:** Stefano Bruno, Ying Zhang, Dong-Young Lim, Ömer Deniz Akyildiz, Sotirios Sabanis

**Abstract:** We provide full theoretical guarantees for the convergence behaviour of
diffusion-based generative models under the assumption of strongly logconcave
data distributions while our approximating class of functions used for score
estimation is made of Lipschitz continuous functions. We demonstrate via a
motivating example, sampling from a Gaussian distribution with unknown mean,
the powerfulness of our approach. In this case, explicit estimates are provided
for the associated optimization problem, i.e. score approximation, while these
are combined with the corresponding sampling estimates. As a result, we obtain
the best known upper bound estimates in terms of key quantities of interest,
such as the dimension and rates of convergence, for the Wasserstein-2 distance
between the data distribution (Gaussian with unknown mean) and our sampling
algorithm.
  Beyond the motivating example and in order to allow for the use of a diverse
range of stochastic optimizers, we present our results using an $L^2$-accurate
score estimation assumption, which crucially is formed under an expectation
with respect to the stochastic optimizer and our novel auxiliary process that
uses only known information. This approach yields the best known convergence
rate for our sampling algorithm.


---

## 7. [Adaptive Sampling for Deep Learning via Efficient Nonparametric Proxies](https://arxiv.org/abs/2311.13583) <a id="link7"></a>
**ArXiv ID:** 2311.13583
**Authors:** Shabnam Daghaghi, Benjamin Coleman, Benito Geordie, Anshumali Shrivastava

**Abstract:** Data sampling is an effective method to improve the training speed of neural
networks, with recent results demonstrating that it can even break the neural
scaling laws. These results critically rely on high-quality scores to estimate
the importance of an input to the network. We observe that there are two
dominant strategies: static sampling, where the scores are determined before
training, and dynamic sampling, where the scores can depend on the model
weights. Static algorithms are computationally inexpensive but less effective
than their dynamic counterparts, which can cause end-to-end slowdown due to
their need to explicitly compute losses. To address this problem, we propose a
novel sampling distribution based on nonparametric kernel regression that
learns an effective importance score as the neural network trains. However,
nonparametric regression models are too computationally expensive to accelerate
end-to-end training. Therefore, we develop an efficient sketch-based
approximation to the Nadaraya-Watson estimator. Using recent techniques from
high-dimensional statistics and randomized algorithms, we prove that our
Nadaraya-Watson sketch approximates the estimator with exponential convergence
guarantees. Our sampling algorithm outperforms the baseline in terms of
wall-clock time and accuracy on four datasets.


---

## 8. [PaSS: Parallel Speculative Sampling](https://arxiv.org/abs/2311.13581) <a id="link8"></a>
**ArXiv ID:** 2311.13581
**Authors:** Giovanni Monea, Armand Joulin, Edouard Grave

**Abstract:** Scaling the size of language models to tens of billions of parameters has led
to impressive performance on a wide range of tasks. At generation, these models
are used auto-regressively, requiring a forward pass for each generated token,
and thus reading the full set of parameters from memory. This memory access
forms the primary bottleneck for generation and it worsens as the model size
increases. Moreover, executing a forward pass for multiple tokens in parallel
often takes nearly the same time as it does for just one token. These two
observations lead to the development of speculative sampling, where a second
smaller model is used to draft a few tokens, that are then validated or
rejected using a single forward pass of the large model. Unfortunately, this
method requires two models that share the same tokenizer and thus limits its
adoption. As an alternative, we propose to use parallel decoding as a way to
draft multiple tokens from a single model with no computational cost, nor the
need for a second model. Our approach only requires an additional input token
that marks the words that will be generated simultaneously. We show promising
performance (up to $30\%$ speed-up) while requiring only as few as $O(d_{emb})$
additional parameters.


---

## 9. [$σ$-PCA: a unified neural model for linear and nonlinear principal
  component analysis](https://arxiv.org/abs/2311.13580) <a id="link9"></a>
**ArXiv ID:** 2311.13580
**Authors:** Fahdi Kanavati, Lucy Katsnith, Masayuki Tsuneki

**Abstract:** Linear principal component analysis (PCA), nonlinear PCA, and linear
independent component analysis (ICA) -- those are three methods with
single-layer autoencoder formulations for learning linear transformations from
data. Linear PCA learns orthogonal transformations (rotations) that orient axes
to maximise variance, but it suffers from a subspace rotational indeterminacy:
it fails to find a unique rotation for axes that share the same variance. Both
nonlinear PCA and linear ICA reduce the subspace indeterminacy from rotational
to permutational by maximising statistical independence under the assumption of
unit variance. The main difference between them is that nonlinear PCA only
learns rotations while linear ICA learns not just rotations but any linear
transformation with unit variance. The relationship between all three can be
understood by the singular value decomposition of the linear ICA transformation
into a sequence of rotation, scale, rotation. Linear PCA learns the first
rotation; nonlinear PCA learns the second. The scale is simply the inverse of
the standard deviations. The problem is that, in contrast to linear PCA,
conventional nonlinear PCA cannot be used directly on the data to learn the
first rotation, the first being special as it reduces dimensionality and orders
by variances. In this paper, we have identified the cause, and as a solution we
propose $\sigma$-PCA: a unified neural model for linear and nonlinear PCA as
single-layer autoencoders. One of its key ingredients: modelling not just the
rotation but also the scale -- the variances. This model bridges the disparity
between linear and nonlinear PCA. And so, like linear PCA, it can learn a
semi-orthogonal transformation that reduces dimensionality and orders by
variances, but, unlike linear PCA, it does not suffer from rotational
indeterminacy.


---
