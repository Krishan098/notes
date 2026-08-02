# Scaling Laws for Neural Language Models

### Abstract

- This paper is to study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amounts of data and stopping significantly before convergence.

#### Introduction

- Language provides a natural domain for the study of artificial intelligence, as the vast majority of reasoning tasks can be efficiently expressed and evaluated in language, and the world's text provides a wealth of language modeling, with state of the art models approaching human-level performance on many specific tasks, including the composition of coherent multi-paragraph prompted text samples.

- One might expect language modeling performance to depend on model architecture, the size of neural models, the computing power used to train them, and the data available for this training process. In this work, we will empirically investigate the dependence of language modeling loss on all of these factors, focusing on the Transformer architecture. The high ceiling and low floor for performance on language tasks allows us to study trends over more than seven orders of magnitude in scale.

- Throughout we will observe precise power-law scalings for performance as a function of training time, context length, dataset size, model size and compute budget.

#### Summary

- Key findings for Transformer language models are as follows:

![alt text](image-1.png)

- Language modeling performancr improves smoothly as we increase the model size, dataset size and amount of compute used for training.

- **Performance depends strongly on scale, weakly on model shape**: Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N (excluding embeddings), the size of the dataset D, and the amount of compute C used for training. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs width.

- **Smooth power laws**: Performance has a power-law relationship with each of the three scale factors N,D,C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude. They observed no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss.

- **Univerality of overfitting**: performance improves predictably as long as we scale up N and D in tandem, but enters a regime of diminishing returns if either N or D is held fixed while the other increases. The performance penalty depends predictably on the ratio $N^{0.74}/D$, meaning that every time we increase the model size 8x, we only need to increase the data by rougly 5x to avoid a penalty.

- **Universality of training**: Training curves follow predictable power-laws whose parameters are roughly independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the loss that would be achieved if we trained for much longer.

- **Transfer improves with test performance**: When we evaluate models on text with a different distribution than they were trained on, the results are strongly correlated to those on the training validation set with a roughly constant offset in the loss - in other words, transfer to a different distribution incurs a constant penalty but otherwise improves roughly in line with performance on the training set.

- **Sample Efficiency**: Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps and using fewer data points.

- **Convergence is inefficient**: When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models and stopping significantly short of convergence. Maximally compute-efficient training would therefore be far more sample efficient than one might expect based on training small models to convergence, with data requirements growing very slowly as $D~C^{0.27}$ with training compute.

- **Optimal batch size**: The ideal batch size for training these models is roughly a power of the loss only, and continues to be determinable by measuring the gradient noise scale; it is roughly 1-2 million tokens at convergence for the largest models we can train.

- Taken together, these results show that language modeling performance improves smoothly and predictably as we appropriately scale up model size, data and compute. We expect that larger language models will perform better and be more sample efficient than current models.

#### Summary of Scaling Laws

- The test loss of a transformer trained to autoregressively model language can be predicted using a power-law when performance is limited by only either the number of non-embedding parameters N, the dataset size D, or the optimally allocated compute budget $C_{min}$:

    1. For models with a limited number of parameters, trained to convergence on sufficiently large datasets:
        $$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N \approx 0.076,\quad N_c \approx 8.8 \times 10^{13}$$ (non-embedding parameters)
    
    2. For large models trained with a limited dataset with early stopping:
        $$L(D) = \left(\frac{D_{c}}{D}\right)^{\alpha_D} ; \alpha_D \approx 0.095, D_c \approx 5.4 \times 10^{13}$$ (tokens)
    3. When training with a limited amount of compute, a sufficiently large dataset, an optimally-sized model, and a sufficiently small batch size(making optimal use of compute):
        $$L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}} ; {\alpha_C}^{min} \approx 0.050, {C_c}^{min} \approx 3.1 \times 10^{8}$$(PF-days)

![alt text](image-2.png)

- These relations hold across eight orders of magnitude in $C_{min}$, six orders of magnitude in N, and over two orders of magnitude in D. They depend very weakly on model shape and other Transformer hyperparameters(depth, width, number of self-attention heads), with specific numerical values associated with the Webtext2 training set. The power laws  $\alpha_N, \alpha_D, {\alpha_C}^{min}$ specify the degree of performance improvement expected as we scale up N,D or $C_{min}$; for example, doubling the number of parameters yields a loss that is smaller by a factor $2^{-\alpha_N}=0.95$. The precise numerical values of $N_c, {C_c}^{min}, and D_c$ depend on the vocabulary size and tokenization and hence do not have a fundamental meaning.

- The critical batch size, which determines the speed/efficiency tradeoff for data parallelism, also roughly obeys a power law in L:
        $$B_{crit}(L)= \left(\frac{B_*}{L^{1/{\alpha_B}}}\right), B_* \approx 2.10^{8} tokens, \alpha_B \approx 0.21$$

- The first 2 equations together suggest that as we increase the model size, we should increase the dataset size sublinearly according to $D \propto N^{\alpha_N/{\alpha_D}} \approx N^{0.74}$. In fact, we find that there is a single equation combining the 2 equations that governs the simultaneous dependence on N and D and governs the degree of overfitting:
        $$L(N,D)=[(N_c/{N})^{\alpha_N/{\alpha_D}}+D_c/D]^{\alpha_D}$$. We conjucture that this functional form may also parameterize the trained log-likelihood for other generative modeling tasks.
        When training a given model for a finite number of parameter update steps S in the infinite data limit, after an initial transient period, the learning curves can be accurately fit by:
        $$L(N,S)=(N_c/N)^{\alpha_N}+(S_c/{S_{min}(S)})^{\alpha_S} where S_c \approx 2.1 \times 10^3 and \alpha_S \approx 0.76, and S_{min}(S)$$ is the minimum possible number of optimization steps(paramter updates) estimated using the equation $$ S_{min}(S)=S/({1+({B_{crit}(L)})/B})$$
        When training within a fixed compute budget C, but with no other constraints, the above equations leads to the prediction that the optimal model size N, optimal batch size B, optimal number of steps S, and dataset size D should grow as $$ N \propto C^{{{\alpha_C}^{min}}/\alpha_N}, B \propto C^{{{\alpha_C}^{min}}/\alpha_B}, S \propto C^{{{\alpha_C}^{min}}/\alpha_S}, D=B.S$$ with
        $${\alpha_C}^{min}=1/{(1/{\alpha_S}+1/{\alpha_B}+1/{\alpha_N})}$$ which closely matches the empirically optimal results $N \propto C_{min}^{0.73},B \propto C_{min}^{0.24},S \propto C_{min}^{0.03}$. As the computational budget C increases, it should be spent primarily on larger models, without dramatic increases in training time or dataset size. This also implies that as models grow larger, they become increasingly sample efficient. In practice, resarchers typically train smaller models for longer than would be maximally compute-efficient because of hardware constraints. Optimal performance depends on total compute as a power law.

#### Notation

- L: the cross-entropy loss in nats. Typically it will be averaged over the tokens in a context, but in some cases we report the loss for specific tokens within the context.

- N: the number of model parameters, excluding all vocabulary and positional embeddings.

- $C \approx 6N BS$- an estimate of the total non-embedding training compute, where B is the batch size, and S is the number of training steps. We quote numerical values in PF-days, where one PF-day= $10^{15} \times 24 \times 3600 = 8.64 \times 10^{19}$ floating point operations.

- D- the dataset size in tokens.

- $B_{crit}$- the critical batch size. Training at the critical batch size provides a roughly optimal compromise between time and compute efficiency.

- $C_{min}$- an estimate of the minimum amount of non-embedding compute to reach a given value of the loss. This is the training compute that would be used if the model were trained at a batch size much less than the critical batch size.

- $S_{min}$- an estimate of the minimal number of training steps needed to reach a given value of the loss. This is also the number of training steps that would be used if the model were trained at a batch size much greater than the critical batch size.

- $\alpha_x$- power-law exponents for the scaling of the loss as $L(X) \propto 1/{X^{\alpha_x}}$ where X can be any of N,D,C,S,B,$C^{min}$.

#### Background and Methods

- Language models were trained on WebText2, an extended version of the WebText datset, tokenized using byte-pair encoding with a vocabulary size $n_{vocab}=50257$. We optimize the autoregressive log-likelihood(i.e cross-entropy loss) averaged over a 1024-token context, which is also our principal performancr metric. We record the loss on the WebText2 test distribution and on a selection  of other text distributions. We primarily train decoder-only transformer models, though we also train LSTM models and Universal Transformers for comparison.