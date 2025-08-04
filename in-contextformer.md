# In-Context Former: Lightning-fast Compressing Context for Large Language Model

## Abstract

- One effective approach to reduce high inference costs is to compress the long input contexts. Existing LLMs use self attention mechanism for context-compression. 

- Despite these methods, the compression process still has a quadratic time complexity. 

- Unlike these methods, IC-Former doesnot depend on the target LLMs. Instead it leverages the cross-attention mechanism and a small number of learnable digest tokens to directly condense information from the contextual word embeddings. This approach significantly reduces inference time, whcih achieves linear growth in time complexity within the compression range. 

- It requires only 1/32 of the floating-point operations of the baseline during compression and improves processing speed by 68 to 112 times while achieving over 90% of the baseline performance on evaluation metrics.

## Introduction

![alt text](image.png)

- The self attention improvement mechanisms mitigate the overhead of long context processing, they inevitably introduce modifications to the original structure of LLMs, potentially impacting the capabilities of the original model.

- A preliminary context compression process is based on a core assumption: most natural language texts contain redundant information, which makes context compression feasible. 

- These methods design compression models to condense lengthy contexts into short, context-rich soft prompts,which then serve as substitutes for the original context when input into the LLM. However, these methods still suffer the issue of expensive time costs during the compression process. This limitation restricts their application in real-time compression scenarios, such as compressing retrieved or real-time Internet documents immediately.

- The In-context former aims at optimizing resource consumption during the compression of long context in existing models. This model is based on two assumptions regarding semantic compression:

    1. Word embeddings already contains sufficient semantic information, suggesting that the interactions between embeddings may not be necessary prior to the extraction process.

    2. Learnable tokens within an elaborate structure can effectively aggregate information to a certain extent.

- Based on these, the paper tries to discard the costly self-attention interaction of text content in previous models. Instead, it uses the cross-attention mechanism for information extraction. This strategy ensures that the computational overhead of compression grows linearly with the context length within the compression range.

- The IC-former consists of a few cross-attention blocks and some learnable digest tokens. 

- Through this structure, the IC-Former leverages the digest tokens to extract information from lengthy contextual content and refine it into compact digest vectors.

- Subsequently these digest vectors directly replace the original, verbose context and serve as input to the LLMs while ensuring that the generated texts are faithful to the original context.

- In the training phase, it employs a strategy that combines pre-training and fine-tuning to optimize the IC-Former. 

- During pre-training phase, the IC-Former engages in a context reconstruction task. It is used to generate digest vectors from which an LLM can reconstruct the original context. 

- In the fine-tuning phase, the IC-Former is trained to ensure that the generated digest vectors correctly respond to various context-related prompts.

![alt text](image-1.png)

## Method 

### Task Formulation

- Context compression aims to transform lengthy contexxts into brief, compact representations while endeavoring to preserve the fundamental semantics and integrity of the original contexts.

- Formally, we define the original context that is to be compressed as w=(w1,w2,....,wn) where w$\_i$ represents the i$\^th$ token of context and n is the number of tokens in context.

- Then we denote e(.) as the word embedding lookup in the LLM and $\^~e$(.) as the learnable embeddings of soft tokens. 

- A context compressor model &#920; utilizes the embeddings of soft tokens $\^{~}e$(d)= ($\^{~}e$(d$\_1$),$\^~e$(d\_2$),.....$\^~e$(d$\_k$)) and context embeddings e(w)=(e(w~1~),e(w~2~),....e(w~n~)) to generate compact representations d&#7771;= (^~^d~1~,^~^d~2~,....,^~^d~k~) of context, where k is the length of compressed context and k<<n.

- The condensed vectors d&#7771; can substitute the original context and can be combined with other prompt e(p)=(e(p~1~),e(p~2~),....,e(p~l~)) for input to an LLM $\phi$. 

- The output y=(y~1~,.....,y~m~) remains faithful to the content of the original context w.