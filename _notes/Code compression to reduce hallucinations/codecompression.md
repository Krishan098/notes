---
layout: default
title: "Summary for the three papers"
categories: [Code compression to reduce hallucinations]
---
# Orchestrating Structure and Efficiency in LLM contexts

- **AST-T5**: enhances code understanding by leveraging **Abstract Syntax Tree** structures without complex architectural changes, leading to superior parameter efficiency.

- **Graph Retrieal-Augmented Generation for Customized LLM**: It revolutionizes LLM customization for specialised domains by utilizing graph-structured knowledge representation for multi-hop reasoning and enhanced knowledge retrieval, improving interpretability and token efficiency.

- **In-Context Former:** An efficient context compression model that dramatically reduces LLM inference costs with a linear time complexity by using learnable "digit tokens" and operating independently of the LLM. 

## AST-T5: Structure-Aware Pretraining for Code Generation and Understanding

- AST-T5 is a novel pretraining paradigm designed to enhance large language models for code-related tasks by leveraging Abstract Syntax Tree structure of code. Traditional LLMs treat code as simple sequences, ignoring its inherent structure, which limits their performance. 

- ASTs could improve performance but are computationally expensive.

- AST-T5 overcomes these limitations by integrating AST awareness without complex program analysis or architectural changes, making it seamlessly compatible with any encoder-decoder Transformer, similar to Vanilla T5.

- The core contributions are:
    * **AST-Aware Segmentation**: This dynamic programming-based algorithm is used to split lengthy code files into chunks while preserving the structural integrity of the code, minimizing disruptions to AST structures. This is a significant improvement over "Greedy Segmentation".

    * **AST-Aware Span Corruption:**It specifically masks code spans that correspond to AST subtrees, ranging from individual tokens to entire function bodies. This trains the model to reconstruct coherent code structures. __theta:__ hyperparameter that controls the granularity of masking, enabling diverse training scenarios, from single-token completion to full function body generation.

### Interesting Fact:
- high performance gains through data preparation and masking strategies, rather than complex architectural changes or computationally expensive analysis.

- Granularity control for masking

- Empirical Component Analysis: increasing the mask ratio improved generation without compromising understanding tasks.

- Parameter Efficiency: only 277M parameters

### Identified gaps:
- Natural language performance: specialised code masking might lead to suboptimal performance in natural language generation.

- Syntactically Invalid code: the method assumes syntactically valid for parsing. 

### Methodology Thinking Inspiration

- Domain-specific structure is Gold: This paper strongly represents that explicitly incorporating domain-specific structural knowledge is often more effective than treating inputs as flat sequences.

- Smart Data Preparation: The dynamic programming approach for structure-preserving data segmentation is a valuable technique that could be generalised to other structured data types to optimize inputs for LLMs.

- Targeted Pretraining Objectives: Designing pretraining objectives that directly align with the desired output structure is a powerful way to guide model learning. 

- The paper acknowledges that AST-leveraging methods "necessitate parsing or static analysis for downstream tasks, which is less feasible for incomplete or incorrect code scenarios. "

- yes , AST-T5's pretraining objective and architecture are compatible with Vanilla T5. It can seamlessly integrate as a drop-in replacement for any T5 variant, suggesting it can leverage existing T5 model architectures without requiring complete re-training of the base model.

- An AST is a tree representation of the syntactic structure of code. Each node in the tree represents a construct from the source code (e.g., variable, loop, function call). 


---

## A survey of Graph Retrieval Augmented Generation for customised Large Language Models

- This survey introduces Graph-based Retrieval-Augmented Generation(GraphRAG) as a revolutionary paradigm for customizing LLMs for specialised domains.

- Traditional LLMs struggle with domain-specific knowledge due to limitations in knowledge depth, reasoning complexity, and contet sensitivity. Standard RAG systems rely on flat text retrieval, face challenges with complex query understanding, integrating distributed knowledge, ontxt window limitations and scalbility bottlenecks.

- GraphRAG addresses these through 3 key innovations:

1. **Graph-structured knowledge representation:** It explicitly captures entity relationships and domain hierarchies using Knowledge graphs.

2. **Efficient graph-based retrieval techniques:** It enables context-preserving knowledge retrieval with multi-hop reasoning capabilities by traversing graph structures.

3. **Structure-aware knowledge integration algorithms:** It leverages retrieved knowledge for accurate and logically coherent LLM generation.

- The survy categorizes existing GraphRAG models into 3 main paradigms for knowledge organisation:

- **Knowledge-based GraphRAG(Graph as knowledge Carriers):** Transforms unstructured text into explicit, structured KGs, where nodes represent concepts and edges capture semantic relationships. This enables better representation of hierarchical relationships and complex knowledge dependencies, facilitating logic-guided chain retrieval and multi-step reasoning.

- **Index-based GraphRAG(Graph for knowledge indexing):** maintains the original textual form but uses graph structures as an indexing mechanism to organise and retrieve relevant text chunks efficiently.

- **Hybrid GraphRAG:** Combines the strengths of both, using graphs as carriers of key information while linking to original text chunks for detailed contextual information.

- The knowledge retrieval process in GraphRAG involves query/graph preprocessing, matching(semantic similarity,structural relationships), and knowledge pruning to refine retrieved subgraphs. Various retrieval techniques are employed, including semantic-similarity based, logical reasoning based, GNN based,LLm-based and RL based retrievers.

- Retrieval can also be enhanced through multi-round retrieval, post-retrieval validation, and hybrid retrieval from multiple sources.

- For **Knowledge integration**, GraphRAG uses either fine-tuning(injecting node-level,path-level or subgraph-level knowledge into opensource LLMs) or in-context learning(for closed-source LLMs, using graph-enhanced Chain-of-Thought or collaborative KG refinement via prompt engineering)

- GraphRAG offers several advantages over traditional RAG: enhanced knowledge representation, flexibility in knowledge sources, improved efficiency and scalability and better interpretability through traceable reasoning paths. 

- Built-in Interpretability: The inherent transparency of GraphRAG, allowing users to "trace the path of reasoning" through the knowledge graph 

- Token Efficiency Claim: The claim that GraphRAG systems can achieve LLM responses with 26% to 97% fewer tokens than traditional methods is highly significant for reducing computational costs and latency.

### Limitations:

- **Lack of high quality KGs**: scarcity of high quality, comprehensive KGs, especially for new or niche domain.

- **KG construction Trade-offs:** When building KGs from text, there's a tradeoff between retaining fine-grained detail and compactness. LLM-based summarization for KG construction can also be computationally expensive.

- **LLM's Graph-structured input challenge:** LLMs are not inherently designed to process graph-structured data and often require conversion to natural language.

- **Error accumulation:** In-context learning methods like Graph-enhanced Chain-of-Thought still face the problem of error accumulation in multi-step reasoning.

- For a new GraphRAG we can start by using Open Information Extraction techniques to extract relational facts and triples from unstructured domain-specific texts. 

### Addressing fixed context window limitation

- While token reduction is part of it, it provides more concise,relevant and distillled information. Instead of providing large, potentially noisy text chunks, it retrieves pre-pruned factual information or specific subgraphs directly relevant to the query. This allows the LLM to fit more essential, semantically rich knowledge into its context window, facilitating multi-hop reasoning within that compressed and structured context.

### knowledge-based GraphRAG vs Index-based GraphRAG

- Knowledge-based GraphRAG is preferred when deep, precise, multi-step reasoning and explicit semantic relationships are crucial, as it transforms text into structured KGs that are highly amenable to logical inference and auditing.

- Index-based GraphRAG is better when the primary goal is efficient retrieval of raw, detailed textual information while still benefitting from some graph-based organization to establish semantic connections between text chunks. It's often used when maintaing the original text is important for context.

- Knowledge-based GraphRAG treats graphs as "knowledge carriers" by transforming unstructured text into explicit, structured knowledge graphs (KGs), where nodes are concepts and edges are semantic relationships. Index-based GraphRAG, conversely, uses graphs as "indexing tools" by maintaining the original textual form but employing graph structures to organize and efficiently retrieve relevant text chunks.

## In-context former

- In-context Former is a novel and highly efficient context compression model designed to significantly reduce the high infernece costs of Transformer-based LLMs, which typically suffer from quadratic time complexity due to their self attention mechanism with long input contexts.

- operates independently of the LLM

- uses a cross-attention mechanism and a small fixed number of learnable "digest tokens". These digest tokens act as queries that directly condense information from the contextual word embeddings(keys and values) into compact "digest vectors". 

- This achieves a linear time complexity w.r.t context length n.

- Rotational positional embeddings are applied to ensure the model captures are applied to ensure the model captures positional relationships within the context.

### Involves 2 phases:

1. **Pretraining**: The odel is trained to generate digest vectors such that a fixed LLM can accurately reconstruct the original context from these vectors, forcing IC-former to preserve essential contextual information.

2. **Instruction Fine-tuning:** IC-Former is further fine-tuned on instruction data to ensure that the generated digested vectors enable the LLM to correctly respond to various context-related prompts. For long contexts , a divide and conquer strategy is employed, where contexts are split into manageable chunks, individually compressed and then concatenated.


- It requires only 1/32 of the floating-point operations (FLOPs) compared to a baseline (ICAE) for 4x compression

- It achieves compression speeds 68 to 112  times faster than the baseline

- Analysis of attention maps reveals that IC-Former compresses context by aggregating information from adjacent tokens in sequential order, with different layers focusing on different grammatical categories.

- Model Decoupling: The ability to operate as a separate, lightweight compressor that does not modify or depend on the target LLM is highly valuable.

### Limitations:

- The paper explicitly mentions resource constraints preventing testing on larger models and longer contexts, which limits the full validation of its scalability

- Sub-baseline Performance: While "over 90%," the fact that it "has not surpassed the baseline’s performance in downstream tasks" implies a slight quality trade-off for the massive speed gains. This might not be acceptable for all high-accuracy critical applications.


### inspiration

- Modular AI Systems: The success of the decoupled compressor highlights the benefits of a modular design in LLM pipelines, where specialized components handle specific tasks (like compression) without affecting the core LLM.

## EVALUATION

- The primary method for assessing the correctness of generated code across multiple papers is through execution on unit tests. Benchmarks like HumanEval, MBPP, and APPS measure performance using metrics such as "Pass@1," which indicate whether the generated code successfully passes a set of provided test cases