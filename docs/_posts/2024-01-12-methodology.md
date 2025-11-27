---
layout: post
title: "Methodology"
date: 2024-01-12
description: Notes on Methodology
tags: [code-compression]
categories: [Code Compression]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

The proposed methodology aims to significantly reduce syntactic and logical hallucinations in code generation by employing a **structure-aware compressed Retrieval-Augmented Generation (RAG) system** [Methodology Statement]. This system will combine **AST-T5 segmentation** with an **AST-fidelity-aware In-Context Former (IC-Former)** [Methodology Statement].

### Proposed Methodology

This method directly addresses the limitations of Large Language Models (LLMs) when reasoning over long or unstructured code by preserving critical structural information ]. The pipeline involves three main stages:

1.  **Code Parsing and AST-Aware Segmentation with AST-T5**:
    *   Initially, code will be parsed into Abstract Syntax Trees (ASTs) using a **lightweight, multi-language parser such as Tree-sitter** . This parsing process is fundamental to capturing the inherent structure of the code.
    *   Following parsing, **AST-T5's dynamic programming-based segmentation algorithm** will be applied to chunk the code . This "AST-aware code segmentation" is a key contribution of AST-T5, designed to **maintain the structural integrity of the input code** during chunking, unlike "Greedy Segmentation" which can fragment AST structures and introduce noisy partial expressions. This ensures that code segments remain coherent, similar to complete function definitions often found in downstream tasks.

2.  **AST-Fidelity-Aware Context Compression with IC-Former**:
    *   The segmented code chunks will then be compressed using a **modified In-Context Former (IC-Former)** model . The original IC-Former efficiently compresses lengthy inputs into "digest vectors" (compact representations) using **cross-attention mechanisms** and learnable digest tokens, achieving linear time complexity within the compression range and significantly faster compression speeds compared to baselines.
    *   The crucial modification is the "AST-fidelity-aware" aspect: this IC-Former will be **trained to reconstruct *both* the code and its AST representation** from these digest vectors [Detailed Explanation]. This training objective is inspired by models like StructCoder, which explicitly train a Transformer decoder using novel auxiliary tasks such as **Abstract Syntax Tree (AST) path prediction (APP)** and **dataflow prediction (DFP)** to enhance the quality and structural correctness of generated code. By incorporating similar objectives, the digest vectors are intended to retain both semantic and syntactic fidelity, guiding the LLM towards generating syntactically correct and semantically coherent code 

3.  **Retrieval-Augmented Generation (RAG) with Downstream Generative Model**:
    *   The **structure-aware digest vectors** produced by the modified IC-Former will serve as the **retrieved knowledge** input into a downstream generative model, such as **CodeLlama** or **CodeT5+**, within a **Retrieval-Augmented Generation (RAG) framework** 
    *   This structure-aware compression pipeline is designed to enable the LLM to focus on relevant context while avoiding overload from long, flat token sequences, which is a common limitation of LLMs in reasoning over long or unstructured code . The use of structured knowledge, as is central to **GraphRAG systems**, can improve contextual reasoning and handle complex relationships more effectively than plain text chunks. This focused and structurally informed context is expected to reduce hallucinations and improve the executability of the generated output [.

### Success Criteria and Their Computation

The success of this methodology will be evaluated against specific, measurable criteria:

1.  **≥30% reduction in hallucinated code (syntax errors, test failures) compared to baseline (non-structured RAG)** [Success Criteria].
    *   **Computation:** "Hallucinated code" is defined as code containing "syntax errors" or "test failures" [Success Criteria].
        *   **Syntax Errors:** These would typically be detected when attempting to compile or execute the generated code. A syntax error would prevent the code from running and thus lead to a test failure.
        *   **Test Failures:** This is directly measured by the `Pass@1` metric, which evaluates whether the generated code successfully executes against a set of unit tests provided in the benchmark.
    *   **Comparison:** The reduction will be measured against a "baseline (non-structured RAG)" system [Success Criteria]. This implies comparing the percentage of generated code that exhibits syntax errors or fails tests in the proposed system versus a RAG system that does not incorporate AST-aware segmentation or AST-fidelity-aware compression, likely relying on simple text chunking. The rationale is that "structure-preserving compression ensures more relevant and executable context reaches the model" [Methodology Statement].

2.  **AST reconstruction accuracy ≥85% from digest vectors** [Success Criteria].
    *   **Computation:** This metric specifically assesses the effectiveness of the proposed "AST-fidelity-aware" modification to the IC-Former. It will be computed by taking the digest vectors generated by the modified IC-Former and attempting to reconstruct the original AST representation of the code [Detailed Explanation]. The accuracy would then be the percentage of correctly reconstructed ASTs (or components thereof, such as node types or paths, similar to how StructCoder evaluates AST path prediction) compared to the ground-truth ASTs from the original code segments. The original IC-Former demonstrates high text reconstruction accuracy (e.g., BLEU-4 scores over 0.99 for shorter contexts), and this criterion extends that concept to the AST structure.

3.  **Improved Pass@1 score on both HumanEval and MBPP EvalPlus** [Success Criteria].
    *   **Computation:** `Pass@1` is a widely recognized metric for code generation, indicating the functional correctness of a single generated code per problem.
        *   For `Pass@k` (where k=1), the computation involves generating a number of code samples (`n`) per problem, counting the correct samples (`c`) that pass all unit tests, and then calculating an unbiased estimator. For `Pass@1` specifically, using greedy decoding, it corresponds to the direct pass rate.
        *   **HumanEval** and **MBPP** are standard Python code generation benchmarks. **EvalPlus** is a more rigorous benchmark that augments these with a substantial number of additional test cases for more accurate evaluation.
    *   **Comparison:** The `Pass@1` scores achieved by the proposed system on these benchmarks will be compared against existing baselines. AST-T5 models, even without a RAG framework, have already demonstrated improved `Pass@1` scores on HumanEval (14.0) and MBPP (19.3) compared to T5 baselines and similar-sized models like CodeT5 and CodeT5+. CodeLlama models also report Pass@1 scores on these benchmarks (e.g., CodeLlama 70B achieved 54.7% on MultiPL-E HumanEval Python and 45.0% on MBPP). The expectation is that the structure-aware compression within RAG will lead to further improvements in these functional correctness scores [Methodology Statement].