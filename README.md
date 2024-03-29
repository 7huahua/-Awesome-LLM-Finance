# Awesome-LLM-Finance [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of Awesome things about **Large Language Models (LLMs) for Finance**

Large Language Models (LLMs) have made significant advancements in natural language processing, but their application in the finance field is less explored. This repository aims to fill this gap by offering a collection of research papers that investigate how LLMs can be integrated with financial data and models. By focusing on the finance domain, the repository seeks to uncover novel insights and techniques for leveraging LLMs to solve complex problems in economics, market prediction, risk assessment, and financial decision-making processes.


## Table of Contents

- [Awesome-LLM-Finance ](#awesome-llm-finance-)
  - [Table of Contents](#table-of-contents)
  - [Datasets, Benchmarks \& Surveys](#datasets-benchmarks--surveys)
  - [Prompting](#prompting)
  - [General Finance Model](#general-finance-model)
  - [Large Multimodal Models (LMMs)](#large-multimodal-models-lmms)
  - [Applications](#applications)
    <!-- - [Basic Graph Reasoning](#basic-graph-reasoning)
    - [Node Classification](#node-classification)
    - [Graph Classification/Regression](#graph-classificationregression)
    - [Knowledge Graph](#knowledge-graph) -->
    - [Others](#others)
  - [Resources \& Tools](#resources--tools)
  - [Contributing](#contributing)
  - [Star History](#star-history)


## Datasets, Benchmarks & Surveys
<!-- - (*NAACL'21*) Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training [[paper](https://aclanthology.org/2021.naacl-main.278/)][[code](https://github.com/google-research-datasets/KELM-corpus)]
- (*NeurIPS'23*) Can Language Models Solve Graph Problems in Natural Language? [[paper](https://arxiv.org/abs/2305.10037)][[code](https://github.com/Arthur-Heng/NLGraph)]
- (*IEEE Intelligent Systems 2023*) Integrating Graphs with Large Language Models: Methods and Prospects [[paper](https://arxiv.org/abs/2310.05499)]
- (*ICLR'24*) Talk like a Graph: Encoding Graphs for Large Language Models [[paper](https://arxiv.org/abs/2310.04560)]
- (*arXiv 2023.05*) GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking [[paper](https://arxiv.org/abs/2305.15066)][[code](https://github.com/SpaceLearner/Graph-GPT)]
- (*arXiv 2023.08*) Graph Meets LLMs: Towards Large Graph Models [[paper](http://arxiv.org/abs/2308.14522)]
- (*arXiv 2023.10*) Towards Graph Foundation Models: A Survey and Beyond [[paper](https://arxiv.org/abs/2310.11829v1)]
- (*arXiv 2023.10*) LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs? [[paper](https://arxiv.org/abs/2310.17110)]
- (*arXiv 2023.11*) Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey [[paper](https://arxiv.org/abs/2311.07914v1)]
- (*arXiv 2023.11*) A Survey of Graph Meets Large Language Model: Progress and Future Directions [[paper](https://arxiv.org/abs/2311.12399)][[code](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks)]
- (*arXiv 2023.12*) Large Language Models on Graphs: A Comprehensive Survey [[paper](https://arxiv.org/abs/2312.02783)][[code](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]
- (*arXiv 2024.02*) Towards Versatile Graph Learning Approach: from the Perspective of Large Language Models [[paper](https://arxiv.org/abs/2402.11641)] -->
  
## Prompting
<!-- - (*EMNLP'23*) StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper](https://arxiv.org/abs/2305.09645)][[code](https://github.com/RUCAIBox/StructGPT)]
- (*AAAI'24*) Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper](https://arxiv.org/abs/2308.09687)][[code](https://github.com/spcl/graph-of-thoughts)]
- (*arXiv 2023.05*) PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs [[paper](https://arxiv.org/abs/2305.12392)][[code](https://github.com/Jiuzhouh/PiVe)]
- (*arXiv 2023.08*) Boosting Logical Reasoning in Large Language Models through a New Framework: The Graph of Thought [[paper](https://arxiv.org/abs/2308.08614)]
- (*arxiv 2023.10*) Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models [[paper](https://arxiv.org/abs/2310.03965v2)]
- (*arxiv 2024.01*) Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts [[paper](https://arxiv.org/abs/2401.14295)] -->


## General Finance Model
<!-- - (*ICLR'24*) One for All: Towards Training One Graph Model for All Classification Tasks [[paper](https://arxiv.org/abs/2310.00149)][[code](https://github.com/LechengKong/OneForAll)]
- (*arXiv 2023.08*) Natural Language is All a Graph Needs [[paper](https://arxiv.org/abs/2308.07134)][[code](https://github.com/agiresearch/InstructGLM)]
- (*arXiv 2023.10*) GraphGPT: Graph Instruction Tuning for Large Language Models [[paper](https://arxiv.org/abs/2310.13023)][[code](https://github.com/HKUDS/GraphGPT)][[blog in Chinese](https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw)]
- (*arXiv 2023.10*) Graph Agent: Explicit Reasoning Agent for Graphs [[paper](https://arxiv.org/abs/2310.16421)]
- (*arXiv 2024.02*) Let Your Graph Do the Talking: Encoding Structured Data for LLMs [[paper](https://arxiv.org/abs/2402.05862)]
- (*arXiv 2024.02*) G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering [[paper](https://arxiv.org/abs/2402.07630)][[code](https://github.com/XiaoxinHe/G-Retriever)][[blog](https://medium.com/@xxhe/graph-retrieval-augmented-generation-rag-beb19dc30424)]
- (*arXiv 2024.02*) InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment [[paper](https://arxiv.org/abs/2402.08785)][[code](https://github.com/wjn1996/InstructGraph)]
- (*arXiv 2024.02*) LLaGA: Large Language and Graph Assistant [[paper](https://arxiv.org/abs/2402.08170)][[code](https://github.com/VITA-Group/LLaGA)]
- (*arXiv 2024.02*) HiGPT: Heterogeneous Graph Language Model [[paper](https://arxiv.org/abs/2402.16024)][[code](https://github.com/HKUDS/HiGPT)] -->



## Large Multimodal Models (LMMs)
<!-- - (*NeurIPS'23*) GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph [[paper](https://arxiv.org/abs/2309.13625)][[code](https://github.com/lixinustc/GraphAdapter)]
- (*arXiv 2023.10*) Multimodal Graph Learning for Generative Tasks [[paper](https://arxiv.org/abs/2310.07478)][[code](https://github.com/minjiyoon/MMGL)]
- (*arXiv 2024.02*) Rendering Graphs for Graph Reasoning in Multimodal Large Language Models [[paper](https://arxiv.org/abs/2402.02130)] -->


## Applications