# 🦜️🔗 LangChain Milvus

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/LangChainAI.svg?style=social&label=Follow%20%40LangChain)](https://x.com/LangChainAI)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/milvusio.svg?style=social&label=Follow%20%40Milvus)](https://x.com/milvusio)

![](https://github.com/langchain-ai/langchain-milvus/blob/main/readme_img.png)

This repository contains the Milvus integration package for LangChain:

- [langchain-milvus](https://pypi.org/project/langchain-milvus/) - A powerful integration between LangChain and Milvus, enabling vector search and retrievers for AI applications.

## Overview

LangChain Milvus provides seamless integration between LangChain, a framework for building applications with large language models (LLMs), and Milvus, a powerful vector database designed for similarity search and AI applications. This integration enables efficient vector storage and retrieval for AI applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

## Features

- **Vector Storage**: Store embeddings from any LangChain embedding model in Milvus
- **Similarity Search**: Perform efficient similarity searches on vector data
- **Hybrid Search**: Combine vector search with keyword search for improved results
- **Maximal Marginal Relevance**: Filter for diversity in search results
- **Multiple Vector Fields**: Support for multiple vector fields in a single collection
- **Sparse Embeddings**: Support for sparse vector embeddings
- **Built-in Functions**: Support for Milvus built-in functions like BM25
- **Async Support**: Full support for async operations and APIs

## Installation

```bash
pip install -U langchain-milvus
```

## Usage
- [Basic Usage](https://milvus.io/docs/basic_usage_langchain.md): Learn how to get started with basic vector operations in Milvus using LangChain.

- [Build RAG(Retrieval Augmented Generation)](https://milvus.io/docs/integrate_with_langchain.md): Discover how to build powerful RAG applications by combining LangChain with Milvus.

- [Full-text Search](https://milvus.io/docs/full_text_search_with_langchain.md): Explore how to implement full-text search capabilities using LangChain and Milvus.

- [Hybrid Search](https://milvus.io/docs/milvus_hybrid_search_retriever.md): Learn how to combine vector and keyword search for more accurate results.



## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/langchain-ai/langchain-milvus/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/langchain-ai/langchain-milvus/blob/main/LICENSE) file for details.
