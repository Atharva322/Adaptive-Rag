def retriever_node(state: State) -> dict:
    query = state["latest_query"]
    metadata_filter = state.get("metadata_filter")
    retriever = get_retriever(
        search_type=settings.SEARCH_TYPE,
        k=settings.RETRIEVER_K,
        metadata_filter=metadata_filter,
    )
    documents = retriever.invoke(query)
    return {"documents": documents}