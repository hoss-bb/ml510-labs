"""Format retrieved passages for explicit answer prompts."""



def format_context(documents):
    return "\n\n".join(f"[{d.metadata['passage_id']}]\n{d.page_content}" for d in documents)
