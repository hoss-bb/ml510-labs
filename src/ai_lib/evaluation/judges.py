"""Native judges and local-only evaluation clients."""



def build_judge(model, rubric, *, name):
    from openevals.llm import create_llm_as_judge
    return create_llm_as_judge(judge=model, prompt=rubric, feedback_key=name,
                              continuous=False, use_reasoning=True)


def local_evaluation_client():
    """Use local examples and forbid accidental hosted requests from this client."""
    import requests
    from langsmith import Client

    class LocalSession(requests.Session):
        def request(self, *args, **kwargs):
            raise RuntimeError("This evaluation client is local-only.")

    return Client(auto_batch_tracing=False, info={"version": "local"}, session=LocalSession())
