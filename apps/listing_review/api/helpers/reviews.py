"""Run one saved review after the API response; never repeat a claimed job."""
import time
from ai_lib.models.errors import expected_call_error
from lab_helpers.s10_shared.reviews.chain import review
from lab_helpers.s10_shared.reviews.errors import ReviewError
from lab_helpers.s10_shared.runtime.logging import event


def process_review(store, job_id, runner=None):
    if not store.claim_job(job_id):
        return False
    runner = runner or review
    started = time.perf_counter()
    try:
        def save(stage, output):
            store.save_stage(job_id, stage, output)
            event("review_checkpoint", job_id=job_id, name=stage, outcome="saved")
        snapshot = {**store.job(job_id)["snapshot"], "job_id": job_id}
        result = runner(snapshot, save)
        store.finish_job(job_id, result=result)
        event("review", job_id=job_id, outcome="completed", seconds=time.perf_counter() - started)
    except Exception as error:
        code = error.code if isinstance(error, ReviewError) else type(error).__name__
        store.finish_job(job_id, error=code)
        event("review", job_id=job_id, outcome="failed", error_type=type(error).__name__, reason=code)
        if not expected_call_error(error):
            raise
    return True
