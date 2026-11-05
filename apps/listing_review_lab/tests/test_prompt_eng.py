"""Prompt selection and selected instruction files; no provider calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from lab_helpers.s10_shared.reviews.prompts import save_prompts, load_prompts, PromptError
from apps.listing_review.jobs.prompt_eng.helpers.workflow import select_prompt, export_selection, ROOT
from apps.listing_review.jobs.prompt_eng.helpers.templates import assemble


def results(starting, candidate, error=None):
    return pd.DataFrame([{"variant": variant, "case": case, "action": "no_change", "action_matches": passed,
                          "error": error if variant == "candidate" and case == "conflict" else None}
                         for variant, checks in [("starting", starting), ("candidate", candidate)]
                         for case, passed in zip(["conflict", "legitimate"], checks)])


class PromptJobTests(unittest.TestCase):
    def test_workflow_records_constructed_fixture(self):
        from apps.listing_review.jobs.prompt_eng.helpers import workflow as job
        from ml_lib.experiment.artifacts import RunArtifacts
        with patch.object(job, "RunArtifacts", autospec=RunArtifacts) as artifacts, \
             patch.object(job, "generation_settings", return_value={}), \
             patch.object(job, "compare", return_value=results([True, True], [True, True])), \
             patch.object(job, "export_selection", return_value=Path("candidate")), \
             patch("builtins.print"):
            artifacts.return_value.directory = Path("artifacts/example")
            job.run_job()
            self.assertTrue(artifacts.call_args.kwargs["dataset_path"].is_file())

    def test_improvement_tie_and_regression(self):
        self.assertEqual(select_prompt(results([False,True],[True,True]))["selected"],"candidate")
        self.assertEqual(select_prompt(results([True,True],[True,True]))["selected"],"starting")
        self.assertEqual(select_prompt(results([True,False],[False,True]))["selected"],"starting")
        report=select_prompt(results([False,False],[False,False]))
        self.assertEqual(report["passed_cases"],{"starting":0,"candidate":0})
        self.assertEqual(report["selected"],"starting")

    def test_incomplete_and_missing_case(self):
        report=select_prompt(results([True,True],[True,True],error="evidence_format"))
        self.assertFalse(report["complete"])
        self.assertIsNone(report["selected"])
        with self.assertRaises(ValueError):select_prompt(results([True,True],[True,True]).iloc[:3])

    def test_prompt_files_roundtrip_and_required_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "prompts"
            instructions = assemble("starting")
            save_prompts(folder, instructions)
            self.assertEqual(load_prompts(folder), instructions)
            self.assertEqual({p.name for p in folder.iterdir()}, {"evidence.txt", "assessment.txt", "proposal.txt"})
            with self.assertRaises(FileExistsError):
                save_prompts(folder, instructions)
            # Historical manifests do not govern the new text-file loader.
            (folder / "manifest.json").write_text("old manifest")
            self.assertEqual(load_prompts(folder), instructions)
            (folder / "evidence.txt").write_text("  ")
            with self.assertRaises(PromptError):
                load_prompts(folder)
            (folder / "evidence.txt").unlink()
            with self.assertRaises(PromptError):
                load_prompts(folder)
            with self.assertRaises(PromptError):
                save_prompts(Path(tmp) / "invalid", {"evidence": "Only one stage"})

    def test_export_matches_selected_and_failure_exports_nothing(self):
        variants={name:assemble(name) for name in ['starting','candidate']}
        run=Mock();run.directory=ROOT/'artifacts/unit-prompt-job'
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            self.assertIsNone(export_selection(run,results([True,True],[True,True],error='provider'),{},variants,folder))
            self.assertEqual(list(folder.iterdir()),[])
            selected=export_selection(run,results([False,True],[True,True]),{'model':'fixed'},variants,folder)
            self.assertEqual(load_prompts(selected),variants['candidate'])
            self.assertFalse((selected / 'manifest.json').exists())
            experiment = next(call.args[1] for call in run.record.call_args_list if call.args[0] == 'experiment')
            self.assertEqual(experiment['generation'], {'model': 'fixed'})


if __name__ == '__main__':unittest.main()
