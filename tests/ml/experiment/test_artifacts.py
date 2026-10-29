"""Small artifact checks: record sources and leave raw input unchanged."""
from pathlib import Path
import tempfile
import unittest
from ml_lib.experiment.artifacts import RunArtifacts, write_json


class ArtifactTests(unittest.TestCase):
    def test_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): write_json(Path(tmp)/"x.json", {"bad": float("nan")})

    def test_run_snapshots_and_no_raw_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"src"; source.mkdir()
            (source/"a.py").write_text("x = 1\n")
            data=root/"data.csv"; data.write_text("y\n1\n")
            original=data.read_bytes()
            run=RunArtifacts(root,"check",config={"seed":42},dataset_path=data)
            import json
            info=json.loads((run.directory/"manifest.json").read_text())
            self.assertEqual(len(info["dataset"]["sha256"]),64)
            self.assertIn(str(Path("src") / "a.py"), info["source_sha256"])
            self.assertEqual(data.read_bytes(),original)
            with self.assertRaises(ValueError): run.record("../unsafe",{})

    def make_run(self, root, label="example"):
        data = root / "data.csv"
        data.write_text("value\n1\n", encoding="utf-8")
        return RunArtifacts(root, label, config={}, dataset_path=data)

    def test_runs_are_grouped_and_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = self.make_run(root), self.make_run(root)
            self.assertEqual(first.directory.parent, root / "artifacts" / "example")
            self.assertEqual(first.directory.parent, second.directory.parent)
            self.assertNotEqual(first.directory, second.directory)

    def test_generated_notebooks_excluded(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notebooks" / "s4_regression" / "example.ipynb"
            source.parent.mkdir(parents=True)
            source.write_text("{}", encoding="utf-8")
            first = self.make_run(root)
            for folder in (
                root / "notebooks" / "corrections",
                root / "notebooks" / "s4_corrections",
                root / "notebooks" / "s5_corrections",
                source.parent / ".ipynb_checkpoints",
            ):
                folder.mkdir(parents=True)
                (folder / "example.ipynb").write_text('{"outputs": []}', encoding="utf-8")
            second = self.make_run(root)
            before = json.loads((first.directory / "manifest.json").read_text(encoding="utf-8"))
            after = json.loads((second.directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(before["source_tree_sha256"], after["source_tree_sha256"])
            self.assertEqual(list(after["source_sha256"]), [str(source.relative_to(root))])

    def test_tables_and_text_use_stable_names(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            table = pd.DataFrame({"value": [1]}, index=pd.Index(["row"], name="sample"))
            path = run.save_table("summary", table, index=True)
            self.assertEqual(path, run.directory / "tables" / "summary.csv")
            self.assertEqual(pd.read_csv(path).columns.tolist(), ["sample", "value"])
            run.save_table("summary", pd.DataFrame({"value": [2]}))
            self.assertEqual(pd.read_csv(path).value.tolist(), [2])
            self.assertEqual(len(list(path.parent.glob("*.csv"))), 1)
            text = "Model summary: caf\u00e9"
            saved = run.save_text("model-summary", text)
            self.assertEqual(saved.read_text(encoding="utf-8"), text)
            with self.assertRaises(ValueError): run.save_table("../escape", table)
            with self.assertRaises(ValueError): run.save_text("../escape", text)

    def test_overview_links_only_existing_files(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            run.record("experiment", {"status": "complete"})
            run.save_table("summary", pd.DataFrame({"mae": [1.0]}))
            overview = run.write_overview("Example", "Validation results saved.")
            text = overview.read_text(encoding="utf-8")
            self.assertIn("[summary.csv](tables/summary.csv)", text)
            self.assertIn("[experiment.json](experiment.json)", text)
            self.assertNotIn("## Figures", text)
            self.assertNotIn("[README.md]", text)
            run.write_overview("Example", "Updated outcome.")
            self.assertIn("Updated outcome.", overview.read_text(encoding="utf-8"))

    def test_consolidated_reports_replace_reruns_and_omit_training_rows(self):
        from types import SimpleNamespace
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            def report(split, prediction):
                return SimpleNamespace(model_name="linear", split=split,
                    metrics={"mae": abs(prediction - 1), "n": 1},
                    predictions=pd.DataFrame({"actual": [1], "prediction": [prediction]}))
            run.save_report(report("train", 1), consolidated=True)
            run.save_report(report("validation", 3), consolidated=True)
            run.save_report(report("validation", 2), consolidated=True)
            metrics = pd.read_csv(run.directory / "tables" / "metrics.csv")
            predictions = pd.read_csv(run.directory / "tables" / "predictions.csv")
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics.loc[metrics.split.eq("validation"), "mae"].tolist(), [1])
            self.assertEqual(predictions.split.tolist(), ["validation"])
            self.assertEqual(predictions.prediction.tolist(), [2])

    def test_overview_priority_order(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            run.save_table("summary", pd.DataFrame({"mae": [1.0]}))
            run.save_text("decision", "DRAFT")
            text = run.write_overview("Example", "Saved.", inspect_first=[
                "tables/summary.csv", "decision.txt", "tables/summary.csv",
            ]).read_text(encoding="utf-8")
            self.assertLess(text.index("## Inspect first"), text.index("## Run details"))
            self.assertLess(text.index("[summary.csv]"), text.index("[decision.txt]"))
            self.assertEqual(text.count("[summary.csv]"), 1)
            self.assertEqual(text.count("[decision.txt]"), 1)

    def test_overview_missing_priority_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            text = run.write_overview("Example", "Started.", inspect_first=[
                "tables/not-created.csv",
            ]).read_text(encoding="utf-8")
            self.assertNotIn("not-created", text)
            self.assertNotIn("## Inspect first", text)

    def test_overview_external_priority_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            for path in ("../outside.csv", str(Path(tmp).resolve() / "data.csv")):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    run.write_overview("Example", "Saved.", inspect_first=[path])
