"""Existing assignments validated without application-specific assumptions."""
import unittest
import pandas as pd
from ml_lib.data.validation import validate_partitions


class PartitionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"record": [1, 2, 3, 4], "customer": ["a", "a", "b", "c"],
                             "role": ["development", "development", "assessment", "assessment"]})

    def validate(self, frame, **changes):
        options = dict(case_id="record", group_id="customer", partition="role",
                       expected_partitions=["assessment", "development"])
        return validate_partitions(frame, **{**options, **changes})

    def test_declared_order_and_independent_frames(self):
        frame = self.frame()
        parts = self.validate(frame)
        self.assertEqual(list(parts), ["assessment", "development"])
        self.assertEqual(parts["development"].record.tolist(), [1, 2])
        parts["development"].loc[0, "record"] = 99
        self.assertEqual(frame.loc[0, "record"], 1)

    def test_missing_duplicate_and_unknown_assignments(self):
        for column, value in [("record", None), ("record", 2), ("customer", None), ("role", "other")]:
            frame = self.frame()
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.validate(frame)
        with self.assertRaises(ValueError):
            self.validate(self.frame().iloc[:2])
        with self.assertRaises(ValueError):
            self.validate(self.frame().drop(columns="record"))

    def test_customer_separation_is_optional(self):
        frame = self.frame()
        frame.loc[2, "customer"] = "a"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.validate(frame)
        self.validate(frame, group_id=None)
