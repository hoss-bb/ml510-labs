import tempfile
import unittest
import numpy as np
import pandas as pd
from ml_lib.models.discovery import AnomalyModel, ClusterModel
from ml_lib.evaluation.discovery import cluster_diagnostics, assignment_agreement, reference_threshold, evaluate_alerts, require_matching_windows, select_detector


class DiscoveryTests(unittest.TestCase):
    def test_snapshot(self):
        p = {'n_clusters':2,'n_init':10}
        m = ClusterModel('kmeans',p)
        p['n_clusters']=99
        self.assertEqual(m.parameters['n_clusters'],2)
        with self.assertRaises(ValueError):
            AnomalyModel('isolation',{'n_estimators':10})

    def test_preparation(self):
        x = np.arange(40.).reshape(20,2)
        m = AnomalyModel('deviation',{}).fit(x,feature_names=['a','b'])
        mean = m.scaler.mean_.copy()
        self.assertGreater(m.score_samples([[100,100]],feature_names=['a','b'])[0],m.score_samples([[20,21]],feature_names=['a','b'])[0])
        np.testing.assert_array_equal(mean,m.scaler.mean_)
        with self.assertRaises(ValueError):
            m.score_samples(x,feature_names=['b','a'])

    def test_roundtrip(self):
        x = np.random.default_rng(42).normal(size=(40,3,2))
        m = AnomalyModel('pca',{'n_components':2}).fit(x,feature_names=['a','b'])
        np.testing.assert_allclose(m.contributions(x,feature_names=['a','b']).mean(axis=1),m.score_samples(x,feature_names=['a','b']))
        with tempfile.TemporaryDirectory() as d:
            m.save(d)
            with self.assertRaises(ValueError):
                AnomalyModel.load(d)
            restored=AnomalyModel.load(d,trusted=True)
            np.testing.assert_allclose(restored.score_samples(x,feature_names=['a','b']),m.score_samples(x,feature_names=['a','b']))

    def test_undefined_silhouette(self):
        result=cluster_diagnostics(np.ones((5,2)),[-1]*5)
        self.assertIsNone(result['silhouette'])
        self.assertEqual(result['noise_fraction'],1)

    def test_arbitrary_labels(self):
        self.assertEqual(assignment_agreement(pd.Series([0,0,1]),pd.Series([3,3,2])),1)

    def test_episodes(self):
        starts=pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:10','2020-01-01 00:40'])
        table=pd.DataFrame({'window_id':['a','b','c'],'start':starts,'end':starts+pd.Timedelta('10min'),'score':[2.,2.,2.]})
        events=pd.DataFrame({'incident':['e'],'start':[pd.Timestamp('2020-01-01')],'end':[pd.Timestamp('2020-01-01 00:20')]})
        metrics,_,episodes,evidence=evaluate_alerts(table,1,events)
        self.assertEqual(metrics['alert_episodes'],2)
        self.assertEqual(metrics['outside_report_episodes'],1)
        self.assertEqual(evidence.delay_minutes.iloc[0],10)
        self.assertEqual(metrics['detected_incidents'],1)
        self.assertEqual(evaluate_alerts(table,2,events)[0]['alert_episodes'],0)

    def test_matching(self):
        table=pd.DataFrame({'window_id':['a'],'start':[1],'end':[2]})
        with self.assertRaises(ValueError):
            require_matching_windows([table,table.assign(window_id='b')])

    def test_cutoff(self):
        self.assertEqual(reference_threshold([1,2,2],.5),2)
        with self.assertRaises(ValueError):
            reference_threshold([np.nan])

    def test_selection(self):
        table=pd.DataFrame({'model':['a','b'],'covered_incidents':[2,2],'detected_incidents':[1,1],'outside_report_episodes_per_day':[2.,2.]})
        self.assertEqual(select_detector(table),'a')

    def test_score_at_incident_start(self):
        table=pd.DataFrame({'window_id':['a','b'],
            'start':pd.to_datetime(['2020-01-01 00:00','2020-01-01 00:10']),
            'end':pd.to_datetime(['2020-01-01 00:10','2020-01-01 00:20']), 'score':[2.,0.]})
        events=pd.DataFrame({'incident':['e'],'start':[pd.Timestamp('2020-01-01 00:10')],
                             'end':[pd.Timestamp('2020-01-01 00:21')]})
        metrics,_,_,evidence=evaluate_alerts(table,1,events)
        self.assertFalse(evidence.detected.iloc[0])
        self.assertEqual(metrics['detected_incidents'],0)

    def test_poor_incident_coverage(self):
        table=pd.DataFrame({'window_id':['a'],'start':[pd.Timestamp('2020-01-01')],
            'end':[pd.Timestamp('2020-01-01 00:10')],'score':[2.]})
        events=pd.DataFrame({'incident':['e'],'start':[pd.Timestamp('2020-01-01')],
            'end':[pd.Timestamp('2020-01-02')]})
        metrics,_,_,evidence=evaluate_alerts(table,1,events)
        self.assertEqual(metrics['covered_incidents'],0)
        self.assertTrue(evidence.detected.iloc[0])

    def duration_fixture(self, intervals):
        starts = pd.date_range('2020-01-01', periods=3, freq='10min')
        table = pd.DataFrame({'window_id': ['a', 'b', 'c'], 'start': starts,
                              'end': starts + pd.Timedelta('10min'), 'score': [2., 2., 2.]})
        events = pd.DataFrame([
            {'incident': str(i), 'start': starts[0] + pd.Timedelta(minutes=a),
             'end': starts[0] + pd.Timedelta(minutes=b)}
            for i, (a, b) in enumerate(intervals)
        ], columns=['incident', 'start', 'end'])
        return table, events

    def test_partial_episode_duration(self):
        table, events = self.duration_fixture([(5, 15)])
        metrics, _, episodes, _ = evaluate_alerts(table, 1, events)
        self.assertEqual(metrics['outside_report_episodes'], 0)
        self.assertAlmostEqual(metrics['inside_report_alerted_hours'], 10 / 60)
        self.assertAlmostEqual(metrics['outside_report_alerted_hours'], 20 / 60)
        self.assertAlmostEqual(episodes.duration_hours.iloc[0], .5)
        self.assertAlmostEqual(episodes.outside_report_hours.iloc[0], 20 / 60)

    def test_overlapping_report_duration(self):
        table, events = self.duration_fixture([(5, 15), (10, 20)])
        metrics, _, episodes, evidence = evaluate_alerts(table, 1, events)
        self.assertEqual(len(evidence), 2)
        self.assertAlmostEqual(metrics['inside_report_alerted_hours'], .25)
        self.assertAlmostEqual(metrics['outside_report_alerted_hours'], .25)
        self.assertAlmostEqual(episodes.inside_report_hours.sum(), .25)

    def test_report_boundary_duration(self):
        table, events = self.duration_fixture([(-10, 0), (30, 40)])
        metrics, _, episodes, _ = evaluate_alerts(table, 1, events)
        self.assertEqual(metrics['inside_report_alerted_hours'], 0)
        self.assertEqual(metrics['outside_report_alerted_hours'], .5)
        self.assertFalse(episodes.overlaps_report.any())

    def test_empty_report_duration(self):
        table, events = self.duration_fixture([])
        metrics, _, episodes, _ = evaluate_alerts(table, 1, events)
        self.assertEqual(metrics['covered_incidents'], 0)
        self.assertEqual(metrics['outside_report_alerted_hours'], .5)
        self.assertEqual(episodes.outside_report_hours.iloc[0], .5)
        metrics, _, episodes, _ = evaluate_alerts(table, 2, events)
        self.assertEqual(metrics['outside_report_alerted_hours'], 0)
        self.assertEqual(metrics['inside_report_alerted_hours'], 0)
        self.assertTrue(episodes.empty)

    def test_missing_window_time(self):
        table, events = self.duration_fixture([])
        table.loc[0, 'end'] = pd.NaT
        with self.assertRaises(ValueError):
            evaluate_alerts(table, 1, events)
