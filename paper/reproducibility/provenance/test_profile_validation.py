#!/usr/bin/env python3
"""Regression tests for study-independent metadata validation (no training)."""
import copy
import json
from pathlib import Path
import unittest

from profile_validation import COLLECTIONS, validate_links, validate_profile
from query_claim_evidence import build_query_results

ROOT = Path(__file__).resolve().parent


def smaller_negative_fixture(audit):
    """A synthetic two-run, one-claim record; intentionally not new evidence."""
    result = copy.deepcopy(audit)
    result['record_id'] = 'synthetic:two-run-negative-control'
    result['protocols'] = result['protocols'][:1]
    protocol = result['protocols'][0]
    protocol['expected_runs'] = 2
    protocol['design']['seeds'] = [5101]
    protocol['design']['arms'].pop('zero_rk4', None)
    result['runs'] = [run for run in result['runs'] if run['seed'] == 5101 and run['arm'] in {'fixed_rate1', 'grouped_rate1'}]
    run_ids = {run['id'] for run in result['runs']}
    for name in ('configurations', 'checkpoints', 'predictions', 'evaluations'):
        result[name] = [item for item in result[name] if item['run_id'] in run_ids]
    result['controller_traces'] = []
    result['spatial_domains'] = [item for item in result['spatial_domains'] if item['campaign'] == protocol['campaign']]
    metric = result['metrics'][0]
    metric.update(value=0.1, ci95=[0.08, 0.12], n=1, wins_lower_error=0)
    metric['prediction_ids'] = [identifier for identifier in metric['prediction_ids'] if any(item['id'] == identifier for item in result['predictions'])]
    metric['evaluation_ids'] = [identifier for identifier in metric['evaluation_ids'] if any(item['id'] == identifier for item in result['evaluations'])]
    result['metrics'] = [metric]
    claim = result['claims'][0]
    claim.update(id='C99', statement='Synthetic control: the proposed reduction is not supported.', evidence_assessment='not_supported', supporting_evidence_ids=[metric['id']] + metric['evaluation_ids'], limiting_evidence_ids=[], limiting_artifact_ids=[], metric_ids=[metric['id']])
    result['claims'] = [claim]
    for entity in result['prov']['entities']:
        if entity['record_id'] == 'C1':
            entity['record_id'] = 'C99'
            entity['id'] = 'entity:C99'
    for relation in result['prov']['relations']:
        for key in ('subject_id', 'object_id'):
            if relation[key] == 'entity:C1':
                relation[key] = 'entity:C99'
    known = {item['id'] for name in COLLECTIONS if name != 'generations' for item in result[name]}
    result['generations'] = [item for item in result['generations'] if (not item.get('run_id') or item['run_id'] in run_ids) and set(item['input_ids'] + item['output_ids']).issubset(known)]
    known.update(item['id'] for item in result['generations'])
    for kind in ('entities', 'activities'):
        result['prov'][kind] = [item for item in result['prov'][kind] if item['record_id'] in known]
    ids = {item['id'] for kind in ('entities', 'activities', 'agents') for item in result['prov'][kind]}
    result['prov']['relations'] = [item for item in result['prov']['relations'] if item['subject_id'] in ids and item['object_id'] in ids]
    return result


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / 'allen_cahn_audit_record.json').read_text())

    def check_pass(self, audit):
        report = validate_profile(audit)
        self.assertTrue(report['passed'], '\n'.join(report['errors']))

    def test_frozen_record_has_valid_profile(self):
        self.check_pass(self.original)

    def test_smaller_negative_study_is_not_rejected(self):
        audit = smaller_negative_fixture(self.original)
        self.assertEqual((len(audit['runs']), len(audit['claims'])), (2, 1))
        self.check_pass(audit)
        query = build_query_results(audit)
        self.assertEqual(query['queries'][0]['query_id'], 'claim-evidence:C99')

    def test_claim_and_competency_identifiers_are_distinct(self):
        result = build_query_results(self.original)
        self.assertEqual([item['competency_question_id'] for item in result['competency_results']], [f'CQ{i}' for i in range(1, 7)])
        self.assertTrue(all(item['query_id'].startswith('claim-evidence:') for item in result['queries']))
        c4 = next(item for item in result['queries'] if item['claim_id'] == 'C4')
        self.assertEqual(c4['competency_question_ids'], ['CQ1', 'CQ4'])
        self.assertFalse(result['competency_results'][4]['performed_by_query'])

    def test_interval_boundary_error(self):
        audit = copy.deepcopy(self.original)
        audit['evaluations'][0]['interval']['end'] += 10
        self.assertTrue(validate_links(audit)['passed'])
        self.assertTrue(any('outside prediction' in error for error in validate_profile(audit)['errors']))

    def test_missing_disclosure_differs_from_empty_disclosure(self):
        audit = copy.deepcopy(self.original)
        audit['claims'][0]['limiting_evidence_ids'] = []
        self.check_pass(audit)
        del audit['claims'][0]['limiting_evidence_ids']
        self.assertFalse(validate_profile(audit)['passed'])

    def test_paired_metric_needs_run_level_evidence(self):
        audit = copy.deepcopy(self.original)
        audit['metrics'][0]['prediction_ids'] = []
        audit['metrics'][0]['evaluation_ids'] = []
        self.assertFalse(validate_profile(audit)['passed'])

    def test_primary_promotion_is_rejected(self):
        audit = copy.deepcopy(self.original)
        audit['claims'][0]['inferential_status'] = 'locked_confirmatory_primary'
        self.assertTrue(any('exceeds linked metric' in error for error in validate_profile(audit)['errors']))

    def test_duplicate_prov_identifier_is_rejected(self):
        audit = copy.deepcopy(self.original)
        audit['prov']['entities'].append(copy.deepcopy(audit['prov']['entities'][0]))
        self.assertFalse(validate_profile(audit)['passed'])

    def test_missing_application_projection_is_rejected(self):
        audit = copy.deepcopy(self.original)
        removed = audit['prov']['entities'].pop(0)['id']
        audit['prov']['relations'] = [item for item in audit['prov']['relations'] if removed not in {item['subject_id'], item['object_id']}]
        self.assertTrue(validate_links(audit)['passed'])
        self.assertFalse(validate_profile(audit)['passed'])

    def test_prov_edge_types_are_checked(self):
        audit = copy.deepcopy(self.original)
        relation = audit['prov']['relations'][0]
        relation['subject_id'] = audit['prov']['entities'][0]['id']
        self.assertTrue(validate_links(audit)['passed'])
        self.assertFalse(validate_profile(audit)['passed'])

    def test_consistent_false_prose_remains_a_blind_spot(self):
        audit = copy.deepcopy(self.original)
        audit['claims'][0]['statement'] = 'A scientifically false assertion cannot be rejected by metadata consistency alone.'
        self.check_pass(audit)


if __name__ == '__main__':
    unittest.main()
