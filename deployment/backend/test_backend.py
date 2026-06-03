import unittest
import json
import time
import os
import sys

# Append current directory to import path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend Flask app
from app import app, GENE_SEQUENCES, STRUCTURAL_DB

class TestHCMBackend(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        """Test health endpoint returns loaded models status."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertEqual(data['model_version'], 'v1.0')
        self.assertIn('two_tower', data)
        self.assertIn('rf_model', data)

    def test_presets_list(self):
        """Test preset variants list is dynamically returned."""
        response = self.app.get('/variants')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('variants', data)
        self.assertIn('count', data)

    def test_input_validation_missing_fields(self):
        """Test missing fields return 400 with descriptive errors."""
        response = self.app.post('/predict', 
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Input validation failed')
        self.assertTrue(any("gene" in m for m in data['messages']))

    def test_input_validation_invalid_gene(self):
        """Test wrong gene name returns 400."""
        payload = {
            "gene": "INVALID_GENE",
            "position": 100,
            "ref_aa": "A",
            "alt_aa": "P"
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertTrue(any("Invalid gene" in m for m in data['messages']))

    def test_input_validation_position_range(self):
        """Test position exceeding maximum residue size returns 400."""
        payload = {
            "gene": "MYL3",
            "position": 5000, # MYL3 only has length 195
            "ref_aa": "A",
            "alt_aa": "P"
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertTrue(any("exceeds maximum residue position" in m for m in data['messages']))

    def test_input_validation_amino_acid_matching(self):
        """Test reference and mutant residues being identical returns 400."""
        payload = {
            "gene": "MYL3",
            "position": 152,
            "ref_aa": "E",
            "alt_aa": "E" # identical
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertTrue(any("cannot be identical" in m for m in data['messages']))

    def test_prediction_success_with_window_resolution(self):
        """Test valid variant predicts successfully with auto-resolved sequence window."""
        # MYL3 E152K is a known variant
        payload = {
            "gene": "MYL3",
            "position": 152,
            "ref_aa": "E",
            "alt_aa": "K"
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['gene'], 'MYL3')
        self.assertEqual(data['position'], 152)
        self.assertEqual(data['ref_aa'], 'E')
        self.assertEqual(data['alt_aa'], 'K')
        
        # Check auto-resolved sequence window centered at E (TVMGAELRHVL)
        self.assertEqual(data['sequence_window'], 'TVMGAELRHVL')
        self.assertEqual(data['sequence_window'][5], 'E') # Center must be WT (E)
        
        # Check metrics and version
        self.assertIn('raw_score', data)
        self.assertIn('calibrated_score', data)
        self.assertIn('rf_score', data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertEqual(data['model_version'], 'v1.0')
        self.assertIn('explanations', data)
        
        # Verify explanations shape
        exps = data['explanations']
        self.assertTrue(len(exps) > 0)
        self.assertEqual(exps[0]['feature'], 'ESM-2 Delta Embedding')

if __name__ == '__main__':
    unittest.main()
