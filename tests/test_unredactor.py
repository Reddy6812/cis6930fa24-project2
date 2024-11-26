import unittest
import pandas as pd

# Mock predict_on_test function for simplicity
def mock_predict_on_test(test_path, submission_path):
    """Mock function to simulate predictions without using models."""
    # Load mock test data
    test_df = pd.DataFrame({
        "id": [1, 2],
        "context": [
            "His wife, scheming ██████████████, longs for the finer things in life.",
            "This movie starred ██████████████ in a great role."
        ]
    })
    
    # Mock predictions
    test_df['predicted_name'] = ["Catherine Zeta", "William Holden"]
    
    # Save to submission file
    test_df[['id', 'predicted_name']].to_csv(submission_path, sep='\t', index=False, header=False)
    return test_df[['id', 'predicted_name']]

class TestMockPredictOnTest(unittest.TestCase):
    
    def setUp(self):
        """Set up mock file paths."""
        self.test_path = "mock_test.tsv"  # Mock input path
        self.submission_path = "mock_submission.tsv"  # Mock output path
    
    def test_mock_predictions(self):
        """Test mock predictions."""
        # Call the mock function
        result_df = mock_predict_on_test(self.test_path, self.submission_path)
        
        # Check if the result DataFrame has expected predictions
        self.assertEqual(result_df.iloc[0]['predicted_name'], "Catherine Zeta")
        self.assertEqual(result_df.iloc[1]['predicted_name'], "William Holden")
        self.assertEqual(len(result_df), 2, "Expected 2 rows in submission.")
        print("Mock prediction test passed.")
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            os.remove(self.submission_path)
        except FileNotFoundError:
            pass

# Run the test
suite = unittest.TestLoader().loadTestsFromTestCase(TestMockPredictOnTest)
unittest.TextTestRunner().run(suite)
