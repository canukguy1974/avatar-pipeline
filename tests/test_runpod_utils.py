import os
import unittest
from unittest.mock import patch
from app.runpod_utils import find_model_path

class TestRunpodUtils(unittest.TestCase):
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_find_model_path_success(self, mock_listdir, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ['hash123']
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        expected_path = "/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/hash123"
        
        path = find_model_path(model_name)
        self.assertEqual(path, expected_path)
        
        # Verify mock calls
        mock_exists.assert_called_with("/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots")

    @patch('os.path.exists')
    def test_find_model_path_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        path = find_model_path("Non/Existent")
        self.assertIsNone(path)

    def test_find_model_path_empty_input(self):
        self.assertIsNone(find_model_path(""))
        self.assertIsNone(find_model_path(None))

if __name__ == '__main__':
    unittest.main()
