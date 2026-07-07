import json
from pathlib import Path
from datetime import datetime

class DataManager:
    def __init__(self, data_dir="backend/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.samples_file = self.data_dir / "samples.json"
        self.results_file = self.data_dir / "results.json"
    
    def create_default_samples(self):
        """Create 30 default samples"""
        instructions = [
            "Make it sound happy and cheerful",
            "Speak with a calm, soothing tone",
            "Sound excited and energetic",
            "Use a sad, melancholic voice",
            "Neutral professional tone"
        ]
        
        reference_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Good morning, welcome to our service",
            "Thank you for using our application",
            "Please try again later",
            "Your request has been processed successfully"
        ]
        
        samples = []
        models = ["Model_A", "Model_B", "Model_C"]
        
        for i in range(1, 31):
            sample = {
                "id": i,
                "instruction": instructions[(i - 1) % len(instructions)],
                "reference_text": reference_texts[(i - 1) % len(reference_texts)],
                "models": {
                    model: {
                        "name": model,
                        "audio_path": f"assets/audio/sample_{i}_{model}.wav",
                        "duration": 3.5
                    } for model in models
                }
            }
            samples.append(sample)
        
        return samples
    
    def load_samples(self):
        """Load samples from file"""
        if self.samples_file.exists():
            with open(self.samples_file, 'r') as f:
                return json.load(f)
        return self.create_default_samples()
    
    def save_samples(self, samples):
        """Save samples to file"""
        with open(self.samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
    
    def load_results(self):
        """Load evaluation results"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_result(self, result):
        """Append evaluation result"""
        results = self.load_results()
        result['timestamp'] = datetime.now().isoformat()
        results.append(result)
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_evaluator_stats(self, evaluator_name):
        """Get statistics for a specific evaluator"""
        results = self.load_results()
        evaluator_results = [r for r in results if r.get('evaluator') == evaluator_name]
        
        if not evaluator_results:
            return None
        
        return {
            "total_evaluations": len(evaluator_results),
            "average_score": sum(r['total_score'] for r in evaluator_results) / len(evaluator_results),
            "bonus_count": len([r for r in evaluator_results if r.get('bonus_info', {}).get('bonus', 0) > 0])
        }