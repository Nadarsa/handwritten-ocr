import numpy as np
import pandas as pd
import time
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from typing import List, Tuple, Dict


class MetricsCalculator:
    """Расчёт метрик валидации OCR"""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        
    def _normalize_text(self, text: str) -> str:
        if not self.normalize:
            return text
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    def calculate_cer(self, reference: str, hypothesis: str) -> Dict[str, float]:
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        edit_distance = levenshtein_distance(ref, hyp)
        cer = (edit_distance / len(ref) * 100) if len(ref) > 0 else 0
        normalized_cer = min(cer, 100.0)
        accuracy = ((len(ref) - edit_distance) / len(ref) * 100) if len(ref) > 0 else 0
        
        return {
            'cer': cer,
            'normalized_cer': normalized_cer,
            'accuracy': accuracy,
            'edit_distance': edit_distance,
            'reference_length': len(ref),
            'hypothesis_length': len(hyp)
        }
    
    def calculate_wer(self, reference: str, hypothesis: str) -> Dict[str, float]:
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        edit_distance = levenshtein_distance(' '.join(ref_words), ' '.join(hyp_words))
        wer = (edit_distance / len(ref_words) * 100) if len(ref_words) > 0 else 0
        
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        correct_words = len(ref_set & hyp_set)
        bwer = ((len(ref_words) - correct_words) / len(ref_words) * 100) if len(ref_words) > 0 else 0
        
        word_accuracy = ((len(ref_words) - edit_distance) / len(ref_words) * 100) if len(ref_words) > 0 else 0
        
        return {
            'wer': wer,
            'bwer': bwer,
            'delta_wer': wer - bwer,
            'word_accuracy': word_accuracy,
            'reference_words': len(ref_words),
            'hypothesis_words': len(hyp_words)
        }
    
    def analyze_character_errors(self, reference: str, hypothesis: str) -> Dict:
        ref = self._normalize_text(reference)
        hyp = self._normalize_text(hypothesis)
        
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        min_len = min(len(ref), len(hyp))
        
        substitutions = []
        for i in range(min_len):
            if ref[i] != hyp[i]:
                confusion_matrix[ref[i]][hyp[i]] += 1
                substitutions.append((ref[i], hyp[i], i))
        
        deletions = len(ref) - min_len if len(ref) > len(hyp) else 0
        insertions = len(hyp) - min_len if len(hyp) > len(ref) else 0
        
        errors = []
        for true_char, pred_dict in confusion_matrix.items():
            for pred_char, count in pred_dict.items():
                errors.append((true_char, pred_char, count))
        errors.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'confusion_matrix': dict(confusion_matrix),
            'substitutions': len(substitutions),
            'deletions': deletions,
            'insertions': insertions,
            'top_errors': errors[:10]
        }
    
    def evaluate_batch(self, references: List[str], hypotheses: List[str]) -> pd.DataFrame:
        if len(references) != len(hypotheses):
            raise ValueError("Length mismatch")
        
        results = []
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            cer_metrics = self.calculate_cer(ref, hyp)
            wer_metrics = self.calculate_wer(ref, hyp)
            error_analysis = self.analyze_character_errors(ref, hyp)
            
            result = {
                'sample_id': f"sample_{i}",
                'reference': ref,
                'hypothesis': hyp,
                'cer': cer_metrics['cer'],
                'normalized_cer': cer_metrics['normalized_cer'],
                'char_accuracy': cer_metrics['accuracy'],
                'wer': wer_metrics['wer'],
                'bwer': wer_metrics['bwer'],
                'delta_wer': wer_metrics['delta_wer'],
                'word_accuracy': wer_metrics['word_accuracy'],
                'edit_distance': cer_metrics['edit_distance'],
                'substitutions': error_analysis['substitutions'],
                'deletions': error_analysis['deletions'],
                'insertions': error_analysis['insertions']
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        return {
            'mean_cer': df['cer'].mean(),
            'std_cer': df['cer'].std(),
            'median_cer': df['cer'].median(),
            'mean_normalized_cer': df['normalized_cer'].mean(),
            'mean_wer': df['wer'].mean(),
            'std_wer': df['wer'].std(),
            'mean_bwer': df['bwer'].mean(),
            'mean_delta_wer': df['delta_wer'].mean(),
            'mean_char_accuracy': df['char_accuracy'].mean(),
            'mean_word_accuracy': df['word_accuracy'].mean(),
            'total_samples': len(df),
            'perfect_samples': (df['cer'] == 0).sum(),
            'perfect_rate': (df['cer'] == 0).sum() / len(df) * 100
        }
    
class PerformanceMetrics:
    """Метрики производительности"""
    
    def measure_latency(self, inference_fn, test_images: List, 
                       warmup: int = 3, iterations: int = 10) -> Dict:
        """
        Измерение латентности
        
        warmup: прогрев GPU/кэша (первые запуски медленнее)
        iterations: повторные измерения для статистики
        """
        # Warmup
        for i in range(warmup):
            _ = inference_fn(test_images[i % len(test_images)])
        
        # Измерение
        latencies = []
        for i in range(iterations):
            img = test_images[i % len(test_images)]
            start = time.perf_counter()
            _ = inference_fn(img)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        latencies = np.array(latencies)
        mean_latency_ms = np.mean(latencies)
        
        return {
            'mean_latency_ms': mean_latency_ms,
            'mean_latency_sec': mean_latency_ms / 1000,  # В секундах
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_per_sec': 1000 / mean_latency_ms,
            'iterations': iterations
        }
    
    def measure_gpu_memory(self):
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                    'device': torch.cuda.get_device_name(0)
                }
            return {'error': 'CUDA not available'}
        except ImportError:
            return {'error': 'PyTorch not installed'}