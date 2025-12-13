from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from models import AVAILABLE_OCR_MODELS, get_ocr_model
from metrics import MetricsCalculator, PerformanceMetrics

from typing import List, Tuple, Dict


class ValidationPipeline:
    """Pipeline валидации без сохранения файлов"""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator(normalize=True)
        self.perf_metrics = PerformanceMetrics()
    
    def evaluate_model(self, model_name: str, inference_fn, test_data: List[Tuple[str, str]], 
                      measure_performance: bool = True, 
                      show_examples: int = 3) -> Dict:
        """
        Оценка модели
        
        Args:
            model_name: название модели
            inference_fn: функция инференса
            test_data: List[(image_path, ground_truth)]
            measure_performance: измерять производительность
            show_examples: сколько примеров показать (0 = не показывать)
        """
        print(f"\n{'='*60}")
        print(f"Модель: {model_name}")
        print(f"{'='*60}")
        print(f"Образцов: {len(test_data)}")
        
        images = [item[0] for item in test_data]
        ground_truths = [item[1] for item in test_data]
        
        # Инференс
        print("Инференс...")
        predictions = [inference_fn(img) for img in images]
        
        # Метрики точности
        print("Расчёт метрик...")
        df_metrics = self.metrics_calc.evaluate_batch(ground_truths, predictions)
        summary = self.metrics_calc.get_summary(df_metrics)
        
        # Показываем примеры
        if show_examples > 0:
            self._print_examples(df_metrics, show_examples)
        
        # Производительность
        performance = {}
        if measure_performance:
          print("Измерение производительности...")
          latency = self.perf_metrics.measure_latency(
              inference_fn, images[:min(10, len(images))], warmup=3, iterations=10
          )
          gpu_mem = self.perf_metrics.measure_gpu_memory()
          
          performance = {**latency, 'gpu_memory': gpu_mem}
        
        report = {
            'model_name': model_name,
            'test_samples': len(test_data),
            'accuracy_metrics': summary,
            'performance_metrics': performance,
            'detailed_metrics': df_metrics
        }
        
        self._print_summary(model_name, summary, performance)
        
        return report
    
    def _print_examples(self, df: pd.DataFrame, n: int):
        """Выводит примеры prediction vs ground truth"""
        print(f"\n{'='*60}")
        print(f"Примеры (первые {n}):")
        print(f"{'='*60}")
        
        for i in range(min(n, len(df))):
            row = df.iloc[i]
            print(f"\n[{row['sample_id']}]")
            print(f"Ground Truth: {row['reference'][:80]}...")
            print(f"Prediction:   {row['hypothesis'][:80]}...")
            print(f"CER: {row['cer']:.1f}% | WER: {row['wer']:.1f}%")
    
    def _print_summary(self, model_name: str, summary: Dict, performance: Dict):
        print(f"\n{'='*60}")
        print(f"Результаты: {model_name}")
        print(f"{'='*60}")
        print(f"\nТочность:")
        print(f"  CER:           {summary['mean_cer']:.2f}% (±{summary['std_cer']:.2f}%)")
        print(f"  WER:           {summary['mean_wer']:.2f}% (±{summary['std_wer']:.2f}%)")
        print(f"  Char Accuracy: {summary['mean_char_accuracy']:.2f}%")
        print(f"  Perfect Rate:  {summary['perfect_rate']:.2f}%")
        
        if performance:
            print(f"\nПроизводительность:")
            print(f"  Latency:    {performance['mean_latency_ms']:.2f} ms ({performance['mean_latency_sec']:.3f} sec)")
            print(f"  P95:        {performance.get('p95_latency_ms', 0):.2f} ms")
            print(f"  Throughput: {performance.get('throughput_per_sec', 0):.2f} samples/sec")
            
            gpu = performance.get('gpu_memory', {})
            if 'max_allocated_mb' in gpu:
                print(f"  GPU Memory: {gpu['max_allocated_mb']:.2f} MB")
        
        print(f"{'='*60}\n")
    
    def compare_models(self, reports: List[Dict]) -> pd.DataFrame:
        comparison = []
        for report in reports:
            acc = report['accuracy_metrics']
            perf = report['performance_metrics']
            
            row = {
                'Model': report['model_name'],
                'CER (%)': f"{acc['mean_cer']:.2f}",
                'WER (%)': f"{acc['mean_wer']:.2f}",
                'Char Acc (%)': f"{acc['mean_char_accuracy']:.2f}%",
                'Latency (sec)': f"{perf.get('mean_latency_sec', 0):.3f}",
                'Throughput': f"{perf.get('throughput_per_sec', 0):.2f}/sec"
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        print("\nСравнение моделей:")
        print(df.to_string(index=False))
        
        return df
