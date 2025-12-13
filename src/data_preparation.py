import numpy as np
import random
from typing import List, Tuple, Dict
from pathlib import Path

class TestDataPreparer:
    """Подготовка объединённого датасета из HWR200 + School Notebooks"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def _load_hwr200(self, processed: bool) -> List[Tuple[str, str]]:
      """Внутренняя загрузка HWR200"""
      data_type = "processed" if processed else "raw"
      img_dir = self.base_path / "data" / data_type / "hwr200"
      txt_dir = self.base_path / "data" / "raw" / "hwr200" / "annotations"
      
      if (img_dir / "images").exists():
          img_dir = img_dir / "images"
      
      dataset = []
      
      # Ищем все форматы изображений
      for pattern in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
          for img_path in sorted(img_dir.glob(pattern)):
              base_name = img_path.stem
              parts = base_name.split('_')
              
              for i in range(len(parts), 0, -1):
                  candidate = '_'.join(parts[:i])
                  txt_path = txt_dir / f"{candidate}.txt"
                  if txt_path.exists():
                      text = txt_path.read_text(encoding='utf-8').strip()
                      dataset.append((str(img_path), text))
                      break
      
      return dataset


    def _load_school_notebooks(self, processed: bool) -> List[Tuple[str, str]]:
      """Внутренняя загрузка School Notebooks"""
      data_type = "processed" if processed else "raw"
      img_dir = self.base_path / "data" / data_type / "school_notebooks_ru"
      csv_path = self.base_path / "data" / "raw" / "school_notebooks_ru" / "annotations.csv"
      
      if (img_dir / "images").exists():
          img_dir = img_dir / "images"
      
      import pandas as pd
      df = pd.read_csv(csv_path)
      
      texts = {}
      for img_name, group in df.groupby('image'):
          if 'bbox' in group.columns:
              group = group.copy()
              try:
                  group['bbox_dict'] = group['bbox'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                  group['top'] = group['bbox_dict'].apply(lambda b: b['top'])
                  group['left'] = group['bbox_dict'].apply(lambda b: b['left'])
                  
                  # Группируем по строкам (±50 пикселей по высоте)
                  group['row'] = (group['top'] // 100).astype(int)
                  
                  # Сортируем: сначала по строке (сверху вниз), потом слева направо
                  group = group.sort_values(by=['row', 'left'])
              except:
                  pass
          
          full_text = ' '.join(group['text'].values)
          texts[img_name.lower()] = full_text
      
      dataset = []
      
      for pattern in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
          for img_path in sorted(img_dir.glob(pattern)):
              base_name = img_path.stem
              parts = base_name.split('_')
              
              for i in range(len(parts), 0, -1):
                  for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                      candidate = ('_'.join(parts[:i]) + ext).lower()
                      if candidate in texts:
                          dataset.append((str(img_path), texts[candidate]))
                          break
                  else:
                      continue
                  break
      
      return dataset
    
    def load_combined_dataset(self, processed: bool = False, max_samples: int = None, 
                             shuffle: bool = True, seed: int = 42) -> List[Tuple[str, str]]:
        """
        Загружает объединённый датасет HWR200 + School Notebooks
        
        Args:
            processed: использовать аугментированные данные
            max_samples: общее ограничение на весь датасет (не на каждый отдельно!)
            shuffle: перемешать датасет
            seed: seed для воспроизводимости
        
        Returns:
            List[(image_path, ground_truth)]
        """
        print(f"Загрузка объединённого датасета ({'processed' if processed else 'raw'})...")
        
        hwr200 = self._load_hwr200(processed)
        print(f"  HWR200: {len(hwr200)} образцов")
        
        school = self._load_school_notebooks(processed)
        print(f"  School Notebooks: {len(school)} образцов")
        
        # Объединяем
        combined = hwr200 + school
        
        # Перемешиваем
        if shuffle:
            random.seed(seed)
            random.shuffle(combined)
            print(f"  Датасет перемешан (seed={seed})")
        
        # Ограничиваем
        if max_samples and len(combined) > max_samples:
            combined = combined[:max_samples]
            print(f"  Ограничено до {max_samples} образцов")
        
        print(f"Итого: {len(combined)} образцов")
        return combined
    
    def get_statistics(self, dataset: List[Tuple[str, str]]) -> Dict:
        if len(dataset) == 0:
            return {'error': 'Empty dataset'}
        
        texts = [text for _, text in dataset]
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        all_chars = set(''.join(texts))
        
        return {
            'total_samples': len(dataset),
            'mean_text_length': np.mean(lengths),
            'std_text_length': np.std(lengths),
            'min_text_length': np.min(lengths),
            'max_text_length': np.max(lengths),
            'mean_word_count': np.mean(word_counts),
            'total_characters': sum(lengths),
            'unique_characters': len(all_chars)
        }