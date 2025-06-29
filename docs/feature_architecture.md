# Feature Extraction Architecture

## Overview

The pipeline supports multiple feature extraction strategies to accommodate different ML approaches:

1. **Statistical Features** (Current) - Traditional ML features (mean, std, percentiles)
2. **Image Features** (Planned) - CNN-compatible 2D representations
3. **Sequence Features** (Planned) - LSTM/Transformer-compatible sequences
4. **Text Features** (Planned) - NLP features from user text files

## Current Architecture

### Feature Configuration

Features are defined in `scripts/pipeline/extract_features.py`:

```python
FEATURE_CONFIGS = {
    "typenet_ml_user_platform": FeatureConfig(
        name="typenet_ml_user_platform",
        description="TypeNet ML features aggregated by user and platform",
        aggregation_level="user_platform",
        imputation_strategy="global",
        keep_outliers=False
    ),
    # ... more configs
}
```

### Adding a New Feature Set

#### Step 1: Define Feature Configuration

```python
# In scripts/pipeline/extract_features.py
FEATURE_CONFIGS["lstm_sequences"] = FeatureConfig(
    name="lstm_sequences",
    description="Sequential features for LSTM models",
    aggregation_level="session",  # or "video", "user_platform"
    output_format="sequences",    # New format type
    sequence_length=100,          # Custom parameter
    keep_outliers=True            # Keep all data for sequences
)
```

#### Step 2: Create Feature Extractor

Create a new file: `scripts/pipeline/features/lstm_extractor.py`

```python
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

class LSTMFeatureExtractor:
    """Extract sequential features for LSTM models"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.sequence_length = config.sequence_length
    
    def extract(self, keypairs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract LSTM-ready sequences from keypairs"""
        
        # Group by session for sequence continuity
        sequences = []
        labels = []
        
        for (user_id, session_id), group in keypairs_df.groupby(['user_id', 'session_id']):
            # Sort by timestamp
            group = group.sort_values('key1_timestamp')
            
            # Extract timing features
            hl_sequence = group['HL'].values
            il_sequence = group['IL'].values
            pl_sequence = group['PL'].values
            rl_sequence = group['RL'].values
            
            # Stack features
            features = np.column_stack([hl_sequence, il_sequence, pl_sequence, rl_sequence])
            
            # Create fixed-length sequences
            for i in range(0, len(features) - self.sequence_length + 1, 10):
                seq = features[i:i + self.sequence_length]
                sequences.append(seq)
                labels.append(user_id)
        
        return {
            'sequences': np.array(sequences),
            'labels': np.array(labels),
            'feature_names': ['HL', 'IL', 'PL', 'RL']
        }
    
    def save(self, features: Dict[str, np.ndarray], output_dir: Path):
        """Save sequences in NumPy format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sequences
        np.save(output_dir / 'sequences.npy', features['sequences'])
        np.save(output_dir / 'labels.npy', features['labels'])
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'num_sequences': len(features['sequences']),
            'feature_names': features['feature_names'],
            'shape': features['sequences'].shape
        }
        
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
```

#### Step 3: Register Extractor

```python
# In scripts/pipeline/extract_features.py
from scripts.pipeline.features.lstm_extractor import LSTMFeatureExtractor

# Add to the extract_features function
if feature_type == "lstm_sequences":
    extractor = LSTMFeatureExtractor(config)
    features = extractor.extract(valid_data)
    extractor.save(features, feature_dir)
```

## Image Features for CNN

Example implementation for CNN features:

```python
# scripts/pipeline/features/cnn_extractor.py
class CNNFeatureExtractor:
    """Convert keystroke patterns to images"""
    
    def __init__(self, config: FeatureConfig):
        self.image_size = (224, 224)
        self.channels = 3
    
    def create_keystroke_image(self, user_keypairs: pd.DataFrame) -> np.ndarray:
        """Create image representation of keystroke patterns"""
        
        # Create timing matrix
        timing_matrix = user_keypairs[['HL', 'IL', 'PL', 'RL']].values
        
        # Normalize to 0-255
        normalized = (timing_matrix - timing_matrix.min()) / (timing_matrix.max() - timing_matrix.min())
        normalized = (normalized * 255).astype(np.uint8)
        
        # Reshape to image
        # Option 1: Direct reshape with padding
        # Option 2: Create spectrogram-like representation
        # Option 3: Use key-pair relationships as 2D structure
        
        # Example: Create heatmap of key transitions
        from scipy.ndimage import zoom
        
        # Create transition matrix
        keys = pd.concat([user_keypairs['key1'], user_keypairs['key2']]).unique()
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        
        transition_matrix = np.zeros((len(keys), len(keys)))
        for _, row in user_keypairs.iterrows():
            i = key_to_idx.get(row['key1'], 0)
            j = key_to_idx.get(row['key2'], 0)
            transition_matrix[i, j] += row['HL']  # Use hold latency as weight
        
        # Resize to standard size
        image = zoom(transition_matrix, 
                    (self.image_size[0] / transition_matrix.shape[0],
                     self.image_size[1] / transition_matrix.shape[1]))
        
        # Create 3-channel image (RGB)
        image_rgb = np.stack([image, image, image], axis=-1)
        
        return image_rgb
```

## Text Feature Extraction

For processing user text files:

```python
# scripts/pipeline/features/text_extractor.py
from transformers import AutoTokenizer, AutoModel
import torch

class TextFeatureExtractor:
    """Extract features from user text files"""
    
    def __init__(self, config: FeatureConfig):
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
    
    def extract_embeddings(self, text: str) -> np.ndarray:
        """Extract BERT embeddings from text"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Extract statistical text features"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]),
            'vocabulary_size': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
        }
```

## Testing New Features

Create tests for your feature extractor:

```python
# tests/pipeline/features/test_lstm_extractor.py
import pytest
import numpy as np
import pandas as pd
from scripts.pipeline.features.lstm_extractor import LSTMFeatureExtractor

def test_lstm_sequence_extraction():
    """Test LSTM sequence extraction"""
    # Create sample keypairs data
    keypairs = pd.DataFrame({
        'user_id': ['user1'] * 200,
        'session_id': [1] * 200,
        'key1_timestamp': range(200),
        'HL': np.random.normal(100, 20, 200),
        'IL': np.random.normal(50, 10, 200),
        'PL': np.random.normal(80, 15, 200),
        'RL': np.random.normal(120, 25, 200),
        'valid': True
    })
    
    config = FeatureConfig(
        name="lstm_test",
        sequence_length=100
    )
    
    extractor = LSTMFeatureExtractor(config)
    features = extractor.extract(keypairs)
    
    # Check output shape
    assert features['sequences'].shape[1] == 100  # sequence length
    assert features['sequences'].shape[2] == 4    # number of features
    assert len(features['labels']) == len(features['sequences'])
```

## Best Practices

1. **Consistency**: Follow existing patterns for configuration and output formats
2. **Documentation**: Include docstrings and examples
3. **Testing**: Write unit tests for edge cases
4. **Performance**: Consider memory usage for large datasets
5. **Compatibility**: Ensure outputs work with common ML frameworks (scikit-learn, PyTorch, TensorFlow)

## Future Considerations

- **Streaming Processing**: For very large datasets
- **GPU Acceleration**: For complex transformations
- **Feature Versioning**: Track feature extraction logic changes
- **Feature Store Integration**: For production ML systems