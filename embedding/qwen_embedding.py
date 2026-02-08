"""
Qwen2.5-VL-32B-Instruct Embedding Extractor for Patient Descriptions

This script extracts embeddings from patient text descriptions using Qwen2.5-VL-32B-Instruct.
Supports both local inference and API-based inference.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import logging
from dataclasses import dataclass
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction"""
    model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1  # For 32B model, typically process one at a time
    max_length: int = 2048  # Maximum sequence length
    use_8bit: bool = False  # Use 8-bit quantization to reduce memory
    use_4bit: bool = True  # Use 4-bit quantization for even lower memory
    pooling_method: str = "mean"  # Options: "mean", "cls", "last"
    output_layer: int = -1  # Which layer to extract embeddings from (-1 = last layer)


class QwenEmbeddingExtractor:
    """Extract embeddings using Qwen2.5-VL-32B-Instruct"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5-VL model with optimization"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from transformers import BitsAndBytesConfig
            
            logger.info(f"Loading {self.config.model_name} on {self.config.device}...")
            
            # Configure quantization if needed
            quantization_config = None
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            elif self.config.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                logger.info("Using 8-bit quantization")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.config.device
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except ImportError as e:
            logger.error(f"Required packages not found: {e}")
            logger.error("Please install: pip install transformers accelerate bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings to get a single vector per sample"""
        if self.config.pooling_method == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_method == "cls":
            # Use the first token (CLS-like)
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_method == "last":
            # Use the last non-padding token
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            return hidden_states[torch.arange(batch_size), sequence_lengths]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling_method}")
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a single text"""
        try:
            # Prepare input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            # Process text
            text_input = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_input],
                padding=True,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get hidden states from the specified layer
                hidden_states = outputs.hidden_states[self.config.output_layer]
                
                # Pool to get single vector
                attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
                pooled_embedding = self._pool_embeddings(hidden_states, attention_mask)
                
                # Convert to numpy
                embedding = pooled_embedding.cpu().float().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of texts"""
        embeddings = []
        
        for text in tqdm(texts, desc="Extracting embeddings"):
            embedding = self.extract_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class PatientEmbeddingProcessor:
    """Process patient text files and extract embeddings"""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 config: EmbeddingConfig = None):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = QwenEmbeddingExtractor(config)
        
    def load_patient_text(self, file_path: Path) -> str:
        """Load patient text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def process_all_patients(self) -> Dict[str, np.ndarray]:
        """Process all patient files and extract embeddings"""
        patient_files = sorted(self.data_dir.glob("patient_*.txt"))
        
        logger.info(f"Found {len(patient_files)} patient files")
        
        embeddings_dict = {}
        
        for patient_file in tqdm(patient_files, desc="Processing patients"):
            try:
                # Extract patient ID from filename
                patient_id = patient_file.stem  # e.g., "patient_1279399"
                
                # Load patient text
                patient_text = self.load_patient_text(patient_file)
                
                # Extract embedding
                embedding = self.extractor.extract_embedding(patient_text)
                
                embeddings_dict[patient_id] = embedding
                
                logger.info(f"Processed {patient_id}: embedding shape {embedding.shape}")
                
            except Exception as e:
                logger.error(f"Error processing {patient_file}: {e}")
                continue
        
        return embeddings_dict
    
    def save_embeddings(self, embeddings_dict: Dict[str, np.ndarray], format: str = "npz"):
        """Save embeddings to disk"""
        
        if format == "npz":
            # Save as compressed numpy archive
            output_file = self.output_dir / "patient_embeddings.npz"
            np.savez_compressed(output_file, **embeddings_dict)
            logger.info(f"Saved embeddings to {output_file}")
        
        elif format == "npy":
            # Save individual numpy files
            for patient_id, embedding in embeddings_dict.items():
                output_file = self.output_dir / f"{patient_id}_embedding.npy"
                np.save(output_file, embedding)
            logger.info(f"Saved {len(embeddings_dict)} individual embedding files")
        
        elif format == "pkl":
            # Save as pickle
            output_file = self.output_dir / "patient_embeddings.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            logger.info(f"Saved embeddings to {output_file}")
        
        elif format == "json":
            # Save as JSON (embeddings as lists)
            output_file = self.output_dir / "patient_embeddings.json"
            embeddings_json = {
                k: v.tolist() for k, v in embeddings_dict.items()
            }
            with open(output_file, 'w') as f:
                json.dump(embeddings_json, f)
            logger.info(f"Saved embeddings to {output_file}")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Also save metadata
        metadata = {
            "num_patients": len(embeddings_dict),
            "embedding_dim": next(iter(embeddings_dict.values())).shape[0],
            "patient_ids": list(embeddings_dict.keys()),
            "model_name": self.extractor.config.model_name,
            "pooling_method": self.extractor.config.pooling_method,
        }
        
        metadata_file = self.output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_embeddings(self, format: str = "npz") -> Dict[str, np.ndarray]:
        """Load previously saved embeddings"""
        
        if format == "npz":
            input_file = self.output_dir / "patient_embeddings.npz"
            data = np.load(input_file)
            return {key: data[key] for key in data.files}
        
        elif format == "pkl":
            input_file = self.output_dir / "patient_embeddings.pkl"
            with open(input_file, 'rb') as f:
                return pickle.load(f)
        
        elif format == "json":
            input_file = self.output_dir / "patient_embeddings.json"
            with open(input_file, 'r') as f:
                embeddings_json = json.load(f)
            return {k: np.array(v) for k, v in embeddings_json.items()}
        
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    """Main function to run embedding extraction"""
    
    # Configuration
    config = EmbeddingConfig(
        model_name="Qwen/Qwen2.5-VL-32B-Instruct",
        use_4bit=True,  # Use 4-bit quantization to fit in memory
        pooling_method="mean",
        max_length=2048,
    )
    
    # Paths
    data_dir = Path("/Users/finleyyu/Desktop/research/disease prediction/data/preprocessed/natural_text_train")
    output_dir = Path("/Users/finleyyu/Desktop/research/disease prediction/data/preprocessed/embeddings")
    
    # Process patients
    processor = PatientEmbeddingProcessor(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config
    )
    
    # Extract embeddings
    logger.info("Starting embedding extraction...")
    embeddings_dict = processor.process_all_patients()
    
    # Save embeddings
    processor.save_embeddings(embeddings_dict, format="npz")
    
    logger.info("Embedding extraction completed!")
    logger.info(f"Total patients processed: {len(embeddings_dict)}")
    
    # Print sample information
    sample_id = list(embeddings_dict.keys())[0]
    sample_embedding = embeddings_dict[sample_id]
    logger.info(f"Sample embedding shape: {sample_embedding.shape}")
    logger.info(f"Sample embedding (first 10 dims): {sample_embedding[:10]}")


if __name__ == "__main__":
    main()

