//! RTEN-based CLIP embedder implementation.
//!
//! This module provides a multimodal (text + image) embedder using the RTEN
//! inference runtime with OpenAI CLIP ONNX models. It loads ONNX models
//! directly (no `.rten` conversion required) and produces L2-normalized
//! embeddings in CLIP's shared 512-dimensional vector space.
//!
//! Requires the `embeddings-rten-multimodal` feature to be enabled.

#[cfg(feature = "embeddings-rten-multimodal")]
use std::any::Any;
#[cfg(feature = "embeddings-rten-multimodal")]
use std::path::Path;

#[cfg(feature = "embeddings-rten-multimodal")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-rten-multimodal")]
use hf_hub::api::sync::ApiBuilder;
#[cfg(feature = "embeddings-rten-multimodal")]
use rten::{Model, NodeId, ValueOrView};
#[cfg(feature = "embeddings-rten-multimodal")]
use rten_tensor::Tensor;
#[cfg(feature = "embeddings-rten-multimodal")]
use rten_tensor::prelude::*;
#[cfg(feature = "embeddings-rten-multimodal")]
use tokenizers::Tokenizer;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;

/// Default image size for CLIP ViT-B/32.
#[cfg(feature = "embeddings-rten-multimodal")]
const DEFAULT_IMAGE_SIZE: usize = 224;

/// Default embedding dimension for CLIP ViT-B/32.
#[cfg(feature = "embeddings-rten-multimodal")]
const DEFAULT_DIM: usize = 512;

/// RTEN-based CLIP embedder using ONNX models.
///
/// This embedder uses the RTEN runtime to run CLIP models locally, providing
/// embeddings for both text and images in the same vector space. This enables
/// cross-modal similarity search (e.g. text-to-image search).
///
/// # Requirements
///
/// - A CLIP ONNX model file exported via `optimum-cli export onnx`.
///   The model must accept `input_ids`, `pixel_values`, and `attention_mask`
///   inputs and expose `text_embeds` and `image_embeds` outputs.
/// - A `tokenizer.json` in the same directory as the model file.
///
/// # Supported Models
///
/// Any HuggingFace CLIP model exported to ONNX, for example:
/// - `openai/clip-vit-base-patch32`
/// - `onnx-community/clip-vit-base-patch32-ONNX`
///
/// # Examples
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::rten_clip_embedder::RtenClipEmbedder;
///
/// # async fn example() -> laurus::Result<()> {
/// let embedder = RtenClipEmbedder::new_from_hf("onnx-community/clip-vit-base-patch32-ONNX")?;
///
/// let text_vec = embedder.embed(&EmbedInput::Text("a photo of a cat")).await?;
///
/// let img_bytes = std::fs::read("cat.jpg")?;
/// let img_vec = embedder.embed(&EmbedInput::Bytes(&img_bytes, None)).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-rten-multimodal")]
pub struct RtenClipEmbedder {
    /// The RTEN CLIP model.
    model: Model,
    /// HuggingFace tokenizer for text input.
    tokenizer: Tokenizer,
    /// Dimension of the shared embedding space.
    dim: usize,
    /// Expected image size (width = height in pixels).
    image_size: usize,
    /// Path to the ONNX model file (or HF model ID).
    model_path: String,
}

#[cfg(feature = "embeddings-rten-multimodal")]
impl std::fmt::Debug for RtenClipEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtenClipEmbedder")
            .field("model_path", &self.model_path)
            .field("dimension", &self.dim)
            .field("image_size", &self.image_size)
            .finish()
    }
}

#[cfg(feature = "embeddings-rten-multimodal")]
impl RtenClipEmbedder {
    /// Return the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Create a new RTEN-based CLIP embedder from a local ONNX model file.
    ///
    /// The tokenizer is loaded from `tokenizer.json` in the same directory as
    /// the model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded, or if the
    /// required input/output nodes are missing.
    pub fn new(model_path: &str) -> Result<Self> {
        let model = Model::load_file(model_path)
            .map_err(|e| LaurusError::InvalidOperation(format!("RTEN model load failed: {}", e)))?;

        let tokenizer_path = Path::new(model_path)
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .filter(|p| p.exists())
            .ok_or_else(|| {
                LaurusError::InvalidOperation(
                    "tokenizer.json not found next to the ONNX model".to_string(),
                )
            })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenizer load failed: {}", e)))?;

        Self::validate_nodes(&model)?;

        Ok(Self {
            model,
            tokenizer,
            dim: DEFAULT_DIM,
            image_size: DEFAULT_IMAGE_SIZE,
            model_path: model_path.to_string(),
        })
    }

    /// Create a new RTEN-based CLIP embedder from a HuggingFace model ID.
    ///
    /// The model files are downloaded from HuggingFace Hub and cached locally.
    /// Both `onnx/model.onnx` and `model.onnx` are tried.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model identifier
    ///   (e.g. `"onnx-community/clip-vit-base-patch32-ONNX"`).
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be resolved, downloaded, or loaded.
    pub fn new_from_hf(model_id: &str) -> Result<Self> {
        let api = ApiBuilder::new()
            .build()
            .map_err(|e| LaurusError::InvalidOperation(format!("HF API initialization failed: {}", e)))?;
        let repo = api.model(model_id.to_string());

        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| {
                LaurusError::InvalidOperation(format!(
                    "ONNX model download failed for '{}': {}",
                    model_id, e
                ))
            })?;

        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            LaurusError::InvalidOperation(format!("Tokenizer download failed: {}", e))
        })?;

        let model_dir = model_path
            .parent()
            .ok_or_else(|| {
                LaurusError::InvalidOperation("Downloaded model has no parent directory".to_string())
            })?;

        std::fs::copy(&tokenizer_path, model_dir.join("tokenizer.json")).map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "Failed to copy tokenizer next to model: {}",
                e
            ))
        })?;

        Self::new(model_path.to_str().unwrap_or(model_id))
    }

    /// Verify that all required input and output nodes exist in the model.
    fn validate_nodes(model: &Model) -> Result<()> {
        for name in &["input_ids", "pixel_values", "attention_mask"] {
            Self::node_id(model, name)?;
        }
        for name in &["text_embeds", "image_embeds"] {
            Self::node_id(model, name)?;
        }
        Ok(())
    }

    fn node_id(model: &Model, name: &str) -> Result<NodeId> {
        model.node_id(name).map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "Node '{}' not found in RTEN CLIP model: {}",
                name, e
            ))
        })
    }

    /// Build a 2-D RTEN tensor from a flat vector.
    fn tensor_2d_i32(data: Vec<i32>, rows: usize, cols: usize) -> Tensor<i32> {
        Tensor::from_vec(data).into_shape([rows, cols]).into()
    }

    /// Tokenize text for CLIP, returning token IDs and attention mask.
    fn tokenize(&self, text: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;
        let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let attention_mask: Vec<i32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i32)
            .collect();
        Ok((token_ids, attention_mask))
    }

    /// Embed text synchronously.
    fn embed_text_sync(&self, text: &str) -> Result<Vector> {
        let (token_ids, attention_mask) = self.tokenize(text)?;
        let seq_length = token_ids.len();

        let input_ids = Self::tensor_2d_i32(token_ids, 1, seq_length);
        let attention_mask_tensor = Self::tensor_2d_i32(attention_mask, 1, seq_length);

        // Dummy pixel values (unused for text embedding but required by the
        // combined CLIP model).
        let pixel_values: Tensor<f32> =
            Tensor::zeros(&[1, 3, self.image_size, self.image_size]);

        let input_ids_id = Self::node_id(&self.model, "input_ids")?;
        let pixel_values_id = Self::node_id(&self.model, "pixel_values")?;
        let attention_mask_id = Self::node_id(&self.model, "attention_mask")?;
        let text_embeds_id = Self::node_id(&self.model, "text_embeds")?;

        let inputs = vec![
            (input_ids_id, ValueOrView::from(input_ids)),
            (pixel_values_id, ValueOrView::from(pixel_values)),
            (attention_mask_id, ValueOrView::from(attention_mask_tensor)),
        ];

        let [output] = self
            .model
            .run_n(inputs, [text_embeds_id], None)
            .map_err(|e| LaurusError::InvalidOperation(format!("CLIP text forward failed: {}", e)))?;

        let embed_tensor: Tensor<f32> = output.try_into().map_err(|e| {
            LaurusError::InvalidOperation(format!("Output conversion failed: {:?}", e))
        })?;

        // text_embeds shape: [1, dim]
        let data = embed_tensor.to_vec();
        let mut vec = data;
        Self::l2_normalize(&mut vec);

        Ok(Vector::new(vec))
    }

    /// Embed image bytes synchronously.
    fn embed_image_bytes_sync(&self, bytes: &[u8]) -> Result<Vector> {
        let pixel_values = self.preprocess_image_bytes(bytes)?;

        // Dummy text inputs (unused for image embedding but required by the
        // combined CLIP model).
        let dummy_seq = 2usize;
        let input_ids = Self::tensor_2d_i32(vec![0i32; dummy_seq], 1, dummy_seq);
        let attention_mask = Self::tensor_2d_i32(vec![0i32; dummy_seq], 1, dummy_seq);

        let input_ids_id = Self::node_id(&self.model, "input_ids")?;
        let pixel_values_id = Self::node_id(&self.model, "pixel_values")?;
        let attention_mask_id = Self::node_id(&self.model, "attention_mask")?;
        let image_embeds_id = Self::node_id(&self.model, "image_embeds")?;

        let inputs = vec![
            (input_ids_id, ValueOrView::from(input_ids)),
            (pixel_values_id, ValueOrView::from(pixel_values)),
            (attention_mask_id, ValueOrView::from(attention_mask)),
        ];

        let [output] = self
            .model
            .run_n(inputs, [image_embeds_id], None)
            .map_err(|e| LaurusError::InvalidOperation(format!("CLIP image forward failed: {}", e)))?;

        let embed_tensor: Tensor<f32> = output.try_into().map_err(|e| {
            LaurusError::InvalidOperation(format!("Output conversion failed: {:?}", e))
        })?;

        // image_embeds shape: [1, dim]
        let data = embed_tensor.to_vec();
        let mut vec = data;
        Self::l2_normalize(&mut vec);

        Ok(Vector::new(vec))
    }

    /// Preprocess image bytes into the normalized CHW float tensor expected
    /// by CLIP.
    ///
    /// Steps (matching the CLIP `preprocessor_config.json`):
    /// 1. Resize the shortest side to `image_size`, preserving aspect ratio.
    /// 2. Center-crop to `(image_size, image_size)`.
    /// 3. Scale pixels to `[0, 1]` and normalize with CLIP mean/std.
    fn preprocess_image_bytes(&self, bytes: &[u8]) -> Result<Tensor<f32>> {
        use image::ImageReader;
        use std::io::Cursor;

        let img = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| {
                LaurusError::InvalidOperation(format!("Image format guess failed: {}", e))
            })?
            .decode()
            .map_err(|e| LaurusError::InvalidOperation(format!("Image decode failed: {}", e)))?;

        let (input_w, input_h) = (img.width(), img.height());
        let target = self.image_size as u32;

        // Resize shortest side to target, preserving aspect ratio.
        let (resized_w, resized_h) = if input_w < input_h {
            let scale = target as f32 / input_w as f32;
            (target, (input_h as f32 * scale) as u32)
        } else {
            let scale = target as f32 / input_h as f32;
            ((input_w as f32 * scale) as u32, target)
        };

        let img = img.resize_exact(
            resized_w,
            resized_h,
            image::imageops::FilterType::Triangle,
        );

        // Center crop.
        let crop_top = resized_h.saturating_sub(target) / 2;
        let crop_left = resized_w.saturating_sub(target) / 2;
        let img = img.crop_imm(crop_left, crop_top, target, target);

        // Convert to RGB.
        let rgb = match img {
            image::DynamicImage::ImageRgb8(img) => img,
            img => img.to_rgb8(),
        };

        // Normalize: (pixel / 255.0 - mean) / std
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std_dev = [0.26862954f32, 0.2613026, 0.2757771];

        let size = self.image_size;
        let mut flat = vec![0.0f32; 3 * size * size];

        for y in 0..size as u32 {
            for x in 0..size as u32 {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    let normalized = (val - mean[c]) / std_dev[c];
                    let chw_idx = c * size + (y as usize) * size + (x as usize);
                    flat[chw_idx] = normalized;
                }
            }
        }

        // Shape [1, 3, H, W] for batch dimension.
        let tensor = Tensor::from_vec(flat)
            .into_shape([1, 3, size, size])
            .into();
        Ok(tensor)
    }

    /// L2-normalize a vector in place.
    fn l2_normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vector {
                *v /= norm;
            }
        }
    }
}

#[cfg(feature = "embeddings-rten-multimodal")]
#[async_trait]
impl Embedder for RtenClipEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => {
                let text = text.to_string();
                tokio::task::block_in_place(|| self.embed_text_sync(&text))
            }
            EmbedInput::Bytes(bytes, mime) => {
                if let Some(mime_type) = mime
                    && mime_type.starts_with("text/")
                {
                    let text = std::str::from_utf8(bytes).map_err(|e| {
                        LaurusError::invalid_argument(format!("invalid utf-8 for text bytes: {}", e))
                    })?;
                    let text = text.to_string();
                    return tokio::task::block_in_place(|| self.embed_text_sync(&text));
                }
                let bytes = bytes.to_vec();
                tokio::task::block_in_place(|| self.embed_image_bytes_sync(&bytes))
            }
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text, EmbedInputType::Image]
    }

    fn name(&self) -> &str {
        &self.model_path
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(all(test, feature = "embeddings-rten-multimodal"))]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0f32, 4.0];
        RtenClipEmbedder::l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
