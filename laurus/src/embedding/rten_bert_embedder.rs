//! RTEN-based BERT embedder implementation.
//!
//! This module provides a text embedder using the RTEN inference runtime.
//! It can load ONNX models directly, so no `.rten` conversion is required.
//! Requires the `embeddings-rten` feature to be enabled.
//!
//! The embedder follows the same pattern as [`CandleBertEmbedder`](super::candle_bert_embedder):
//! tokenize the input, run a BERT-style model, mean-pool the token embeddings
//! (when the model outputs `last_hidden_state`), and L2-normalize the result.

#[cfg(feature = "embeddings-rten")]
use std::any::Any;
#[cfg(feature = "embeddings-rten")]
use std::path::Path;

#[cfg(feature = "embeddings-rten")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-rten")]
use rten::{Model, NodeId, ValueOrView};
#[cfg(feature = "embeddings-rten")]
use rten_tensor::Tensor;
#[cfg(feature = "embeddings-rten")]
use rten_tensor::prelude::*;
#[cfg(feature = "embeddings-rten")]
use tokenizers::Tokenizer;
#[cfg(feature = "embeddings-rten")]
use hf_hub::api::sync::ApiBuilder;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;

/// Default output node for BERT-style ONNX models exported with HuggingFace
/// Optimum. RTEN embedders can override this via the `output_node` field if
/// they expose a pre-pooled `sentence_embedding` output.
#[cfg(feature = "embeddings-rten")]
const DEFAULT_OUTPUT_NODE: &str = "last_hidden_state";

/// Pooling strategy applied to a token-level model output
/// (`last_hidden_state`, shape `[batch, seq, dim]`).
///
/// For pre-pooled 2-D outputs (`sentence_embedding`) the value is ignored.
#[cfg(feature = "embeddings-rten")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// Arithmetic mean over non-padded tokens.
    Mean,
    /// First-token ([CLS]) hidden state.
    Cls,
}

#[cfg(feature = "embeddings-rten")]
impl Default for Pooling {
    fn default() -> Self {
        Pooling::Mean
    }
}

/// RTEN-based BERT embedder using ONNX models.
///
/// This embedder uses the RTEN runtime to run BERT-style models locally.
/// Models must be provided as ONNX files (`.onnx`); RTEN can load them
/// directly without a separate conversion step.
///
/// # Requirements
///
/// - An ONNX model file (e.g. exported via `optimum-cli export onnx`).
/// - A `tokenizer.json` in the same directory as the model file.
///
/// # Features
///
/// - Offline inference (no API calls)
/// - Pure Rust dependency tree (no ONNX Runtime C++ library)
/// - CPU inference with SIMD acceleration
///
/// # Examples
///
/// ```no_run
/// use laurus::embedding::embedder::{Embedder, EmbedInput};
/// use laurus::embedding::rten_bert_embedder::RtenBertEmbedder;
///
/// # async fn example() -> laurus::Result<()> {
/// let embedder = RtenBertEmbedder::new("model/model.onnx")?;
/// let vector = embedder.embed(&EmbedInput::Text("Rust is awesome!")).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "embeddings-rten")]
pub struct RtenBertEmbedder {
    /// The RTEN model for generating embeddings.
    model: Model,
    /// Tokenizer for converting text to token IDs.
    tokenizer: Tokenizer,
    /// Dimension of the output embeddings.
    dim: usize,
    /// Path to the ONNX model file.
    model_path: String,
    /// Name of the output node to read.
    output_node: String,
    /// Pooling strategy for token-level outputs.
    pooling: Pooling,
}

#[cfg(feature = "embeddings-rten")]
impl std::fmt::Debug for RtenBertEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtenBertEmbedder")
            .field("model_path", &self.model_path)
            .field("output_node", &self.output_node)
            .field("pooling", &self.pooling)
            .field("dimension", &self.dim)
            .finish()
    }
}

#[cfg(feature = "embeddings-rten")]
impl RtenBertEmbedder {
    /// Return the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Create a new RTEN-based BERT embedder from an ONNX model file.
    ///
    /// The tokenizer is loaded from `tokenizer.json` in the same directory as
    /// the model. The default output node is `last_hidden_state`; if the model
    /// exposes a pre-pooled `sentence_embedding` output, use
    /// [`RtenBertEmbedder::new_with_output_node`] instead.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model file cannot be loaded.
    /// - `tokenizer.json` is not found next to the model.
    /// - The tokenizer cannot be loaded.
    /// - The configured output node is not present in the model.
    pub fn new(model_path: &str) -> Result<Self> {
        Self::new_with_output_node(model_path, DEFAULT_OUTPUT_NODE)
    }

    /// Create a new embedder with an explicit output node name.
    ///
    /// Use this constructor when the ONNX model already contains a pooling
    /// layer and exposes a node such as `sentence_embedding`.
    pub fn new_with_output_node(model_path: &str, output_node: &str) -> Result<Self> {
        Self::new_with_output_node_and_pooling(model_path, output_node, Pooling::default())
    }

    /// Create a new embedder with an explicit output node name and pooling
    /// strategy.
    ///
    /// `pooling` only affects token-level (`last_hidden_state`) outputs; for
    /// pre-pooled 2-D outputs it is ignored. Use [`Pooling::Cls`] for models
    /// trained with CLS pooling (e.g. IBM Granite embeddings).
    pub fn new_with_output_node_and_pooling(
        model_path: &str,
        output_node: &str,
        pooling: Pooling,
    ) -> Result<Self> {
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

        // Verify the output node exists up front.
        let _ = Self::output_node_id(&model, output_node)?;

        let dim = Self::infer_dim(&model, &tokenizer, output_node)?;

        Ok(Self {
            model,
            tokenizer,
            dim,
            model_path: model_path.to_string(),
            output_node: output_node.to_string(),
            pooling,
        })
    }

    /// Create a new RTEN-based BERT embedder from a HuggingFace model ID.
    ///
    /// The model files are downloaded from HuggingFace Hub and cached locally.
    /// Both `onnx/model.onnx` and `model.onnx` are tried, and `tokenizer.json`
    /// is copied next to the model so it can be loaded automatically.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model identifier (e.g., `"sirasagi62/ruri-v3-30m-ONNX"`).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model ID cannot be resolved or downloaded.
    /// - Neither `onnx/model.onnx` nor `model.onnx` exists in the repo.
    /// - `tokenizer.json` cannot be downloaded.
    pub fn new_from_hf(model_id: &str) -> Result<Self> {
        Self::new_from_hf_with_output_node(model_id, DEFAULT_OUTPUT_NODE)
    }

    /// Create a new embedder from a HuggingFace model ID with an explicit
    /// output node name.
    pub fn new_from_hf_with_output_node(model_id: &str, output_node: &str) -> Result<Self> {
        Self::new_from_hf_with_output_node_and_pooling(model_id, output_node, Pooling::default())
    }

    /// Create a new embedder from a HuggingFace model ID with an explicit
    /// output node name and pooling strategy.
    pub fn new_from_hf_with_output_node_and_pooling(
        model_id: &str,
        output_node: &str,
        pooling: Pooling,
    ) -> Result<Self> {
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

        Self::new_with_output_node_and_pooling(
            model_path.to_str().unwrap_or(model_id),
            output_node,
            pooling,
        )
    }

    fn output_node_id(model: &Model, name: &str) -> Result<NodeId> {
        model.node_id(name).map_err(|e| {
            LaurusError::InvalidOperation(format!(
                "Output node '{}' not found in RTEN model: {}",
                name, e
            ))
        })
    }

    /// Build a 2-D RTEN tensor from a flat vector.
    ///
    /// The caller must ensure `data.len() == rows * cols`.
    fn tensor_2d_i32(data: Vec<i32>, rows: usize, cols: usize) -> Tensor<i32> {
        Tensor::from_vec(data).into_shape([rows, cols]).into()
    }

    /// Infer the embedding dimension by running a dummy forward pass.
    fn infer_dim(model: &Model, tokenizer: &Tokenizer, output_node: &str) -> Result<usize> {
        let encoding = tokenizer
            .encode("", true)
            .map_err(|e| LaurusError::InvalidOperation(format!("Tokenization failed: {}", e)))?;
        let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let seq_length = token_ids.len();

        let input_ids = Self::tensor_2d_i32(token_ids, 1, seq_length);
        let attention_mask = Self::tensor_2d_i32(vec![1i32; seq_length], 1, seq_length);

        let input_ids_id = model.node_id("input_ids").map_err(|e| {
            LaurusError::InvalidOperation(format!("input_ids node not found: {}", e))
        })?;
        let attention_mask_id = model.node_id("attention_mask").map_err(|e| {
            LaurusError::InvalidOperation(format!("attention_mask node not found: {}", e))
        })?;

        let mut inputs = vec![
            (input_ids_id, ValueOrView::from(input_ids)),
            (attention_mask_id, ValueOrView::from(attention_mask)),
        ];

        if let Some(token_type_ids_id) = model.find_node("token_type_ids") {
            let token_type_ids = Self::tensor_2d_i32(vec![0i32; seq_length], 1, seq_length);
            inputs.push((token_type_ids_id, ValueOrView::from(token_type_ids)));
        }

        let output_id = Self::output_node_id(model, output_node)?;
        let [output] = model
            .run_n(inputs, [output_id], None)
            .map_err(|e| LaurusError::InvalidOperation(format!("Model forward failed: {}", e)))?;

        let tensor: Tensor<f32> = output.try_into().map_err(|e| {
            LaurusError::InvalidOperation(format!("Output conversion failed: {:?}", e))
        })?;
        let shape = tensor.shape();

        match shape.len() {
            2 => Ok(shape[1]),
            3 => Ok(shape[2]),
            _ => Err(LaurusError::InvalidOperation(format!(
                "Unexpected RTEN output shape {:?}",
                shape
            ))),
        }
    }

    /// Embed text directly (internal implementation).
    async fn embed_text(&self, text: &str) -> Result<Vector> {
        let text = text.to_string();
        tokio::task::block_in_place(|| self.embed_text_sync(&text))
    }

    /// Synchronous embedding implementation for a single text.
    fn embed_text_sync(&self, text: &str) -> Result<Vector> {
        let (token_ids, attention_mask) = self.tokenize(text)?;
        let seq_length = token_ids.len();

        let input_ids = Self::tensor_2d_i32(token_ids, 1, seq_length);
        let attention_mask_tensor = Self::tensor_2d_i32(attention_mask.clone(), 1, seq_length);

        let input_ids_id = self.model.node_id("input_ids").map_err(|e| {
            LaurusError::InvalidOperation(format!("input_ids node not found: {}", e))
        })?;
        let attention_mask_id = self.model.node_id("attention_mask").map_err(|e| {
            LaurusError::InvalidOperation(format!("attention_mask node not found: {}", e))
        })?;

        let mut inputs = vec![
            (input_ids_id, ValueOrView::from(input_ids)),
            (attention_mask_id, ValueOrView::from(attention_mask_tensor)),
        ];

        if let Some(token_type_ids_id) = self.model.find_node("token_type_ids") {
            let token_type_ids = Self::tensor_2d_i32(vec![0i32; seq_length], 1, seq_length);
            inputs.push((token_type_ids_id, ValueOrView::from(token_type_ids)));
        }

        let output_id = self
            .model
            .node_id(&self.output_node)
            .map_err(|e| LaurusError::InvalidOperation(format!("output node not found: {}", e)))?;
        let [output] = self
            .model
            .run_n(inputs, [output_id], None)
            .map_err(|e| LaurusError::InvalidOperation(format!("Model forward failed: {}", e)))?;

        let output_tensor: Tensor<f32> = output.try_into().map_err(|e| {
            LaurusError::InvalidOperation(format!("Output conversion failed: {:?}", e))
        })?;

        let shape = output_tensor.shape().to_vec();
        let data = output_tensor.to_vec();

        let mut pooled = self.pool(&data, &shape, &attention_mask)?;
        Self::l2_normalize(&mut pooled);

        Ok(Vector::new(pooled))
    }

    /// Embed a batch of texts in a single forward pass.
    fn embed_batch_sync(&self, texts: &[String]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut token_ids_batch: Vec<Vec<i32>> = Vec::with_capacity(texts.len());
        let mut masks_batch: Vec<Vec<i32>> = Vec::with_capacity(texts.len());
        let mut max_length = 0;

        for text in texts {
            let (token_ids, attention_mask) = self.tokenize(text)?;
            max_length = max_length.max(token_ids.len());
            token_ids_batch.push(token_ids);
            masks_batch.push(attention_mask);
        }

        let batch_size = texts.len();
        let mut ids_array = Vec::with_capacity(batch_size * max_length);
        let mut mask_array = Vec::with_capacity(batch_size * max_length);
        let mut type_array = Vec::with_capacity(batch_size * max_length);

        for (token_ids, mask) in token_ids_batch.iter().zip(&masks_batch) {
            ids_array.extend(token_ids.iter().cloned());
            ids_array.extend(std::iter::repeat(0i32).take(max_length - token_ids.len()));
            mask_array.extend(mask.iter().cloned());
            mask_array.extend(std::iter::repeat(0i32).take(max_length - mask.len()));
            type_array.extend(std::iter::repeat(0i32).take(max_length));
        }

        let input_ids = Self::tensor_2d_i32(ids_array, batch_size, max_length);
        let attention_mask_tensor = Self::tensor_2d_i32(mask_array, batch_size, max_length);

        let input_ids_id = self.model.node_id("input_ids").map_err(|e| {
            LaurusError::InvalidOperation(format!("input_ids node not found: {}", e))
        })?;
        let attention_mask_id = self.model.node_id("attention_mask").map_err(|e| {
            LaurusError::InvalidOperation(format!("attention_mask node not found: {}", e))
        })?;

        let mut inputs = vec![
            (input_ids_id, ValueOrView::from(input_ids)),
            (attention_mask_id, ValueOrView::from(attention_mask_tensor)),
        ];

        if let Some(token_type_ids_id) = self.model.find_node("token_type_ids") {
            let token_type_ids = Self::tensor_2d_i32(type_array, batch_size, max_length);
            inputs.push((token_type_ids_id, ValueOrView::from(token_type_ids)));
        }

        let output_id = self
            .model
            .node_id(&self.output_node)
            .map_err(|e| LaurusError::InvalidOperation(format!("output node not found: {}", e)))?;
        let [output] = self
            .model
            .run_n(inputs, [output_id], None)
            .map_err(|e| LaurusError::InvalidOperation(format!("Model forward failed: {}", e)))?;

        let output_tensor: Tensor<f32> = output.try_into().map_err(|e| {
            LaurusError::InvalidOperation(format!("Output conversion failed: {:?}", e))
        })?;

        let shape = output_tensor.shape().to_vec();
        let data = output_tensor.to_vec();

        let mut vectors = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let row_mask = &masks_batch[i];
            let mut pooled = self.pool_for_item(&data, &shape, i, row_mask)?;
            Self::l2_normalize(&mut pooled);
            vectors.push(Vector::new(pooled));
        }

        Ok(vectors)
    }

    /// Tokenize a single text and return token IDs plus the attention mask.
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

    /// Pool a single-item output tensor into a fixed-size embedding vector.
    fn pool(&self, data: &[f32], shape: &[usize], attention_mask: &[i32]) -> Result<Vec<f32>> {
        self.pool_for_item(data, shape, 0, attention_mask)
    }

    /// Pool the output for the `item_index`-th element of a batch.
    fn pool_for_item(
        &self,
        data: &[f32],
        shape: &[usize],
        item_index: usize,
        attention_mask: &[i32],
    ) -> Result<Vec<f32>> {
        match shape.len() {
            // Pre-pooled output: [batch, dim] or [dim].
            2 => {
                let dim = shape[1];
                let start = item_index * dim;
                let end = start + dim;
                if end > data.len() {
                    return Err(LaurusError::InvalidOperation(
                        "RTEN output data shorter than expected".to_string(),
                    ));
                }
                Ok(data[start..end].to_vec())
            }
            // Token-level output: [batch, seq, dim].
            3 => {
                let max_seq = shape[1];
                let dim = shape[2];

                match self.pooling {
                    Pooling::Mean => {
                        let mut pooled = vec![0.0f32; dim];
                        let mask_sum: f32 = attention_mask.iter().map(|&m| m as f32).sum();

                        if mask_sum > 0.0 {
                            for (token_idx, &mask_val) in attention_mask.iter().enumerate() {
                                if mask_val == 0 {
                                    continue;
                                }
                                let src_offset = (item_index * max_seq + token_idx) * dim;
                                if src_offset + dim > data.len() {
                                    return Err(LaurusError::InvalidOperation(
                                        "RTEN output data shorter than expected".to_string(),
                                    ));
                                }
                                for d in 0..dim {
                                    pooled[d] += data[src_offset + d];
                                }
                            }
                            for v in &mut pooled {
                                *v /= mask_sum;
                            }
                        }
                        Ok(pooled)
                    }
                    Pooling::Cls => {
                        // First token of the sequence ([CLS]).
                        let src_offset = item_index * max_seq * dim;
                        if src_offset + dim > data.len() {
                            return Err(LaurusError::InvalidOperation(
                                "RTEN output data shorter than expected".to_string(),
                            ));
                        }
                        Ok(data[src_offset..src_offset + dim].to_vec())
                    }
                }
            }
            _ => Err(LaurusError::InvalidOperation(format!(
                "Unexpected RTEN output shape {:?}",
                shape
            ))),
        }
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

#[cfg(feature = "embeddings-rten")]
#[async_trait]
impl Embedder for RtenBertEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => self.embed_text(text).await,
            _ => Err(LaurusError::invalid_argument(
                "RtenBertEmbedder only supports text input",
            )),
        }
    }

    async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
        let texts: Vec<String> = inputs
            .iter()
            .map(|input| match input {
                EmbedInput::Text(text) => Ok(text.to_string()),
                _ => Err(LaurusError::invalid_argument(
                    "RtenBertEmbedder only supports text input",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        tokio::task::block_in_place(|| self.embed_batch_sync(&texts))
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        &self.model_path
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(all(test, feature = "embeddings-rten"))]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0f32, 4.0];
        RtenBertEmbedder::l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
