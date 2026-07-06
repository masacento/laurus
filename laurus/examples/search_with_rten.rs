//! Japanese vector search example using the RTEN runtime and the Ruri model.
//!
//! This binary downloads `sirasagi62/ruri-v3-30m-ONNX` from HuggingFace Hub on
//! first run, then indexes a small Japanese corpus and performs semantic search.
//!
//! Run with:
//!
//! ```bash
//! cargo run --bin ruri_vector_search --features embeddings-rten
//! ```
//!
//! To use an already-downloaded model directory:
//!
//! ```bash
//! cargo run --bin ruri_vector_search --features embeddings-rten -- --model-dir /path/to/model_dir
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use laurus::lexical::core::field::TextOption;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::{FlatOption, VectorQueryParser};
use laurus::{
    DataValue, Document, Engine, PerFieldEmbedder, QueryPayload, Result, RtenBertEmbedder, Schema,
    SearchRequestBuilder, SearchResult, VectorSearchQuery,
};

#[derive(Parser, Debug)]
#[command(
    name = "ruri_vector_search",
    about = "Japanese vector search with Ruri + RTEN"
)]
struct Args {
    /// HuggingFace model ID to download.
    #[arg(long, default_value = "sirasagi62/ruri-v3-30m-ONNX")]
    model: String,

    /// Use a local model directory instead of downloading.
    /// The directory must contain `model.onnx` and `tokenizer.json`.
    #[arg(long, value_name = "DIR")]
    model_dir: Option<PathBuf>,

    /// Number of results to return.
    #[arg(short, long, default_value_t = 3)]
    limit: usize,
}

fn memory_storage() -> Result<Arc<dyn Storage>> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))
}

fn print_results(results: &[SearchResult]) {
    if results.is_empty() {
        println!("  (結果なし)");
        return;
    }
    for (i, hit) in results.iter().enumerate() {
        println!("  {}. score: {:.4}", i + 1, hit.score);
        if let Some(doc) = &hit.document {
            if let Some(DataValue::Text(title)) = doc.fields.get("title") {
                println!("     title: {}", title);
            }
            if let Some(DataValue::Text(text)) = doc.fields.get("text") {
                let preview = if text.chars().count() > 80 {
                    format!("{}...", text.chars().take(80).collect::<String>())
                } else {
                    text.clone()
                };
                println!("     text: {}", preview);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading RTEN embedder ...");
    // This ONNX export already provides a pooled `sentence_embedding` output.
    let embedder = match args.model_dir {
        Some(dir) => {
            let model_path = dir.join("model.onnx");
            if !model_path.exists() {
                eprintln!("model.onnx not found in {}", dir.display());
                std::process::exit(1);
            }
            RtenBertEmbedder::new_with_output_node(
                model_path.to_str().unwrap(),
                "sentence_embedding",
            )?
        }
        None => RtenBertEmbedder::new_from_hf_with_output_node(&args.model, "sentence_embedding")?,
    };
    let dimension = embedder.dimension();
    println!("Embedder loaded: {} dimensions\n", dimension);

    let embedder: Arc<dyn laurus::Embedder> = Arc::new(embedder);
    let per_field_embedder = PerFieldEmbedder::new(embedder.clone());
    per_field_embedder.add_embedder("text_vec", embedder.clone());
    let per_field_embedder = Arc::new(per_field_embedder);

    let storage = memory_storage()?;
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default().indexed(false).stored(true))
        .add_text_field("text", TextOption::default().indexed(false).stored(true))
        .add_text_field(
            "category",
            TextOption::default().indexed(false).stored(true),
        )
        .add_flat_field("text_vec", FlatOption::default().dimension(dimension))
        .build();

    let engine = Engine::builder(storage, schema)
        .embedder(per_field_embedder.clone())
        .build()
        .await?;

    // ─── 日本語コーパスをインデックス ─────────────────────────────────────
    println!("--- Indexing Japanese documents ---\n");

    let docs = vec![
        (
            "doc1",
            "寿司について",
            "寿司は日本の伝統的な食文化です。酢飯に新鮮な魚介を乗せて食べます。",
            "food",
        ),
        (
            "doc2",
            "ラーメンの歴史",
            "ラーメンは中華麺をベースにしたスープ料理で、日本全国に多様な流派があります。",
            "food",
        ),
        (
            "doc3",
            "東京観光ガイド",
            "東京は日本の首都で、浅草や渋谷、新宿など多くの観光地があります。",
            "travel",
        ),
        (
            "doc4",
            "京都の寺院",
            "京都には金閣寺や清水寺など、美しい寺院が数多く残されています。",
            "travel",
        ),
        (
            "doc5",
            "Rustの所有権",
            "Rustの所有権システムはコンパイル時にメモリ安全性を保証する革新的な仕組みです。",
            "programming",
        ),
        (
            "doc6",
            "非同期プログラミング",
            "async/awaitを使うと、少ないリソースで多くの並行処理を効率的に記述できます。",
            "programming",
        ),
    ];

    for (id, title, text, category) in docs {
        let doc = Document::builder()
            .add_text("title", title)
            .add_text("text", text)
            .add_text("category", category)
            .add_text("text_vec", text)
            .build();
        engine.add_document(id, doc).await?;
    }
    engine.commit().await?;
    println!("Indexed {} documents.\n", 6);

    // ─── ベクトル検索 ────────────────────────────────────────────────────
    let queries = vec!["日本の食べ物", "観光スポット", "プログラミング言語"];

    for query in queries {
        println!("{}", "=".repeat(60));
        println!("Query: '{}'", query);
        println!("{}", "=".repeat(60));

        let results = engine
            .search(
                SearchRequestBuilder::new()
                    .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload::new(
                        "text_vec",
                        DataValue::Text(query.into()),
                    )]))
                    .limit(args.limit)
                    .build(),
            )
            .await?;
        print_results(&results);
        println!();
    }

    // ─── DSL ベクトル検索 ────────────────────────────────────────────────
    println!("{}", "=".repeat(60));
    println!("DSL Query: text_vec:\"寿司\"");
    println!("{}", "=".repeat(60));
    let vector_parser = VectorQueryParser::new(per_field_embedder.clone());
    let parsed = vector_parser.parse("text_vec:\"寿司\"").await?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(parsed.query)
                .limit(args.limit)
                .build(),
        )
        .await?;
    print_results(&results);
    println!();

    println!("\nRuri vector search example completed successfully!");
    Ok(())
}
