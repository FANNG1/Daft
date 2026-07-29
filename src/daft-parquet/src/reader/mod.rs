mod chunk_source;
mod field_reader;
mod rg_processor;
mod util;

// Parquet read flow (high level):
//
// 1. `stream_parquet` opens the source and reads footer metadata.
// 2. `resolve_column_plan` separates projected columns, predicate-only columns,
//    and the schema that will be visible to the caller.
// 3. `prune_row_groups` eliminates RGs from statistics and user constraints;
//    `build_base_selections` then expresses offset/deletes/limit within each
//    surviving RG.
// 4. Each RG is decoded as a stream of column-aligned batches. The local path
//    uses the established `ChunkSource` implementation. The remote path uses
//    the PR1 owned-RG pipeline below.
// 5. `apply_cross_rg_limit` is the final owner of a global result limit, so
//    concurrent RG work cannot race while deciding which rows to return.

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use arrow::{array::ArrayRef, datatypes::Schema as ArrowSchema};
use chunk_source::{
    ChunkSource, ChunkSourceBuilder, LocalChunkSource, RgAccess, open_local_file,
    prepare_remote_chunk_source,
};
use common_error::DaftResult;
use common_runtime::{JoinSet, get_compute_runtime};
use daft_core::prelude::*;
use daft_dsl::{ExprRef, expr::bound_expr::BoundExpr, optimization::get_required_columns};
use daft_recordbatch::RecordBatch;
use futures::{Stream, StreamExt, stream::BoxStream};
use parquet::{
    arrow::arrow_reader::{ArrowReaderMetadata, RowSelection, RowSelector},
    file::metadata::ParquetMetaData,
};
use rg_processor::{
    process_rg_predicate_only, process_rg_with_data_cols, recv_one_chunk, spawn_col_decoders,
};
use snafu::ResultExt;
use util::{
    cap_selection_to, eval_predicate_mask, filter_arrays_by_mask, project_schema,
    record_batch_from_arrow, schema_from_indices, truncate_mask_to_n_trues,
};

use crate::{
    ArrowSnafu,
    helpers::{
        bool_array_to_row_selection, build_offset_row_selection, build_single_rg_delete_selection,
        combine_selections, predicate_pushable_cols, prune_row_groups, refine_selection,
        substitute_missing_cols,
    },
    metadata::{
        apply_field_ids_to_arrowrs_parquet_metadata, strip_string_types_from_parquet_metadata,
    },
    read::{ParquetReadOptions, ParquetSchemaInferenceOptions, StringEncoding},
};

pub enum ParquetSource<'a> {
    Local {
        path: &'a str,
    },
    Url {
        uri: &'a str,
        io_client: Arc<daft_io::IOClient>,
        io_stats: Option<daft_io::IOStatsRef>,
    },
}

impl ParquetSource<'_> {
    fn label(&self) -> &str {
        match self {
            Self::Local { path } => path,
            Self::Url { uri, .. } => uri,
        }
    }
}

const DEFAULT_BATCH_SIZE: usize = 128 * 1024;

async fn open_chunk_source(
    source: &ParquetSource<'_>,
    opts: &ParquetReadOptions,
) -> crate::Result<(ChunkSourceBuilder, ArrowReaderMetadata)> {
    match source {
        ParquetSource::Local { path } => {
            let (file, file_len, arrow_metadata) = open_local_file(path).await?;
            let cs = LocalChunkSource {
                path: Arc::from(*path),
                file,
                file_len,
                metadata: arrow_metadata.metadata().clone(),
            };
            Ok((ChunkSourceBuilder::Local(cs), arrow_metadata))
        }
        ParquetSource::Url {
            uri,
            io_client,
            io_stats,
        } => {
            Box::pin(prepare_remote_chunk_source(
                uri,
                io_client.clone(),
                io_stats.clone(),
                opts,
            ))
            .await
        }
    }
}

struct PreparedMetadata {
    parquet_metadata: Arc<ParquetMetaData>,
    arrow_schema: Arc<ArrowSchema>,
}

fn prepare_metadata(
    arrow_metadata: ArrowReaderMetadata,
    field_id_mapping: Option<&Arc<BTreeMap<i32, Field>>>,
    opts: ParquetSchemaInferenceOptions,
    path: &str,
) -> crate::Result<PreparedMetadata> {
    let mut parquet_metadata = arrow_metadata.metadata().clone();
    if let Some(mapping) = field_id_mapping {
        parquet_metadata =
            apply_field_ids_to_arrowrs_parquet_metadata(parquet_metadata, mapping, path)?;
    }
    let raw_encoding = opts.string_encoding == StringEncoding::Raw;
    if raw_encoding {
        parquet_metadata = strip_string_types_from_parquet_metadata(parquet_metadata, path)?;
    }
    let arrow_schema = crate::schema_inference::infer_schema_from_parquet_metadata_arrowrs(
        &parquet_metadata,
        Some(opts.coerce_int96_timestamp_unit),
        raw_encoding,
    )
    .with_context(|_| ArrowSnafu {
        path: path.to_string(),
    })?;
    Ok(PreparedMetadata {
        parquet_metadata,
        arrow_schema: Arc::new(arrow_schema),
    })
}

#[derive(Clone)]
pub(super) struct ColumnPlan {
    /// Physical Parquet column indices that must be read, including projection and filter columns.
    pub(super) read_col_indices: Vec<usize>,
    /// Physical Parquet column indices used only for predicate pushdown evaluation.
    pub(super) pred_col_indices: Vec<usize>,
    /// Physical Parquet column indices returned as data after predicate pushdown.
    pub(super) data_col_indices: Vec<usize>,
    /// Whether the predicate can be evaluated while reading Parquet row groups.
    pub(super) predicate_pushed: bool,
    /// Schema visible to callers after applying the requested projection.
    pub(super) return_daft_schema: Arc<Schema>,
    /// Schema for all columns read from Parquet, including predicate helper columns.
    pub(super) read_daft_schema: Schema,
}

fn resolve_column_plan(
    prepared: &PreparedMetadata,
    opts: &ParquetReadOptions,
) -> DaftResult<ColumnPlan> {
    let daft_schema = Schema::try_from(prepared.arrow_schema.as_ref())?;
    let user_cols = opts.columns.as_deref();

    let mut read_col_names: HashSet<String> = match user_cols {
        Some(cols) => cols.iter().cloned().collect(),
        None => daft_schema.field_names().map(str::to_string).collect(),
    };

    let (pred_cols, predicate_pushed) = match opts.predicate.as_ref() {
        Some(pred) => match predicate_pushable_cols(pred, &daft_schema) {
            Some(cols) => (Some(cols), true),
            None => {
                let extra: HashSet<String> = get_required_columns(pred)
                    .into_iter()
                    .filter(|c| daft_schema.get_field(c).is_ok())
                    .collect();
                (Some(extra), false)
            }
        },
        None => (None, false),
    };

    if let Some(filter_cols) = &pred_cols {
        read_col_names.extend(filter_cols.iter().cloned());
    }

    let read_daft_schema = project_schema(&daft_schema, &read_col_names);
    let return_daft_schema = match user_cols {
        Some(cols) => Arc::new(project_schema(
            &daft_schema,
            &cols.iter().cloned().collect(),
        )),
        None => Arc::new(daft_schema),
    };

    let arrow_fields = prepared.arrow_schema.fields();
    let read_col_indices: Vec<usize> = arrow_fields
        .iter()
        .enumerate()
        .filter(|(_, f)| read_col_names.contains(f.name()))
        .map(|(i, _)| i)
        .collect();

    let (pred_col_indices, data_col_indices): (Vec<usize>, Vec<usize>) = if predicate_pushed {
        let pred_set: HashSet<&str> = pred_cols
            .as_ref()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect();
        read_col_indices
            .iter()
            .copied()
            .partition(|&i| pred_set.contains(arrow_fields[i].name().as_str()))
    } else {
        (Vec::new(), read_col_indices.clone())
    };

    Ok(ColumnPlan {
        read_col_indices,
        pred_col_indices,
        data_col_indices,
        predicate_pushed,
        return_daft_schema,
        read_daft_schema,
    })
}

fn build_global_starts(metadata: &ParquetMetaData) -> Vec<usize> {
    let mut starts = Vec::with_capacity(metadata.num_row_groups());
    let mut acc = 0usize;
    for i in 0..metadata.num_row_groups() {
        starts.push(acc);
        acc += metadata.row_group(i).num_rows() as usize;
    }
    starts
}

fn build_base_selections(
    metadata: &ParquetMetaData,
    rg_indices: &[usize],
    opts: &ParquetReadOptions,
) -> Vec<Option<RowSelection>> {
    // With a predicate, limit is enforced post-filter, not at RG-build time:
    // `num_rows=10` means ten *matching* rows, not ten input rows. The owned
    // path also runs RGs concurrently, so the global post-filter limit cannot
    // be mutated here by individual RG tasks.
    let global_starts = build_global_starts(metadata);
    let start_offset = opts.start_offset.unwrap_or(0);
    let delete_rows = opts.delete_rows.as_deref();
    let mut limit_remaining: Option<usize> =
        opts.predicate.is_none().then_some(opts.num_rows).flatten();
    rg_indices
        .iter()
        .map(|&rg_idx| {
            let rg_rows = metadata.row_group(rg_idx).num_rows() as usize;
            let mut sel = build_offset_selection(start_offset, global_starts[rg_idx], rg_rows);
            if let Some(deletes) = delete_rows
                && !deletes.is_empty()
            {
                sel = combine_selections(
                    sel,
                    Some(build_single_rg_delete_selection(
                        deletes,
                        global_starts[rg_idx],
                        rg_rows,
                    )),
                );
            }
            if let Some(remaining) = limit_remaining.as_mut() {
                sel = apply_limit_to_selection(sel, remaining, rg_rows);
            }
            sel
        })
        .collect()
}

/// Row-level offset selection within the RG straddling `start_offset`. RGs
/// entirely before the offset are already pruned by `prune_row_groups`, so
/// only the straddling case (or no-skip) reaches here.
fn build_offset_selection(
    global_offset: usize,
    rg_global_start: usize,
    rg_rows: usize,
) -> Option<RowSelection> {
    if global_offset > rg_global_start {
        Some(build_offset_row_selection(
            global_offset - rg_global_start,
            rg_rows,
        ))
    } else {
        None
    }
}

/// Cap the selection for the RG straddling the `num_rows` boundary. RGs past
/// the limit are already pruned by `prune_row_groups`, so we only need to
/// cap-or-passthrough — never produce an all-skip selection.
fn apply_limit_to_selection(
    sel: Option<RowSelection>,
    remaining: &mut usize,
    rg_rows: usize,
) -> Option<RowSelection> {
    let contrib = sel.as_ref().map(|s| s.row_count()).unwrap_or(rg_rows);
    if contrib > *remaining {
        let cap = *remaining;
        *remaining = 0;
        Some(cap_selection_to(sel.as_ref(), cap, rg_rows))
    } else {
        *remaining -= contrib;
        sel
    }
}

/// Per-RG inputs to the streaming decode pass.
///
/// - `selection`: row selection to apply when decoding data columns —
///   offset/delete-derived, refined to skip rows the predicate already rejected.
/// - `pred_arrays`: predicate-column arrays already filtered by the predicate
///   mask, so the main pass slices them directly without re-decoding or
///   re-filtering. Empty when there's no predicate to prefilter.
struct RgInputs {
    selection: Option<RowSelection>,
    pred_arrays: Vec<ArrayRef>,
}

/// Build the per-RG input bundles for the streaming decoder.
///
/// Without a pushed predicate prefilter, this just forwards offset/delete/limit
/// selections. With one, it first decodes predicate columns, evaluates the mask,
/// and uses that same mask for both filtered predicate arrays and a data-column
/// `RowSelection`.
#[allow(clippy::too_many_arguments)]
async fn build_rg_inputs(
    chunk_source: &Arc<ChunkSource>,
    metadata: &Arc<ParquetMetaData>,
    arrow_schema: &Arc<ArrowSchema>,
    rg_indices: &[usize],
    plan: &ColumnPlan,
    opts: &ParquetReadOptions,
    path: &Arc<str>,
) -> DaftResult<Vec<RgInputs>> {
    let base_selections = build_base_selections(metadata, rg_indices, opts);

    // If no predicate, then we can just return the base selections for the whole RG immediately.
    // Otherwise, we can pre-filter based on the predicate
    let Some(prefilter_predicate) = opts
        .predicate
        .as_ref()
        .filter(|_| plan.predicate_pushed && !plan.data_col_indices.is_empty())
    else {
        return Ok(base_selections
            .into_iter()
            .map(|selection| RgInputs {
                selection,
                pred_arrays: Vec::new(),
            })
            .collect());
    };

    // Limit set → chunked streaming for early termination. Unbounded → one
    // chunk per col so per-chunk arrow setup costs collapse to per-RG.
    let chunk_size = match opts.num_rows {
        Some(_) => opts.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1),
        None => usize::MAX,
    };
    let pred_arrow_schema = schema_from_indices(arrow_schema, &plan.pred_col_indices);
    // Bind predicate once across all RGs — same chunk schema everywhere.
    let chunk_daft_schema = Arc::new(Schema::try_from(pred_arrow_schema.as_ref())?);
    let bound_pred = BoundExpr::try_new(
        substitute_missing_cols(prefilter_predicate, &chunk_daft_schema)?,
        &chunk_daft_schema,
    )?;

    let mut selected_rows_remaining = opts.num_rows.unwrap_or(usize::MAX);
    let mut out: Vec<RgInputs> = Vec::with_capacity(rg_indices.len());
    let access = RgAccess::Source(chunk_source.clone());

    for (&rg_idx, base_sel) in rg_indices.iter().zip(base_selections) {
        if selected_rows_remaining == 0 {
            break;
        }
        let total_selected = base_sel
            .as_ref()
            .map(|s| s.row_count())
            .unwrap_or_else(|| metadata.row_group(rg_idx).num_rows() as usize);

        let (mut col_receivers, _col_handles) = spawn_col_decoders(
            &plan.pred_col_indices,
            &access,
            metadata,
            arrow_schema,
            base_sel.as_ref(),
            rg_idx,
            chunk_size,
            path,
        )
        .await?;

        let mut predicate_selectors: Vec<RowSelector> = Vec::new();
        let mut filtered_pred_by_col: Vec<Vec<ArrayRef>> = (0..plan.pred_col_indices.len())
            .map(|_| Vec::new())
            .collect();
        let mut processed_rows = 0usize;

        loop {
            let Some(chunks) = recv_one_chunk(&mut col_receivers).await? else {
                break;
            };
            let chunk_rows = chunks[0].len();
            let daft_pred =
                record_batch_from_arrow(pred_arrow_schema.clone(), chunks.clone(), path.as_ref())?;
            let mask = eval_predicate_mask(&daft_pred, &bound_pred)?;

            let selected_rows = mask.true_count();
            let (mask, selected_rows) = if selected_rows > selected_rows_remaining {
                (
                    truncate_mask_to_n_trues(&mask, selected_rows_remaining),
                    selected_rows_remaining,
                )
            } else {
                (mask, selected_rows)
            };

            let filtered = filter_arrays_by_mask(&chunks, &mask, path.as_ref())?;
            for (col_pos, filtered) in filtered.into_iter().enumerate() {
                filtered_pred_by_col[col_pos].push(filtered);
            }
            predicate_selectors.extend(bool_array_to_row_selection(&mask).iter().copied());
            processed_rows += chunk_rows;
            selected_rows_remaining -= selected_rows;
            if selected_rows_remaining == 0 {
                break;
            }
        }
        // Dropping receivers closes the channels → spawned decoders abort.
        drop(col_receivers);

        // Concat per-col filtered chunks into one ArrayRef per col. Zero rows
        // processed (e.g. base_sel selected 0 rows) → empty array of the col's
        // data type.
        let pred_arrays: Vec<ArrayRef> = plan
            .pred_col_indices
            .iter()
            .enumerate()
            .map(|(col_pos, &col_idx)| {
                let chunks = &filtered_pred_by_col[col_pos];
                if chunks.is_empty() {
                    arrow::array::new_empty_array(arrow_schema.field(col_idx).data_type())
                } else {
                    let refs: Vec<&dyn arrow::array::Array> =
                        chunks.iter().map(|a| a.as_ref()).collect();
                    arrow::compute::concat(&refs).expect("concat per-col chunks")
                }
            })
            .collect();

        let unprocessed = total_selected - processed_rows;
        if unprocessed > 0 {
            predicate_selectors.push(RowSelector::skip(unprocessed));
        }
        let pred_sel = RowSelection::from(predicate_selectors);
        let selection = Some(match &base_sel {
            Some(base) => refine_selection(base, &pred_sel),
            None => pred_sel,
        });

        out.push(RgInputs {
            selection,
            pred_arrays,
        });
    }

    Ok(out)
}

/// Predicate prefilter state shared by independently scheduled remote RG tasks.
///
/// PR1 choice: bind the predicate once, but evaluate it inside the lifetime of
/// one owned RG. This avoids retaining predicate-column buffers for every RG
/// in the file. The tradeoff is that a later in-flight RG can do work which an
/// earlier RG's global limit will ultimately make unnecessary; ordered output
/// and bounded lookahead keep that speculative work small and deterministic.
struct PredPrefilter {
    pred_arrow_schema: Arc<ArrowSchema>,
    bound_pred: BoundExpr,
    chunk_size: usize,
}

impl PredPrefilter {
    fn try_new(
        arrow_schema: &Arc<ArrowSchema>,
        plan: &ColumnPlan,
        predicate: &ExprRef,
        opts: &ParquetReadOptions,
    ) -> DaftResult<Self> {
        let chunk_size = match opts.num_rows {
            Some(_) => opts.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1),
            None => usize::MAX,
        };
        let pred_arrow_schema = schema_from_indices(arrow_schema, &plan.pred_col_indices);
        let chunk_daft_schema = Arc::new(Schema::try_from(pred_arrow_schema.as_ref())?);
        let bound_pred = BoundExpr::try_new(
            substitute_missing_cols(predicate, &chunk_daft_schema)?,
            &chunk_daft_schema,
        )?;
        Ok(Self {
            pred_arrow_schema,
            bound_pred,
            chunk_size,
        })
    }
}

/// Run a pushed predicate for one owned row group.
///
/// `max_selected_rows` is only a per-RG early-stop hint. The global limit is
/// intentionally not consumed here: concurrently executing RGs must not share
/// mutable limit state; the ordered outer stream owns the final truncation.
#[allow(clippy::too_many_arguments)]
async fn prefilter_one_rg(
    access: &RgAccess,
    metadata: &Arc<ParquetMetaData>,
    arrow_schema: &Arc<ArrowSchema>,
    plan: &ColumnPlan,
    path: &Arc<str>,
    setup: &PredPrefilter,
    rg_idx: usize,
    base_sel: Option<RowSelection>,
    max_selected_rows: Option<usize>,
) -> DaftResult<RgInputs> {
    let total_selected = base_sel
        .as_ref()
        .map(|selection| selection.row_count())
        .unwrap_or_else(|| metadata.row_group(rg_idx).num_rows() as usize);
    let (mut col_receivers, _col_handles) = spawn_col_decoders(
        &plan.pred_col_indices,
        access,
        metadata,
        arrow_schema,
        base_sel.as_ref(),
        rg_idx,
        setup.chunk_size,
        path,
    )
    .await?;

    let mut selected_rows_remaining = max_selected_rows.unwrap_or(usize::MAX);
    let mut selectors = Vec::new();
    let mut filtered_by_col: Vec<Vec<ArrayRef>> = (0..plan.pred_col_indices.len())
        .map(|_| Vec::new())
        .collect();
    let mut processed_rows = 0;
    while selected_rows_remaining > 0 {
        let Some(chunks) = recv_one_chunk(&mut col_receivers).await? else {
            break;
        };
        let chunk_rows = chunks[0].len();
        let batch = record_batch_from_arrow(setup.pred_arrow_schema.clone(), chunks.clone(), path)?;
        let mut mask = eval_predicate_mask(&batch, &setup.bound_pred)?;
        let mut selected_rows = mask.true_count();
        if selected_rows > selected_rows_remaining {
            mask = truncate_mask_to_n_trues(&mask, selected_rows_remaining);
            selected_rows = selected_rows_remaining;
        }
        let filtered = filter_arrays_by_mask(&chunks, &mask, path)?;
        for (position, array) in filtered.into_iter().enumerate() {
            filtered_by_col[position].push(array);
        }
        selectors.extend(bool_array_to_row_selection(&mask).iter().copied());
        processed_rows += chunk_rows;
        selected_rows_remaining -= selected_rows;
    }
    drop(col_receivers);

    let pred_arrays = plan
        .pred_col_indices
        .iter()
        .enumerate()
        .map(|(position, &column)| {
            let chunks = &filtered_by_col[position];
            if chunks.is_empty() {
                arrow::array::new_empty_array(arrow_schema.field(column).data_type())
            } else {
                let refs: Vec<&dyn arrow::array::Array> =
                    chunks.iter().map(|array| array.as_ref()).collect();
                arrow::compute::concat(&refs).expect("concat per-col chunks")
            }
        })
        .collect();
    let unprocessed = total_selected - processed_rows;
    if unprocessed > 0 {
        selectors.push(RowSelector::skip(unprocessed));
    }
    let predicate_selection = RowSelection::from(selectors);
    let selection = Some(match &base_sel {
        Some(base) => refine_selection(base, &predicate_selection),
        None => predicate_selection,
    });
    Ok(RgInputs {
        selection,
        pred_arrays,
    })
}

/// Shared per-file state for an RG-decoding task. One `Arc<RgTaskCtx>` is
/// cloned per RG task — replaces the previous fistful-of-Arcs cloning ritual.
///
/// Per-RG processor selection happens in `build_rg_stream` based on
/// `plan.data_col_indices.is_empty()`: empty → PredOnly path, non-empty →
/// Default path.
pub(super) struct RgTaskCtx {
    pub(super) path: Arc<str>,
    pub(super) metadata: Arc<ParquetMetaData>,
    pub(super) arrow_schema: Arc<ArrowSchema>,
    pub(super) plan: ColumnPlan,
    pub(super) predicate: Option<ExprRef>,
    pub(super) chunk_size: usize,
}

/// One bounded channel per RG, drained in RG order. In-file output is always
/// RG-ordered: `maintain_order=false` at the scan layer only reorders BETWEEN
/// scan tasks; downstream code (and tests) expect file-order output within
/// a single file.
fn build_rg_stream(
    ctx: Arc<RgTaskCtx>,
    chunk_source: Arc<ChunkSource>,
    rg_indices: Vec<usize>,
    rg_inputs: Vec<RgInputs>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    use tokio_stream::wrappers::ReceiverStream;

    // Capacity 1: each task runs ~one batch ahead — cross-RG overlap, bounded memory.
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..rg_inputs.len())
        .map(|_| tokio::sync::mpsc::channel::<DaftResult<RecordBatch>>(1))
        .unzip();

    let compute = get_compute_runtime();
    let mut joinset: JoinSet<DaftResult<()>> = JoinSet::new();
    for (rg_pos, (sender, inputs)) in senders.into_iter().zip(rg_inputs).enumerate() {
        let ctx = ctx.clone();
        let access = RgAccess::Source(chunk_source.clone());
        let rg_idx = rg_indices[rg_pos];
        joinset.spawn_on(
            async move {
                let mut sub_stream = if ctx.plan.data_col_indices.is_empty() {
                    process_rg_predicate_only(ctx.clone(), access, rg_idx, inputs.selection).await
                } else {
                    process_rg_with_data_cols(
                        ctx,
                        access,
                        rg_idx,
                        inputs.selection,
                        inputs.pred_arrays,
                    )
                    .await
                };
                while let Some(item) = sub_stream.next().await {
                    if sender.send(item).await.is_err() {
                        break;
                    }
                }
                DaftResult::Ok(())
            },
            &compute,
        );
    }

    let inner_streams = receivers.into_iter().map(ReceiverStream::new);
    let merged: BoxStream<'static, DaftResult<RecordBatch>> =
        Box::pin(futures::stream::iter(inner_streams).flatten());
    common_runtime::combine_stream(merged, async move { joinset.join_all().await }).boxed()
}

fn count_only_stream(
    metadata: &ParquetMetaData,
    rg_indices: &[usize],
    num_rows: Option<usize>,
    return_schema: Arc<Schema>,
) -> DaftResult<(Arc<Schema>, BoxStream<'static, DaftResult<RecordBatch>>)> {
    let total: usize = rg_indices
        .iter()
        .map(|&i| metadata.row_group(i).num_rows() as usize)
        .sum();
    let n = num_rows.map(|n| n.min(total)).unwrap_or(total);
    let batch = RecordBatch::new_with_size(return_schema.clone(), Vec::new(), n)?;
    Ok((
        return_schema,
        futures::stream::once(async move { Ok(batch) }).boxed(),
    ))
}

fn apply_cross_rg_limit(
    stream: impl Stream<Item = DaftResult<RecordBatch>> + Send + 'static,
    limit: Option<usize>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    let Some(limit) = limit else {
        return Box::pin(stream);
    };
    // `scan` ends the stream when the closure yields `None`. This is more than
    // an optimization: once the caller has enough rows, closing the upstream
    // receiver lets the owned-RG coordinator cancel remaining decode work and
    // drop its resident bytes. `filter_map` would keep polling upstream.
    let bounded = stream.scan(limit, |remaining, res| {
        let out = match res {
            Err(e) => Some(Err(e)),
            Ok(_) if *remaining == 0 => None,
            Ok(b) if b.num_rows() <= *remaining => {
                *remaining -= b.num_rows();
                Some(Ok(b))
            }
            Ok(b) => {
                let take = std::mem::replace(remaining, 0);
                Some(b.head(take))
            }
        };
        std::future::ready(out)
    });
    Box::pin(bounded)
}

pub async fn stream_parquet(
    source: ParquetSource<'_>,
    opts: &ParquetReadOptions,
) -> DaftResult<(Arc<Schema>, BoxStream<'static, DaftResult<RecordBatch>>)> {
    let (cs_builder, arrow_metadata) = open_chunk_source(&source, opts).await?;
    let path = cs_builder.path().clone();

    let chunk_size = opts.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let prepared = prepare_metadata(
        arrow_metadata,
        opts.field_id_mapping.as_ref(),
        opts.schema_infer,
        &path,
    )?;
    let plan = resolve_column_plan(&prepared, opts)?;

    // Single RG-level pruning pass: user row_groups + positional (start_offset,
    // num_rows) + predicate stats. It must precede any remote data GET: footer
    // metadata is cheap, whereas downloading a pruned RG is pure waste.
    let rg_indices = prune_row_groups(
        &prepared.parquet_metadata,
        opts.row_groups.as_deref(),
        opts.start_offset.unwrap_or(0),
        opts.num_rows,
        opts.predicate.as_ref(),
        &plan.read_daft_schema,
        source.label(),
    )?;
    if rg_indices.is_empty() {
        // Empty stream — NOT a single empty batch. Downstream sinks
        // (e.g. iceberg writer) treat any received batch as "there's
        // something to write" and emit metadata for it; a 0-row batch
        // would still land an empty snapshot.
        return Ok((
            plan.return_daft_schema.clone(),
            futures::stream::empty().boxed(),
        ));
    }
    if opts.num_rows == Some(0) {
        return Ok((
            plan.return_daft_schema.clone(),
            futures::stream::empty().boxed(),
        ));
    }
    if plan.read_col_indices.is_empty() {
        return count_only_stream(
            &prepared.parquet_metadata,
            &rg_indices,
            opts.num_rows,
            plan.return_daft_schema,
        );
    }

    let return_schema = plan.return_daft_schema.clone();
    let ctx = Arc::new(RgTaskCtx {
        path: path.clone(),
        metadata: prepared.parquet_metadata.clone(),
        arrow_schema: prepared.arrow_schema.clone(),
        plan,
        predicate: opts.predicate.clone(),
        chunk_size,
    });

    // [PR1 remote-reader change]
    // Remote reads use an owned row-group pipeline. Planning does not issue
    // GETs; each requested RG occurrence downloads only when it enters the
    // bounded lookahead. This replaces eager whole-file prefetch for remote
    // sources, while deliberately leaving the mature local `ChunkSource` path
    // unchanged. PR1 does not attempt a process-wide byte budget; its scope is
    // the more fundamental ownership/lifetime boundary.
    let cs_builder = match cs_builder.build_plan(&prepared.parquet_metadata, &rg_indices) {
        Ok(remote_plan) => {
            let base_selections =
                build_base_selections(&prepared.parquet_metadata, &rg_indices, opts);
            let pred_setup = match ctx
                .predicate
                .as_ref()
                .filter(|_| ctx.plan.predicate_pushed && !ctx.plan.data_col_indices.is_empty())
            {
                Some(predicate) => Some(Arc::new(PredPrefilter::try_new(
                    &prepared.arrow_schema,
                    &ctx.plan,
                    predicate,
                    opts,
                )?)),
                None => None,
            };
            let stream = stream_parquet_owned(
                Arc::new(remote_plan),
                ctx,
                rg_indices,
                base_selections,
                pred_setup,
                opts.num_rows,
            );
            return Ok((return_schema, apply_cross_rg_limit(stream, opts.num_rows)));
        }
        Err(local_builder) => *local_builder,
    };

    // Local path.
    let chunk_source = Arc::new(cs_builder.build(prepared.parquet_metadata.clone(), &rg_indices));

    let rg_inputs = build_rg_inputs(
        &chunk_source,
        &prepared.parquet_metadata,
        &prepared.arrow_schema,
        &rg_indices,
        &ctx.plan,
        opts,
        &path,
    )
    .await?;

    let stream = build_rg_stream(ctx, chunk_source, rg_indices, rg_inputs);

    Ok((return_schema, apply_cross_rg_limit(stream, opts.num_rows)))
}

// [PR1 remote-reader change]
// At most this many RG pipelines per file may have downloaded bytes or be
// decoding at once. Two overlaps I/O/CPU with ordered downstream consumption
// without recreating the previous O(file) eager-prefetch residency. This is a
// per-file bound, not a process-wide memory admission guarantee.
const REMOTE_RG_LOOKAHEAD: usize = 2;

type InflightRg = (
    common_runtime::RuntimeTask<DaftResult<()>>,
    tokio::sync::mpsc::Receiver<DaftResult<RecordBatch>>,
);

/// [PR1 remote-reader change] Ordered remote pipeline with bounded lookahead.
///
/// The coordinator starts up to `REMOTE_RG_LOOKAHEAD` worker-local runtime
/// tasks, but drains only the oldest RG before advancing. That preserves Parquet
/// file order even though downloading/decoding overlaps. It is not a Ray task:
/// the surrounding scan task owns this coordinator. The returned stream owns
/// the coordinator, so early downstream cancellation aborts outstanding tasks
/// and releases their `ResidentRowGroup`s.
fn stream_parquet_owned(
    remote_plan: Arc<chunk_source::RemoteChunkSourcePlan>,
    ctx: Arc<RgTaskCtx>,
    rg_indices: Vec<usize>,
    base_selections: Vec<Option<RowSelection>>,
    pred_setup: Option<Arc<PredPrefilter>>,
    predicate_limit: Option<usize>,
) -> BoxStream<'static, DaftResult<RecordBatch>> {
    use tokio_stream::wrappers::ReceiverStream;

    let compute = get_compute_runtime();
    let (out_tx, out_rx) = tokio::sync::mpsc::channel(1);
    let coordinator = compute.spawn(async move {
        let compute = get_compute_runtime();
        let mut inflight = std::collections::VecDeque::<InflightRg>::new();
        let mut pending = rg_indices
            .into_iter()
            .zip(base_selections)
            .enumerate()
            .collect::<std::collections::VecDeque<_>>();

        loop {
            // Fill a small ordered window. `occurrence`, rather than `rg_idx`,
            // is the identity here because callers may request an RG twice.
            while inflight.len() < REMOTE_RG_LOOKAHEAD {
                let Some((occurrence, (rg_idx, base_sel))) = pending.pop_front() else {
                    break;
                };
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let remote_plan = remote_plan.clone();
                let ctx = ctx.clone();
                let pred_setup = pred_setup.clone();
                let task = compute.spawn(async move {
                    // Downloading constructs the only owner for this RG's
                    // compressed bytes. All decoder readers below clone it,
                    // binding byte lifetime to decoder lifetime by type.
                    let resident = Arc::new(remote_plan.download_occurrence(occurrence).await?);
                    let access = RgAccess::Resident(resident);
                    let inputs = match pred_setup.as_deref() {
                        Some(setup) => {
                            prefilter_one_rg(
                                &access,
                                &ctx.metadata,
                                &ctx.arrow_schema,
                                &ctx.plan,
                                &ctx.path,
                                setup,
                                rg_idx,
                                base_sel,
                                predicate_limit,
                            )
                            .await?
                        }
                        None => RgInputs {
                            selection: base_sel,
                            pred_arrays: Vec::new(),
                        },
                    };
                    let mut stream = if ctx.plan.data_col_indices.is_empty() {
                        process_rg_predicate_only(ctx.clone(), access, rg_idx, inputs.selection)
                            .await
                    } else {
                        process_rg_with_data_cols(
                            ctx,
                            access,
                            rg_idx,
                            inputs.selection,
                            inputs.pred_arrays,
                        )
                        .await
                    };
                    while let Some(batch) = stream.next().await {
                        if tx.send(batch).await.is_err() {
                            break;
                        }
                    }
                    Ok(())
                });
                inflight.push_back((task, rx));
            }

            let Some((task, mut rx)) = inflight.pop_front() else {
                break;
            };
            // Drain in request order. Later RGs can run, but cannot overtake
            // this one in the output stream.
            while let Some(batch) = rx.recv().await {
                if out_tx.send(batch).await.is_err() {
                    return Ok(());
                }
            }
            task.await??;
        }
        Ok(())
    });
    let stream: BoxStream<'static, DaftResult<RecordBatch>> = Box::pin(ReceiverStream::new(out_rx));
    common_runtime::combine_stream(stream, async move { coordinator.await? }).boxed()
}
