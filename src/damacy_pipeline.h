#pragma once

#include "damacy.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct damacy_reader;
  struct damacy_metadata_reader;
  struct damacy_metadata;
  struct damacy_planner;
  struct damacy_executor;

  struct damacy_batch_spec
  {
    enum damacy_dtype dtype;
    int64_t sample_shape[DAMACY_MAX_RANK];
    uint8_t sample_rank;
    uint32_t samples_per_batch;
  };

  struct damacy_queue_limits
  {
    uint32_t lookahead_samples;
    uint32_t prepared_batches;
  };

  struct damacy_plan_limits
  {
    uint32_t max_chunks;
    uint32_t max_chunk_bytes;
    uint32_t max_shards_per_sample;
    uint64_t max_plan_bytes;
  };

  struct damacy_metadata_cache_config
  {
    uint32_t array_entries;
    uint32_t shard_entries;
  };

  struct damacy_cpu_config
  {
    uint32_t decode_workers;
    uint32_t max_encoded_chunk_bytes;
    uint32_t max_decoded_chunk_bytes;
    uint64_t max_memory_bytes;
    uint32_t chunks_per_input_buffer;
  };

  struct damacy_cuda_config
  {
    int device;
    uint64_t max_gpu_memory_bytes;
    uint64_t max_index_bytes;
    uint32_t max_chunk_bytes;
    uint64_t max_read_bytes;
    uint32_t max_chunks_per_wave;
    uint32_t max_substreams_per_chunk;
    uint8_t host_buffer_waves;
    uint32_t chunk_layout_entries;
    enum damacy_numa_strategy numa_strategy;
    int numa_node;
    enum damacy_gds_mode enable_gds;
    uint8_t bypass_decode;
  };

  enum damacy_status damacy_file_reader_create(uint32_t workers,
                                               uint32_t max_inflight_reads,
                                               struct damacy_reader** out);
  void damacy_reader_destroy(struct damacy_reader* reader);

  enum damacy_status damacy_file_metadata_reader_create(
    uint32_t concurrency,
    const struct damacy_latency_model* latency,
    struct damacy_metadata_reader** out);
  void damacy_metadata_reader_destroy(struct damacy_metadata_reader* reader);

  enum damacy_status damacy_zarr_metadata_create(
    struct damacy_metadata_reader* reader,
    const struct damacy_metadata_cache_config* cache,
    struct damacy_metadata** out);
  void damacy_metadata_destroy(struct damacy_metadata* metadata);

  enum damacy_status damacy_chunk_planner_create(
    struct damacy_metadata* metadata,
    const struct damacy_plan_limits* limits,
    struct damacy_planner** out);
  void damacy_planner_destroy(struct damacy_planner* planner);

  enum damacy_status damacy_cpu_executor_create(
    struct damacy_reader* reader,
    const struct damacy_cpu_config* config,
    struct damacy_executor** out);
  enum damacy_status damacy_cuda_executor_create(
    struct damacy_reader* reader,
    const struct damacy_cuda_config* config,
    struct damacy_executor** out);
  void damacy_executor_destroy(struct damacy_executor* executor);

  // Components are borrowed until shutdown; each planner/executor may serve
  // one active pipeline. A reader must outlive its executors. Metadata and
  // metadata readers hold only settings, copied by the objects created from
  // them, so they can be shared or destroyed at any time.
  enum damacy_status damacy_pipeline_create(
    struct damacy_planner* planner,
    struct damacy_executor* executor,
    const struct damacy_batch_spec* output,
    const struct damacy_queue_limits* queues,
    struct damacy** out);

#ifdef __cplusplus
}
#endif
