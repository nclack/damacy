struct resolved_wave_geometry
{
  const struct input_transfer_ops* input;
  struct wave_pool_sizing sizing;
  struct gpu_budget_breakdown predicted;
  uint32_t max_chunks_per_wave;
  uint32_t max_substreams_per_chunk;
  uint8_t host_buffer_waves;
  uint8_t want_gds;
};

enum wave_geometry_step
{
  WAVE_GEOMETRY_SIZING,
  WAVE_GEOMETRY_PREDICT,
};

static enum damacy_status
resolve_wave_geometry(const struct damacy_config* cfg,
                      uint64_t resolver_budget,
                      uint64_t runtime_chunk_cap,
                      struct resolved_wave_geometry* out,
                      enum wave_geometry_step* failed_step)
{
  *out = (struct resolved_wave_geometry){ 0 };
  out->max_chunks_per_wave = resolve_max_chunks_per_wave(cfg);
  out->max_substreams_per_chunk = resolve_max_substreams_per_chunk(cfg);
  out->host_buffer_waves = resolve_host_buffer_waves(cfg);
  out->want_gds = resolve_enable_gds(cfg);
  out->input =
    out->want_gds ? input_transfer_gds() : input_transfer_host_staging();

  uint64_t index_bytes =
    2ull * dispatch_index_capacity(cfg) * sizeof(struct gather_index);
  if (index_bytes > resolver_budget)
    return DAMACY_BUDGET;
  resolver_budget -= index_bytes;

  const struct input_transfer_resources min_input =
    input_transfer_resources(out->input, out->host_buffer_waves, 0);
  enum damacy_status s =
    wave_pool_resolve_sizing(out->max_chunks_per_wave,
                             out->max_substreams_per_chunk,
                             min_input.device_staging_buffers,
                             resolver_budget,
                             cfg->tuning.max_read_op_bytes,
                             runtime_chunk_cap,
                             cfg->samples_per_batch,
                             &out->sizing);
  if (s != DAMACY_OK) {
    if (failed_step)
      *failed_step = WAVE_GEOMETRY_SIZING;
    return s;
  }

  out->sizing.worst_case_total_bytes += index_bytes;
  const struct input_transfer_resources input_resources =
    input_transfer_resources(
      out->input, out->host_buffer_waves, out->sizing.input_staging_per_wave);
  s = gpu_budget_predict(cfg,
                         &input_resources,
                         out->sizing.dev_decompressed_per_wave,
                         &out->predicted);
  if (s != DAMACY_OK && failed_step)
    *failed_step = WAVE_GEOMETRY_PREDICT;
  return s;
}

void
damacy_config_describe(const struct damacy_config* cfg)
{
  if (!cfg) {
    log_info("damacy_config_describe: NULL config");
    return;
  }
  const uint64_t max_gpu = cfg->tuning.max_gpu_memory_bytes;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  uint64_t pool_reserve = 0;
  {
    uint64_t pool_bytes = 0;
    // resolve_sample_volume_bytes rejects rank=0 / non-positive dims; on
    // those, leave pool_reserve at 0 so describe still prints useful
    // info for the rest of the geometry.
    enum damacy_status pvs = resolve_sample_volume_bytes(cfg, &pool_bytes);
    if (pvs == DAMACY_OK)
      pool_reserve = 2ull * pool_bytes;
    else
      log_info(
        "damacy_config_describe: resolve_sample_volume_bytes failed (%s); "
        "pool_reserve=0",
        damacy_status_str(pvs));
  }
  const uint64_t resolver_budget =
    pool_reserve < max_gpu ? max_gpu - pool_reserve : 0;
  log_info("damacy_config_describe: max_gpu_memory_bytes=%llu "
           "(pool_reserve=%llu, resolver_budget=%llu, "
           "max_chunk_uncompressed_bytes=%llu, samples_per_batch=%u)",
           (unsigned long long)max_gpu,
           (unsigned long long)pool_reserve,
           (unsigned long long)resolver_budget,
           (unsigned long long)runtime_chunk_cap,
           (unsigned)cfg->samples_per_batch);

  struct resolved_wave_geometry geom = { 0 };
  enum wave_geometry_step failed_step = WAVE_GEOMETRY_SIZING;
  enum damacy_status rs = resolve_wave_geometry(
    cfg, resolver_budget, runtime_chunk_cap, &geom, &failed_step);
  if (rs != DAMACY_OK) {
    const char* where = failed_step == WAVE_GEOMETRY_PREDICT
                          ? "gpu_budget_predict"
                          : "wave_pool_resolve_sizing";
    log_info(
      "damacy_config_describe: %s failed (%s)", where, damacy_status_str(rs));
    return;
  }
  log_info("damacy_config_describe: input_staging_per_wave=%llu "
           "dev_decompressed_per_wave=%llu",
           (unsigned long long)geom.sizing.input_staging_per_wave,
           (unsigned long long)geom.sizing.dev_decompressed_per_wave);
  log_info("damacy_config_describe: dev_compressed=%llu dev_decompressed=%llu "
           "blosc1_meta=%llu fanout_soa=%llu "
           "nvcomp_temp=%llu batch_metadata=%llu",
           (unsigned long long)geom.predicted.dev_compressed,
           (unsigned long long)geom.predicted.dev_decompressed,
           (unsigned long long)geom.predicted.blosc1_meta,
           (unsigned long long)geom.predicted.fanout_soa,
           (unsigned long long)geom.predicted.nvcomp_temp,
           (unsigned long long)geom.predicted.batch_metadata);
  // predicted.total is the *initial* allocation (initial fanout / decoder
  // floors); sizing.worst_case_total_bytes is the post-grow worst case
  // the resolver pre-reserved against the cap. Their difference is the
  // grow-time headroom; the cap minus the worst case is unused slack.
  const uint64_t initial_alloc = geom.predicted.total;
  const uint64_t worst_case = geom.sizing.worst_case_total_bytes;
  const uint64_t reserved_for_grow =
    worst_case > initial_alloc ? worst_case - initial_alloc : 0;
  const uint64_t slack = max_gpu > worst_case ? max_gpu - worst_case : 0;
  log_info("damacy_config_describe: initial_alloc=%llu reserved_for_grow=%llu "
           "worst_case_total=%llu slack=%llu cap=%llu",
           (unsigned long long)initial_alloc,
           (unsigned long long)reserved_for_grow,
           (unsigned long long)worst_case,
           (unsigned long long)slack,
           (unsigned long long)max_gpu);
}
