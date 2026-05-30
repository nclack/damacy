#include "input_transfer.h"

#include "damacy_limits.h"
#include "nvtx/nvtx.h"
#include "store/store_fs_gds.h"
#include "util/cuda_check.h"
#include "wave/input_slot.h"
#include "wave/wave.h"

static struct input_transfer_resources
h2d_resources(uint8_t n_slots, uint64_t bytes_per_wave)
{
  (void)n_slots;
  return (struct input_transfer_resources){
    .slot_host_bytes = bytes_per_wave,
    .wave_device_bytes = bytes_per_wave,
    .device_staging_bytes = (uint64_t)DAMACY_N_WAVES * bytes_per_wave,
    .device_staging_buffers = DAMACY_N_WAVES,
  };
}

static struct input_transfer_resources
gds_resources(uint8_t n_slots, uint64_t bytes_per_wave)
{
  return (struct input_transfer_resources){
    .slot_device_bytes = bytes_per_wave,
    .device_staging_bytes = (uint64_t)n_slots * bytes_per_wave,
    .device_staging_buffers = n_slots,
  };
}

struct input_transfer_resources
input_transfer_resources(const struct input_transfer_ops* ops,
                         uint8_t n_slots,
                         uint64_t bytes_per_wave)
{
  return ops->resources(n_slots, bytes_per_wave);
}

static void
h2d_bind_stream(struct store* store, CUstream stream)
{
  (void)store;
  (void)stream;
}

static void
gds_bind_stream(struct store* store, CUstream stream)
{
  store_fs_gds_set_stream(store, stream);
}

static void*
h2d_read_base(struct input_slot* slot)
{
  return slot->buf;
}

static void*
gds_read_base(struct input_slot* slot)
{
  return slot->dev_buf;
}

static void*
h2d_wave_input(struct damacy_wave* wave, const struct input_slot* slot)
{
  (void)slot;
  return wave->dev_compressed_owned;
}

static void*
gds_wave_input(struct damacy_wave* wave, const struct input_slot* slot)
{
  (void)wave;
  return slot->dev_buf;
}

static struct store_event
h2d_submit_reads(struct store* store,
                 struct store_read* reads,
                 uint32_t n_reads)
{
  return store_read_submit(store, reads, n_reads);
}

static struct store_event
gds_submit_reads(struct store* store,
                 struct store_read* reads,
                 uint32_t n_reads)
{
  return store_read_submit_dev(store, reads, n_reads);
}

static enum damacy_status
h2d_queue_ready(CUstream stream,
                struct damacy_wave* wave,
                struct input_transfer_submit_state* state)
{
  CU(CudaFail, cuEventRecord(wave->ev.h2d_start, stream));
  state->stream_work_queued = 1;
  damacy_nvtx_range_push("bulk_h2d");
  CU(BulkCudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                       wave->host_input,
                       wave->input_used_bytes,
                       stream));
  CU(BulkCudaFail, cuEventRecord(wave->ev.bulk_h2d_end, stream));
  damacy_nvtx_range_pop(); // bulk_h2d
  return DAMACY_OK;
BulkCudaFail:
  damacy_nvtx_range_pop(); // bulk_h2d
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
gds_queue_ready(CUstream stream,
                struct damacy_wave* wave,
                struct input_transfer_submit_state* state)
{
  CU(CudaFail, cuEventRecord(wave->ev.h2d_start, stream));
  state->stream_work_queued = 1;
  damacy_nvtx_range_push("bulk_h2d");
  CU(BulkCudaFail, cuEventRecord(wave->ev.bulk_h2d_end, stream));
  damacy_nvtx_range_pop(); // bulk_h2d
  return DAMACY_OK;
BulkCudaFail:
  damacy_nvtx_range_pop(); // bulk_h2d
CudaFail:
  return DAMACY_CUDA;
}

static CUevent
h2d_slot_release_gate(const struct damacy_wave* wave)
{
  return wave->ev.bulk_h2d_end;
}

static CUevent
gds_slot_release_gate(const struct damacy_wave* wave)
{
  return wave->ev.h2d_end;
}

static const struct input_transfer_ops k_h2d = {
  .name = "h2d",
  .resources = h2d_resources,
  .bind_stream = h2d_bind_stream,
  .read_base = h2d_read_base,
  .wave_input = h2d_wave_input,
  .submit_reads = h2d_submit_reads,
  .queue_ready = h2d_queue_ready,
  .slot_release_gate = h2d_slot_release_gate,
};

static const struct input_transfer_ops k_gds = {
  .name = "gds",
  .resources = gds_resources,
  .bind_stream = gds_bind_stream,
  .read_base = gds_read_base,
  .wave_input = gds_wave_input,
  .submit_reads = gds_submit_reads,
  .queue_ready = gds_queue_ready,
  .slot_release_gate = gds_slot_release_gate,
};

const struct input_transfer_ops*
input_transfer_h2d(void)
{
  return &k_h2d;
}

const struct input_transfer_ops*
input_transfer_gds(void)
{
  return &k_gds;
}
