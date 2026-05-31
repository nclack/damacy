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
h2d_queue_input(CUstream stream,
                struct damacy_wave* wave,
                struct input_transfer_queue_state* state)
{
  CU(CudaFail, cuEventRecord(wave->ev.input_start, stream));
  state->queued_stream_work = 1;
  damacy_nvtx_range_push("input_transfer");
  CU(BulkCudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                       wave->host_input,
                       wave->input_used_bytes,
                       stream));
  CU(BulkCudaFail, cuEventRecord(wave->ev.input_transfer_done, stream));
  damacy_nvtx_range_pop(); // input_transfer
  return DAMACY_OK;
BulkCudaFail:
  damacy_nvtx_range_pop(); // input_transfer
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
gds_queue_input(CUstream stream,
                struct damacy_wave* wave,
                struct input_transfer_queue_state* state)
{
  CU(CudaFail, cuEventRecord(wave->ev.input_start, stream));
  state->queued_stream_work = 1;
  damacy_nvtx_range_push("input_transfer");
  CU(BulkCudaFail, cuEventRecord(wave->ev.input_transfer_done, stream));
  damacy_nvtx_range_pop(); // input_transfer
  return DAMACY_OK;
BulkCudaFail:
  damacy_nvtx_range_pop(); // input_transfer
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
event_ready(CUevent ev, int* ready)
{
  CUresult q = cuEventQuery(ev);
  if (q == CUDA_SUCCESS) {
    *ready = 1;
    return DAMACY_OK;
  }
  if (q == CUDA_ERROR_NOT_READY) {
    *ready = 0;
    return DAMACY_OK;
  }
  return DAMACY_CUDA;
}

static enum damacy_status
h2d_slot_reuse_ready(const struct damacy_wave* wave, int* ready)
{
  return event_ready(wave->ev.input_transfer_done, ready);
}

static enum damacy_status
gds_slot_reuse_ready(const struct damacy_wave* wave, int* ready)
{
  if (wave->state != WAVE_POST) {
    *ready = 0;
    return DAMACY_OK;
  }
  return event_ready(wave->ev.decomp_end, ready);
}

static const struct input_transfer_ops k_h2d = {
  .name = "h2d",
  .resources = h2d_resources,
  .bind_stream = h2d_bind_stream,
  .read_base = h2d_read_base,
  .wave_input = h2d_wave_input,
  .submit_reads = h2d_submit_reads,
  .queue_input = h2d_queue_input,
  .slot_reuse_ready = h2d_slot_reuse_ready,
};

static const struct input_transfer_ops k_gds = {
  .name = "gds",
  .resources = gds_resources,
  .bind_stream = gds_bind_stream,
  .read_base = gds_read_base,
  .wave_input = gds_wave_input,
  .submit_reads = gds_submit_reads,
  .queue_input = gds_queue_input,
  .slot_reuse_ready = gds_slot_reuse_ready,
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
