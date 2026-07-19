#[cfg(feature = "batching")]
pub mod batch;
#[cfg(feature = "framebuffer")]
pub mod framebuffer;

#[cfg(feature = "pipeline")]
pub mod pipeline;

#[cfg(feature = "batching")]
pub const BATCH_UNITS: usize = batch::PER_BATCH_UNITS;
