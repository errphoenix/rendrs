#[cfg(feature = "batching")]
pub mod batch;
#[cfg(feature = "framebuffer")]
pub mod framebuffer;
#[cfg(feature = "graphics")]
pub mod graphics;
#[cfg(feature = "pack")]
pub mod pack;
#[cfg(feature = "pipeline")]
pub mod pipeline;
#[cfg(feature = "geometry")]
pub mod geometry;

#[cfg(feature = "batching")]
pub const BATCH_UNITS: usize = batch::PER_BATCH_UNITS;

#[allow(unused)]
#[cfg(feature = "pipeline")]
pub use pipeline::{BlitPass, ClearPass, ComputePass, DrawPass, EmptyPassCtx};
