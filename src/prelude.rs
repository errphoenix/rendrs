#[cfg(feature = "batching")]
pub use crate::batch::{Batch, BatchManager};

#[cfg(feature = "framebuffer")]
pub use crate::framebuffer::{
    AttachmentTexture, Framebuffer, FramebufferId, FramebufferView, HasFramebuffer,
};

#[cfg(feature = "pipeline")]
pub use crate::pipeline::{
    ComputePass, DrawPass, ImageAccessKind, ImageObject, ImageObjectTarget, OutputObject, Pass,
    RenderPool, RenderTarget, RenderTargetAccessor, RenderTargetDescriptor, RenderTargetId,
    Sampler, SamplerObject,
};

#[cfg(feature = "graphics")]
pub use crate::graphics::material::{
    MaterialEntryLocation, MaterialGroup, MaterialId, MaterialLocation, MaterialLocationRegistry,
    MaterialParams,
};

#[cfg(feature = "geometry")]
pub use crate::geometry::{DomainData, GeometryBank, HasTriangleBuffers, HasVertexBuffers};
