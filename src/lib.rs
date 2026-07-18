use ethel::shader::{ComputeShaderHandleView, ShaderHandleView, ShaderProgram};
use janus::texture::{TextureTarget, TextureView};

#[cfg(feature = "batching")]
pub mod batch;
#[cfg(feature = "framebuffer")]
pub mod framebuffer;

#[cfg(feature = "batching")]
pub const BATCH_UNITS: usize = batch::PER_BATCH_UNITS;

pub const PASS_READ_UNITS: usize = 16;
pub const PASS_WRITE_UNITS: usize = framebuffer::MAX_ATTACHMENTS;

pub struct PipelineBuilder<const PASSES: usize> {}
impl<const PASSES: usize> PipelineBuilder<PASSES> {}

// pub struct Pipeline<const PASSES: usize> {
//     passes: [Pass; PASSES],
// }

/// A uniform sampler texture.
///
/// Not to be confused with a framebuffer read target.
#[derive(Debug)]
pub struct ReadTarget {
    label: &'static str,
    texture: TextureView,
}
impl ReadTarget {
    pub const fn new(label: &'static str, texture: TextureView) -> Self {
        Self { label, texture }
    }

    pub const fn label(&self) -> &'static str {
        self.label
    }

    pub const fn texture(&self) -> &TextureView {
        &self.texture
    }
}

/// An output framebuffer target used in draw passes.
#[derive(Debug)]
pub struct DrawWriteTarget {
    label: &'static str,
    kind: DrawWriteTargetKind,
    texture: TextureView,
}
impl DrawWriteTarget {
    pub const fn new(label: &'static str, kind: DrawWriteTargetKind, texture: TextureView) -> Self {
        Self {
            label,
            kind,
            texture,
        }
    }

    pub const fn label(&self) -> &'static str {
        self.label
    }

    pub const fn kind(&self) -> DrawWriteTargetKind {
        self.kind
    }

    pub const fn texture(&self) -> &TextureView {
        &self.texture
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DrawWriteTargetKind {
    Color,
    Depth,
}

/// An uniform image used in compute passes.
#[derive(Debug)]
pub struct ImageTarget {
    label: &'static str,
    image: TextureView,
    access: ImageAccessKind,
}
impl ImageTarget {
    pub const fn new(label: &'static str, image: TextureView, access: ImageAccessKind) -> Self {
        Self {
            label,
            image,
            access,
        }
    }

    pub const fn label(&self) -> &'static str {
        self.label
    }

    pub const fn image(&self) -> &TextureView {
        &self.image
    }

    pub const fn access(&self) -> ImageAccessKind {
        self.access
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageAccessKind {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

pub trait PassResources {
    fn shader(&self) -> impl ShaderProgram;

    fn bind_shader(&self) {
        self.shader().bind();
    }
}

#[derive(Debug)]
pub struct DrawPassResources<const R: usize, const W: usize> {
    shader: ShaderHandleView,
    reads: [ReadTarget; R],
    writes: [DrawWriteTarget; W],
}
impl<const R: usize, const W: usize> PassResources for DrawPassResources<R, W> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ShaderHandleView {
        self.shader
    }
}
impl<const R: usize, const W: usize> DrawPassResources<R, W> {
    pub const fn new(
        shader: ShaderHandleView,
        reads: [ReadTarget; R],
        writes: [DrawWriteTarget; W],
    ) -> Self {
        Self {
            shader,
            reads,
            writes,
        }
    }

    pub fn bind_read_targets(&self) {
        self.reads.iter().enumerate().for_each(|(i, target)| {
            let unit = i as u32;
            janus::texture::bind(TextureTarget::Flat, target.texture, unit);
        });
    }

    pub const fn read_targets(&self) -> &[ReadTarget; R] {
        &self.reads
    }

    pub const fn write_targets(&self) -> &[DrawWriteTarget; W] {
        &self.writes
    }
}

#[derive(Debug)]
pub struct ComputePassResources<const I: usize> {
    shader: ComputeShaderHandleView,
    images: [ImageTarget; I],
}
impl<const I: usize> PassResources for ComputePassResources<I> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ComputeShaderHandleView {
        self.shader
    }
}
impl<const I: usize> ComputePassResources<I> {
    pub const fn new(shader: ComputeShaderHandleView, images: [ImageTarget; I]) -> Self {
        Self { shader, images }
    }

    pub const fn image_targets(&self) -> &[ImageTarget; I] {
        &self.images
    }
}
