use ethel::{
    render::Resolution,
    shader::{ComputeShaderHandleView, ShaderHandleView, ShaderProgram},
};
use janus::texture::{
    ImageFormat, ImageType, Texture, TextureFiltering, TextureKey, TextureTarget, TextureView,
};

use crate::framebuffer::{Framebuffer, FramebufferView};

#[cfg(feature = "batching")]
pub mod batch;
#[cfg(feature = "framebuffer")]
pub mod framebuffer;

#[cfg(feature = "batching")]
pub const BATCH_UNITS: usize = batch::PER_BATCH_UNITS;

pub const PASS_READ_UNITS: usize = 16;
pub const PASS_WRITE_UNITS: usize = framebuffer::MAX_ATTACHMENTS;

#[derive(Clone, Copy, Debug)]
pub struct RenderTargetDescriptor {
    format: ImageFormat,
    pixel_type: ImageType,
    filtering: TextureFiltering,
    resolution_relative_scale: f32,
}
impl Default for RenderTargetDescriptor {
    fn default() -> Self {
        Self {
            format: ImageFormat::Rgb,
            pixel_type: ImageType::Bits8,
            filtering: TextureFiltering::Linear,
            resolution_relative_scale: 1.0, // full resolution
        }
    }
}
impl RenderTargetDescriptor {
    pub const fn new(
        format: ImageFormat,
        pixel_type: ImageType,
        filtering: TextureFiltering,
        resolution_relative_scale: f32,
    ) -> Self {
        Self {
            format,
            pixel_type,
            filtering,
            resolution_relative_scale,
        }
    }

    pub const fn format(&self) -> ImageFormat {
        self.format
    }

    pub const fn pixel_type(&self) -> ImageType {
        self.pixel_type
    }

    pub const fn filtering(&self) -> TextureFiltering {
        self.filtering
    }

    pub const fn resolution_relative_scale(&self) -> f32 {
        self.resolution_relative_scale
    }
}

#[derive(Debug)]
pub struct RenderTarget {
    label: &'static str,
    descriptor: RenderTargetDescriptor,
    texture: Texture,
    cached_resolution: (u32, u32),
}
impl RenderTarget {
    pub fn new(
        label: &'static str,
        descriptor: RenderTargetDescriptor,
        resolution: Resolution,
    ) -> Self {
        janus::debug_assert_gl!();

        let resolution = Self::scale_resolution(descriptor.resolution_relative_scale, resolution);
        let texture = Texture::empty(
            resolution.0 as i32,
            resolution.1 as i32,
            descriptor.pixel_type,
            descriptor.format,
        );

        Self {
            label,
            descriptor,
            texture,
            cached_resolution: resolution,
        }
    }

    pub fn resize(&mut self, new_resolution: Resolution) {
        let scaled_resolution =
            Self::scale_resolution(self.descriptor.resolution_relative_scale, new_resolution);

        if scaled_resolution != self.cached_resolution {
            self.cached_resolution = scaled_resolution;
            self.texture = Texture::empty(
                scaled_resolution.0 as i32,
                scaled_resolution.1 as i32,
                self.descriptor.pixel_type,
                self.descriptor.format,
            );
        }
    }

    fn scale_resolution(scale: f32, resolution: Resolution) -> (u32, u32) {
        (
            ((resolution.width * scale).round() as u32).max(1),
            ((resolution.height * scale).round() as u32).max(1),
        )
    }

    pub fn view(&self) -> TextureView {
        self.texture.view()
    }

    pub fn cached_resolution(&self) -> (u32, u32) {
        self.cached_resolution
    }

    pub fn label(&self) -> &'static str {
        self.label
    }

    pub fn descriptor(&self) -> RenderTargetDescriptor {
        self.descriptor
    }
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
