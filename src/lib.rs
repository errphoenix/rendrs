use ethel::{
    render::{Resolution, buffer::StorageSection},
    shader::{ComputeShaderHandleView, ShaderHandleView, ShaderProgram},
};
use janus::texture::{
    ImageFormat, ImageType, Texture, TextureFiltering, TextureTarget, TextureView,
};

use crate::framebuffer::{Framebuffer, FramebufferError, FramebufferView, HasFramebuffer};

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

/// Resolution dependant render output buffer.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderTargetId(u32);

/// An view into a [`RenderTarget`] from [`RenderPool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderTargetAccessor {
    id: RenderTargetId,
    texture: TextureView,
}
impl RenderTargetAccessor {
    pub const fn id(&self) -> RenderTargetId {
        self.id
    }

    pub const fn texture(&self) -> TextureView {
        self.texture
    }

    pub fn revalidate(&mut self, render_pool: &RenderPool) {
        *self = render_pool
            .accessor(self.id())
            .expect("accessor's render target must exist in pool");
    }
}

/// A global collection of [`render targets`](RenderTarget).
#[derive(Debug, Default)]
pub struct RenderPool {
    targets: Vec<RenderTarget>,
}
impl RenderPool {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            targets: Vec::with_capacity(capacity),
        }
    }

    pub fn with_targets<const N: usize>(targets: [RenderTarget; N]) -> (Self, [RenderTargetId; N]) {
        let targets = {
            let mut vec = Vec::with_capacity(N);
            for target in targets {
                vec.push(target);
            }
            vec
        };
        let ids = std::array::from_fn(|i| RenderTargetId(i as u32));
        (Self { targets }, ids)
    }

    /// Revalidate all targets with a new `resolution`.
    ///
    /// Each target will only be revalidated if the resolution has
    /// effectively changed from last time.
    pub fn revalidate_targets(&mut self, resolution: Resolution) {
        self.targets
            .iter_mut()
            .for_each(|target| target.resize(resolution));
    }

    pub fn add(&mut self, target: RenderTarget) -> RenderTargetId {
        let id = RenderTargetId(self.targets.len() as u32);
        self.targets.push(target);
        id
    }

    pub fn get(&self, id: RenderTargetId) -> Option<&RenderTarget> {
        self.targets.get(id.0 as usize)
    }

    pub fn get_mut(&mut self, id: RenderTargetId) -> Option<&mut RenderTarget> {
        self.targets.get_mut(id.0 as usize)
    }

    pub fn accessor(&self, id: RenderTargetId) -> Option<RenderTargetAccessor> {
        let texture = self.get(id)?.view();
        Some(RenderTargetAccessor { id, texture })
    }
}

/// todo: refactor like [`OutputObject`]
///
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

/// An uniform sampler object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplerObject(TextureView);
impl SamplerObject {
    pub fn new(texture: impl Into<TextureView>) -> Self {
        Self(texture.into())
    }

    pub const fn texture(&self) -> TextureView {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OutputObject {
    Color(RenderTargetAccessor),
    Depth(RenderTargetAccessor),
}
impl OutputObject {
    pub fn color(target: RenderTargetAccessor) -> Self {
        Self::Color(target)
    }

    pub fn depth(target: RenderTargetAccessor) -> Self {
        Self::Depth(target)
    }

    pub const fn accessor(&self) -> RenderTargetAccessor {
        match self {
            OutputObject::Color(render_target_accessor) => *render_target_accessor,
            OutputObject::Depth(render_target_accessor) => *render_target_accessor,
        }
    }

    pub fn revalidate(&mut self, render_pool: &RenderPool) {
        self.accessor().revalidate(render_pool);
    }

    pub const fn texture(&self) -> TextureView {
        self.accessor().texture
    }

    pub const fn target_id(&self) -> RenderTargetId {
        self.accessor().id
    }
}

pub trait Pass<Ctx> {
    fn shader(&self) -> impl ShaderProgram;

    fn bind_shader(&self) {
        self.shader().bind();
    }

    fn execute<F: Fn(StorageSection, &Ctx)>(
        &mut self,
        frame_index: StorageSection,
        render_pool: &RenderPool,
        ctx: &Ctx,
        submit: F,
    );
}

#[derive(Debug)]
pub struct DrawPass<const S: usize, const O: usize> {
    shader: ShaderHandleView,
    samplers: [SamplerObject; S],
    outputs: [OutputObject; O],
    framebuffer: Option<Framebuffer>,
    valid: bool,
}
impl<Ctx, const S: usize, const O: usize> Pass<Ctx> for DrawPass<S, O> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ShaderHandleView {
        self.shader
    }

    fn execute<F: Fn(StorageSection, &Ctx)>(
        &mut self,
        frame_index: StorageSection,
        render_pool: &RenderPool,
        ctx: &Ctx,
        submit: F,
    ) {
        if !self.is_valid() {
            if let Err(err) = self.revalidate_framebuffer(render_pool) {
                tracing::error!("failed to revalidate framebuffer: {err}");
            }
        }

        Pass::<Ctx>::bind_shader(self);
        self.bind_samplers();
        self.bind_framebuffer();

        submit(frame_index, ctx);
    }
}
impl<const S: usize, const O: usize> DrawPass<S, O> {
    /// Initialize resource descriptions for a draw-pass.
    ///
    /// This does not yet create a full `Framebuffer`: it will be initialized
    /// lazily when needed, i.e. on the first execution.
    ///
    /// The `outputs` described can be multiple [`OutputObject::Color`]
    /// variants, but up to only one (optional) [`OutputObject::Depth`].
    pub const fn new(
        shader: ShaderHandleView,
        samplers: [SamplerObject; S],
        outputs: [OutputObject; O],
    ) -> Self {
        Self {
            shader,
            samplers,
            outputs,
            framebuffer: None,
            valid: false,
        }
    }

    /// Invalidate draw-pass framebuffer (e.g. due to resolution changes).
    ///
    /// This will cause a recreation of the framebuffer on the next executon.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn revalidate_framebuffer(
        &mut self,
        render_pool: &RenderPool,
    ) -> Result<(), FramebufferError> {
        self.outputs
            .iter_mut()
            .for_each(|output| output.revalidate(render_pool));

        let (colors, fb_size, depth) = {
            // includes depth and must be explicitly ignored later
            let mut outputs: [TextureView; O] = std::array::from_fn(|i| self.outputs[i].texture());

            // since all attachments must have the same size, any will do
            let fb_size = outputs.get(0).map(TextureView::size).unwrap_or((1, 1));

            let depth_i = self
                .outputs
                .iter()
                .position(|output| matches!(output, OutputObject::Depth(_)));

            if O > 0 {
                if let Some(depth_i) = depth_i {
                    if O == 1 {
                        // no color attachments, the only output was depth
                        // default/null textures are ignored
                        outputs = [TextureView::default(); O];
                    } else {
                        if depth_i == O {
                            // its last, just set to default and go on
                            outputs[depth_i] = TextureView::default();
                        } else {
                            // shift elements after depth to the left to preserve
                            // color outputs order, then set depth to null
                            outputs[depth_i..].rotate_left(1);
                            outputs[O - 1] = TextureView::default();
                        }
                    }
                }
            }

            let depth_output = depth_i.map(|i| outputs[i]);
            (outputs, fb_size, depth_output)
        };

        let framebuffer = Framebuffer::new(fb_size.0 as u32, fb_size.1 as u32, &colors, depth)?;
        framebuffer.set_default_buffers_state();
        self.framebuffer = Some(framebuffer);
        self.valid = true;
        Ok(())
    }

    pub fn bind_framebuffer(&self) {
        if let Some(fb) = &self.framebuffer {
            fb.bind();
        }
    }

    pub fn bind_samplers(&self) {
        self.samplers.iter().enumerate().for_each(|(i, sampler)| {
            let unit = i as u32;
            let texture = sampler.texture();
            janus::texture::bind(TextureTarget::Flat, texture, unit);
        });
    }

    pub const fn samplers(&self) -> &[SamplerObject; S] {
        &self.samplers
    }

    pub fn sampler(&self, index: usize) -> &SamplerObject {
        &self.samplers[index]
    }

    pub fn outputs(&self) -> &[OutputObject; O] {
        &self.outputs
    }

    pub fn output(&self, index: usize) -> &OutputObject {
        &self.outputs[index]
    }

    pub fn output_mut(&mut self, index: usize) -> &mut OutputObject {
        &mut self.outputs[index]
    }

    /// Returns `None` if the framebuffer is not initialized.
    ///
    /// The framebuffer is always initialized after the first execution, but
    /// it may not be valid if it has been invalidated before the next
    /// execution.
    pub fn framebuffer(&self) -> Option<&Framebuffer> {
        self.framebuffer.as_ref()
    }

    /// See [`Self::framebuffer`].
    pub fn framebuffer_view(&self) -> Option<FramebufferView> {
        self.framebuffer.as_ref().map(Framebuffer::as_view)
    }
}

#[derive(Debug)]
pub struct ComputePass<const S: usize, const I: usize> {
    shader: ComputeShaderHandleView,
    samplers: [SamplerObject; S],
    images: [ImageTarget; I],
}
impl<Ctx, const S: usize, const I: usize> Pass<Ctx> for ComputePass<S, I> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ComputeShaderHandleView {
        self.shader
    }

    fn execute<F: Fn(StorageSection, &Ctx)>(
        &mut self,
        frame_index: StorageSection,
        render_pool: &RenderPool,
        ctx: &Ctx,
        submit: F,
    ) {
        Pass::<Ctx>::bind_shader(self);
        self.bind_samplers();
        self.bind_images(render_pool);
        submit(frame_index, ctx);
    }
}
impl<const S: usize, const I: usize> ComputePass<S, I> {
    pub const fn new(
        shader: ComputeShaderHandleView,
        samplers: [SamplerObject; S],
        images: [ImageTarget; I],
    ) -> Self {
        Self {
            shader,
            samplers,
            images,
        }
    }

    pub fn bind_samplers(&self) {
        self.samplers.iter().enumerate().for_each(|(i, sampler)| {
            let unit = i as u32;
            let texture = sampler.texture();
            janus::texture::bind(TextureTarget::Flat, texture, unit);
        });
    }

    pub const fn samplers(&self) -> &[SamplerObject; S] {
        &self.samplers
    }

    pub fn sampler(&self, index: usize) -> &SamplerObject {
        &self.samplers[index]
    }

    pub fn bind_images(&mut self, _render_pool: &RenderPool) {
        todo!()
    }

    pub fn image_target(&self, index: usize) -> &ImageTarget {
        &self.images[index]
    }

    pub const fn image_targets(&self) -> &[ImageTarget; I] {
        &self.images
    }
}
