use ethel::{
    render::{Resolution, buffer::StorageSection},
    shader::{ComputeShaderHandleView, ShaderHandleView, ShaderProgram},
};
use janus::{
    GlProperty, GpuResource,
    texture::{
        ImageFormat, ImageType, MipLevels, Tex, Texture, TextureFiltering, TextureKind, TextureView,
    },
};

use crate::framebuffer::{Framebuffer, FramebufferError, FramebufferView, HasFramebuffer};

#[derive(Clone, Copy, Debug)]
pub struct RenderTargetDescriptor {
    format: ImageFormat,
    pixel_type: ImageType,
    filtering: TextureFiltering,
    hardware_mip_levels: MipLevels,
    resolution_relative_scale: f32,
}
impl Default for RenderTargetDescriptor {
    fn default() -> Self {
        Self {
            format: ImageFormat::Rgb,
            pixel_type: ImageType::Bits8,
            filtering: TextureFiltering::Linear,
            hardware_mip_levels: MipLevels::default(),
            resolution_relative_scale: 1.0, // full resolution
        }
    }
}
impl RenderTargetDescriptor {
    pub const fn new(
        format: ImageFormat,
        pixel_type: ImageType,
        filtering: TextureFiltering,
        hardware_mip_levels: MipLevels,
        resolution_relative_scale: f32,
    ) -> Self {
        Self {
            format,
            pixel_type,
            filtering,
            hardware_mip_levels,
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

    pub const fn hardware_mip_levels(&self) -> MipLevels {
        self.hardware_mip_levels
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
        let texture = Texture::new_2d(
            resolution.0 as i32,
            resolution.1 as i32,
            descriptor.hardware_mip_levels,
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
            self.texture = Texture::new_2d(
                scaled_resolution.0 as i32,
                scaled_resolution.1 as i32,
                self.descriptor.hardware_mip_levels,
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

/// A uniform image object with compute shader access and layout metadata.
///
/// Also see [`ImageObject`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageObjectTarget {
    object: ImageObject,
    access: ImageAccessKind,
    unit: u32,
    layer: Option<u32>,
}
impl ImageObjectTarget {
    pub const fn new(
        object: ImageObject,
        access: ImageAccessKind,
        unit: u32,
        layer: Option<u32>,
    ) -> Self {
        Self {
            object,
            access,
            unit,
            layer,
        }
    }

    pub fn revalidate_if_pooled(&mut self, render_pool: &RenderPool) {
        self.object.revalidate_if_pooled(render_pool);
    }

    pub const fn is_pool_target(&self) -> bool {
        self.object.is_pool_target()
    }

    pub const fn is_direct_texture(&self) -> bool {
        self.object.is_direct_texture()
    }

    pub const fn accessor(&self) -> Option<RenderTargetAccessor> {
        self.object.accessor()
    }

    pub const fn accessor_mut(&mut self) -> Option<&mut RenderTargetAccessor> {
        self.object.accessor_mut()
    }

    pub const fn texture(&self) -> TextureView {
        self.object.texture()
    }

    pub const fn inner_image(&self) -> ImageObject {
        self.object
    }

    pub const fn inner_image_mut(&mut self) -> &mut ImageObject {
        &mut self.object
    }

    pub const fn unit(&self) -> u32 {
        self.unit
    }

    pub const fn layer(&self) -> Option<u32> {
        self.layer
    }

    pub const fn is_layer(&self) -> bool {
        self.layer.is_some()
    }

    pub fn bind(&self) {
        self.object.bind(self.unit, self.layer, self.access);
    }
}

/// A uniform image object used in compute passes.
///
/// Can either refer to a [`RenderPool`] target as [`ImageTarget::FromPool`]
/// or a to direct [`TextureView`] handle with [`ImageTarget::DirectTexture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageObject {
    PoolTarget(RenderTargetAccessor),
    DirectTexture(TextureView),
}
impl ImageObject {
    pub const fn from_pool_target(accessor: RenderTargetAccessor) -> Self {
        Self::PoolTarget(accessor)
    }

    pub const fn from_direct_texture(texture: TextureView) -> Self {
        Self::DirectTexture(texture)
    }

    pub fn revalidate_if_pooled(&mut self, render_pool: &RenderPool) {
        if let Self::PoolTarget(accessor) = self {
            accessor.revalidate(render_pool);
        }
    }

    pub const fn is_pool_target(&self) -> bool {
        matches!(self, Self::PoolTarget(_))
    }

    pub const fn is_direct_texture(&self) -> bool {
        matches!(self, Self::DirectTexture(_))
    }

    pub const fn accessor(&self) -> Option<RenderTargetAccessor> {
        match self {
            Self::PoolTarget(render_target_accessor) => Some(*render_target_accessor),
            Self::DirectTexture(_) => None,
        }
    }

    pub const fn accessor_mut(&mut self) -> Option<&mut RenderTargetAccessor> {
        match self {
            ImageObject::PoolTarget(render_target_accessor) => Some(render_target_accessor),
            ImageObject::DirectTexture(_) => None,
        }
    }

    pub const fn texture(&self) -> TextureView {
        match self {
            Self::PoolTarget(render_target_accessor) => render_target_accessor.texture,
            Self::DirectTexture(texture_view) => *texture_view,
        }
    }

    pub fn bind(&self, unit: u32, layer: Option<u32>, access: ImageAccessKind) {
        let layered = layer.is_none() as u8;
        let layer = layer.unwrap_or_default() as i32;
        let texture = self.texture().texture_id();
        let access = access.property_enum();
        let format = self.texture().metadata().internal_format();
        unsafe {
            janus::gl::BindImageTexture(
                unit,
                texture,
                0,
                layered,
                layer,
                access,
                format.glenum_internal_format(),
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageAccessKind {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}
impl janus::GlProperty for ImageAccessKind {
    fn property_enum(self) -> u32 {
        match self {
            ImageAccessKind::ReadOnly => janus::gl::READ_ONLY,
            ImageAccessKind::WriteOnly => janus::gl::WRITE_ONLY,
            ImageAccessKind::ReadWrite => janus::gl::READ_WRITE,
        }
    }
}

/// An uniform sampler object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplerObject(TextureView);
impl SamplerObject {
    pub fn from_texture(texture: impl Into<TextureView>) -> Self {
        Self(texture.into())
    }

    pub const fn new(texture: TextureView) -> Self {
        Self(texture)
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

pub trait CtxType {
    type Ctx<'ctx>;
}
impl CtxType for () {
    type Ctx<'ctx> = ();
}

#[macro_export]
macro_rules! context_wrapper {
    (for<$lt:lifetime> $context:ident) => {
        paste::paste! {
            #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
            pub struct [< $context Wrapper >];
            impl $crate::pipeline::CtxType for [< $context Wrapper >] {
                type Ctx<$lt> = $context<$lt>;
            }
        }
    };
    ($context:ident) => {
        paste::paste! {
            #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
            pub struct [< $context Wrapper >];
            impl $crate::pipeline::CtxType for [< $context Wrapper >] {
                type Ctx<'ctx> = $context;
            }
        }
    };
}

pub trait Pass<K: CtxType> {
    fn shader(&self) -> impl ShaderProgram;

    fn bind_shader(&self) {
        self.shader().bind();
    }

    fn revalidate(&mut self, render_pool: &RenderPool);

    fn execute(&self, frame_index: StorageSection, render_pool: &RenderPool, ctx: &K::Ctx<'_>);
}

#[derive(Debug)]
pub struct DrawPass<K: CtxType, const S: usize, const O: usize> {
    shader: ShaderHandleView,
    samplers: [SamplerObject; S],
    outputs: [OutputObject; O],
    dispatch: for<'ctx> fn(StorageSection, &K::Ctx<'ctx>),
    framebuffer: Option<Framebuffer>,
}
impl<K: CtxType, const S: usize, const O: usize> Pass<K> for DrawPass<K, S, O> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ShaderHandleView {
        self.shader
    }

    fn revalidate(&mut self, render_pool: &RenderPool) {
        if O == 0 {
            return;
        }
        if let Err(err) = self.revalidate_framebuffer(render_pool) {
            tracing::error!("failed to revalidate framebuffer: {err}");
        }
    }

    fn execute(&self, frame_index: StorageSection, _render_pool: &RenderPool, ctx: &K::Ctx<'_>) {
        self.bind_shader();
        self.bind_samplers();
        self.bind_framebuffer();
        (self.dispatch)(frame_index, ctx);
    }
}
impl<K: CtxType, const S: usize, const O: usize> DrawPass<K, S, O> {
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
        dispatch: fn(StorageSection, &K::Ctx<'_>),
    ) -> Self {
        Self {
            shader,
            samplers,
            outputs,
            dispatch,
            framebuffer: None,
        }
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
            let fb_size = outputs.get(0).map(Tex::size).unwrap_or((1, 1));

            let depth_i = self
                .outputs
                .iter()
                .position(|output| matches!(output, OutputObject::Depth(_)));

            if O > 0 {
                if let Some(depth_i) = depth_i {
                    if O == 1 {
                        // no color attachments, the only output was depth
                        // default/null textures are ignored
                        outputs = [TextureView::null(TextureKind::Dim2D); O];
                    } else {
                        // shift elements after depth to the left to preserve
                        // color outputs order, then set depth to null
                        outputs[depth_i..].rotate_left(1);
                        outputs[O - 1] = TextureView::null(TextureKind::Dim2D);
                    }
                }
            }

            let depth_output = depth_i.map(|i| outputs[i]);
            (outputs, fb_size, depth_output)
        };

        let framebuffer = Framebuffer::new(fb_size.0 as u32, fb_size.1 as u32, &colors, depth)?;
        framebuffer.set_default_buffers_state();
        self.framebuffer = Some(framebuffer);
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
            texture.bind(unit);
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
pub struct ComputePass<K: CtxType, const S: usize, const I: usize> {
    shader: ComputeShaderHandleView,
    samplers: [SamplerObject; S],
    images: [ImageObjectTarget; I],
    pre_dispatch: for<'ctx> fn(StorageSection, &K::Ctx<'ctx>) -> [u32; 3],
}
impl<K: CtxType, const S: usize, const I: usize> Pass<K> for ComputePass<K, S, I> {
    #[allow(refining_impl_trait)]
    fn shader(&self) -> ComputeShaderHandleView {
        self.shader
    }

    fn revalidate(&mut self, render_pool: &RenderPool) {
        self.images
            .iter_mut()
            .for_each(|image| image.revalidate_if_pooled(render_pool));
    }

    fn execute(&self, frame_index: StorageSection, _render_pool: &RenderPool, ctx: &K::Ctx<'_>) {
        self.bind_shader();
        self.bind_samplers();
        self.bind_images();
        let workgroups = (self.pre_dispatch)(frame_index, ctx);
        self.shader.dispatch_compute(workgroups);
    }
}
impl<K: CtxType, const S: usize, const I: usize> ComputePass<K, S, I> {
    pub const fn new(
        shader: ComputeShaderHandleView,
        samplers: [SamplerObject; S],
        images: [ImageObjectTarget; I],
        pre_dispatch: fn(StorageSection, &K::Ctx<'_>) -> [u32; 3],
    ) -> Self {
        Self {
            shader,
            samplers,
            images,
            pre_dispatch,
        }
    }

    pub fn bind_samplers(&self) {
        self.samplers.iter().enumerate().for_each(|(i, sampler)| {
            let unit = i as u32;
            let texture = sampler.texture();
            texture.bind(unit);
        });
    }

    pub const fn samplers(&self) -> &[SamplerObject; S] {
        &self.samplers
    }

    pub fn sampler(&self, index: usize) -> &SamplerObject {
        &self.samplers[index]
    }

    pub fn bind_images(&self) {
        self.images.iter().for_each(ImageObjectTarget::bind);
    }

    pub fn image_target(&self, index: usize) -> &ImageObjectTarget {
        &self.images[index]
    }

    pub const fn image_targets(&self) -> &[ImageObjectTarget; I] {
        &self.images
    }
}

#[derive(Debug)]
struct BlitPassCtx;
impl CtxType for BlitPassCtx {
    type Ctx<'ctx> = ();
}

#[derive(Debug)]
pub struct BlitPass {
    inner: DrawPass<BlitPassCtx, 0, 0>,
    source: RenderTargetAccessor,
}
impl BlitPass {
    pub fn new(source: RenderTargetAccessor) -> Self {
        Self {
            source,
            inner: DrawPass::new(ShaderHandleView::default(), [], [], |_, _| {}),
        }
    }

    pub fn revalidate_source(&mut self, render_pool: &RenderPool) {
        self.source.revalidate(render_pool);
    }

    pub const fn source(&self) -> &RenderTargetAccessor {
        &self.source
    }

    pub const fn source_mut(&mut self) -> &mut RenderTargetAccessor {
        &mut self.source
    }

    pub const fn set_source(&mut self, source: RenderTargetAccessor) -> RenderTargetAccessor {
        std::mem::replace(&mut self.source, source)
    }

    pub fn execute(&self, render_pool: &RenderPool) {
        self.inner.execute(StorageSection::Front, render_pool, &());
        let framebuffer = self.inner.framebuffer().unwrap();
        framebuffer.set_read_buffer(Some(0));

        let read_framebuffer = framebuffer.resource_id();
        let (w, h) = self.source.texture.size();

        unsafe {
            janus::gl::BlitNamedFramebuffer(
                read_framebuffer,
                0,
                0,
                0,
                w,
                h,
                0,
                0,
                w,
                h,
                janus::gl::COLOR_BUFFER_BIT,
                janus::gl::NEAREST,
            );
        }
    }
}
