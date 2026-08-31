use ethel::{
    render::buffer::{SingleBuffer, StorageSection, ViewMut},
    shader::{ComputeShader, ShaderProgram},
};

use crate::{
    ComputePass,
    geometry::DomainData,
    pipeline::{CtxType, ImageObjectTarget, Pass, RenderPool, Sampler},
};

#[derive(Debug)]
pub struct DomainDataWriter<'buf> {
    data: ViewMut<'buf, DomainData>,
    write_len: usize,
}
impl<'buf> DomainDataWriter<'buf> {
    pub const unsafe fn new(ssbo: &'buf SingleBuffer<DomainData>) -> Self {
        let view = unsafe { ssbo.view_mut() };
        Self {
            data: view,
            write_len: 0,
        }
    }

    pub const fn write(&mut self, data: DomainData) -> bool {
        let index = self.write_len;

        if index as u32 >= super::MAX_DOMAIN_COUNT {
            return false;
        }

        unsafe {
            self.data.as_mut_ptr().add(index).write(data);
        }
        self.write_len += 1;
        true
    }

    pub const fn len(&self) -> usize {
        self.write_len
    }

    pub const fn is_empty(&self) -> bool {
        self.write_len == 0
    }
}

pub struct GeomPass<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> {
    shader: CS,
    inner_pass: ComputePass<K, S, I>,
    domain_data: SingleBuffer<DomainData>,
    pre_dispatch: fn(StorageSection, &<K as CtxType>::Ctx<'_>, &mut DomainDataWriter),
}
impl<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> std::fmt::Debug
    for GeomPass<CS, K, S, I>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeomPass")
            .field("shader", &self.shader)
            .field("inner_pass", &self.inner_pass)
            .field("domain_data", &self.domain_data)
            .field("pre_dispatch", &self.pre_dispatch)
            .finish()
    }
}
impl<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> Pass<K>
    for GeomPass<CS, K, S, I>
{
    fn shader(&self) -> impl ShaderProgram {
        self.shader.compute_handle().view()
    }

    fn revalidate(&mut self, render_pool: &RenderPool) {
        self.inner_pass.revalidate(render_pool);
    }

    fn execute(
        &self,
        frame_index: StorageSection,
        _render_pool: &RenderPool,
        ctx: &<K as CtxType>::Ctx<'_>,
    ) {
        self.bind_shader();
        self.bind_samplers();
        self.bind_images();

        let mut writer = unsafe { DomainDataWriter::new(&self.domain_data) };
        (self.pre_dispatch)(frame_index, ctx, &mut writer);
        let domain_count = writer.len();
        if domain_count < 1 {
            return;
        }
        if domain_count as u32 >= super::MAX_DOMAIN_COUNT {
            tracing::error!(
                "failed to dispatch geometry job: too many domains {}",
                domain_count
            );
            return;
        }

        self.domain_data
            .bind_shader_storage(super::SSBO_BINDING_DOMAINS, 0);

        self.shader
            .compute_handle()
            .dispatch_compute([domain_count as u32, 1, 1]);
    }
}
impl<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> GeomPass<CS, K, S, I> {
    pub fn new(
        shader: CS,
        samplers: [Sampler; S],
        images: [ImageObjectTarget; I],
        pre_dispatch: fn(StorageSection, &<K as CtxType>::Ctx<'_>, &mut DomainDataWriter),
    ) -> Self {
        let handle_view = shader.compute_handle().view();
        Self {
            shader,
            inner_pass: ComputePass::new(
                handle_view,
                samplers,
                images,
                |_, _| [0, 0, 0], // ignored
            ),
            domain_data: SingleBuffer::zeroed(super::MAX_DOMAIN_COUNT as usize),
            pre_dispatch,
        }
    }

    pub fn inner_pass(&self) -> &ComputePass<K, S, I> {
        &self.inner_pass
    }

    pub fn bind_samplers(&self) {
        self.inner_pass.bind_samplers();
    }

    pub const fn samplers(&self) -> &[Sampler; S] {
        self.inner_pass.samplers()
    }

    pub fn sampler(&self, index: usize) -> &Sampler {
        self.inner_pass.sampler(index)
    }

    pub fn bind_images(&self) {
        self.inner_pass.bind_images();
    }

    pub fn image_target(&self, index: usize) -> &ImageObjectTarget {
        self.inner_pass.image_target(index)
    }

    pub const fn image_targets(&self) -> &[ImageObjectTarget; I] {
        self.inner_pass.image_targets()
    }
}
