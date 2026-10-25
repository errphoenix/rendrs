use std::cell::{RefCell, RefMut};

use ethel::{
    render::buffer::{SingleBuffer, StorageSection},
    shader::{ComputeShader, ShaderProgram},
};

use crate::{
    ComputePass,
    geometry::DomainData,
    pipeline::{CtxType, ImageObjectTarget, Pass, RenderPool, Sampler},
};

#[derive(Debug)]
pub struct DomainDataWriter<'buf> {
    local_buffer: RefMut<'buf, Vec<DomainData>>,
    dst_buf: &'buf SingleBuffer<DomainData>,
    // in number of elements
    write_len: usize,
}
impl<'buf> DomainDataWriter<'buf> {
    pub unsafe fn new(
        ssbo: &'buf SingleBuffer<DomainData>,
        local_buffer: &'buf RefCell<Vec<DomainData>>,
    ) -> Self {
        Self {
            local_buffer: local_buffer.borrow_mut(),
            dst_buf: ssbo,
            write_len: 0,
        }
    }

    pub fn blit(&mut self, data: &[DomainData]) {
        let len = data.len();
        self.local_buffer.extend_from_slice(data);
        self.write_len = len;
    }

    pub fn write(&mut self, data: DomainData) -> bool {
        let index = self.write_len;
        if index as u32 >= super::MAX_DOMAIN_COUNT {
            return false;
        }
        self.local_buffer.push(data);
        self.write_len += 1;
        true
    }

    pub fn upload(&mut self) {
        self.dst_buf.write(&self.local_buffer, 0);
        self.local_buffer.clear();
    }

    pub const fn len(&self) -> usize {
        self.write_len
    }

    pub const fn is_empty(&self) -> bool {
        self.write_len == 0
    }
}

pub type GeomDispatchFn<CS, K> =
    fn(StorageSection, &CS, &<K as CtxType>::Ctx<'_>, &mut DomainDataWriter);

#[derive(Debug)]
pub struct GeomPass<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> {
    shader: CS,
    inner_pass: ComputePass<K, S, I>,
    domain_data: SingleBuffer<DomainData>,
    local_domain_buffer: RefCell<Vec<DomainData>>,
    pre_dispatch: GeomDispatchFn<CS, K>,
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

    /// Caller must ensure the [`GeometryBank`]'s SSBO bindings are active
    /// and that this pass will not override any binding from 0 to 4.
    fn execute(
        &self,
        frame_index: StorageSection,
        _render_pool: &RenderPool,
        ctx: &<K as CtxType>::Ctx<'_>,
    ) {
        self.bind_shader();
        self.bind_samplers();
        self.bind_images();
        let mut writer =
            unsafe { DomainDataWriter::new(&self.domain_data, &self.local_domain_buffer) };
        let shader = &self.shader;
        (self.pre_dispatch)(frame_index, shader, ctx, &mut writer);
        let domain_count = writer.len();
        if domain_count < 1 {
            return;
        }
        if domain_count as u32 >= super::MAX_DOMAIN_COUNT {
            tracing::error!(
                "failed to dispatch geometry composition job: too many domains {}",
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
        pre_dispatch: GeomDispatchFn<CS, K>,
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
            local_domain_buffer: RefCell::new(Vec::new()),
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
