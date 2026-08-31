use ethel::{
    render::buffer::{SingleBuffer, StorageSection},
    shader::{ComputeShader, ShaderProgram},
};

use crate::{
    ComputePass,
    geometry::DomainData,
    pipeline::{CtxType, ImageObjectTarget, Pass, RenderPool, Sampler},
};

pub struct GeomPass<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> {
    shader: CS,
    inner_pass: ComputePass<K, S, I>,
    domain_data: SingleBuffer<DomainData>,
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
        render_pool: &RenderPool,
        ctx: &<K as CtxType>::Ctx<'_>,
    ) {
        self.bind_shader();
        self.bind_samplers();
        self.bind_images();

        //todo
        //let workgroups = (self.pre_dispatch)(frame_index, ctx);
        //self.shader.dispatch_compute(workgroups);

        todo!()
    }
}
impl<CS: ComputeShader, K: CtxType, const S: usize, const I: usize> GeomPass<CS, K, S, I> {
    pub fn new(
        shader: CS,
        samplers: [Sampler; S],
        images: [ImageObjectTarget; I],
        /*todo pre_dispatch: fn(StorageSection, &K::Ctx<'_>),*/
    ) -> Self {
        let handle_view = shader.compute_handle().view();
        Self {
            shader,
            inner_pass: ComputePass::new(
                handle_view,
                samplers,
                images,
                |_, _| [0, 0, 0], // overriden
            ),
            domain_data: SingleBuffer::zeroed(super::MAX_DOMAIN_COUNT as usize),
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
