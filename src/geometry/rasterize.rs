use ethel::{
    render::{Resolution, buffer::StorageSection},
    shader::ShaderKind,
};
use janus::texture::{ImageFormat, ImageType, MipLevels, TextureFiltering};

use crate::{
    DrawPass,
    geometry::GeometryBank,
    pipeline::{OutputObject, Pass, RenderPool, RenderTarget, RenderTargetDescriptor},
};

/// Issues a `shader_storage` and `atomic_counter` memory barrier.
pub fn barrier_geom_compose() {
    unsafe {
        janus::gl::MemoryBarrier(
            janus::gl::SHADER_STORAGE_BARRIER_BIT | janus::gl::ATOMIC_COUNTER_BARRIER_BIT,
        );
    }
}

/// Issues a `framebuffer` memory barrier.
pub fn barrier_geom_rasterize() {
    janus::gl::barrier_framebuffers();
}

pub fn geom_rasterize_target(
    resolution: Resolution,
    resolution_relative_scale: f32,
) -> RenderTarget {
    RenderTarget::new(
        "rendrs_target.geometry_rasterize",
        RenderTargetDescriptor::new(
            ImageFormat::DualChannelInteger,
            ImageType::Integer32U,
            TextureFiltering::Nearest,
            MipLevels::default(),
            resolution_relative_scale,
        ),
        resolution,
    )
}

#[derive(Debug)]
pub struct GeomRasterizePass {
    inner: DrawPass<GeomRasterizeCtxWrapper, 0, 1>,
    shader: ShaderGeomRasterize,
}
impl GeomRasterizePass {
    /// Expects an RG32UI `output` color attachment, as returned by
    /// [`geom_rasterize_target`].
    pub fn new(output: OutputObject) -> Self {
        let shader = ShaderGeomRasterize::new_compiled();
        Self {
            inner: DrawPass::new(shader.handle().view(), [], [output], |_, ctx| {
                let GeomRasterizeCtx {
                    gbank,
                    shader,
                    m_proj,
                    m_view,
                } = ctx;

                shader.uniform_proj_mat_mat4v([*m_proj]);
                shader.uniform_view_mat_mat4v([*m_view]);

                // SAFETY: safe access to the gcounter ssbo is only guaranteed
                // if the geometry composition pass has finished writing to it.
                // This is the caller's responsability.
                let gcounter = unsafe { gbank.gcounter_buffer().view() };
                let [v_counter, _t_counter] = gcounter[0];

                gbank.bind_data_buffers();

                unsafe {
                    janus::gl::DrawArrays(janus::gl::TRIANGLES, 0, v_counter as i32);
                }
            }),
            shader,
        }
    }

    pub fn execute(
        &self,
        render_pool: &RenderPool,
        gbank: &GeometryBank,
        #[cfg(feature = "glam")] m_proj: glam::Mat4,
        #[cfg(feature = "glam")] m_view: glam::Mat4,
        #[cfg(not(feature = "glam"))] m_proj: [f32; 16],
        #[cfg(not(feature = "glam"))] m_view: [f32; 16],
    ) {
        let ctx = GeomRasterizeCtx {
            shader: &self.shader,
            gbank,
            #[cfg(feature = "glam")]
            m_proj: m_proj.to_cols_array(),
            #[cfg(feature = "glam")]
            m_view: m_view.to_cols_array(),
            #[cfg(not(feature = "glam"))]
            m_proj,
            #[cfg(not(feature = "glam"))]
            m_view,
        };
        // storage section is ignored
        self.inner.execute(StorageSection::Back, render_pool, &ctx);
    }
}

#[derive(Debug)]
pub struct GeomRasterizeCtx<'ctx> {
    pub gbank: &'ctx GeometryBank,
    pub shader: &'ctx ShaderGeomRasterize,
    pub m_proj: [f32; 16],
    pub m_view: [f32; 16],
}
crate::context_wrapper!(for<'ctx> GeomRasterizeCtx);

ethel::shader_glsl! {
    struct GeomRasterize > [460] {
        common {};

        unit ShaderKind::Vertex => [
            attribs {
                ethel::shader_glsl_attribs! {
                    output rd_TriangleID : uint as flat;
                    output rd_GeoID      : uint as flat;
                }
            };

            uniform {
                length 1, proj_mat : mat4 => [f32; 16];
                length 1, view_mat : mat4 => [f32; 16];
            };
            type {
                crate::geometry::shader::TYPE_RENDERVERTEX
                crate::geometry::shader::TYPE_TRIANGLE_ATTRIBS
            };
            ssbo {
                crate::geometry::shader::SSBO_GBANK_RENDERVERTEX
                crate::geometry::shader::SSBO_GBANK_TRIANGLE
                crate::geometry::shader::SSBO_GBANK_TRIANGLE_ATTRIBS
            };

            src() {
                "
                uint t_i = gl_VertexID / 3;
                uint v_i = gl_VertexID % 3;

                uint tri[3] = rendrs_gbank_triangle[t_i];
                TriangleAttribs tri_attribs = rendrs_gbank_triangle_attribs[t_i];

                uint vert_i = tri[v_i];
                RenderVertex vertex = rendrs_gbank_vertex[vert_i];

                vec3 P_model = vec3(vertex.pos_x, vertex.pos_y, vertex.pos_z);
                vec4 P_world = proj_mat * view_mat * vec4(P_model, 1.0);

                rd_TriangleID = t_i;
                rd_GeoID = tri_attribs.geometry_id;

                gl_Position = P_world;
                ";
            }
        ];

        // assumes rg32 output
        unit ShaderKind::Pixel => [
            attribs {
                ethel::shader_glsl_attribs! {
                    input rd_TriangleID : uint as flat;
                    input rd_GeoID      : uint as flat;
                    output outColor     : uvec4;
                }
            };

            src() {
                "
                //todo: more metadata in g channel (tri-atts), bit-packing
                uint R = rd_TriangleID;
                uint G = rd_GeoID;

                outColor = uvec4(R, G, 0, 0);
                ";
            }
        ];
    }
}
