use ethel::{
    render::{
        Resolution,
        buffer::{InitStrategy, SingleBuffer, StorageSection},
        command::{DrawArraysIndirectCommand, GpuCommandDispatch},
    },
    shader::{GlslStruct, ShaderKind, ShaderProgram},
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

/// Returns an `RG32UI` [`RenderTarget`] for the geometry rasterization pass
/// color output.
pub fn geom_rasterize_target(
    resolution: Resolution,
    resolution_relative_scale: f32,
) -> RenderTarget {
    RenderTarget::new(
        "rendrs_target.geometry.rasterize",
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
    inner: DrawPass<GeomRasterizeCtxWrapper, 0, 2>,
    shader: ShaderGeomRasterize,
    cpy_shader: ComputeShaderGeomRasterCpyOpts,
    opts_buffer: SingleBuffer<DrawArraysIndirectCommand>, //more opts?
}
impl GeomRasterizePass {
    /// Expects an RG32UI `raster_out` color attachment, as returned by
    /// [`geom_rasterize_target`] and a depth attachment.
    pub fn new(raster_out: OutputObject, depth_out: OutputObject) -> Self {
        let shader = ShaderGeomRasterize::new_compiled();
        let cpy_shader = ComputeShaderGeomRasterCpyOpts::new_compiled();

        const DEFAULT_DRAW_CMD: DrawArraysIndirectCommand = DrawArraysIndirectCommand {
            count: 0,
            instance_count: 1,
            first_vertex: 0,
            base_instance: 0,
        };
        let handle_view = shader.handle().view();

        Self {
            shader,
            cpy_shader,
            opts_buffer: SingleBuffer::new(1, InitStrategy::FillWith(|| DEFAULT_DRAW_CMD)),
            inner: DrawPass::new(handle_view, [], [raster_out, depth_out], |_, ctx| {
                let GeomRasterizeCtx {
                    gbank,
                    shader,
                    cpy_shader,
                    opts_buffer,
                    m_proj,
                    m_view,
                } = ctx;

                gbank.bind_data_buffers();
                gbank.bind_gcounter_buffer();

                opts_buffer.bind_shader_storage(G_RASTER_SSBO_BIND_CPYOPTS, 0);
                cpy_shader.bind();
                cpy_shader.dispatch([1, 1, 1]);

                shader.bind();
                // ethel shader uniform interface seems to not
                // work (likely due to array/mat4 mismatch?)
                // shader.uniform_proj_mat_mat4v([*m_proj]);
                // shader.uniform_view_mat_mat4v([*m_view]);
                let l0 = shader.find_uniform_location("proj_mat");
                let l1 = shader.find_uniform_location("view_mat");
                unsafe {
                    janus::gl::UniformMatrix4fv(l0.get(), 1, janus::gl::FALSE, m_proj.as_ptr());
                    janus::gl::UniformMatrix4fv(l1.get(), 1, janus::gl::FALSE, m_view.as_ptr());
                }

                janus::gl::barrier_shader_storage();

                let cmd_view = unsafe {
                    opts_buffer.set_length(1);
                    opts_buffer.view()
                };
                GpuCommandDispatch::from_view(cmd_view).dispatch();
            }),
        }
    }

    pub fn output(&self) -> &OutputObject {
        self.inner.output(0)
    }

    pub fn revalidate(&mut self, render_pool: &RenderPool) {
        self.inner.revalidate(render_pool);
    }

    pub const fn shader(&self) -> &ShaderGeomRasterize {
        &self.shader
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
            cpy_shader: &self.cpy_shader,
            opts_buffer: &self.opts_buffer,
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
    pub cpy_shader: &'ctx ComputeShaderGeomRasterCpyOpts,
    pub opts_buffer: &'ctx SingleBuffer<DrawArraysIndirectCommand>,
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

        // assumes rg32ui color output
        unit ShaderKind::Pixel => [
            attribs {
                ethel::shader_glsl_attribs! {
                    input rd_TriangleID : uint as flat;
                    input rd_GeoID      : uint as flat;
                    output outColor     : uvec2;
                }
            };

            src() {
                "
                //todo: more metadata in g channel (tri-atts), bit-packing
                uint R = rd_TriangleID;
                uint G = rd_GeoID;

                outColor = uvec2(R, G);
                ";
            }
        ];
    }
}

macro_rules! ssbo_binding {
    (rendrs_GeomRasterCpyOpts_Outbuf) => {
        10
    };
}

pub const G_RASTER_SSBO_BIND_CPYOPTS: u32 = ssbo_binding!(rendrs_GeomRasterCpyOpts_Outbuf);

ethel::shader_glsl_compute! {
    struct GeomRasterCpyOpts > [460] {
        workgroup [1, 1, 1];

        type {
            TYPE_DRAWCMD_INDIRECT_ARRAYS
            TYPE_DRAWCMD_INDIRECT_ELEMENTS
        };
        ssbo {
            super::shader::SSBO_GBANK_GCOUNTER

            ethel::shader_glsl_ssbo! {
                buf rendrs_GeomRasterCpyOpts_Outbuf => {
                    DrawArraysIndirectCommand : out_cmd;
                }
            }
        };

        src() {
            "
            uint gc_vert = atomicExchange(rendrs_gbank_gcounter_vertex, 0u);
            uint gc_tris = atomicExchange(rendrs_gbank_gcounter_triangle, 0u);
            out_cmd.count = gc_vert;
            ";
        }
    }
}

pub const TYPE_DRAWCMD_INDIRECT_ARRAYS: GlslStruct =
    DrawArraysIndirectCommandGlslStruct::as_definition();
pub const TYPE_DRAWCMD_INDIRECT_ELEMENTS: GlslStruct =
    DrawElementsIndirectCommandGlslStruct::as_definition();

ethel::shader_glsl_struct! {
    struct DrawArraysIndirectCommand {
        count: u32 => uint,
        instance_count: u32 => uint,
        first_vertex: u32 => uint,
        base_instance: u32 => uint
    }
}

ethel::shader_glsl_struct! {
    struct DrawElementsIndirectCommand {
        count: u32 => uint,
        instance_count: u32 => uint,
        first_vertex: u32 => uint,
        base_vertex: i32 => int,
        base_instance: u32 => uint
    }
}
