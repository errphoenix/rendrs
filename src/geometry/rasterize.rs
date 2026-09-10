use ethel::{
    render::{
        Resolution,
        buffer::{InitStrategy, SingleBuffer, StorageSection},
        command::DrawElementsIndirectCommand,
    },
    shader::{GlslLib, GlslStruct, ShaderKind},
};
use janus::{
    GpuResource,
    texture::{ImageFormat, ImageType, MipLevels, TextureFiltering},
};

use crate::{
    DrawPass,
    geometry::GeometryBank,
    pack::{PACK_OCTAHEDRON_DECODE, PACK_OCTAHEDRON_ENCODE, PACK_OCTAHEDRON_WRAP_UTIL},
    pipeline::{OutputObject, Pass, RenderPool, RenderTarget, RenderTargetDescriptor},
};

/// Issues a `shader_storage` and `atomic_counter` memory barrier.
pub fn barrier_geom_compose() {
    unsafe {
        janus::gl::MemoryBarrier(
            janus::gl::SHADER_STORAGE_BARRIER_BIT
                | janus::gl::ATOMIC_COUNTER_BARRIER_BIT
                | janus::gl::ELEMENT_ARRAY_BARRIER_BIT,
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
    opts_buffer: SingleBuffer<DrawElementsIndirectCommand>, //more opts?
}
impl GeomRasterizePass {
    /// Expects an RG32UI `raster_out` color attachment, as returned by
    /// [`geom_rasterize_target`] and a depth attachment.
    pub fn new(raster_out: OutputObject, depth_out: OutputObject) -> Self {
        let shader = ShaderGeomRasterize::new_compiled();
        let cpy_shader = ComputeShaderGeomRasterCpyOpts::new_compiled();

        const DEFAULT_DRAW_CMD: DrawElementsIndirectCommand = DrawElementsIndirectCommand {
            count: 0,
            instance_count: 1,
            first_vertex: 0,
            base_vertex: 0,
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
                shader.uniform_proj_mat_mat4v([*m_proj]);
                shader.uniform_view_mat_mat4v([*m_view]);

                janus::gl::barrier_shader_storage();
                janus::gl::barrier_commands();

                gbank.bind_index_buffer();

                unsafe {
                    janus::gl::BindBuffer(
                        janus::gl::DRAW_INDIRECT_BUFFER,
                        opts_buffer.resource_id(),
                    );
                    janus::gl::MultiDrawElementsIndirect(
                        janus::gl::TRIANGLES,
                        janus::gl::UNSIGNED_INT,
                        std::ptr::null(),
                        1,
                        0,
                    );
                }
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

    /// Expects a VAO bound with a valid EBO for the rasterizing geometry to
    /// be bound.
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
    pub opts_buffer: &'ctx SingleBuffer<DrawElementsIndirectCommand>,
    pub m_proj: [f32; 16],
    pub m_view: [f32; 16],
}
crate::context_wrapper!(for<'ctx> GeomRasterizeCtx);

ethel::shader_glsl! {
    struct GeomRasterize > [460] {
        common {};

        unit ShaderKind::Vertex => [
            uniform {
                length 1, proj_mat : mat4 => [f32; 16];
                length 1, view_mat : mat4 => [f32; 16];
            };
            type {
                crate::geometry::shader::TYPE_RENDERVERTEX
            };
            ssbo {
                crate::geometry::shader::SSBO_GBANK_RENDERVERTEX
            };

            src() {
                "
                RenderVertex vertex = rendrs_gbank_vertex[gl_VertexID];

                vec3 P_model = vec3(vertex.pos_x, vertex.pos_y, vertex.pos_z);
                vec4 P_world = proj_mat * view_mat * vec4(P_model, 1.0);

                gl_Position = P_world;
                ";
            }
        ];

        // assumes rg32ui color output
        unit ShaderKind::Pixel => [
            attribs {
                ethel::shader_glsl_attribs! {
                    output outColor : uvec2;
                }
            };
            type {
                crate::geometry::shader::TYPE_TRIANGLE_ATTRIBS
            };
            ssbo {
                crate::geometry::shader::SSBO_GBANK_TRIANGLE_ATTRIBS
            };

            src() {
                "
                TriangleAttribs tri_attribs = rendrs_gbank_triangle_attribs[gl_PrimitiveID];

                //todo: more metadata in g channel (tri-atts), bit-packing
                uint R = gl_PrimitiveID;
                uint G = tri_attribs.geometry_id;

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
                    DrawElementsIndirectCommand : out_cmd;
                }
            }
        };

        src() {
            "
            uint gc_vert = atomicExchange(rendrs_gbank_gcounter_vertex, 0u);
            uint gc_tris = atomicExchange(rendrs_gbank_gcounter_triangle, 0u);
            out_cmd.count = gc_tris * 3;
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

pub const ATTRIB_INTERP_SAMPLER_UNIT_RASTER: u32 = 0;
pub const ATTRIB_INTERP_IMAGE_BIND_FRAME: u32 = 0;
pub const ATTRIB_INTERP_IMAGE_BIND_GRADS: u32 = 1;

ethel::shader_glsl_compute! {
    struct AttribsInterp > [460] {
        workgroup [8, 8, 1];

        uniform {
            length 1, proj_mat : mat4 => [f32; 16];
            length 1, view_mat : mat4 => [f32; 16];
        };
        sampler {
            on ATTRIB_INTERP_SAMPLER_UNIT_RASTER => tex_raster : usampler2D;
        };
        image {
            on ATTRIB_INTERP_IMAGE_BIND_FRAME => ima_space : image2D as rgba16  writeonly;
            on ATTRIB_INTERP_IMAGE_BIND_GRADS => ima_grads : image2D as rgba16f writeonly;
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
        lib {
            PACK_OCTAHEDRON_WRAP_UTIL;
            PACK_OCTAHEDRON_ENCODE;
            PACK_OCTAHEDRON_DECODE;
            INTERNAL_UTIL_ATTR_INTERP_PROJECT_TO_SCREEN;
        };

        src() {
            "
            ivec2 px = ivec2(gl_GlobalInvocationID.xy);
            uvec2 raster_data = texelFetch(tex_raster, px, 0).rg;

            uint Tid = raster_data.x;
            uint Gid = raster_data.y;

            uint b_tri[3] = rendrs_gbank_triangle[Tid];
            RenderVertex w_v0 = rendrs_gbank_vertex[b_tri[0]];
            RenderVertex w_v1 = rendrs_gbank_vertex[b_tri[1]];
            RenderVertex w_v2 = rendrs_gbank_vertex[b_tri[2]];
            vec2 b_uv0 = vec2(w_v0.uv_x, w_v0.uv_y);
            vec2 b_uv1 = vec2(w_v1.uv_x, w_v1.uv_y);
            vec2 b_uv2 = vec2(w_v2.uv_x, w_v2.uv_y);

            uvec2 resolution = uvec2(textureSize(tex_raster));
            mat4 MVP = proj_mat * view_mat;

            vec2 s_v0 = _rendrs_Project_ScreenSpace(MVP, resolution, w_v0);
            vec2 s_v1 = _rendrs_Project_ScreenSpace(MVP, resolution, w_v1);
            vec2 s_v2 = _rendrs_Project_ScreenSpace(MVP, resolution, w_v2);

            vec2 px_c = vec2(px) + 0.5;
            float inv_det = 1.0 / ((s_v1.x - s_v0.x) * (s_v2.y - s_v0.y) - (s_v2.x - s_v0.x) * (s_v1.y - s_v0.y));
            float B_v = ((px_c.x - s_v0.x) * (s_v2.y - s_v0.y) - (px_c.y - s_v0.y) * (s_v2.x - s_v0.x)) * inv_det;
            float B_w = ((s_v1.x - s_v0.x) * (px_c.y - s_v0.y) - (s_v1.y - s_v0.y) * (px_c.x - s_v0.x)) * inv_det;
            float B_u = 1.0 - B_v - B_w;

            vec2 b_n0e = vec2(w_v0.norm_oct_x, w_v0.norm_oct_y);
            vec3 b_n0  = rendrs_unpackOctahedron(b_n0e);
            vec2 b_n1e = vec2(w_v1.norm_oct_x, w_v1.norm_oct_y);
            vec3 b_n1  = rendrs_unpackOctahedron(b_n1e);
            vec2 b_n2e = vec2(w_v2.norm_oct_x, w_v2.norm_oct_y);
            vec3 b_n2  = rendrs_unpackOctahedron(b_n2e);
            vec3 N  = b_n0 * B_u + b_n1 * B_v + b_n2 * B_w;
            vec2 Ne = rendrs_packOctahedron(N);
            imageStore(ima_space, px, vec4(Ne.x, Ne.y, B_u, B_v));

            vec2 dUv1 = b_uv1 - b_uv0;
            vec2 dUv2 = b_uv2 - b_uv0;
            vec2 ddxUv = (dUv1 * (s_v2.y - s_v0.y) - dUv2 * (s_v1.y - s_v0.y)) * inv_det;
            vec2 ddyUv = (dUv2 * (s_v1.x - s_v0.x) - dUv1 * (s_v2.x - s_v0.x)) * inv_det;
            imageStore(ima_grads, px, vec4(ddxUv.x, ddxUv.y, ddyUv.x, ddyUv.y));
            ";
        }
    }
}

/// Helper function to extract barycentric-weights from the `frame/space`
/// image target produced by the deferred attribute interpolation pass.
///
/// Creates the `rendrs_GetBWeights` function, which takes the `vec4` color
/// queried from the relevant image target.
///
/// The function will retrieve the last `BA` components of the sample,
/// which correspond to `u` and `v` barycentric weightrs, in addition to
/// reconstructing the `w` weight via the formula `w = 1 - u - v` and returns
/// the result as a `vec3` in the standard order `uvw`.
const LIB_UTIL_GET_BWEIGHTS: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_GetBWeights[
        s_space : vec4
    ] => "
        vec2 B_uv = s_space.zw;
        float B_w = 1.0 - B_uv.x - B_uv.y;
        return vec3(B_uv, B_w);
    "
};

/// Helper function for barycentric vertex attribute interpolation.
///
/// Creates the `rendrs_InterpAttrib` function, which takes the following
/// arguments:
/// * the vertex attributes `a0`, `a1`, `a2`, which must be any of the types
///   `float`, `vec2`, `vec3`, `vec4`. The same type must be used for all 3
///   parameters.
/// * the barycentric weights `vec3` as obtained from [`rendrs_GetBWeights`]
///
/// [`rendrs_GetBWeights`]: LIB_UTIL_GET_BWEIGHTS
const LIB_INTERP_ATTRIB: GlslLib = GlslLib::new(
    "
    float rendrs_InterpAttrib(float a0, float a1, float a2, vec3 w) {
        return a0 * w.x + a1 * w.y + a2 * w.z;
    }
    vec2 rendrs_InterpAttrib(vec2 a0, vec2 a1, vec2 a2, vec3 w) {
        return a0 * w.x + a1 * w.y + a2 * w.z;
    }
    vec3 rendrs_InterpAttrib(vec3 a0, vec3 a1, vec3 a2, vec3 w) {
        return a0 * w.x + a1 * w.y + a2 * w.z;
    }
    vec4 rendrs_InterpAttrib(vec4 a0, vec4 a1, vec4 a2, vec4 w) {
        return a0 * w.x + a1 * w.y + a2 * w.z;
    }
    ",
);

const INTERNAL_UTIL_ATTR_INTERP_PROJECT_TO_SCREEN: GlslLib = ethel::shader_glsl_lib! {
    vec2 _rendrs_Project_ScreenSpace[
        MVP        : mat4,
        resolution : uvec2,
        point      : vec3
    ] => "
        vec4 C   = MVP * vec4(point, 1.0);
        vec2 ndc = C.xy / C.w;
        return vec2(
            (ndc.x * 0.5 + 0.5) * float(resolution.x),
            (ndc.y * 0.5 + 0.5) * float(resolution.y)
        );
    "
};
