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
    ComputePass, DrawPass,
    geometry::{GeometryBank, HasTriangleBuffers, HasVertexBuffers},
    graphics::{PixelResolution, UTIL_DERIVE_COTANGENT_GRAD},
    pack::{PACK_OCTAHEDRON_DECODE, PACK_OCTAHEDRON_ENCODE, PACK_OCTAHEDRON_WRAP_UTIL},
    pipeline::{
        CtxType, ImageAccessKind, ImageObject, ImageObjectTarget, OutputObject, Pass, RenderPool,
        RenderTarget, RenderTargetDescriptor,
    },
};

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

pub fn geom_attribs_framespace_target(
    resolution: Resolution,
    resolution_relative_scale: f32,
) -> RenderTarget {
    RenderTarget::new(
        "rendrs_target.geometry.attribs.frame",
        RenderTargetDescriptor::new(
            ImageFormat::Rgba,
            ImageType::Bits16,
            TextureFiltering::Nearest,
            MipLevels::default(),
            resolution_relative_scale,
        ),
        resolution,
    )
}

pub fn geom_attribs_gradients_target(
    resolution: Resolution,
    resolution_relative_scale: f32,
) -> RenderTarget {
    RenderTarget::new(
        "rendrs_target.geometry.attribs.gradients",
        RenderTargetDescriptor::new(
            ImageFormat::Rgba,
            ImageType::Float16,
            TextureFiltering::Nearest,
            MipLevels::default(),
            resolution_relative_scale,
        ),
        resolution,
    )
}

#[derive(Debug)]
pub struct GeomRasterizePass<V: HasVertexBuffers, T: HasTriangleBuffers> {
    inner: DrawPass<GeomRasterizeCtxWrapper<V, T>, 0, 2>,
    variant: ShaderGeomRasterizeVariants,
    shader: ShaderGeomRasterize,
    cpy_shader: ComputeShaderGeomRasterCpyOpts,
    opts_buffer: SingleBuffer<DrawElementsIndirectCommand>, //more opts?
}
impl<V: HasVertexBuffers, T: HasTriangleBuffers> GeomRasterizePass<V, T> {
    /// Expects an RG32UI `raster_out` color attachment, as returned by
    /// [`geom_rasterize_target`] and a depth attachment.
    pub fn new(
        variant: ShaderGeomRasterizeVariants,
        raster_out: OutputObject,
        depth_out: OutputObject,
    ) -> Self {
        let shader = ShaderGeomRasterize::new_compiled_variant(variant);

        let cpy_variant = if matches!(variant, ShaderGeomRasterizeVariants::Instanced) {
            ComputeShaderGeomRasterCpyOptsVariants::Instanced
        } else {
            ComputeShaderGeomRasterCpyOptsVariants::Default
        };
        let cpy_shader = ComputeShaderGeomRasterCpyOpts::new_compiled_variant(cpy_variant);

        const DEFAULT_DRAW_CMD: DrawElementsIndirectCommand = DrawElementsIndirectCommand {
            count: 0,
            instance_count: 0,
            first_vertex: 0,
            base_vertex: 0,
            base_instance: 0,
        };
        let handle_view = shader.handle().view();

        const COMMAND_CAP: usize = super::MAX_INSTANCELIST_COUNT as usize + 1;

        Self {
            shader,
            variant,
            cpy_shader,
            opts_buffer: SingleBuffer::new(
                COMMAND_CAP,
                InitStrategy::FillWith(|| DEFAULT_DRAW_CMD),
            ),
            inner: DrawPass::new(handle_view, [], [raster_out, depth_out], |_, ctx| {
                let GeomRasterizeCtx {
                    gbank,
                    shader,
                    variant,
                    cpy_shader,
                    opts_buffer,
                    m_proj,
                    m_view,
                } = ctx;

                gbank
                    .vertex_buffers()
                    .bind_positions(G_RASTER_SSBO_BIND_VERTEX_POSITIONS);
                gbank
                    .vertex_buffers()
                    .bind_normals(G_RASTER_SSBO_BIND_VERTEX_NORMALS);
                gbank
                    .vertex_buffers()
                    .bind_uvs(G_RASTER_SSBO_BIND_VERTEX_UVS);
                gbank
                    .triangle_buffers()
                    .bind_indices(G_RASTER_SSBO_BIND_TRIANGLE_INDICES);
                gbank
                    .triangle_buffers()
                    .bind_attribs(G_RASTER_SSBO_BIND_TRIANGLE_ATTRIBS);

                gbank.bind_gcounter_buffer();
                gbank.bind_instancing_buffers();

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
                    match variant {
                        Instanced => janus::gl::MultiDrawElementsIndirect(
                            janus::gl::TRIANGLES,
                            janus::gl::UNSIGNED_INT,
                            size_of::<DrawElementsIndirectCommand>() as *const _,
                            super::MAX_INSTANCELIST_COUNT as i32,
                            0,
                        ),
                        _ => janus::gl::MultiDrawElementsIndirect(
                            janus::gl::TRIANGLES,
                            janus::gl::UNSIGNED_INT,
                            std::ptr::null(),
                            1,
                            0,
                        ),
                    }
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
        gbank: &GeometryBank<V, T>,
        #[cfg(feature = "glam")] m_proj: glam::Mat4,
        #[cfg(feature = "glam")] m_view: glam::Mat4,
        #[cfg(not(feature = "glam"))] m_proj: [f32; 16],
        #[cfg(not(feature = "glam"))] m_view: [f32; 16],
    ) {
        let ctx = GeomRasterizeCtx {
            shader: &self.shader,
            variant: self.variant,
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
pub struct GeomRasterizeCtx<'ctx, V: HasVertexBuffers, T: HasTriangleBuffers> {
    pub shader: &'ctx ShaderGeomRasterize,
    pub variant: ShaderGeomRasterizeVariants,
    pub gbank: &'ctx GeometryBank<V, T>,
    pub cpy_shader: &'ctx ComputeShaderGeomRasterCpyOpts,
    pub opts_buffer: &'ctx SingleBuffer<DrawElementsIndirectCommand>,
    pub m_proj: [f32; 16],
    pub m_view: [f32; 16],
}
#[derive(Debug)]
pub struct GeomRasterizeCtxWrapper<V: HasVertexBuffers, T: HasTriangleBuffers> {
    _marker: std::marker::PhantomData<(V, T)>,
}
impl<V: HasVertexBuffers, T: HasTriangleBuffers> CtxType for GeomRasterizeCtxWrapper<V, T> {
    type Ctx<'ctx> = GeomRasterizeCtx<'ctx, V, T>;
}

macro_rules! ssbo_binding {
    (rendrs_Geom_Rasterize_VertexPositions) => {
        0
    };
    // 2 reserved for gcounters
    (rendrs_Geom_Rasterize_VertexNormals) => {
        1
    };
    (rendrs_Geom_Rasterize_VertexUvs) => {
        3
    };
    // 4 reserved for instancing data
    (rendrs_Geom_Rasterize_TriangleIndices) => {
        5
    };
    (rendrs_Geom_Rasterize_TriangleAttribs) => {
        6
    };
    (rendrs_Geom_Rasterize_CpyOptsOut) => {
        10
    };
}

pub const G_RASTER_SSBO_BIND_VERTEX_POSITIONS: u32 =
    ssbo_binding!(rendrs_Geom_Rasterize_VertexPositions);
pub const G_RASTER_SSBO_BIND_VERTEX_NORMALS: u32 =
    ssbo_binding!(rendrs_Geom_Rasterize_VertexNormals);
pub const G_RASTER_SSBO_BIND_VERTEX_UVS: u32 = ssbo_binding!(rendrs_Geom_Rasterize_VertexUvs);
pub const G_RASTER_SSBO_BIND_TRIANGLE_INDICES: u32 =
    ssbo_binding!(rendrs_Geom_Rasterize_TriangleIndices);
pub const G_RASTER_SSBO_BIND_TRIANGLE_ATTRIBS: u32 =
    ssbo_binding!(rendrs_Geom_Rasterize_TriangleAttribs);
pub const G_RASTER_SSBO_BIND_CPYOPTS: u32 = ssbo_binding!(rendrs_Geom_Rasterize_CpyOptsOut);
pub const G_RASTER_SSBO_BIND_INSTANCE_DATA: u32 = super::SSBO_BINDING_GBANK_INSTANCE_DATA;

use crate::graphics::LIB_QUATERNION_MUL;
use crate::graphics::LIB_QUATERNION_MUL_VEC;
use ShaderGeomRasterizeVariants::*;

ethel::shader_glsl! {
    struct GeomRasterize > [460] {
        common {
            variants {
                Batched;
                Instanced;
            };
        };

        unit ShaderKind::Vertex => [
            attribs {
                ethel::shader_glsl_attribs! {
                    output instanceId : uint as flat;
                }
            };
            uniform {
                length 1, proj_mat : mat4 => [f32; 16];
                length 1, view_mat : mat4 => [f32; 16];

                length 1, instance_command_offset : uint => u32;
            };
            type {
                super::TYPE_INSTANCELIST
                super::TYPE_INSTANCE_TRANSFORM
            };
            ssbo {
                // gbank ssbos bound using runtime arrays as there is no
                // concern about ssbo binding limit here
                ethel::shader_glsl_ssbo! {
                    buf rendrs_Geom_Rasterize_VertexPositions => {
                        [dyn_array float : rendrs_vertex_positions => each 3]
                    }
                }
                super::SSBO_GBANK_INSTANCING_DATA
            };
            lib {
                GlslLib::new("");
                >Instanced => LIB_QUATERNION_MUL;
                >Instanced => LIB_QUATERNION_MUL_VEC;
            };

            src() {
                "
                float position[] = rendrs_vertex_positions[gl_VertexID];
                vec3 P_model = vec3(position[0], position[1], position[2]);
                vec4 P_world = vec4(P_model, 1.0);
                ";
                match {
                    Instanced => {
                        "
                        uint draw_id    = gl_DrawID - instance_command_offset;
                        InstanceList il = rendrs_gbank_instance_lists[draw_id];
                        uint instance_base = il.instance_base_count >> 16;
                        uint instance_id   = gl_InstanceID;

                        InstanceTransform transform = rendrs_gbank_instance_transforms[instance_id];
                        float t_scaling  = transform.s;
                        vec4  t_rotation = vec4(transform.qx, transform.qy, transform.qz, transform.qw);
                        vec3  t_position = vec3(transform.px, transform.py, transform.pz);

                        P_world.xyz *= t_scaling;
                        P_world.xyz  = rendrs_QuaternionMul(P_world.xyz, t_rotation);
                        P_world.xyz += t_position;

                        instanceId = instance_id;
                        ";
                    };
                }
                "
                P_world = proj_mat * view_mat * P_world;
                gl_Position = P_world;
                ";
            }
        ];

        // assumes rg32ui color output
        unit ShaderKind::Pixel => [
            attribs {
                ethel::shader_glsl_attribs! {
                    input instanceId : uint as flat;
                    output outColor  : uvec2;
                }
            };
            type {
                super::TYPE_TRIANGLE_ATTRIBS
            };
            ssbo {
                // gbank ssbos bound using runtime arrays as there is no
                // concern about ssbo binding limit here
                ethel::shader_glsl_ssbo! {
                    buf rendrs_Geom_Rasterize_TriangleAttribs => {
                        [dyn_array TriangleAttribs : rendrs_triangle_attribs]
                    }
                }
            };

            src() {
                "
                TriangleAttribs tri_attribs = rendrs_triangle_attribs[gl_PrimitiveID];

                // full 32 bits for triangle index
                uint R = gl_PrimitiveID + 1;

                // 16 bits for optional instance id
                // 1  bit  for 'is instanced' flag
                // 15 bits for geometry id

                #define G_MASK_15B 0x7fff

                const uint geometry_id = tri_attribs.geometry_id;

                ";
                match {
                    Instanced => "
                        const uint is_instanced = 1 << 15;
                        const uint instance_id  = instanceId << 16;
                    ";
                    Batched => "
                        const uint is_instanced = 0;
                        const uint instance_id  = 0;
                    ";
                }
                "

                uint G = instance_id | is_instanced | (geometry_id & G_MASK_15B);

                outColor = uvec2(R, G);
                ";
            }
        ];
    }
}

ethel::shader_glsl_compute! {
    struct GeomRasterCpyOpts > [460] {
        workgroup [1, 1, 1];

        type {
            TYPE_DRAWCMD_INDIRECT_ARRAYS
            TYPE_DRAWCMD_INDIRECT_ELEMENTS
            super::TYPE_INSTANCELIST
            super::TYPE_INSTANCE_TRANSFORM
        };
        ssbo {
            super::SSBO_GBANK_GCOUNTER
            super::SSBO_GBANK_INSTANCING_DATA

            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_CpyOptsOut => {
                    [dyn_array DrawElementsIndirectCommand : out_cmd]
                }
            }
        };
        variants {
            Batched;
            Instanced;
        };

        src() {
            match {
                Instanced => {
                    "
                    atomicExchange(rendrs_gbank_gcounter_instance, 0u);
                    const uint gc_instancelist = atomicExchange(rendrs_gbank_gcounter_instancelist, 0u);

                    //0 reserved to batch draw command
                    for (uint i = 0; i < gc_instancelist; ++i) {
                        const InstanceList list   = rendrs_gbank_instance_lists[i];
                        const uint instance_base  = list.instance_base_count >> 16;
                        const uint instance_count = list.instance_base_count & 0x0000ffff;
                        out_cmd[i + 1] = DrawElementsIndirectCommand(
                            list.tri_count * 3,
                            instance_count,
                            list.tri_base * 3,
                            0,
                            instance_base
                        );
                    }

                    // set rest to 0
                    #define INSTANCELIST_CAP 1024
                    const uint rest = INSTANCELIST_CAP - gc_instancelist;
                    if (rest > 0) {
                        for (uint i = gc_instancelist; i < INSTANCELIST_CAP; ++i) {
                            out_cmd[i + 1].instance_count = 0;
                        }
                    }
                    ";
                };
                _ => {
                    "
                    uint gc_vert = atomicExchange(rendrs_gbank_gcounter_vertex, 0u);
                    uint gc_tris = atomicExchange(rendrs_gbank_gcounter_triangle, 0u);
                    //0 reserved to batch draw command
                    out_cmd[0].count = gc_tris * 3;
                    out_cmd[0].instance_count = 1;
                    ";
                };
            }
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

#[derive(Debug)]
pub struct AttribInterpolationCtx<'ctx, V: HasVertexBuffers, T: HasTriangleBuffers> {
    pub shader: &'ctx ComputeShaderAttribsInterp,
    pub gbank: &'ctx GeometryBank<V, T>,
    pub resolution: PixelResolution,
    pub m_proj: [f32; 16],
    pub m_view: [f32; 16],
}
#[derive(Debug)]
pub struct AttribInterpolationCtxWrapper<V: HasVertexBuffers, T: HasTriangleBuffers> {
    _marker: std::marker::PhantomData<(V, T)>,
}
impl<V: HasVertexBuffers, T: HasTriangleBuffers> CtxType for AttribInterpolationCtxWrapper<V, T> {
    type Ctx<'ctx> = AttribInterpolationCtx<'ctx, V, T>;
}

#[derive(Debug)]
pub struct AttribInterpolationPass<V: HasVertexBuffers, T: HasTriangleBuffers> {
    inner: ComputePass<AttribInterpolationCtxWrapper<V, T>, 0, 3>,
    shader: ComputeShaderAttribsInterp,
}
impl<V: HasVertexBuffers, T: HasTriangleBuffers> AttribInterpolationPass<V, T> {
    /// Expects the geometry raster target (generated by the rasterization
    /// pass) as an input image object, and the output framespace and output
    /// gradients targets as created, respectively, by
    /// [`geom_attribs_framespace_target`] and [`geom_attribs_gradients_target`].
    pub fn new(
        in_raster: ImageObject,
        out_framespace: ImageObject,
        out_gradients: ImageObject,
    ) -> Self {
        let shader = ComputeShaderAttribsInterp::new_compiled();
        let handle_view = shader.compute_handle().view();

        let in_raster = ImageObjectTarget::new(
            in_raster,
            ImageAccessKind::ReadOnly,
            ATTRIB_INTERP_IMAGE_BIND_RASTER,
            None,
        );
        let out_framespace = ImageObjectTarget::new(
            out_framespace,
            ImageAccessKind::WriteOnly,
            ATTRIB_INTERP_IMAGE_BIND_FRAME,
            None,
        );
        let out_gradients = ImageObjectTarget::new(
            out_gradients,
            ImageAccessKind::WriteOnly,
            ATTRIB_INTERP_IMAGE_BIND_GRADS,
            None,
        );

        Self {
            shader,
            inner: ComputePass::new(
                handle_view,
                [],
                [in_raster, out_framespace, out_gradients],
                |_, ctx| {
                    let AttribInterpolationCtx {
                        shader,
                        gbank,
                        resolution,
                        m_proj,
                        m_view,
                    } = ctx;

                    gbank
                        .vertex_buffers()
                        .bind_positions(G_RASTER_SSBO_BIND_VERTEX_POSITIONS);
                    gbank
                        .vertex_buffers()
                        .bind_normals(G_RASTER_SSBO_BIND_VERTEX_NORMALS);
                    gbank
                        .vertex_buffers()
                        .bind_uvs(G_RASTER_SSBO_BIND_VERTEX_UVS);
                    gbank
                        .triangle_buffers()
                        .bind_indices(G_RASTER_SSBO_BIND_TRIANGLE_INDICES);
                    gbank
                        .triangle_buffers()
                        .bind_attribs(G_RASTER_SSBO_BIND_TRIANGLE_ATTRIBS);

                    let wg_x = resolution.width().div_ceil(8);
                    let wg_y = resolution.height().div_ceil(8);

                    shader.uniform_resolution_uvec2v([[resolution.width(), resolution.height()]]);
                    shader.uniform_proj_mat_mat4v([*m_proj]);
                    shader.uniform_view_mat_mat4v([*m_view]);

                    [wg_x, wg_y, 1]
                },
            ),
        }
    }

    pub const fn shader(&self) -> &ComputeShaderAttribsInterp {
        &self.shader
    }

    pub const fn inner(&self) -> &ComputePass<AttribInterpolationCtxWrapper<V, T>, 0, 3> {
        &self.inner
    }

    pub const fn inner_mut(
        &mut self,
    ) -> &mut ComputePass<AttribInterpolationCtxWrapper<V, T>, 0, 3> {
        &mut self.inner
    }

    pub fn input_raster(&self) -> &ImageObjectTarget {
        self.inner.image_target(0)
    }

    pub fn output_framespace(&self) -> &ImageObjectTarget {
        self.inner.image_target(1)
    }

    pub fn output_gradients(&self) -> &ImageObjectTarget {
        self.inner.image_target(2)
    }

    pub fn revalidate(&mut self, render_pool: &RenderPool) {
        self.inner.revalidate(render_pool);
    }

    pub fn execute(
        &self,
        render_pool: &RenderPool,
        resolution: PixelResolution,
        geometry_bank: &GeometryBank<V, T>,
        #[cfg(feature = "glam")] m_proj: glam::Mat4,
        #[cfg(feature = "glam")] m_view: glam::Mat4,
        #[cfg(not(feature = "glam"))] m_proj: [f32; 16],
        #[cfg(not(feature = "glam"))] m_view: [f32; 16],
    ) {
        let ctx = AttribInterpolationCtx {
            shader: &self.shader,
            gbank: geometry_bank,
            resolution,
            #[cfg(feature = "glam")]
            m_proj: m_proj.to_cols_array(),
            #[cfg(feature = "glam")]
            m_view: m_view.to_cols_array(),
            #[cfg(not(feature = "glam"))]
            m_proj,
            #[cfg(not(feature = "glam"))]
            m_view,
        };
        self.inner.execute(StorageSection::Back, render_pool, &ctx);
    }
}

pub const ATTRIB_INTERP_IMAGE_BIND_RASTER: u32 = 3;
pub const ATTRIB_INTERP_IMAGE_BIND_FRAME: u32 = 4;
pub const ATTRIB_INTERP_IMAGE_BIND_GRADS: u32 = 5;

ethel::shader_glsl_compute! {
    struct AttribsInterp > [460] {
        workgroup [8, 8, 1];

        uniform {
            length 1, proj_mat   : mat4 => [f32; 16];
            length 1, view_mat   : mat4 => [f32; 16];
            length 1, resolution : uvec2 => [u32; 2];
        };
        image {
            on ATTRIB_INTERP_IMAGE_BIND_RASTER => ima_raster : uimage2D as rg32ui   readonly;
            on ATTRIB_INTERP_IMAGE_BIND_FRAME => ima_space   : image2D  as rgba16  writeonly;
            on ATTRIB_INTERP_IMAGE_BIND_GRADS => ima_grads   : image2D  as rgba16f writeonly;
        };
        type {
            super::TYPE_TRIANGLE_ATTRIBS
            super::TYPE_INSTANCELIST
            super::TYPE_INSTANCE_TRANSFORM
        };
        ssbo {
            // gbank ssbos bound using runtime arrays as there is no
            // concern about ssbo binding limit here
            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_VertexPositions => {
                    [dyn_array float : rendrs_vertex_positions => each 3]
                }
            }
            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_VertexNormals => {
                    [dyn_array float : rendrs_vertex_normals => each 2]
                }
            }
            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_VertexUvs => {
                    [dyn_array float : rendrs_vertex_uvs => each 2]
                }
            }
            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_TriangleIndices => {
                    [dyn_array uint : rendrs_triangle_indices => each 3]
                }
            }
            ethel::shader_glsl_ssbo! {
                buf rendrs_Geom_Rasterize_TriangleAttribs => {
                    [dyn_array TriangleAttribs : rendrs_triangle_attribs]
                }
            }
            super::SSBO_GBANK_INSTANCING_DATA
        };
        lib {
            PACK_OCTAHEDRON_WRAP_UTIL;
            PACK_OCTAHEDRON_ENCODE;
            PACK_OCTAHEDRON_DECODE;
            UTIL_DERIVE_COTANGENT_GRAD;
            INTERNAL_UTIL_ATTR_INTERP_PROJECT_TO_SCREEN;
            LIB_QUATERNION_MUL;
            LIB_QUATERNION_MUL_VEC;
        };

        src() {
            "
            ivec2 px = ivec2(gl_GlobalInvocationID.xy);

            if (px.x >= resolution.x || px.y >= resolution.y) return;
            uvec2 raster_data = imageLoad(ima_raster, px).rg;

            // full 32 bits for triangle index
            uint Tid = raster_data.x;

            // 16 bits for optional instance id
            // 1  bit  for 'is instanced' flag
            // 15 bits for geometry id
            #define G_MASK_15B 0x7fff
            uint G   = raster_data.y;
            const uint G16br = G & uint(0x0000ffff);
            const uint G16bl = G & uint(0xffff0000);

            const bool is_inst = bool(G16br >> 15);
            const uint inst_id = G16bl;

            uint Gid = G16br & G_MASK_15B;

            if (Tid == 0) {
                imageStore(ima_space, px, vec4(0.0));
                imageStore(ima_grads, px, vec4(0.0));
                return;
            }
            Tid -= 1; //0 is a valid index but rasterizer offsets valid tris to 1

            uint b_tri[3] = rendrs_triangle_indices[Tid];
            float b_p0[3] = rendrs_vertex_positions[b_tri[0]];
            float b_p1[3] = rendrs_vertex_positions[b_tri[1]];
            float b_p2[3] = rendrs_vertex_positions[b_tri[2]];
            vec3 p0 = vec3(b_p0[0], b_p0[1], b_p0[2]);
            vec3 p1 = vec3(b_p1[0], b_p1[1], b_p1[2]);
            vec3 p2 = vec3(b_p2[0], b_p2[1], b_p2[2]);

            if (is_inst) {
                InstanceTransform transform = rendrs_gbank_instance_transforms[inst_id];
                float t_scaling  = transform.s;
                vec4  t_rotation = vec4(transform.qx, transform.qy, transform.qz, transform.qw);
                vec3  t_position = vec3(transform.px, transform.py, transform.pz);

                p0 *= t_scaling;
                p1 *= t_scaling;
                p2 *= t_scaling;
                p0  = rendrs_QuaternionMul(p0, t_rotation);
                p1  = rendrs_QuaternionMul(p1, t_rotation);
                p2  = rendrs_QuaternionMul(p2, t_rotation);
                p0 += t_position;
                p1 += t_position;
                p2 += t_position;
            }

            const mat4 VP = proj_mat * view_mat;

            vec2 s_v0 = _rendrs_Project_ScreenSpace(VP, resolution, p0);
            vec2 s_v1 = _rendrs_Project_ScreenSpace(VP, resolution, p1);
            vec2 s_v2 = _rendrs_Project_ScreenSpace(VP, resolution, p2);

            vec2 px_c = vec2(px) + 0.5;
            float inv_det = 1.0 / ((s_v1.x - s_v0.x) * (s_v2.y - s_v0.y) - (s_v2.x - s_v0.x) * (s_v1.y - s_v0.y));
            float B_v = ((px_c.x - s_v0.x) * (s_v2.y - s_v0.y) - (px_c.y - s_v0.y) * (s_v2.x - s_v0.x)) * inv_det;
            float B_w = ((s_v1.x - s_v0.x) * (px_c.y - s_v0.y) - (s_v1.y - s_v0.y) * (px_c.x - s_v0.x)) * inv_det;
            float B_u = 1.0 - B_v - B_w;

            float w_p0 = (VP * vec4(p0, 1.0)).w;
            float w_p1 = (VP * vec4(p1, 1.0)).w;
            float w_p2 = (VP * vec4(p2, 1.0)).w;
            float iw0 = 1.0 / w_p0;
            float iw1 = 1.0 / w_p1;
            float iw2 = 1.0 / w_p2;

            float y_0 = B_u * iw0;
            float y_1 = B_v * iw1;
            float y_2 = B_w * iw2;
            float y_sum = y_0 + y_1 + y_2;
            float iy_sum = 1.0 / y_sum;
            B_u = y_0 * iy_sum;
            B_v = y_1 * iy_sum;
            B_w = y_2 * iy_sum;

            float b_u0[2] = rendrs_vertex_uvs[b_tri[0]];
            float b_u1[2] = rendrs_vertex_uvs[b_tri[1]];
            float b_u2[2] = rendrs_vertex_uvs[b_tri[2]];

            float ddxIw = ((iw1 - iw0) * (s_v2.y - s_v0.y) - (iw2 - iw0) * (s_v1.y - s_v0.y)) * inv_det;
            float ddyIw = ((iw2 - iw0) * (s_v1.x - s_v0.x) - (iw1 - iw0) * (s_v2.x - s_v0.x)) * inv_det;

            float b_n0[2] = rendrs_vertex_normals[b_tri[0]];
            float b_n1[2] = rendrs_vertex_normals[b_tri[1]];
            float b_n2[2] = rendrs_vertex_normals[b_tri[2]];

            vec2 uv0 = vec2(b_u0[0], b_u0[1]);
            vec2 uv1 = vec2(b_u1[0], b_u1[1]);
            vec2 uv2 = vec2(b_u2[0], b_u2[1]);

            vec2 ddxUw = ((uv1 * iw1 - uv0 * iw0) * (s_v2.y - s_v0.y) - (uv2 * iw2 - uv0 * iw0) * (s_v1.y - s_v0.y)) * inv_det;
            vec2 ddyUw = ((uv2 * iw2 - uv0 * iw0) * (s_v1.x - s_v0.x) - (uv1 * iw1 - uv0 * iw0) * (s_v2.x - s_v0.x)) * inv_det;
            vec3 ddxPw = ((p1 * iw1 - p0 * iw0) * (s_v2.y - s_v0.y) - (p2 * iw2 - p0 * iw0) * (s_v1.y - s_v0.y)) * inv_det;
            vec3 ddyPw = ((p2 * iw2 - p0 * iw0) * (s_v1.x - s_v0.x) - (p1 * iw1 - p0 * iw0) * (s_v2.x - s_v0.x)) * inv_det;
            vec2 uv_p  = uv0 * B_u + uv1 * B_v + uv2 * B_w;
            vec3 p_p   = p0  * B_u + p1  * B_v + p2  * B_w;
            vec2 ddxUv = (ddxUw - uv_p * ddxIw) * iy_sum;
            vec2 ddyUv = (ddyUw - uv_p * ddyIw) * iy_sum;
            vec3 ddxP  = (ddxPw - p_p  * ddxIw) * iy_sum;
            vec3 ddyP  = (ddyPw - p_p  * ddyIw) * iy_sum;

            imageStore(ima_grads, px, vec4(ddxUv.x, ddxUv.y, ddyUv.x, ddyUv.y));

            vec3 b_n0d = rendrs_unpackOctahedron(vec2(b_n0[0], b_n0[1]));
            vec3 b_n1d = rendrs_unpackOctahedron(vec2(b_n1[0], b_n1[1]));
            vec3 b_n2d = rendrs_unpackOctahedron(vec2(b_n2[0], b_n2[1]));
            vec3 N   = normalize(b_n0d * B_u + b_n1d * B_v + b_n2d * B_w);
            mat3 TBN = rendrs_deriveCotangentGrad(N, ddxP, ddyP, ddxUv, ddyUv);
            vec3 T   = TBN[0];
            vec2 Te = rendrs_packOctahedron(T); //unorm16
            imageStore(ima_space, px, vec4(Te.x, Te.y, B_u, B_v));
            ";
        }
    }
}

/// Helper function to extract barycentric-weights from the `frame/space`
/// image target produced by the deferred attribute interpolation pass.
///
/// Creates the `rendrs_FrameSpace_GetBWeights` function, which takes the
/// `vec4` sample fetched from the relevant image target, which is the
/// 'framespace'.
///
/// The function will retrieve the `BA` components of the sample,
/// which correspond to `u` and `v` barycentric weights, in addition to
/// reconstructing the `w` weight via the formula `w = 1 - u - v` and returns
/// the result as a `vec3` in the standard order `uvw`.
///
/// This is meant to be used to reconstruct interpolated attributes in an
/// eventual shading (or intermediate) pass.
pub const LIB_UTIL_FRAMESPACE_GET_BWEIGHTS: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_FrameSpace_GetBWeights[
        vS_framespace : vec4
    ] => "
        vec2 B_uv = vS_framespace.ba;
        float B_w = 1.0 - B_uv.x - B_uv.y;
        return vec3(B_uv, B_w);
    "
};

/// Helper function to extract the tangent-framre from the `frame/space`
/// image target produced by the deferred attribute interpolation pass.
///
/// Creates the `rendrs_FrameSpace_GetTanFrame` function, which takes the
/// `vec4` sample fetched from the relevant image target, which is the
/// 'framespace'.
///
/// The function will retrieve the `RG` components of the sample,
/// which correspond to *normalized* octahedron-encoded coordinates of the
/// vector.
/// From there, the tangent is reconstructed to a 3d vector and returned.
///
/// Requires [`rendrs_unpackOctahedron`](crate::pack::PACK_OCTAHEDRON_DECODE).
///
/// This is meant to be used to reconstruct interpolated attributes in an
/// eventual shading (or intermediate) pass.
pub const LIB_UTIL_FRAMESPACE_GET_TANFRAME: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_FrameSpace_GetTanFrame[
        vS_framespace : vec4
    ] => "
        vec2 T_oct = vS_framespace.rg;
        return rendrs_unpackOctahedron(T_oct);
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
///
/// This is meant to be used to reconstruct interpolated attributes in an
/// eventual shading (or intermediate) pass.
pub const LIB_INTERP_ATTRIB: GlslLib = GlslLib::new(
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

/// Utility function to reconstruct world-position from a depth value.
///
/// Creates the `rendrs_DepthWorldPosition` function, which requires the
/// following arguments:
/// * the scalar depth value sampled at the relevant point in screen-space
/// * the UV coordinate (0,1) of the relevant point in screen-space
/// * the inverse view-projection matrix
///
/// Returns the world-position as a 3d vector.
///
/// This is meant to be used to reconstruct interpolated attributes in an
/// eventual shading (or intermediate) pass.
///
/// ## Depth Range
/// The scalar `depth` value is also used as the NDC's Z component, so it must
/// be supplied according to the application's depth range convention.
///
/// E.g.:
///
/// For reverse-z, `depth` can be provided directly as sampled as it is
/// already within the common reverse-z range (0,1). For standard depth
/// convention where the range is (-1,1), `depth` must be normalized to that
/// range.
pub const LIB_UTIL_DEPTH_WORLDPOS: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_DepthWorldPosition[
        s_depth     : float,
        v_uv_screen : vec2,
        m_vp_inv    : mat4
    ] => "
        vec4 NDC = vec4(v_uv_screen * 2.0 - 1.0, s_depth, 1.0);
        vec4 WHV = m_vp_inv * NDC;
        return WHV.xyz / WHV.w;
    "
};

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
