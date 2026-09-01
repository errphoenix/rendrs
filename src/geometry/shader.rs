use ethel::shader::{GlslStorage, GlslStruct};

ethel::shader_glsl_struct! {
    struct RenderVertex {
        pos_x : f32 => float
        pos_y : f32 => float
        pos_z : f32 => float
        norm_oct_x : f32 => float
        norm_oct_y : f32 => float
        tan_oct_x : f32 => float
        tan_oct_y : f32 => float
        uv_x : f32 => float
        uv_y : f32 => float
    }
}

ethel::shader_glsl_struct! {
    struct DomainData {
        idx8_geoid24 : u32 => uint
        thread_count : u32 => uint
    }
}

ethel::shader_glsl_struct! {
    struct TriangleAttribs {
        geometry_id : u32 => uint
    }
}

pub const TYPE_RENDERVERTEX: GlslStruct = RenderVertexGlslStruct::as_definition();
pub const TYPE_DOMAIN_DATA: GlslStruct = DomainDataGlslStruct::as_definition();
pub const TYPE_TRIANGLE_ATTRIBS: GlslStruct = TriangleAttribsGlslStruct::as_definition();

macro_rules! ssbo_binding {
    (Rendrs_GBANK_RenderVertex) => {
        0
    };
    (Rendrs_GBANK_Triangle) => {
        1
    };
    (Rendrs_GBANK_TriangleAttribs) => {
        2
    };
    (Rendrs_GBANK_GCounter) => {
        3
    };
    (Rendrs_Domains) => {
        4
    };
}

pub const SSBO_BINDING_GBANK_RENDERVERTEX: u32 = ssbo_binding!(Rendrs_GBANK_RenderVertex);
pub const SSBO_BINDING_GBANK_TRIANGLE: u32 = ssbo_binding!(Rendrs_GBANK_Triangle);
pub const SSBO_BINDING_GBANK_TRIANGLE_ATTRIBS: u32 = ssbo_binding!(Rendrs_GBANK_TriangleAttribs);
pub const SSBO_BINDING_GBANK_GCOUNTER: u32 = ssbo_binding!(Rendrs_GBANK_GCounter);
pub const SSBO_BINDING_DOMAINS: u32 = ssbo_binding!(Rendrs_Domains);

pub const SSBO_GBANK_RENDERVERTEX: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_RenderVertex => {
        [dyn_array RenderVertex : rendrs_gbank_vertex]
    }
};
pub const SSBO_GBANK_TRIANGLE: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_Triangle => {
        [dyn_array uint : rendrs_gbank_triangle => each 3]
    }
};
pub const SSBO_GBANK_TRIANGLE_ATTRIBS: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_TriangleAttribs => {
        [dyn_array TriangleAttribs : rendrs_gbank_triangle_attribs]
    }
};
pub const SSBO_GBANK_GCOUNTER: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_GCounter => {
        rendrs_gbank_gcounter_vertex   : uint;
        rendrs_gbank_gcounter_triangle : uint;
    }
};
pub const SSBO_DOMAINS: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_Domains => {
        [dyn_array DomainData : rendrs_domains]
    }
};

/// Create a geometry submission job to be attached to a geometry pass.
///
/// This is a compute shader that runs arbitrary logic with the purpose of
/// gathering, modifying, and submitting geometry organized with `domains`.
///
/// The macro's syntax is similar to [`ethel's compute shaders`].
///
/// Optional blocks for additional data can be defined, these are, in order:
/// `uniform`, `sampler`, `image`, `type`, `ssbo`, `lib`, and `share`.
/// The definition syntax for each of these is identical to
/// [`ethel's compute shaders`].
///
/// **NOTE**: Any additional SSBO must begin at index 5, as the first 4 binding indices
/// are reserved for geometry data.
///
/// ## Context
///
/// There is also an additional (also optional) block `context`: this is where
/// the inner [`compute pass`]' context data is defined. This will create a
/// [`CtxType`] struct to be initialized and passed to the [`compute pass`]
/// when dispatched.
///
/// Context structs allow storing borrowed data. Borrowed data must be
/// defined with the `'ctx` lifetime like in the example below.
///
/// ### Context definition example:
/// ```rust,ignore
/// context {
///     some_data : u32;
///     some_borrowed_data : TriBuffer<u32>, for 'ctx;
/// }
/// ```
///
/// [`Compute Pass`]: crate::pipeline::ComputePass
/// [`CtxType`]: crate::pipeline::CtxType
///
/// # Source
///
/// After the optional blocks, the shader's 'source' is defined: this is a
/// single string literal containing the relevant GLSL code.
///
/// ## Constants
///
/// The shader's source has access to the following parameters:
/// * GLSL's standard compute shader variables (`gl_GlobalInvocationID`, etc.)*
/// * `rendrs_GeometryID` the index of the current working geometric entity
/// * `rendrs_DomainIndex` the local index of the current working domain of the
///   current geometric entity
/// * `rendrs_WorkGroupID` the global index of the current working domain,
///   equal to `gl_WorkGroupID.x`
/// * `rendrs_ThreadID` the thread index (invocation) local to the current
///   working geometric entity
/// * `rendrs_DomainThreadID` the thread index (invocation) local to the
///   current working domain, equal to `gl_LocalInvocationID.x`
/// * `rendrs_GlobalThreadID` the global thread index (invocation), equal
///   to `gl_GlobalInvocationID.x`
///
/// *Note that the shader's workgroup (frequently referred to as `domain`) is
/// of linear size 64 (x=64,y=1,z=1).
///
/// Threads that would be out-of-bounds return before reaching any geometry
/// submission job.
///
/// [`ethel's compute shaders`]: ethel::shader_glsl_compute
///
/// ## Functions
///
/// Rendrs' normal encode/decode functions are all available by default,
/// these are:
/// * Octahedron encoding: `rendrs_packOctahedron` and
///   `rendrs_unpackOctahedron`
/// * Spherical encoding: `rendrs_packSpherical` and `rendrs_unpackSpherical`
///
/// Output normals (and tangents) must be octahedron-encoded.
///
/// ### Geometry submission functions
///
/// Geometry submission functions can submit arbitrary vertices and triangles
/// data for rendering:
/// * `uint Vertex(vec3 position, vec2 normal_oct, vec2 tangent_oct, vec2 uv)`:
///     Submits a vertex at `position` with the specified `normal` and
///     `tangent` vectors. The last 2 are octahedron-encoded.
///     Returns the index of the submitted vertex.
/// * `uint Vertex(vec3 position, vec3 normal, vec3 tangent, vec2 uv)`:
///     Submits a vertex at `position` with the specified `normal` and
///     `tangent` vectors. These are *not* octahedron-encoded, octahedron
///     encoding is performed inside the function with `rendrs'` packing
///     functions.
///     Returns the index of the submitted vertex.
/// * `uint Triangle(uint v0, uint v1, uint v2, uint geom_id)`:
///    Submits a triangle formed by the given `v0, v1, v2` vertex indices as
///    returned by `Vertex`.
///    `geom_id` is the geometry ID as provided by Rendrs.
///    Returns the index of the submitted triangle.
///
/// [`rendrs_packOctahedron`]: crate::pack::PACK_OCTAHEDRON_ENCODE
/// [`rendrs_unpackOctahedron`]: crate::pack::PACK_OCTAHEDRON_DECODE
/// [`rendrs_packSpherical`]: crate::pack::PACK_SPHERICAL_ENCODE
/// [`rendrs_unpackSpherical`]: crate::pack::PACK_SPHERICAL_DECODE
#[macro_export]
macro_rules! geometry_submission_job {
    (
        $name:ident => {
            $(uniform {
                $(length $u_len:literal, $u_gl_name:ident: $u_gl_type:ident => $u_r_type:ty;)+
            })?
            $(sampler {
                $(on $s_idx:expr $(, for $s_len:expr)? => $us_name:ident : $sampler_type:ident ; )+
            })?
            $(image {
                $(on $idx:expr $(, for $len:expr)? => $ui_name:ident : $image_type:ident as $format:ident $($m:ident)* ; )+
            })?
            $(type {
                $($type_glsl:expr)+
            })?
            $(ssbo {
                $($ssbo_glsl:expr)+
            })?
            $(lib {
                $($e_lib:expr)+
            })?
            $(share {
                $($share_t:ident $share_n:ident $([$arr_c:expr])*;)*
            })?

            $(context {
                $($ctx_field:ident : $ctx_type:ty $(, for $ctx_lt:lifetime)? ;)+
            })?

            $source:literal
        }
    ) => {
        paste::paste! {

        const [< $name:upper GEOM_SAMPLER_COUNT >]: usize = $($(1 + $($s_len - 1 +)?)*)? 0;
        const [< $name:upper GEOM_IMAGE_COUNT >]: usize = $($(1 + $($len - 1 +)?)*)? 0;

        #[derive(Debug)]
        pub struct [< $name GeomCtx >]<'ctx> {
            pub shader: &'ctx [< ComputeShader $name GeomSubmit >],
            $($(pub $ctx_field: $(&$ctx_lt)? $ctx_type,)+)?
        }
        $crate::context_wrapper!(for<'ctx> [< $name GeomCtx >]);

        pub type [< $name GeomPass >] = $crate::geometry::GeomPass<
            [< ComputeShader $name GeomSubmit >],
            [< $name GeomCtxWrapper >],
            [< $name:upper GEOM_SAMPLER_COUNT >],
            [< $name:upper GEOM_IMAGE_COUNT >],
        >;

        ethel::shader_glsl_compute! {
            struct [< $name GeomSubmit >] > [460] {
                workgroup [64, 1, 1];

                $(uniform {
                    $(length $u_len, $u_gl_name: $u_gl_type => $u_r_type;)+
                };)?
                $(sampler {
                    $(on $s_idx $(, for $s_len)? => $us_name : $sampler_type ; )+
                };)?
                $(image {
                    $(on $idx $(, for $len)? => $ui_name : $image_type as $format $($m)* ; )+
                };)?
                type {
                    $crate::geometry::shader::TYPE_RENDERVERTEX
                    $crate::geometry::shader::TYPE_DOMAIN_DATA
                    $crate::geometry::shader::TYPE_TRIANGLE_ATTRIBS

                    $($($type_glsl)+)?
                };
                ssbo {
                    $crate::geometry::shader::SSBO_GBANK_RENDERVERTEX
                    $crate::geometry::shader::SSBO_GBANK_TRIANGLE
                    $crate::geometry::shader::SSBO_GBANK_TRIANGLE_ATTRIBS
                    $crate::geometry::shader::SSBO_GBANK_GCOUNTER
                    $crate::geometry::shader::SSBO_DOMAINS

                    $($($ssbo_glsl)+)?
                };
                lib {
                    $crate::pack::PACK_OCTAHEDRON_WRAP_UTIL;
                    $crate::pack::PACK_OCTAHEDRON_ENCODE;
                    $crate::pack::PACK_OCTAHEDRON_DECODE;
                    $crate::pack::PACK_SPHERICAL_ENCODE;
                    $crate::pack::PACK_SPHERICAL_DECODE;

                    // domain data bit-packing helpers (internal)
                    ethel::shader::GlslLib::new(
                        "
                        const uint _iDOMAIN_INDEX_BITSHIFT = 24;
                        const uint _iDOMAIN_GEOID_BITMASK = 0x00ffffff;

                        uint _iDomain_unpackIndex(uint packed) {
                            return packed >> _iDOMAIN_INDEX_BITSHIFT;
                        }
                        uint _iDomain_unpackGeoID(uint packed) {
                            return packed & _iDOMAIN_GEOID_BITMASK;
                        }",
                    );
                    // emit triangle function
                    ethel::shader::GlslLib::new(
                        "
                        uint Triangle(uint v0, uint v1, uint v2, uint geom_id) {
                            uint triangle_index = atomicAdd(rendrs_gbank_gcounter_triangle, 1);

                            rendrs_gbank_triangle[triangle_index][0] = v0;
                            rendrs_gbank_triangle[triangle_index][1] = v1;
                            rendrs_gbank_triangle[triangle_index][2] = v2;

                            TriangleAttribs attribs = TriangleAttribs(geom_id);
                            rendrs_gbank_triangle_attribs[triangle_index] = attribs;

                            return triangle_index;
                        }
                    ",
                    );
                    // emit vertex functions
                    ethel::shader::GlslLib::new(
                        "
                        uint Vertex(vec3 p, vec2 n_oct, vec2 t_oct, vec2 uv) {
                            uint vertex_index = atomicAdd(rendrs_gbank_gcounter_vertex, 1);

                            rendrs_gbank_vertex[vertex_index] = RenderVertex(
                                p.x, p.y, p.z,
                                n_oct.x, n_oct.y,
                                t_oct.x, t_oct.y,
                                uv.x, uv.y
                            );

                            vec4 nota = vec4(n_oct.x, n_oct.y, t_oct.x, t_oct.y);
                            rendrs_gbank_nota[vertex_index] = nota;

                            return vertex_index;
                        }

                        uint Vertex(vec3 p, vec3 n, vec3 t, vec2 uv) {
                            vec2 n_oct = rendrs_packOctahedron(n);
                            vec2 t_oct = rendrs_packOctahedron(t);
                            return Vertex(p, n_oct, t_oct, uv);
                        }
                    ",
                    );

                    $($($e_lib;)+)?

                    ethel::shader::GlslLib::new(concat!(
                        "void _submitGeometry(
                            in uint rendrs_GeometryID,
                            in uint rendrs_DomainIndex,
                            in uint rendrs_WorkGroupID,
                            in uint rendrs_ThreadID,
                            in uint rendrs_DomainThreadID,
                            in uint rendrs_GlobalThreadID
                        ) {\n", $source, "\n}"
                    ));
                };
                $(share {
                    $($share_t:ident $share_n:ident $([$arr_c:expr])*;)*
                };)?

                src() {
                    "
                    DomainData _domain = rendrs_domains[gl_WorkGroupID.x];

                    uint _d_threads = _domain.thread_count;
                    if (gl_LocalInvocationID.x >= _d_threads) {
                        return;
                    }

                    uint _d_packed = _domain.idx8_geoid24;
                    uint _d_index  = _iDomain_unpackIndex(_d_packed);
                    uint _d_geoid  = _iDomain_unpackGeoID(_d_packed);

                    const uint rendrs_GeometryID  = _d_geoid;
                    const uint rendrs_DomainIndex = _d_index;
                    const uint rendrs_WorkGroupID = gl_WorkGroupID.x;
                    const uint rendrs_ThreadID = 64 * _d_index + gl_LocalInvocationID.x;
                    const uint rendrs_DomainThreadID = gl_LocalInvocationID.x;
                    const uint rendrs_GlobalThreadID = gl_GlobalInvocationID.x;

                    _submitGeometry(
                        rendrs_GeometryID,
                        rendrs_DomainIndex,
                        rendrs_WorkGroupID,
                        rendrs_ThreadID,
                        rendrs_DomainThreadID,
                        rendrs_GlobalThreadID
                    );
                    ";
                }
            }
        }}
    };
}
