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
        uint : rendrs_gbank_gcounter_vertex;
        uint : rendrs_gbank_gcounter_triangle;
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
/// (Also ensure no types are named exactly 'Vertex' or 'Triangle', as these
/// names are already used for functions)
///
/// The triangle indices data can be bound as an EBO for indexed drawing.
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
/// * **Allocation**
///   * `uint Alloc[Vertex|Triangle](optional uint count)`:
///     allocates one or a sequence of length `count` vertices/triangles to the
///     global counter and returns the base handle.
///     If `count` is not provided or is `1`, the base handle is the index of the
///     triangle/vertex itself. If `count > 1` then a span from `(base,base+count)`
///     will be allocated.
/// * **Fill**
///   * `void VertexData(uint handle, vec3 position, vec2|vec3 normal,
///      vec2|vec3 tangent, vec2 uv)`:
///      Fills vertex data for the vertex corresponding to `handle` with the
///      given data. `normal` and `tangent` can be either `vec2`s if
///      octahedron encoded or `vec3`s if not (if they are given as `vec3`s,
///      they will be encoded anyways internally).
///   * `void VertexData(uint base, vec3 positions[], vec2|vec3 normals[]
///      vec2|vec3 tangents[], vec2 uvs[], uint count)`:
///      See above. Bulk-fills contiguous vertex data with the given parallel
///      data arrays.
///      `count` must be the amount of vertices to fill starting from `base`.
///   * `void TriangleData(uint handle, uint indices[3], uint geom_id)`:
///      Fills the triangle data for the triangle corresponding to `handle`
///      with the given data.
///   * `void TriangleData(uint base, uint indices[][3], uint geom_id,
///      uint count)`:
///      See above. Bulk-fills contiguous triangle data with the given parallel
///      data arrays.
///      `count` must be the amount of triangles to fill starting from `base`.
///  * **One-off alloc + fill**
///    * `uint Alloc[Vertex|Triangle]Data(DATA data)`:
///      allocate and feed a single vertex/triangle with the given `data`,
///      returning the index of the allocated vertex/triangle.
///      The `DATA data` parameter(s) must correspond to the parameter list
///      as seen in the entry for `VertexData` or `TriangleData` methods
///      (non-bulk variants).
///  * **Getters**
///   * `RenderVertex GetVertex(uint handle)`:
///     returns the vertex data corresponding to the given `handle`.
///   * `uint[3] GetTriangle(uint handle)`:
///     returns the triangle indexing data corresponding to the given
///     `handle`.
///   * `TriangleAttribs GetTriangleAttribs(uint handle)`:
///     returns the triangle attribute data corresponding to the given
///     `handle`.
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
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        const uint _iDOMAIN_INDEX_BITSHIFT = 24;
                        const uint _iDOMAIN_GEOID_BITMASK = 0x00ffffff;

                        uint _iDomain_unpackIndex(uint idx8_geoid24) {
                            return idx8_geoid24 >> _iDOMAIN_INDEX_BITSHIFT;
                        }
                        uint _iDomain_unpackGeoID(uint idx8_geoid24) {
                            return idx8_geoid24 & _iDOMAIN_GEOID_BITMASK;
                        }"
                    });

                    // vertex/triangle allocation functions
                    // these functions are akin to OpenGL's Create*/Gen* functions
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        // alloc 1, return
                        uint AllocVertex() {
                            return atomicAdd(rendrs_gbank_gcounter_vertex, 1);
                        }
                        uint AllocTriangle() {
                            return atomicAdd(rendrs_gbank_gcounter_triangle, 1);
                        }

                        // alloc N, return base
                        uint AllocVertex(uint count) {
                            //if (count == 0) return 0;
                            return atomicAdd(rendrs_gbank_gcounter_vertex, count);
                        }
                        uint AllocTriangle(uint count) {
                            //if (count == 0) return 0;
                            return atomicAdd(rendrs_gbank_gcounter_triangle, count);
                        }
                        "
                    });
                    // vertex/triangle data feeding functions
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        void VertexData(uint index, vec3 p, vec2 n_oct, vec2 t_oct, vec2 uv) {
                            rendrs_gbank_vertex[index] = RenderVertex(
                                p.x, p.y, p.z,
                                n_oct.x, n_oct.y,
                                t_oct.x, t_oct.y,
                                uv.x, uv.y
                            );
                        }
                        void VertexData(uint index, vec3 p, vec3 n, vec3 t, vec2 uv) {
                            vec2 n_oct = rendrs_packOctahedron(n);
                            vec2 t_oct = rendrs_packOctahedron(t);
                            VertexData(index, p, n_oct, t_oct, uv);
                        }
                        void TriangleData(uint index, uint data[3], uint geom_id) {
                            rendrs_gbank_triangle[index] = data;
                            TriangleAttribs attribs = TriangleAttribs(geom_id);
                            rendrs_gbank_triangle_attribs[index] = attribs;
                        }
                        "
                    });
                    // vertex/triangle getters
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        uint[3] GetTriangle(uint index) {
                            return rendrs_gbank_triangle[index];
                        }
                        TriangleAttribs GetTriangleAttribs(uint index) {
                            return rendrs_gbank_triangle_attribs[index];
                        }

                        RenderVertex GetVertex(uint index) {
                            return rendrs_gbank_vertex[index];
                        }
                        "
                    });
                    // all-in-one alloc+data functions for convenience/testing
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        uint AllocTriangleData(uint indices[3], uint geom_id) {
                            uint triangle_index = AllocTriangle();
                            TriangleData(triangle_index, indices, geom_id);
                            return triangle_index;
                        }

                        uint AllocVertexData(vec3 p, vec2 n_oct, vec2 t_oct, vec2 uv) {
                            uint vertex_index = AllocVertex();
                            VertexData(vertex_index, p, n_oct, t_oct, uv);
                            return vertex_index;
                        }
                        uint AllocVertexData(vec3 p, vec3 n, vec3 t, vec2 uv) {
                            vec2 n_oct = rendrs_packOctahedron(n);
                            vec2 t_oct = rendrs_packOctahedron(t);
                            return AllocVertexData(p, n_oct, t_oct, uv);
                        }
                        "
                    });

                    $($($e_lib;)+)?

                    ethel::shader::GlslLib::new(indoc::concatdoc! {
                        "void _submitGeometry(
                            in uint rendrs_GeometryID,
                            in uint rendrs_DomainIndex,
                            in uint rendrs_WorkGroupID,
                            in uint rendrs_ThreadID,
                            in uint rendrs_DomainThreadID,
                            in uint rendrs_GlobalThreadID
                        ) {\n", $source, "\n}"
                    });
                };
                $(share {
                    $($share_t $share_n $([$arr_c])*;)*
                };)?

                src() {
                    "
                    DomainData _domain = rendrs_domains[gl_WorkGroupID.x];
                    uint _d_threads = _domain.thread_count;
                    if (gl_LocalInvocationID.x < _d_threads) {
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
                    }
                    ";
                }
            }
        }}
    };
}
