use ethel::shader::{GlslStorage, GlslStruct};

ethel::shader_glsl_struct! {
    struct DomainData {
        idx8_geoid24 : u32 => uint
        thread_count : u32 => uint
    }
}

ethel::shader_glsl_struct! {
    struct TriangleAttribs {
        geometry_id : u32 => uint
        // 16 bits extra, 16 bits material_id
        extra_material_id : u32 => uint
    }
}

ethel::shader_glsl_struct! {
    struct InstanceList {
        geometry_id : u32 => uint

        tri_base  : u32 => uint
        tri_count : u32 => uint

        // 0xffff0000 - instance base
        // 0x0000ffff - instance count
        instance_base_count : u32 => uint
    }
}
ethel::shader_glsl_struct! {
    struct InstanceTransform {
        px : f32 => float
        py : f32 => float
        pz : f32 => float

        qx : f32 => float
        qy : f32 => float
        qz : f32 => float
        qw : f32 => float

        s  : f32 => float
    }
}

pub const TYPE_DOMAIN_DATA: GlslStruct = DomainDataGlslStruct::as_definition();
pub const TYPE_TRIANGLE_ATTRIBS: GlslStruct = TriangleAttribsGlslStruct::as_definition();
pub const TYPE_INSTANCELIST: GlslStruct = InstanceListGlslStruct::as_definition();
pub const TYPE_INSTANCE_TRANSFORM: GlslStruct = InstanceTransformGlslStruct::as_definition();

macro_rules! ssbo_binding {
    (Rendrs_GBANK_VertexBuffers) => {
        0
    };
    (Rendrs_GBANK_TriangleBuffers) => {
        1
    };
    (Rendrs_GBANK_GCounter) => {
        2
    };
    (Rendrs_Domains) => {
        3
    };
    (Rendrs_GBANK_InstanceData) => {
        4
    };
}

pub const SSBO_BINDING_GBANK_VERTEX: u32 = ssbo_binding!(Rendrs_GBANK_VertexBuffers);
pub const SSBO_BINDING_GBANK_TRIANGLE: u32 = ssbo_binding!(Rendrs_GBANK_TriangleBuffers);
pub const SSBO_BINDING_GBANK_GCOUNTER: u32 = ssbo_binding!(Rendrs_GBANK_GCounter);
pub const SSBO_BINDING_DOMAINS: u32 = ssbo_binding!(Rendrs_Domains);
pub const SSBO_BINDING_GBANK_INSTANCE_DATA: u32 = ssbo_binding!(Rendrs_GBANK_InstanceData);

pub const SSBO_GBANK_GCOUNTER: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_GCounter => {
        uint : rendrs_gbank_gcounter_vertex;
        uint : rendrs_gbank_gcounter_triangle;
        uint : rendrs_gbank_gcounter_instancelist;
        uint : rendrs_gbank_gcounter_instance;
    }
};
pub const SSBO_DOMAINS: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_Domains => {
        [dyn_array DomainData : rendrs_domains]
    }
};

pub const SSBO_GBANK_INSTANCING_DATA: GlslStorage = ethel::shader_glsl_ssbo! {
    buf Rendrs_GBANK_InstanceData => {
        InstanceList      : rendrs_gbank_instance_lists[1024];
        InstanceTransform : rendrs_gbank_instance_transforms[65535];
    }
};

/// Create a geometry submission job to be attached to a geometry pass.
///
/// This is a compute shader that runs arbitrary logic with the purpose of
/// gathering, modifying, and submitting geometry organized with `domains`.
///
/// The first block of the macro is `source`, which must refer to structs
/// implementing [`HasVertexBuffers`] and [`HasTriangleBuffers`], which
/// describe the size and length of the buffers.
/// ```text
/// source {
///     vertex => $vb_source:ty;
///     triangle => $tb_source:ty;
/// };
/// ```
///
/// [`HasVertexBuffers`]: super::HasVertexBuffers
/// [`HasTriangleBuffers`]: super::HasTriangleBuffers
///
/// The rest of macro's syntax is similar to [`ethel's compute shaders`].
///
/// Optional blocks for additional data can be defined, these are, in order:
/// `uniform`, `sampler`, `image`, `type`, `ssbo`, `lib`, and `share`.
/// The definition syntax for each of these is identical to
/// [`ethel's compute shaders`].
///
/// **NOTE**: Any additional SSBO must begin at index 5, as the first 4 binding indices
/// (from 0 to 4) are reserved for geometry data.
///
/// (Also ensure no types are named exactly 'Vertex' or 'Triangle', as these
/// names are already used for functions)
///
/// The triangle indices data array populated by geometry submission jobs
/// should be later used as an `EBO` for indexed drawing.
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
/// * `rendrs_GeometryID` the index of the current working geometric entity.
/// * `rendrs_DomainIndex` the local index of the current working domain of the
///   current geometric entity. A maximum of 255 can be dispatched for one
///   entity.
/// * `rendrs_WorkGroupID` the global index of the current working domain,
///   equal to `gl_WorkGroupID.x`.
/// * `rendrs_ThreadID` the thread index (invocation) local to the current
///   working geometric entity.
/// * `rendrs_DomainThreadID` the thread index (invocation) local to the
///   current working domain, equal to `gl_LocalInvocationID.x`.
/// * `rendrs_GlobalThreadID` the global thread index (invocation), equal
///   to `gl_GlobalInvocationID.x`.
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
///   * `void VertexData{Attribute}(uint handle, vec3|2 data)`:
///      Fills vertex `Attribute` data for the vertex corresponding to
///      `handle`. `Attribute` options are `Position`, `Normal`, `Uv`. The
///      `data` type is `vec3` for `Position`, `vec2` for `Uv`, and can be
///      either `vec3` or `vec2` for `Normal`, depending on whether it is
///      already octahedron-encoded or not.
///   * `void VertexData(uint handle, vec3 position,
///      vec2|vec3 normal, vec2 uv)`:
///      Fills vertex data for the vertex corresponding to `handle` with the
///      given data. `normal` can be either `vec2`s if octahedron encoded or
///      `vec3`s if not (if they are given as `vec3`s, they will be encoded
///      anyways internally).
///   * `void TriangleDataIndices(uint handle, uint indices[3])`:
///      Fills the triangle attribs data for the triangle corresponding to
///      `handle` with the given data.
///   * `void TriangleDataAttribs(uint handle, uint geom_id,
///      uint mat_id, optional uint extra)`:
///      Fills the triangle indices data for the triangle corresponding to
///      `handle` with the given data.
///   * `void TriangleData(uint handle, uint indices[3], uint geom_id
///      uint mat_id, optional uint extra)`:
///      Fills the triangle data for the triangle corresponding to `handle`
///      with the given data.
///  * **One-off alloc + fill**
///    * `uint Alloc[Vertex|Triangle]Data(DATA data)`:
///      allocate and feed a single vertex/triangle with the given `data`,
///      returning the index of the allocated vertex/triangle.
///      The `DATA data` parameter(s) must correspond to the parameter list
///      as seen in the entry for `VertexData` or `TriangleData` methods.
///  * **Getters**
///   * `float[N] GetVertex{Attribute}(uint handle)`:
///     returns the relevant vertex `Attribute` corresponding to the given
///     vertex `handle`. `Attribute` options are `Position`, `Normal`, `Uv`.
///     `N` is `3` for `Position` and `2` otherwise.
///   * `uint[3] GetTriangleIndices(uint handle)`:
///     returns the triangle indexing data corresponding to the given triangle
///     `handle`.
///   * `TriangleAttribs GetTriangleAttribs(uint handle)`:
///     returns the triangle attribute data corresponding to the given triangle
///     `handle`.
///
/// ### Instancing
/// Geometry submission also supports submission of instanced meshes, through
/// the following functions:
/// * `uint AllocInstanceList(optional uint count)`:
///   allocates one instance list or a sequence on the global counter and
///   returns its starting index in the global instance lists array.
/// * `uint AllocInstances(uint count)`:
///   allocates one or a sequence of length `count` vertices/triangles to the
///   global counter and returns the base handle.
///   The functions allocates a span from `base` to `base + count`.
/// * `void InstanceDataTransform(uint index, vec3 position,
///     optional vec4 quaternion, optional float scale)`:
///   Fills instance data for the given `index` with `position`, `rotation`
///   (as a `vec4` quaternion) and a uniform `scale` factor. The last 2
///   parameters are optional, and will default to identities if absent.
/// * `void InstanceListDataGeometry(uint index, uint geometry_id,
///     uint triangle_base, uint triangle_count)`:
///   Fills instance list data for the given `index` with the instanced mesh's
///   `triangle_base` index in the global triangle arrays as well as its
///   `triangle_count`. A `geometry_id` to identify the mesh (or instance) is
///   also required.
/// * `void InstanceListDataBounds(uint index,
///     uint instance_base, uint instance_count)`:
///   Fills instance list data for the given `index` with the base instance
///   index `instance_base` in the global instances array as well as its
///   total `instance_count`.
/// * `void InstanceListData(uint index, uint geo_id,
///     uint triangle_base, uint triangle_count,
///     uint instance_base, uint instance_count)`:
///   Fills all required instance list data for the given `index` in a
///   single operation. This is more efficient than two separate
///   `*DataGeometry` and `*DataBounds` calls and should be preferred when
///   possible.
/// * `uint AllocInstanceListData(uint geo_id,
///     uint triangle_base, uint triangle_count,
///     uint instance_base, uint instance_count)`:
///   Allocate and fill instance list data. Returns the instance list index.
/// * `InstanceTransform GetInstanceTransform(uint index)` and
///    `InstanceList GetInstanceList(uint index)`:
///   Getters for instance transform and instance list.
///
/// [`rendrs_packOctahedron`]: crate::pack::PACK_OCTAHEDRON_ENCODE
/// [`rendrs_unpackOctahedron`]: crate::pack::PACK_OCTAHEDRON_DECODE
/// [`rendrs_packSpherical`]: crate::pack::PACK_SPHERICAL_ENCODE
/// [`rendrs_unpackSpherical`]: crate::pack::PACK_SPHERICAL_DECODE
#[macro_export]
macro_rules! geometry_submission_job {
    (
        $name:ident => {
            source {
                vertex => $vb_source:ty;
                triangle => $tb_source:ty;
            };

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
            pub _marker: std::marker::PhantomData<&'ctx ()>,
            $($(pub $ctx_field: $(&$ctx_lt)? $ctx_type,)+)?
        }
        $crate::context_wrapper!(for<'ctx> [< $name GeomCtx >]);

        pub type [< $name GeomPass >] = $crate::geometry::GeomPass<
            [< ComputeShader $name GeomSubmit >],
            [< $name GeomCtxWrapper >],
            [< $name:upper GEOM_SAMPLER_COUNT >],
            [< $name:upper GEOM_IMAGE_COUNT >],
        >;

        const SSBO_GBANK_VERTEX: ethel::shader::GlslStorage = $vb_source::SSBO_ARRAYS;
        const SSBO_GBANK_TRIANGLE: ethel::shader::GlslStorage = $tb_source::SSBO_ARRAYS;

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
                    $crate::geometry::shader::TYPE_TRIANGLE_ATTRIBS
                    $crate::geometry::shader::TYPE_DOMAIN_DATA
                    $crate::geometry::shader::TYPE_INSTANCELIST
                    $crate::geometry::shader::TYPE_INSTANCE_TRANSFORM

                    $($($type_glsl)+)?
                };
                ssbo {
                    SSBO_GBANK_VERTEX
                    SSBO_GBANK_TRIANGLE
                    $crate::geometry::shader::SSBO_GBANK_GCOUNTER
                    $crate::geometry::shader::SSBO_GBANK_INSTANCING_DATA
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
                            return atomicAdd(rendrs_gbank_gcounter_vertex, count);
                        }
                        uint AllocTriangle(uint count) {
                            return atomicAdd(rendrs_gbank_gcounter_triangle, count);
                        }
                        "
                    });
                    // vertex/triangle data feeding functions
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        void VertexDataPosition(uint index, vec3 position) {
                            rendrs_gbank_vertex_positions[index] = float[](
                                position.x, position.y, position.z
                            );
                        }
                        void VertexDataNormal(uint index, vec2 normal_oct) {
                            rendrs_gbank_vertex_normals[index] = float[](
                                normal_oct.x, normal_oct.y
                            );
                        }
                        void VertexDataNormal(uint index, vec3 normal) {
                            VertexDataNormal(index, rendrs_packOctahedron(normal));
                        }
                        void VertexDataUv(uint index, vec2 uv) {
                            rendrs_gbank_vertex_uvs[index] = float[](
                                uv.x, uv.y
                            );
                        }
                        void VertexData(uint index, vec3 p, vec2 n_oct, vec2 uv) {
                            VertexDataPosition(index, p);
                            VertexDataNormal(index, n_oct);
                            VertexDataUv(index, uv);
                        }
                        void VertexData(uint index, vec3 p, vec3 n, vec2 uv) {
                            VertexDataPosition(index, p);
                            VertexDataNormal(index, n);
                            VertexDataUv(index, uv);
                        }
                        void TriangleDataIndices(uint index, uint data[3]) {
                            rendrs_gbank_triangle_indices[index] = data;
                        }
                        void TriangleDataAttribs(uint index, uint geom_id, uint mat_id, uint extra) {
                            TriangleAttribs attribs = TriangleAttribs(geom_id, mat_id | (extra << 16));
                            rendrs_gbank_triangle_attribs[index] = attribs;
                        }
                        void TriangleDataAttribs(uint index, uint geom_id, uint mat_id) {
                            TriangleAttribs attribs = TriangleAttribs(geom_id, mat_id);
                            rendrs_gbank_triangle_attribs[index] = attribs;
                        }
                        void TriangleData(uint index, uint data[3], uint geom_id, uint mat_id, uint extra) {
                            TriangleDataIndices(index, data);
                            TriangleDataAttribs(index, geom_id, mat_id, extra);
                        }
                        void TriangleData(uint index, uint data[3], uint geom_id, uint mat_id) {
                            TriangleDataIndices(index, data);
                            TriangleDataAttribs(index, geom_id, mat_id);
                        }
                        "
                    });
                    // vertex/triangle getters
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        uint[3] GetTriangleIndices(uint index) {
                            return rendrs_gbank_triangle_indices[index];
                        }
                        TriangleAttribs GetTriangleAttribs(uint index) {
                            return rendrs_gbank_triangle_attribs[index];
                        }
                        uint GetTriangleAttribMaterialID(uint index) {
                            return GetTriangleAttribs(index).extra_material_id & 0xffffu;
                        }
                        uint GetTriangleAttribExtra(uint index) {
                            return GetTriangleAttribs(index).extra_material_id >> 16;
                        }

                        vec3 GetVertexPosition(uint index) {
                            float position[3] = rendrs_gbank_vertex_positions[index];
                            return vec3(position[0], position[1], position[2]);
                        }
                        vec2 GetVertexNormal(uint index) {
                            float normal[2] = rendrs_gbank_vertex_normals[index];
                            return vec2(normal[0], normal[1]);
                        }
                        vec2 GetVertexUv(uint index) {
                            float uv[2] = rendrs_gbank_vertex_uvs[index];
                            return vec2(uv[0], uv[1]);
                        }
                        "
                    });
                    // all-in-one alloc+data functions for convenience/testing
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        uint AllocTriangleData(uint indices[3], uint geom_id, uint mat_id, uint extra) {
                            uint triangle_index = AllocTriangle();
                            TriangleData(triangle_index, indices, geom_id, mat_id, extra);
                            return triangle_index;
                        }
                        uint AllocTriangleData(uint indices[3], uint geom_id, uint mat_id) {
                            uint triangle_index = AllocTriangle();
                            TriangleData(triangle_index, indices, geom_id, mat_id);
                            return triangle_index;
                        }
                        uint AllocVertexData(vec3 p, vec2 n_oct, vec2 uv) {
                            uint vertex_index = AllocVertex();
                            VertexData(vertex_index, p, n_oct, uv);
                            return vertex_index;
                        }
                        uint AllocVertexData(vec3 p, vec3 n, vec2 uv) {
                            vec2 n_oct = rendrs_packOctahedron(n);
                            return AllocVertexData(p, n_oct, uv);
                        }
                        "
                    });

                    // instance functions
                    ethel::shader::GlslLib::new(indoc::indoc! {
                        "
                        // alloc 1, return (list)
                        uint AllocInstanceList() {
                            return atomicAdd(rendrs_gbank_gcounter_instancelist, 1);
                        }
                        // alloc N, return base (list)
                        uint AllocInstanceList(uint count) {
                            return atomicAdd(rendrs_gbank_gcounter_instancelist, count);
                        }
                        // alloc N, return base (instances)
                        uint AllocInstances(uint count) {
                            return atomicAdd(rendrs_gbank_gcounter_instance, count);
                        }

                        void InstanceDataTransform(uint index, vec3 position, vec4 quaternion, float scale) {
                            rendrs_gbank_instance_transforms[index] = InstanceTransform(
                                position.x, position.y, position.z,

                                quaternion.x, quaternion.y,
                                quaternion.z, quaternion.w,

                                scale
                            );
                        }
                        void InstanceDataTransform(uint index, vec3 position, vec4 quaternion) {
                            InstanceDataTransform(index, position, quaternion, 1.0);
                        }
                        void InstanceDataTransform(uint index, vec3 position) {
                            InstanceDataTransform(index, position, vec4(vec3(0.0), 1.0), 1.0);
                        }

                        void InstanceListDataGeometry(uint index, uint geo_id, uint tri_base, uint tri_count) {
                            InstanceList list = rendrs_gbank_instance_lists[index];
                            list.geometry_id = geo_id;
                            list.tri_base = tri_base;
                            list.tri_count = tri_count;
                            rendrs_gbank_instance_lists[index] = list;
                        }
                        void InstanceListDataBounds(uint index, uint instance_base, uint instance_count) {
                            InstanceList list = rendrs_gbank_instance_lists[index];
                            uint m = instance_base << 16;
                            uint l = instance_count & 0xffff;
                            list.instance_base_count = l | m;
                            rendrs_gbank_instance_lists[index] = list;
                        }
                        void InstanceListData(
                            uint index, uint geo_id,
                            uint tri_base, uint tri_count,
                            uint instance_base, uint instance_count
                        ) {
                            uint im = instance_base << 16;
                            uint il = instance_count & 0xffff;
                            rendrs_gbank_instance_lists[index] = InstanceList(
                              geo_id, tri_base, tri_count, il | im
                            );
                        }

                        uint AllocInstanceListData(
                            uint geo_id,
                            uint tri_base, uint tri_count,
                            uint instance_base, uint instance_count
                        ) {
                            uint index = AllocInstanceList();
                            uint im = instance_base << 16;
                            uint il = instance_count & 0xffff;
                            rendrs_gbank_instance_lists[index] = InstanceList(
                              geo_id, tri_base, tri_count, il | im
                            );
                            return index;
                        }

                        InstanceList GetInstanceList(uint index) {
                            return rendrs_gbank_instance_lists[index];
                        }
                        InstanceTransform GetInstanceTransform(uint index) {
                            return rendrs_gbank_instance_transforms[index];
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
