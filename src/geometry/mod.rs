use ethel::{
    render::buffer::SingleBuffer,
    shader::{GlslStorage, GlslStruct},
};

macro_rules! ssbo_binding {
    (Rendrs_GBANK_Vertex) => {
        0
    };
    (Rendrs_GBANK_NoTa) => {
        1
    };
    (Rendrs_GBANK_Triangle) => {
        2
    };
    (Rendrs_GBANK_TriangleMetadata) => {
        3
    };
    (Rendrs_GBANK_Counters) => {
        4
    };
}

/// todo
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TriangleMetadata {
    pub geometry_id: u32,
}
ethel::shader_glsl_struct! {
    struct TriangleMetadata {
        geometry_id : u32 => uint
    }
}

pub const TYPE_TRIANGLE_METADATA: GlslStruct = TriangleMetadataGlslStruct::as_definition();

/// Vertex buffer represented as `x, y, z` 32-bit floats.
pub type VertexBuffer = SingleBuffer<[f32; 3]>;

/// Normals-tangents buffer represented as 2 octahedron encoded 2d vectors.
pub type NoTaBuffer = SingleBuffer<[f32; 4]>;

/// Basic triangle primitive buffer represented as simple vertex indices.
pub type TriangleBuffer = SingleBuffer<[u32; 3]>;

/// Per-triangle metadata used later in rendering.
pub type TriangleMetadataBuffer = SingleBuffer<TriangleMetadata>;

/// Atomic counters buffer.
///
/// Index 0 = vertex counter
///
/// Index 1 = triangle counter
pub type CountersBuffer = SingleBuffer<[u32; 2]>;

#[derive(Debug, Default)]
pub struct GeometryBank {
    vertex_cap: usize,
    triangle_cap: usize,
    vert: VertexBuffer,
    nota: NoTaBuffer,
    triangle: TriangleBuffer,
    triangle_meta: TriangleMetadataBuffer,
    counters: CountersBuffer,
}
impl GeometryBank {
    pub fn new(vertex_cap: usize, triangle_cap: usize) -> Self {
        Self {
            vertex_cap,
            triangle_cap,
            vert: SingleBuffer::zeroed(vertex_cap),
            nota: SingleBuffer::zeroed(vertex_cap),
            triangle: SingleBuffer::zeroed(triangle_cap),
            triangle_meta: SingleBuffer::zeroed(triangle_cap),
            counters: SingleBuffer::zeroed(1),
        }
    }

    pub const fn vertex_cap(&self) -> usize {
        self.vertex_cap
    }

    pub const fn triangle_cap(&self) -> usize {
        self.triangle_cap
    }

    pub const fn vertex_buffer(&self) -> &VertexBuffer {
        &self.vert
    }

    pub const fn nota_buffer(&self) -> &NoTaBuffer {
        &self.nota
    }

    pub const fn triangle_buffer(&self) -> &TriangleBuffer {
        &self.triangle
    }

    pub const fn triangle_metadata_buffer(&self) -> &TriangleMetadataBuffer {
        &self.triangle_meta
    }

    pub const fn counters_buffer(&self) -> &CountersBuffer {
        &self.counters
    }

    pub fn bind_data_buffers_to(
        &self,
        v_index: u32,
        nt_index: u32,
        tri_index: u32,
        trimeta_index: u32,
    ) {
        self.vert.bind_shader_storage(v_index, 0);
        self.nota.bind_shader_storage(nt_index, 0);
        self.triangle.bind_shader_storage(tri_index, 0);
        self.triangle_meta.bind_shader_storage(trimeta_index, 0);
    }

    pub fn bind_data_buffers(&self) {
        self.bind_data_buffers_to(
            ssbo_binding!(Rendrs_GBANK_Vertex),
            ssbo_binding!(Rendrs_GBANK_NoTa),
            ssbo_binding!(Rendrs_GBANK_Triangle),
            ssbo_binding!(Rendrs_GBANK_TriangleMetadata),
        );
    }

    pub fn bind_counters_buffer_to(&self, index: u32) {
        self.counters.bind_shader_storage(index, 0);
    }

    pub fn bind_counters_buffer(&self) {
        self.bind_counters_buffer_to(ssbo_binding!(Rendrs_GBANK_Counters));
    }

    pub const fn glsl_vertex() -> GlslStorage {
        ethel::shader_glsl_ssbo! {
            buf Rendrs_GBANK_Vertex => {
                [dyn_array float : rendrs_gbank_vertex => each 3]
            }
        }
    }

    pub const fn glsl_nota() -> GlslStorage {
        ethel::shader_glsl_ssbo! {
            buf Rendrs_GBANK_NoTa => {
                [dyn_array vec4 : rendrs_gbank_nota]
            }
        }
    }

    pub const fn glsl_triangle() -> GlslStorage {
        ethel::shader_glsl_ssbo! {
            buf Rendrs_GBANK_Triangle => {
                [dyn_array uint : rendrs_gbank_triangle => each 3]
            }
        }
    }

    pub const fn glsl_triangle_meta() -> GlslStorage {
        ethel::shader_glsl_ssbo! {
            buf Rendrs_GBANK_TriangleMetadata => {
                [dyn_array TriangleMetadata : rendrs_gbank_triangle_meta]
            }
        }
    }

    pub const fn glsl_counters() -> GlslStorage {
        ethel::shader_glsl_ssbo! {
            buf Rendrs_GBANK_Counters => {
                rendrs_gbank_counter_vertex   : uint;
                rendrs_gbank_counter_triangle : uint;
            }
        }
    }
}

/// Create a geometry submission job to be attached to a geometry pass.
///
/// This is a compute shader that runs arbitrary logic with the purpose of
/// gathering, modifying, and submitting geometry.
///
/// The macro's syntax is similar to [`ethel's compute shaders`].
///
/// Optional blocks for additional data can be defined, these are, in order:
/// `uniform`, `sampler`, `image`, `type`, `ssbo`, `lib`, and `share`.
/// The definition syntax for each of these is identical to
/// [`ethel's compute shaders`].
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
/// * If the geometry unit is set to `Triangle`:
///   * `rendrs_GlobalTriangleID` the global index of the current working
///     triangle
///   * `rendrs_LocalTriangleID` the local index (to the current geometric
///     entity) of the current working triangle
///   * `rendrs_LocalVertexBase` the local offset (to the current geometric
///     entity) matching the current working triangle, derived as
///     `rendrs_LocalTriangleID * 3`.
///
/// *Note that the shader's workgroup is of linear size 64 (x=64,y=1,z=1).
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
/// * `uint Vertex(vec3 position, vec2 normal_oct, vec2 tangent_oct)`:
///     Submits a vertex at `position` with the specified `normal` and
///     `tangent` vectors. The last 2 are octahedron-encoded.
///     Returns the index of the submitted vertex.
/// * `uint Vertex(vec3 position, vec3 normal, vec3 tangent)`:
///     Submits a vertex at `position` with the specified `normal` and
///     `tangent` vectors. The last 2 are *not* octahedron-encoded, octahedron
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
        $name:ident unit $gkind:ident => {
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

            $source:literal
        }
    ) => {
        paste::paste! {
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
                    TYPE_TRIANGLE_METADATA

                    $($($type_glsl)+)?
                };
                ssbo {
                    $crate::geometry::GeometryBank::glsl_vertex()
                    $crate::geometry::GeometryBank::glsl_nota()
                    $crate::geometry::GeometryBank::glsl_triangle()
                    $crate::geometry::GeometryBank::glsl_triangle_meta()
                    $crate::geometry::GeometryBank::glsl_counters()
                    $($($ssbo_glsl)+)?
                };
                lib {
                    $($($e_lib)+)?

                    crate::pack::PACK_OCTAHEDRON_WRAP_UTIL;
                    crate::pack::PACK_OCTAHEDRON_ENCODE;
                    crate::pack::PACK_OCTAHEDRON_DECODE;
                    crate::pack::PACK_SPHERICAL_ENCODE;
                    crate::pack::PACK_SPHERICAL_DECODE;

                    ethel::shader::GlslLib::new("
                        uint Triangle(uint v0, uint v1, uint v2, uint geom_id) {
                            uint triangle_index = atomicAdd(rendrs_gbank_counter_triangle, 1);

                            rendrs_gbank_triangle[triangle_index][0] = v0;
                            rendrs_gbank_triangle[triangle_index][1] = v1;
                            rendrs_gbank_triangle[triangle_index][2] = v2;

                            TriangleMetadata metadata = TriangleMetadata(geom_id);
                            rendrs_gbank_triangle_meta[triangle_index] = metadata;

                            return triangle_index;
                        }
                    ");

                    ethel::shader::GlslLib::new("
                        uint Vertex(vec3 p, vec2 n_oct, vec2 t_oct) {
                            uint vertex_index = atomicAdd(rendrs_gbank_counter_vertex, 1);

                            rendrs_gbank_vertex[vertex_index][0] = p.x;
                            rendrs_gbank_vertex[vertex_index][1] = p.y;
                            rendrs_gbank_vertex[vertex_index][2] = p.z;

                            vec4 nota = vec4(n_oct.x, n_oct.y, t_oct.x, t_oct.y);
                            rendrs_gbank_nota[vertex_index] = nota;

                            return vertex_index;
                        }

                        uint Vertex(vec3 p, vec3 n, vec3 t) {
                            vec2 n_oct = rendrs_packOctahedron(n);
                            vec2 t_oct = rendrs_packOctahedron(t);
                            return Vertex(p, n_oct, t_oct);
                        }
                    ");

                    ethel::shader::GlslLib::new(concat!(
                        "void _submitGeometry(",
                        $crate::geometry_submission_job!(@unit_args $gkind),
                        "in uint rendrs_GeometryID",
                        ") {\n", $source, "\n}"
                    ));
                };
                $(share {
                    $($share_t:ident $share_n:ident $([$arr_c:expr])*;)*
                };)?

                src() {
                    "
                    // resolve rendrs_* variables
                    _submitGeometry(/**/);
                    ";
                }
            }
        }}
    };

    (@unit_args Geometry) => { "" };
    (@unit_args Triangle) => { "
        in uint rendrs_GlobalTriangleID,
        in uint rendrs_LocalTriangleID,
        in uint rendrs_LocalVertexBase,"
    };
}

geometry_submission_job! {
    Test unit Geometry => {
        "test"
    }
}
