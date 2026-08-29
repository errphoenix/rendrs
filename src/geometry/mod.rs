use ethel::{render::buffer::SingleBuffer, shader::GlslStorage};

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
}

/// Vertex buffer represented as `x, y, z` 32-bit floats.
pub type VertexBuffer = SingleBuffer<[f32; 3]>;

/// Normals-tangents buffer represented as 2 octahedron encoded 2d vectors.
pub type NoTaBuffer = SingleBuffer<[f32; 3]>;

/// Basic triangle primitive buffer represented as simple vertex indices.
pub type TriangleBuffer = SingleBuffer<[u32; 3]>;

#[derive(Debug, Default)]
pub struct GeometryBank {
    vertex_cap: usize,
    triangle_cap: usize,
    vert: VertexBuffer,
    nota: NoTaBuffer,
    triangles: TriangleBuffer,
}
impl GeometryBank {
    pub fn new(vertex_cap: usize, triangle_cap: usize) -> Self {
        Self {
            vertex_cap,
            triangle_cap,
            vert: SingleBuffer::zeroed(vertex_cap),
            nota: SingleBuffer::zeroed(vertex_cap),
            triangles: SingleBuffer::zeroed(triangle_cap),
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
        &self.triangles
    }

    pub fn bind_all_to(&self, v_index: u32, nt_index: u32, tri_index: u32) {
        self.vert.bind_shader_storage(v_index, 0);
        self.nota.bind_shader_storage(nt_index, 0);
        self.triangles.bind_shader_storage(tri_index, 0);
    }

    pub fn bind_all(&self) {
        self.bind_all_to(
            ssbo_binding!(Rendrs_GBANK_Vertex),
            ssbo_binding!(Rendrs_GBANK_NoTa),
            ssbo_binding!(Rendrs_GBANK_Triangle),
        );
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
/// After the optional blocks, the shader's 'source' is defined: this is a
/// single string literal containing the relevant GLSL code.
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
                $(type {
                    $($type_glsl)+
                };)?
                ssbo {
                    $crate::geometry::GeometryBank::glsl_vertex()
                    $crate::geometry::GeometryBank::glsl_nota()
                    $($($ssbo_glsl)+)?
                };
                lib {
                    $($($e_lib)+)?

                    ethel::shader::GlslLib::new("
                        void pushTriangle(){} // todo
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
