use ethel::shader::ShaderKind;

ethel::shader_glsl! {
    struct RasterizeGeom > [460] {
        common {

        };

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
            lib {
                crate::pack::PACK_OCTAHEDRON_WRAP_UTIL;
                crate::pack::PACK_OCTAHEDRON_DECODE;
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
                //todo: more metadata in g channel, bit-packing
                uint R = rd_TriangleID;
                uint G = rd_GeoID;

                outColor = uvec4(R, G, 0, 0);
                ";
            }
        ];
    }
}
