use ethel::shader::GlslLib;

pub mod reflection_filtering;

/// Utility function for cubemap UV conversion.
///
/// Creates the `rendrs_CubemapUV` function, which takes the following
/// arguments:
/// * the 2d vector describing the UV on the singular face, in a [-1,1]
///   coordinate system.
/// * the face index of the cubemap as an unsigned integer ranging from 0 to 5
///
/// Returns a 3d unit vector representing the direction from the origin (the
/// center of the cubemap) to the texel at the given UV on the given face.
///
/// The returned vector can be used directly to sample a cubemap.
pub const LIB_UTIL_CUBEMAP_UV: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_CubemapUV[
        uv   : vec2,
        face : uint
    ] => "
        vec3 dir;
        switch(face) {
            case 0:
                dir = vec3( 1.0, -v, -u);
                break;
            case 1:
                dir = vec3(-1.0, -v,  u);
                break;
            case 2:
                dir = vec3( u,  1.0,  v);
                break;
            case 3:
                dir = vec3( u, -1.0, -v);
                break;
            case 4:
                dir = vec3( u, -v,  1.0);
                break;
            case 5:
                dir = vec3(-u, -v, -1.0);
                break;
        }
        return normalize(dir);
    "
};
