use ethel::shader::GlslLib;

/// Utility function for octahedron encoding.
pub const OCTAHEDRON_UTIL_WRAP: GlslLib = ethel::shader_glsl_lib! {
    vec2 octahedronWrap [ v : vec2 ] => "
        return
            (1.0 - abs(v.xy)) *
            (v.xy >= 0.0 ? 1.0 : - 1.0)
        ;
    "
};

/// Encode a normalized 3-component vector in a 2-component vector.
///
/// Requires [`OCTAHEDRON_UTIL_WRAP`].
pub const OCTAHEDRON_ENCODE: GlslLib = ethel::shader_glsl_lib! {
    vec2 encodeOctahedron [ n : vec3 ] => "
        n /= (abs(n.x) + abs(n.y) + abs(n.z));
        n.xy = n.z >= 0.0 ? n.xy : octahedronWrap(n.xy);
        return n.xy * 0.5 + 0.5;
    "
};

/// Decode a 2-component octahedron-normal vector back into its original vector.
pub const OCTAHEDRON_DECODE: GlslLib = ethel::shader_glsl_lib! {
    vec3 encodeOctahedron [ f : vec2 ] => "
        f = f * 2.0 - 1.0;
        vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
        float t = clamp(-n.x, 0.0, 1.0);
        n.xy += n.xy >= 0.0 ? -t : t;
        return normalize(n);
    "
};

pub const SPHERICAL_ENCODE: GlslLib = ethel::shader_glsl_lib! {
    vec2 encodeSpherical [ n : vec3 ] => "
        vec2 f;
        f.x = atan2(n.y, n.x) * 0.318309886184;
        f.y = n.z;
        return f * 0.5 + 0.5;
    "
};

pub const SPHERICAL_DECODE: GlslLib = ethel::shader_glsl_lib! {
    vec3 decodeSpherical [ f : vec2 ] => "
        vec2 ang = f * 2.0 - 1.0;
        float xpi = ang.x * 3.14159265358979323846264338327950288;
        vec2 scth = vec2(
            sin(xpi),
            cos(xpi)
        );
        vec2 scphi = vec2(sqrt(1.0 - ang.y * ang.y), ang.y);
        return vec3(
            scth.y * scphi.x,
            scth.x * scphi.x,
            scphi.y
        );
    "
};
