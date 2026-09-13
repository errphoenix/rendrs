use ethel::shader::GlslLib;

/// Utility function for octahedron encoding.
///
/// Creates the `rendrs_wrapOctahedron` function. This is an utility function
/// used internally for octahedron encoding.
pub const PACK_OCTAHEDRON_WRAP_UTIL: GlslLib = ethel::shader_glsl_lib! {
    vec2 rendrs_wrapOctahedron [ v : vec2 ] => "
        return
            (1.0 - abs(v.xy)) * vec2(
                v.x >= 0.0 ? 1.0 : - 1.0,
                v.y >= 0.0 ? 1.0 : - 1.0
            )
        ;
    "
};

/// Pack a normalized 3-component vector in a 2-component vector.
///
/// Creates the `rendrs_packOctahedron` function, which takes a 3d vector
/// and returns a 2d vector.
///
/// The 2d vector must be unpacked in order to be used, its purpose is purely
/// to reduce memory footprint of stored unit vectors.
///
/// Requires [`PACK_OCTAHEDRON_WRAP_UTIL`].
pub const PACK_OCTAHEDRON_ENCODE: GlslLib = ethel::shader_glsl_lib! {
    vec2 rendrs_packOctahedron [ n : vec3 ] => "
        n /= (abs(n.x) + abs(n.y) + abs(n.z));
        n.xy = n.z >= 0.0 ? n.xy : rendrs_wrapOctahedron(n.xy);
        return n.xy * 0.5 + 0.5;
    "
};

/// Unpack a 2-component octahedron-normal vector back into its original vector.
///
/// Creates the `rendrs_unpackOctahedron` function, which takes a 2d vector
/// packed by [`PACK_OCTAHEDRON_ENCODE`] and returns the original 3d vector.
pub const PACK_OCTAHEDRON_DECODE: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_unpackOctahedron [ f : vec2 ] => "
        f = f * 2.0 - 1.0;
        vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
        float t = clamp(-n.x, 0.0, 1.0);
        n.xy += vec2(
            n.x >= 0.0 ? -t : t,
            n.y >= 0.0 ? -t : t
        );
        return normalize(n);
    "
};

/// Pack a 3-component unit vector in a 2-component vector using spherical
/// coordinates.
///
/// Creates the `rendrs_packSpherical` function, which takes a 3d vector
/// and returns a 2d vector.
///
/// The 2d vector must be unpacked in order to be used, its purpose is purely
/// to reduce memory footprint of stored unit vectors.
///
/// Note that [`Octahedron Encoding`](PACK_OCTAHEDRON_ENCODE) is usually
/// more efficient, and gives better resuls.
pub const PACK_SPHERICAL_ENCODE: GlslLib = ethel::shader_glsl_lib! {
    vec2 rendrs_packSpherical [ n : vec3 ] => "
        //todo: rewrite?
        vec2 f;
        f.x = atan(n.y, n.x) * 0.318309886184;
        f.y = n.z;
        return f * 0.5 + 0.5;
    "
};

/// Unpack a 2-component spherical-normal vector back into its original vector.
///
/// Creates the `rendrs_unpackSpherical` function, which takes a 2d vector
/// packed by [`PACK_SPHERICAL_ENCODE`] and returns the original 3d vector.
///
/// Note that [`Octahedron Encoding`](PACK_OCTAHEDRON_ENCODE) is usually
/// more efficient, and gives better resuls.
pub const PACK_SPHERICAL_DECODE: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_unpackSpherical [ f : vec2 ] => "
        //todo: rewrite?
        vec2 ang = f * 2.0 - 1.0;
        float xpi = ang.x * 3.1415926;
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
