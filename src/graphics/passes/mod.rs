use ethel::shader::GlslLib;

pub mod brdf_bake_specular;
pub mod reflection_filtering;

/// Utility function for cubemap UV conversion.
///
/// Creates the `rendrs_CubemapUV` function, which takes the following
/// arguments:
/// * the 2d vector describing the UV on the singular face, in a [-1,1]
///   coordinate system.
/// * the face index of the cubemap as an unsigned integer ranging from 0 to 5
///
/// Returns a 3d vector representing the direction from the origin (the
/// center of the cubemap) to the texel at the given UV on the given face.
///
/// The returned vector is not normalized, as OpenGL does not require this.
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
        return dir;
    "
};

/// Util. function for radical inverse Van der Corput sequence.
///
/// Creates the `rendrs_RInv_VanDerCorput` function, which takes a single
/// unsigned integer argument as its bits and returns a floting-point value.
pub const LIB_UTIL_VAN_DER_CORPUT: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_RInv_VanDerCorput[ bits : uint ] => "
        const float _2P32 = 2.3283064365386963e-10;
        uint rbits = bitfieldReverse(bits);
        return float(rbits) * _2P32;
    "
};

/// Util. function for Hammersley quasi-random 2D point generation.
///
/// Creates the `rendrs_Hammersley2D` function, which takes the current sample
/// index and the total number of samples as its arguments as unsigned
/// integers.
///
/// Returns a 2d vector as a quasi-random point, which can be used for
/// importance sampling.
///
/// Requires [`LIB_UTIL_VAN_DER_CORPUT`].
pub const LIB_UTIL_HAMMERSLEY_2D: GlslLib = ethel::shader_glsl_lib! {
    vec2 rendrs_Hammersley2D[
        index : uint,
        N     : uint
    ] => "
        float x = float(index) / float(N);
        float y = rendrs_RInv_VanDerCorput(index);
        return vec2(x, y);
    "
};

/// Importance sampling mapping according to a GGX specular lobe.
///
/// Creates the `rendrs_GGX_ImportanceSample` function, which takes the
/// following arguments:
/// * the 2d vector representing the spherical coordinate/angle
/// * the 3d vector of the unit normal of the surface
/// * the roughness scalar value of this surface
///
/// Returns a 3d vector.
pub const LIB_GGX_IMPSAMPLE: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_GGX_ImportanceSample[
        spherical : vec2,
        normal    : vec3,
        roughness : float
    ] => "
        float r2 = roughness*roughness;
        float phi = 3.14159265359*2.0*spherical.x;
        float cosTheta = sqrt((1.0 - spherical.y) / (1.0 + (r2*r2 - 1.0) * spherical.y));
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        vec3 H = vec3(
            cos(phi) * sinTheta,
            sin(phi) * sinTheta,
            cosTheta
        );

        vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = normalize(cross(up, normal));
        vec3 bitangent = cross(normal, tangent);

        vec3 sample = tangent * H.x + bitangent * H.y + normal * H.z;
    "
};

/// The Smith NDF masking function `G1` used to mask microfacet normals.
///
/// Creates the `rendrs_ndfMask_SmithG1` function, which has the following paramaters:
/// * the angle (dot product) between the **microfacet** surface normal and the
///   vector pointing from the surface to the point
/// * the result of the NDF's `lambda` function for the 3d vector pointing
///   from the surface to the point
///
/// The 'point' is usually the viewpoint or the light's origin.
///
/// The `lambda` function is different depending on the chosen NDF.
///
/// The function returns a floating-point scalar.
///
/// The function assumes the microfacet normal cannot form an angle
/// greater than 90 degrees, therefore they are not clamped. It assumes the
/// use of the 'half-vector' `h` as the microfacet surface normal (param. 1),
/// which is to be perfectly aligned with the microfacet normal vector and
/// points exactly halfway the vectors pointing towards the viewpoint and the
/// light, from which it is derived.
///
/// Mutually exclusive with [`LIB_NDF_MASK_SMITH_G1_GGX_KARIS_APPROX`].
pub const LIB_NDF_MASK_SMITH_G1: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndfMask_SmithG1[
        MdotP        : float,
        lambda_point : float
    ] => "
        float n = MdotP > 0.0 ? 1.0 : 0.0;
        float d = 1.0 + lambda_point;
        return n / d;
    "
};

/// The Smith normal distribution joint masking-shadowing function `G2`, used
/// to mask microfacets from 2 visible directions.
///
/// This is the "separable" form defined by Heitz: the simplest, but prone to
/// over-darkening as it incorrectly uncorrelates masking and shadowing.
/// However, some applications are known to still utilize this approach.
///
/// Creates the `rendrs_ndfMask_SmithG2_Separable` function, which has the following
/// parameters:
/// * the angle (dot product) between the **microfacet** surface normal and the
///   vector pointing from the surface to the viewpoint
/// * the angle (dot product) between the **microfacet** surface normal and the
///   vector pointing from the surface to the light
/// * the result of the NDF's `lambda` function for the 3d vector pointing
///   from the surface to the viewpoint
/// * the result of the NDF's `lambda` function for the 3d vector pointing
///   from the surface to the light
///
/// The `lambda` function is different depending on the chosen NDF.
///
/// The function returns a floating-point scalar.
///
/// Depends on [`LIB_NDF_MASK_SMITH_G1`]
///
/// The function assumes the microfacet normal cannot form an angle
/// greater than 90 degrees, therefore they are not clamped. It assumes the
/// use of the 'half-vector' `h` as the microfacet surface normal (param. 1),
/// which is to be perfectly aligned with the microfacet normal vector and
/// points exactly halfway the vectors pointing towards the viewpoint and the
/// light, from which it is derived.
pub const LIB_NDF_MASK_SMITH_G2_SEPARABLE: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndfMask_SmithG2_Separable[
        MdotV        : float,
        MdotL        : float,
        lambda_view  : float,
        lambda_light : float
    ] => "
        float a = rendrs_ndf_SmithG1(MdotV, lambda_view);
        float b = rendrs_ndf_SmithG1(MdotL, lambda_light);
        return a * b;
    "
};

/// The Smith normal distribution joint masking-shadowing function `G2`, used
/// to mask microfacets from 2 visible directions.
///
/// This is the "height-correlated" form defined by Heitz: this form takes
/// advantage of the fact that the light and view directions are correlated
/// by their relative alignment, but more importantly they both relate to the
/// point's height relative to the rest of the surface.
///
/// Creates the `rendrs_ndfMask_SmithG2_Separable` function, which has the
/// following parameters:
/// * the angle (dot product) between the **microfacet** surface normal and the
///   vector pointing from the surface to the viewpoint
/// * the angle (dot product) between the **microfacet** surface normal and the
///   vector pointing from the surface to the light
/// * the result of the NDF's `lambda` function for the 3d vector pointing
///   from the surface to the viewpoint
/// * the result of the NDF's `lambda` function for the 3d vector pointing
///   from the surface to the light's origin
///
/// The `lambda` function is different depending on the chosen NDF.
///
/// The function returns a floating-point scalar.
///
/// The function assumes the microfacet normal cannot form an angle
/// greater than 90 degrees, therefore they are not clamped. It assumes the
/// use of the 'half-vector' `h` as the microfacet surface normal (param. 1),
/// which is to be perfectly aligned with the microfacet normal vector and
/// points exactly halfway the vectors pointing towards the viewpoint and the
/// light, from which it is derived.
///
/// Mutually exclusive with [`LIB_NDF_MASK_SMITH_G2_HEIGHT_GGX_HAMMON_APPROX`].
pub const LIB_NDF_MASK_SMITH_G2_HEIGHT: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndfMask_SmithG2_Height[
        MdotV        : float,
        MdotL        : float,
        lambda_view  : float,
        lambda_light : float
    ] => "
        float m0 = MdotV > 0.0 ? 1.0 : 0.0;
        float m1 = MdotL > 0.0 ? 1.0 : 0.0;
        float d = 1.0 + lambda_view + lambda_light;
        float n = m0 * m1;
        return n / d;
    "
};

/// The Beckmann normal distribution function.
///
/// Creates the `rendrs_ndf_Beckmann` function, with the following parameters:
/// * the angle (dot product) between the **geometric** surface normal and the
///   **microfacet** surface normal
/// * the scalar roughness value
///
/// The lambda function of the Beckmann NDF corresponds to
/// [`LIB_NDF_LAMBDA_A`].
pub const LIB_NDF_BECKMANN: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_Beckmann[
        NdotM     : float,
        roughness : float
    ] => "
        float m = NdotM > 0.0 ? 1.0 : 0.0;
        float NdotM2 = NdotM*NdotM;
        float NdotM4 = NdotM2*NdotM2;
        float a2 = roughness*roughness;

        float id2  = NdotM2 - 1.0;
        float a2d2 = a2 * NdotM2;
        float g = exp(id2 / a2d2);

        float a2pi = 3.14159 * a2;
        float a2pit = a2p2i * NdotM4;
        float f = m / a2pit;

        return f * g;
    "
};

/// Derive intermediate `a` variable for an NDF `lambda` function.
///
/// Creates the `rendrs_ndf_lambda_A` function, with the following
/// parameters:
/// * the angle (dot product) between the **geometric** surface normal and the
///   vector pointing from the surface to another point, which is usually the
///   viewpoint or the light
/// * the scalar roughness value
///
/// This function is mutually exclusive to [`LIB_NDF_BECKMANN_LAMBDA_A_NOSQRT`].
pub const LIB_NDF_LAMBDA_A: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_lambda_A[
        NdotP     : float,
        roughness : float
    ] => "
        float NdotP2 = NdotP*NdotP;
        float dr = roughness * sqrt(1.0 - NdotP2);
        return NdotP / dr;
    "
};

/// Derive intermediate `a` (squared) variable for an NDF `lambda` function.
///
/// Creates the `rendrs_ndf_lambda_A` function, with the following
/// parameters:
/// * the angle (dot product) between the **geometric** surface normal and the
///   vector pointing from the surface to another point, which is usually the
///   viewpoint or the light
/// * the scalar roughness value
///
/// This returns the squared `a` value used in the GGX lambda function.
///
/// The single difference with the square-root variant is that this function
/// lacks a square-root, which makes it a little cheaper.
///
/// This is meant to be used in a case where a `lambda` function requires only
/// the square of the `a` variable, which makes the square-root unnecessary.
/// An example is the lambda function for the GGX NDF.
///
/// This function is mutually exclusive to [`LIB_NDF_LAMBDA_A`].
pub const LIB_NDF_LAMBDA_A_NOSQRT: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_lambda_A[
        NdotP     : float,
        roughness : float
    ] => "
        float NdotP2 = NdotP*NdotP;
        float r2 = roughness*roughness;
        float d = r2 * (1.0 - NdotP2);
        return NdotP2 / d;
    "
};

/// The Beckmann lambda function, required for the Beckmann NDF.
///
/// Creates the `rendrs_ndf_Beckmann_lambda` function, which takes in a
/// single scalar value as its argument. This value must be the `a` variable
/// as returned by [`LIB_NDF_LAMBDA_A`].
pub const LIB_NDF_BECKMANN_LAMBDA: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_Beckmann_lambda[
        a : float
    ] => "
        if (a < 1.6) {
            float aa = a*a;
            float a0 = 1.259 * a;
            float a1 = 0.396 * aa;
            float a2 = 3.535 * a;
            float a3 = 2.181 * aa;
            float n = 1.0 - a0 + a1;
            float d = a2 + a3;
            return n / d;
        } else {
            return 0.0;
        }
    "
};

/// The GGX normal distribution function.
///
/// Creates the `rendrs_ndf_GGX` function, with the following parameters:
/// * the angle (dot product) between the **geometric** surface normal and the
///   **microfacet** surface normal
/// * the scalar roughness value
///
/// The lambda function of the GGX NDF corresponds to [`LIB_NDF_GGX_LAMBDA`].
pub const LIB_NDF_GGX: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_GGX[
        NdotM     : float,
        roughness : float
    ] => "
        float m = NdotM > 0.0 ? 1.0 : 0.0;
        float a2 = roughness*roughness;
        float am = a2 - 1.0;
        float NdotM2 = NdotM*NdotM;
        float d = NdotM2 * am + 1.0;
        float d2 = d*d;
        float pid2 = 3.14159 * d2;
        float n = m * a2;
        return n / pid2;
    "
};

/// The GGX lambda function, required for the GGX NDF.
///
/// Creates the `rendrs_ndf_GGX_lambda` function, which takes in a single
/// scalar value as its argument. This value must be the squared `a` variable
/// as returned by [`LIB_NDF_LAMBDA_A_NOSQRT`].
///
/// [`LIB_NDF_LAMBDA_A`] can also be used for `a`, but the returned value
/// must be squared first.
///
/// Note that [`LIB_NDF_LAMBDA_A_NOSQRT`] does not require its return value to
/// be squared.
///
/// The function returns a floating-point scalar.
pub const LIB_NDF_GGX_LAMBDA: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_GGX_lambda[
        a2 : float
    ] => "
        float ia2 = 1.0 / a2;
        float s = sqrt(1.0 + ia2);
        float n = -1.0 + s;
        return n / 2.0;
    "
};

/// A GGX-compatible approximation for the Smith NDF masking function `G1`
/// used to mask microfacet normals.
///
/// This approximation drops the requirement for the lambda function, but
/// requires the roughness.
/// It is a specific optimization that is only compatible with the GGX model,
/// proposed by Karis in his 2013 "Real Shading in Unreal Engine 4".
///
/// Creates the `rendrs_ndf_SmithG1` function, which has the following
/// paramaters:
/// * the angle (dot product) between the **geometric** surface normal and the
///   vector pointing from the surface to another point, which is usually the
///   viewpoint or the light
/// * the scalar roughness value of the surface
///
/// The 'point' is usually the viewpoint or the light's origin.
///
/// The function returns a floating-point scalar.
///
/// Mutually exclusive with [`LIB_NDF_MASK_SMITH_G1`].
pub const LIB_NDF_MASK_SMITH_G1_GGX_KARIS_APPROX: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_SmithG1[
        NdotP     : float,
        roughness : float
    ] => "
        float n = 2.0 * NdotP;
        float ma2 = 2.0 - roughness;
        float d = NdotP * ma2 + roughness;
        return n / d;
    "
};

/// A GGX-compatible approximation of the Smith normal distribution joint
/// masking-shadowing function `G2`, used to mask microfacets from 2 visible
/// directions.
///
/// This approximation is described by Hammon in his 2017 GDC talk "PBR Diffuse
/// Lighting for GGX+Smith Microsurfaces".
///
/// This is the "height-correlated" form defined by Heitz: this form takes
/// advantage of the fact that the light and view directions are correlated
/// by their relative alignment, but more importantly they both relate to the
/// point's height relative to the rest of the surface.
///
/// Creates the `rendrs_ndf_SmithG2_Height` function, which has the following
/// parameters:
/// * the absolute angle (abs dot product) between the **geometric** surface
///   normal and the vector pointing from the surface to the viewpoint
/// * the absolute angle (abs dot product) between the **geometric** surface
///   normal and the vector pointing from the surface to the light
/// * the scalar roughness value of the surface
///
/// The function returns a floating-point scalar.
///
/// Mutually exclusive with [`LIB_NDF_MASK_SMITH_G2_HEIGHT`].
///
/// **NOTE**: this optimization includes the term of the specular BRDF
/// denominator `4 * |dot(n,l)| * |dot(n,v)|`. This must be taken into account
/// when evaluating the specular BRDF.
///
/// Practically, the specular BRDF equation should go from:
/// ```
/// (FRESNEL * G2 * NDF) / denom
/// ```
/// to:
/// ```
/// (FRESNEL * NDF) * G2
/// ```
/// where G2 is the Hammon approximation of the G2 Smith function, which
/// already divides by `denom`.
///
/// where `denom` is the specular BRDF denominator as
/// `4 * |dot(n,l)| * |dot(n,v)|`
pub const LIB_NDF_MASK_SMITH_G2_HEIGHT_GGX_HAMMON_APPROX: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_ndf_SmithG2_Height[
        NdotV     : float,
        NdotL     : float,
        roughness : float
    ] => "
        const float N = 0.5;
        float a = 2.0 * NdotL * NdotV;
        float b = NdotL + NdotV;
        float d = mix(a, b, roughness);
        return N / d;
    "
};

ethel::shader_glsl_struct! {
    struct FresnelParams {
        albedo  : [f32; 3] => vec3,
        fresnel : [f32; 3] => vec3
    }
}

/// Evaluate Fresnel-Schlick parameters.
///
/// Creates the `rendrs_FresnelParams` function, which has the following arguments:
/// * a scalar "metalness" factor from 0 to 1
/// * the RGB surface color (albedo)
/// * the RGB "dielectric fallback" color, basically the default specular
///   color of the surface if it is not a metallic (`metalness = 0`, thus
///   dielectric) surface. A good standard value is 0.04 for all 3 channels.
///
/// Returns a `FresnelParams` struct, which first field is the new `albedo`
/// surface color to be used as diffuse color, and then the second field
/// `fresnel` is the `F0` value to be used in the Fresnel function such as
/// the [`Fresnel-Schlick`](LIB_FRESNEL_SCHLICK) approximation function.
pub const LIB_FRESNEL_PARAMS: GlslLib = ethel::shader_glsl_lib! {
    FresnelParams rendrs_FresnelParams[
        metalness           : float,
        surface_color       : vec3,
        dielectric_fallback : vec3
    ] => "
        vec3 f0 = mix(
            dielectric_fallback,
            surface_color,
            metalness
        );
        vec3 ss = mix(
            surface_color,
            vec3(0.0),
            metalness
        );
        return FresnelParams(ss, f0);
    "
};

/// The standard Fresnel-Schlick approximation function.
///
/// Creates `rendrs_FresnelSchlick` function, which has the following
/// arguments:
/// * the positive part of the angle (dot product) between the **microfacet**
///   surface normal and the vector pointing from the surface to the light
/// * the fresnel F0 value as evaluated by [`LIB_FRESNEL_PARAMS`]
///
/// Returns a 3d vector, intended as an RGB color as the Fresnel term of the
/// specular BRDF.
pub const LIB_FRESNEL_SCHLICK: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_FresnelSchlick[
        NdotL   : float,
        fresnel : vec3
    ] => "
        float iNdotL = 1.0 - NdotL;
        float iNdotL5 = iNdotL*iNdotL*iNdotL*iNdotL*iNdotL;
        vec3 f = (1.0 - fresnel) * iNdotL5;
        return fresnel + f;
    "
};
