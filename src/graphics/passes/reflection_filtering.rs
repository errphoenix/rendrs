use ethel::shader::GlslLib;

/// The image binding index for the output texture to write the mipmap to.
///
/// Tip: use `glBindImageTexture`'s `level` field to specify the mip.
pub const IMAGE_BINDING_OUTPUT: u32 = 0;

ethel::shader_glsl_compute! {
    struct BSplineDownscale > [460] {
        workgroup [8, 8, 6];

        uniform {
            length 1, mip_level : uint        => u32;
            length 1, reference : samplerCube => i32;
        };
        image {
            on IMAGE_BINDING_OUTPUT => output : imageCube as rgba16f;
        };

        lib {
            LIB_BSPLINE_SMOOTHING_JACOBIAN;
            super::LIB_UTIL_CUBEMAP_UV;
        };

        src() {
            "
            ivec2 dst_px = gl_GlobalInvocationID.xy;
            uint face = gl_GlobalInvocationID.z;

            ivec2 sourceSize = textureSize(reference, mip_level);
            ivec2 dstSize    = sourceSize / 2;

            if (dst_px.x >= dstSize.x || dst_px.y >= dstSize.y) {
                return;
            }

            vec2 src_origin = vec2(dst_px * 2) + 1.0;
            vec2 px0 = src_origin + vec2(-0.75, -0.75);
            vec2 px1 = src_origin + vec2( 0.75, -0.75);
            vec2 px2 = src_origin + vec2(-0.75,  0.75);
            vec2 px3 = src_origin + vec2( 0.75,  0.75);

            vec2 uv0 = (px0 / vec2(sourceSize)) * 2.0 - 1.0;
            vec2 uv1 = (px1 / vec2(sourceSize)) * 2.0 - 1.0;
            vec2 uv2 = (px2 / vec2(sourceSize)) * 2.0 - 1.0;
            vec2 uv3 = (px3 / vec2(sourceSize)) * 2.0 - 1.0;

            vec3 d0 = rendrs_CubemapUV(uv0, face);
            vec3 d1 = rendrs_CubemapUV(uv1, face);
            vec3 d2 = rendrs_CubemapUV(uv2, face);
            vec3 d3 = rendrs_CubemapUV(uv3, face);

            float mip = float(mip_level);
            vec3 c0 = textureLod(reference, d0, mip).rgb;
            vec3 c1 = textureLod(reference, d1, mip).rgb;
            vec3 c2 = textureLod(reference, d2, mip).rgb;
            vec3 c3 = textureLod(reference, d3, mip).rgb;

            float w0 = rendrs_BSpline_smoothFactor(uv0);
            float w1 = rendrs_BSpline_smoothFactor(uv1);
            float w2 = rendrs_BSpline_smoothFactor(uv2);
            float w3 = rendrs_BSpline_smoothFactor(uv3);
            float wsum = w0+w1+w2+w3;
            c0 *= w0;
            c1 *= w1;
            c2 *= w2;
            c3 *= w3;

            vec3 filtered_rgb = (c0 + c1 + c2 + c3) / wsum;
            vec3 filtered     = vec4(filtered_rgb, 1.0);
            imageStore(output, ivec3(ipx - 1, face), filtered);
            ";
        }
    }
}

/// B-spline smoothing Jacobian factor.
///
/// Creates the `rendrs_BSpline_smoothFactor` function, which takes in a 2d vector
/// representing a 2d coordinate, returning the smoothing factor as a scalar
/// float value.
///
/// The returned value has already been smoothed further as:
/// ```
/// 0.5 * (1.0 + J(x,y))
/// ```
/// where `J(x,y)` is the smoothing Jacobian factor before further smoothing.
pub const LIB_BSPLINE_SMOOTHING_JACOBIAN: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_BSpline_smoothFactor[
        p : vec2
    ] => "
        float sq = p.x*p.x + p.y*p.y + 1.0;
        float J  = pow(sq, -1.5);
        return 0.5 * (1.0 + J);
    "
};
