use ethel::shader::GlslLib;
use janus::texture::{MipLevels, Tex, TextureView};

use crate::{
    ComputePass,
    pipeline::{ImageAccessKind, ImageObject, ImageObjectTarget, SamplerObject},
};

pub type BSplineDownscalePass = ComputePass<BSplineDownscaleCtxWrapper, 0, 0>;

#[derive(Debug)]
pub struct BSplineDownscaleCtx<'ctx> {
    pub shader: &'ctx ComputeShaderBSplineDownscale,
    pub target: TextureView,
    /// Mip-level index to compute
    pub mip_level: MipLevels,
}
crate::context_wrapper!(for<'ctx> BSplineDownscaleCtx);

pub const fn pass(shader: &ComputeShaderBSplineDownscale) -> BSplineDownscalePass {
    let handle_view = shader.compute_handle().view();
    BSplineDownscalePass::new(handle_view, [], [], |_, ctx| {
        let BSplineDownscaleCtx {
            shader,
            target,
            mip_level,
        } = ctx;

        let mip_level = mip_level.get();
        shader.uniform_mip_level_intv([mip_level]);

        let target = *target;
        let input = SamplerObject::with_mip_view(target, mip_level);
        let output = ImageObjectTarget::with_mip_level(
            ImageObject::DirectTexture(target),
            ImageAccessKind::WriteOnly,
            DOWNSCALE_IMAGE_BINDING_OUTPUT,
            None,
            mip_level,
        );

        input.bind(0);
        output.bind();

        let (w, h) = target.size();
        let wg_x = (w as u32).div_ceil(WORKSPACE_SIZE_XY);
        let wg_y = (h as u32).div_ceil(WORKSPACE_SIZE_XY);
        [wg_x, wg_y, 1]
    })
}

/// The image binding index for the output texture to write the mipmap to.
///
/// Tip: use `glBindImageTexture`'s `level` field to specify the mip.
pub const DOWNSCALE_IMAGE_BINDING_OUTPUT: u32 = 0;
pub const WORKSPACE_SIZE_XY: u32 = 8;

ethel::shader_glsl_compute! {
    struct BSplineDownscale > [460] {
        workgroup [8, 8, 6];

        uniform {
            // mip level being computed (never 0)
            length 1, mip_level : int         => i32;
            length 1, reference : samplerCube => i32;
        };
        image {
            on DOWNSCALE_IMAGE_BINDING_OUTPUT => output : imageCube as rgba16f;
        };

        lib {
            LIB_BSPLINE_JACOBIAN_WEIGHT;
            super::LIB_UTIL_CUBEMAP_UV;
        };

        src() {
            "
            ivec3 id = gl_GlobalInvocationID;

            int sourceSize = textureSize(reference, mip_level - 1).x;
            int dstSize    = sourceSize / 2;

            if (id.x >= dstSize || id.y >= dstSize) {
                return;
            }

            float inv_size = 1.0 / float(dstSize);

            float u0 = (float(id.x) * 2.0 + 1.0 - 0.75) * inv_size - 1.0;
            float u1 = (float(id.x) * 2.0 + 1.0 + 0.75) * inv_size - 1.0;
            float v0 = (float(id.y) * 2.0 + 1.0 - 0.75) * -inv_size + 1.0;
            float v1 = (float(id.y) * 2.0 + 1.0 + 0.75) * -inv_size + 1.0;

            float w0 = rendrs_BSpline_Weight(vec2(u0, v0));
            float w1 = rendrs_BSpline_Weight(vec2(u1, v0));
            float w2 = rendrs_BSpline_Weight(vec2(u0, v1));
            float w3 = rendrs_BSpline_Weight(vec2(u1, v1));
            const float wsum = 0.5 / (w0+w1+w2+w3);
            w0 = w0 * wsum + 0.125;
            w1 = w1 * wsum + 0.125;
            w2 = w2 * wsum + 0.125;
            w3 = w3 * wsum + 0.125;

            vec3 d0 = rendrs_CubemapUV(vec2(u0, v0), id.z);
            vec3 d1 = rendrs_CubemapUV(vec2(u1, v0), id.z);
            vec3 d2 = rendrs_CubemapUV(vec2(u0, v1), id.z);
            vec3 d3 = rendrs_CubemapUV(vec2(u1, v1), id.z);

            vec4 color = vec4(vec3(0.0), 1.0);
            color.rgb += textureLod(reference, d0, mip_level).rgb * w0;
            color.rgb += textureLod(reference, d1, mip_level).rgb * w1;
            color.rgb += textureLod(reference, d2, mip_level).rgb * w2;
            color.rgb += textureLod(reference, d3, mip_level).rgb * w3;

            imageStore(output, id, color);
            ";
        }
    }
}

/// B-spline smoothing Jacobian weight.
///
/// Creates the `rendrs_BSpline_Weight` function, which takes in a 2d vector
/// representing a 2d coordinate, returning the smoothing factor as a scalar
/// float value.
///
/// The returned value has already been smoothed further as:
/// ```
/// 0.5 * (1.0 + J(x,y))
/// ```
/// where `J(x,y)` is the smoothing Jacobian factor before further smoothing.
pub const LIB_BSPLINE_JACOBIAN_WEIGHT: GlslLib = ethel::shader_glsl_lib! {
    float rendrs_BSpline_Weight[
        p : vec2
    ] => "
        float sq = p.x*p.x + p.y*p.y + 1.0;
        return sq*sqrt(sq);
    "
};
