use ethel::shader::Constant;
use ethel::shader::GlslLib;
use janus::texture::{MipLevels, Tex, TextureView};

use crate::pipeline::CtxType;
use crate::pipeline::Sampler;
use crate::{
    ComputePass,
    pipeline::{ImageAccessKind, ImageObject, ImageObjectTarget, SamplerObject},
};

pub type BSplineDownscalePass = ComputePass<BSplineDownscaleCtxWrapper, 0, 0>;

#[derive(Debug)]
pub struct BSplineDownscaleCtx {
    /// Acts as both input and output
    pub target: TextureView,
    /// Mip-level index to compute
    ///
    /// fetches `mip_level - 1` and writes to `mip_level`
    pub mip_level: MipLevels,
}
crate::context_wrapper!(BSplineDownscaleCtx);

pub const fn rf_bspline_downsample(shader: &ComputeShaderBSplineDownscale) -> BSplineDownscalePass {
    let handle_view = shader.compute_handle().view();
    BSplineDownscalePass::new(handle_view, [], [], |_, ctx| {
        let BSplineDownscaleCtx { target, mip_level } = ctx;

        let mip_level = mip_level.get();

        let target = *target;
        let input = SamplerObject::from_texture_with_mip_view(target, mip_level - 1);
        let output = ImageObjectTarget::with_mip_level(
            ImageObject::DirectTexture(target),
            ImageAccessKind::WriteOnly,
            DOWNSCALE_IMAGE_BINDING_OUTPUT,
            None,
            mip_level,
        );

        input.bind(DOWNSCALE_IMAGE_BINDING_OUTPUT);
        output.bind();

        let (w, _) = target.size();
        let wg_num = (w as u32).div_ceil(WORKSPACE_SIZE_XY);
        [wg_num, wg_num, 1]
    })
}

pub const FILTERING_MIP_COUNT: u32 = 6;

/// The image binding index for the output texture to write the mipmap to.
///
/// Tip: use `glBindImageTexture`'s `level` field to specify the mip.
pub const DOWNSCALE_IMAGE_BINDING_OUTPUT: u32 = 0;
pub const WORKSPACE_SIZE_XY: u32 = 8;

ethel::shader_glsl_compute! {
    struct BSplineDownscale > [460] {
        workgroup [8, 8, 6];

        uniform {
            length 1, reference : samplerCube => i32;
        };
        image {
            on DOWNSCALE_IMAGE_BINDING_OUTPUT => out_filtered : imageCube as rgba16f writeonly;
        };

        lib {
            LIB_BSPLINE_JACOBIAN_WEIGHT;
            super::LIB_UTIL_CUBEMAP_UV;
        };

        src() {
            "
            uvec3 id = gl_GlobalInvocationID;

            int sourceSize = textureSize(reference, 0).x;
            int dstSize    = sourceSize / 2;

            if (id.x >= dstSize || id.y >= dstSize) {
                return;
            }

            float inv_size = 1.0 / float(dstSize);

            float u0 = ((float(id.x) - 0.75) * inv_size) * 2.0 - 1.0;
            float u1 = ((float(id.x) + 0.75) * inv_size) * 2.0 - 1.0;
            float v0 = ((float(id.y) - 0.75) * inv_size) * 2.0 - 1.0;
            float v1 = ((float(id.y) + 0.75) * inv_size) * 2.0 - 1.0;

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
            color.rgb += textureLod(reference, d0, 0).rgb * w0;
            color.rgb += textureLod(reference, d1, 0).rgb * w1;
            color.rgb += textureLod(reference, d2, 0).rgb * w2;
            color.rgb += textureLod(reference, d3, 0).rgb * w3;

            imageStore(out_filtered, ivec3(id), color);
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

pub type PrefilterCubemapPass =
    ComputePass<PrefilterCubemapCtx, 1, { FILTERING_MIP_COUNT as usize }>;

#[derive(Debug)]
pub struct PrefilterCubemapCtx {
    res_chain: [u32; FILTERING_MIP_COUNT as usize],
    total_pixels: u32,
}
impl PrefilterCubemapCtx {
    pub const fn new(base_resolution: u32) -> Self {
        let mut total_pixels = 0;
        let mut res_chain = [0u32; FILTERING_MIP_COUNT as usize];
        let mut i = 0;
        while i < FILTERING_MIP_COUNT {
            let resi = base_resolution >> i;
            total_pixels += resi * resi;
            res_chain[i as usize] = resi;
            i += 1;
        }
        Self {
            res_chain,
            total_pixels,
        }
    }

    pub const fn total_mip_pixels(&self) -> u32 {
        self.total_pixels
    }

    pub const fn mip_resolution_chain(&self) -> [u32; FILTERING_MIP_COUNT as usize] {
        self.res_chain
    }

    pub const fn base_resolution(&self) -> u32 {
        self.res_chain[0]
    }
}
impl CtxType for PrefilterCubemapCtx {
    type Ctx<'ctx> = Self;
}

pub const fn rf_prefilter_cubemap(
    shader: &ComputeShaderPrefilterCubemap,
    source_sampler: TextureView,
    output: TextureView,
) -> PrefilterCubemapPass {
    let handle_view = shader.compute_handle().view();
    let sampler = SamplerObject::from_texture(source_sampler);
    let image = ImageObject::DirectTexture(output);
    let outputs = ImageObjectTarget::from_texture_mips::<{ FILTERING_MIP_COUNT as usize }>(
        image,
        ImageAccessKind::WriteOnly,
        IMAGE_BINDING_FILTERMIPS_MIP0,
        None,
        0,
    );
    PrefilterCubemapPass::new(
        handle_view,
        [Sampler::wrap(sampler, SAMPLER_UNIT_INPUT_CUBEMAP)],
        outputs,
        |_, ctx| {
            let wg_size = ctx.total_pixels.div_ceil(WORKSPACE_SIZE_XY);
            [wg_size, 6, 1]
        },
    )
}

pub const SAMPLER_UNIT_INPUT_CUBEMAP: u32 = 0;
pub const IMAGE_BINDING_FILTERMIPS_MIP0: u32 = 0;
pub const IMAGE_BINDING_FILTERMIPS_MIP1: u32 = 1;
pub const IMAGE_BINDING_FILTERMIPS_MIP2: u32 = 2;
pub const IMAGE_BINDING_FILTERMIPS_MIP3: u32 = 3;
pub const IMAGE_BINDING_FILTERMIPS_MIP4: u32 = 4;
pub const IMAGE_BINDING_FILTERMIPS_MIP5: u32 = 5;

// todo: adjust indexing to batch many probes under one dispatch call
//       instead of dispatching per-probe like downsampling pass.
//       unlike downsampling, there is no (mip) dependency chain.
ethel::shader_glsl_compute! {
    struct PrefilterCubemap > [460] {
        workgroup [64, 1, 1];

        sampler {
            // resolution of mip 0 / aka probe reflection map res
            on SAMPLER_UNIT_INPUT_CUBEMAP => input_cubemap : samplerCube;
        };
        image {
            on IMAGE_BINDING_FILTERMIPS_MIP0 => out_mip0 : imageCube as rgba16f writeonly;
            on IMAGE_BINDING_FILTERMIPS_MIP1 => out_mip1 : imageCube as rgba16f writeonly;
            on IMAGE_BINDING_FILTERMIPS_MIP2 => out_mip2 : imageCube as rgba16f writeonly;
            on IMAGE_BINDING_FILTERMIPS_MIP3 => out_mip3 : imageCube as rgba16f writeonly;
            on IMAGE_BINDING_FILTERMIPS_MIP4 => out_mip4 : imageCube as rgba16f writeonly;
            on IMAGE_BINDING_FILTERMIPS_MIP5 => out_mip5 : imageCube as rgba16f writeonly;
        };

        type {
            MipSizesGlslStruct::as_definition()
        };

        const {
            Constant::new("NUM_TAPS", 32)
        };

        lib {
            LIB_TABLE_CUBEMAP_COEFFICIENTS;
            LIB_HELPER_MIP_HSIZES;
            super::LIB_UTIL_CUBEMAP_UV;
        };

        src() {
            "
            // id.x = linear address of texel
            // id.y = cubemap face index
            // (id.x and id.y are used to index output)
            // id.x = output texel x
            // id.y = output texel y
            // id.z = face index
            ivec3 id = ivec3(gl_GlobalInvocationID);

            int base_resolution = textureSize(input_cubemap, 0).x;
            const MipSizes mip_hsizes = _helper_Mip_hsizes(base_resolution);

            uint level = 0;
            if (id.x < mip_hsizes.size_0) {
                level = 0;
            } else if (id.x < mip_hsizes.size_1) {
                level = 1;
                id.x -= mip_hsizes.size_0;
            } else if (id.x < mip_hsizes.size_2) {
                level = 2;
                id.x -= mip_hsizes.size_1;
            } else if (id.x < mip_hsizes.size_3) {
                level = 3;
                id.x -= mip_hsizes.size_2;
            } else if (id.x < mip_hsizes.size_4) {
                level = 4;
                id.x -= mip_hsizes.size_3;
            } else if (id.x < mip_hsizes.size_5) {
                level = 5;
                id.x -= mip_hsizes.size_4;
            } else {
                return;
            }

            id.z = id.y;
            int res = base_resolution >> level;
            id.y = id.x / res;
            id.x -= id.y * res;

            vec2 uv = vec2(
                (float(id.x) / float(res)) * 2.0 - 1.0,
                (float(id.y) / float(res)) * 2.0 - 1.0
            );
            vec3 dir = rendrs_CubemapUV(uv, id.z);
            vec3 frameZ = normalize(dir);
            vec3 adir = abs(dir);

            vec4 color = vec4(0.0);
            for (int axis = 0; axis < 3; axis++) {
                const int otherAxis0 = 1 - (axis & 1) - (axis >> 1);
                const int otherAxis1 = 2 - (axis >> 1);

                float frameweight = (max(adir[otherAxis0], adir[otherAxis1]) - 0.75) / 0.25;
                if (frameweight > 0.00001) {
                    vec3 UP;
                    switch (axis) {
                        case 0:
                            UP = vec3(1.0, 0.0, 0.0);
                            break;
                        case 1:
                            UP = vec3(0.0, 1.0, 0.0);
                            break;
                        default:
                            UP = vec3(0.0, 0.0, 1.0);
                            break;
                    }

                    vec3 frameX = normalize(cross(UP, frameZ));
                    vec3 frameY = cross(frameZ, frameX);

                    float Nx = dir[otherAxis0];
                    float Ny = dir[otherAxis1];
                    float Nz = adir[axis];

                    float NmaxXY = max(abs(Ny), abs(Nx));
                    Nx /= NmaxXY;
                    Ny /= NmaxXY;

                    float theta;
                    if (Ny < Nx) {
                        if (Ny <= -0.999) {
                            theta = Nx;
                        } else {
                            theta = Ny;
                        }
                    } else {
                        if (Ny >= 0.999) {
                            theta = -Nx;
                        } else {
                            theta = -Ny;
                        }
                    }

                    float phi;
                    if (Nz <= -0.999) {
                        phi = -NmaxXY;
                    } else if (Nz >= -0.999) {
                        phi = NmaxXY;
                    } else {
                        phi = Nz;
                    }

                    float theta2 = theta*theta;
                    float phi2   = phi*phi;

                    for (int iSuperTap = 0; iSuperTap < NUM_TAPS / 4; iSuperTap++) {
                        const int index = (NUM_TAPS / 4) * axis + iSuperTap;
                        float[4] coeffsDir0[3];
                        float[4] coeffsDir1[3];
                        float[4] coeffsDir2[3];
                        float[4] coeffsLevel[3];
                        float[4] coeffsWeight[3];

                        for (int iCoeff = 0; iCoeff < 3; iCoeff++) {
                            coeffsDir0[iCoeff] = coeffs_quad32[level][0][iCoeff][index];
                            coeffsDir1[iCoeff] = coeffs_quad32[level][1][iCoeff][index];
                            coeffsDir2[iCoeff] = coeffs_quad32[level][2][iCoeff][index];
                            coeffsLevel[iCoeff] = coeffs_quad32[level][3][iCoeff][index];
                            coeffsWeight[iCoeff] = coeffs_quad32[level][4][iCoeff][index];
                        }
                        for (int iSubTap = 0; iSubTap < 4; iSubTap++) {
                            vec3 sample_dir
                            = frameX * (coeffsDir0[0][iSubTap] + coeffsDir0[1][iSubTap] * theta2 + coeffsDir0[2][iSubTap] * phi2)
                            + frameY * (coeffsDir1[0][iSubTap] + coeffsDir1[1][iSubTap] * theta2 + coeffsDir1[2][iSubTap] * phi2)
                            + frameZ * (coeffsDir2[0][iSubTap] + coeffsDir2[1][iSubTap] * theta2 + coeffsDir2[2][iSubTap] * phi2);

                            float sample_level = coeffsLevel[0][iSubTap] + coeffsLevel[1][iSubTap] * theta2 + coeffsLevel[2][iSubTap] * phi2;
                            float sample_weight = coeffsWeight[0][iSubTap] + coeffsWeight[1][iSubTap] * theta2 + coeffsWeight[2][iSubTap] * phi2;
                            sample_weight *= frameweight;

                            // compensate for jacobian
                            sample_dir /= max(abs(sample_dir.x), max(sample_dir.y, sample_dir.z));
                            sample_level += 0.75 * log(dot(sample_dir,sample_dir));

                            color.rgb += textureLod(input_cubemap, sample_dir, sample_level).rgb * sample_weight;
                            color.a   += sample_weight;
                        }
                    }
                }
            }
            color /= color.a;

            color.r = max(0.0, color.r);
            color.g = max(0.0, color.g);
            color.b = max(0.0, color.b);
            color.a = 1.0;

            switch (level) {
                case 0:
                    imageStore(out_mip0, id, color);
                    break;
                case 1:
                    imageStore(out_mip1, id, color);
                    break;
                case 2:
                    imageStore(out_mip2, id, color);
                    break;
                case 3:
                    imageStore(out_mip3, id, color);
                    break;
                case 4:
                    imageStore(out_mip4, id, color);
                    break;
                case 5:
                    imageStore(out_mip5, id, color);
                    break;
            }
            ";
        }
    }
}

ethel::shader_glsl_struct! {
    struct MipSizes {
        size_0 : i32 => int,
        size_1 : i32 => int,
        size_2 : i32 => int,
        size_3 : i32 => int,
        size_4 : i32 => int,
        size_5 : i32 => int
    }
}

const LIB_HELPER_MIP_HSIZES: GlslLib = ethel::shader_glsl_lib! {
    MipSizes _helper_Mip_hsizes[
        base_mip_size : int
    ] => "
        int s0 = base_mip_size >> 0;
        int s1 = base_mip_size >> 1;
        int s2 = base_mip_size >> 2;
        int s3 = base_mip_size >> 3;
        int s4 = base_mip_size >> 4;
        int s5 = base_mip_size >> 5;
        return MipSizes(
            s0*s0,
            s0*s0+s1*s1,
            s0*s0+s1*s1+s2*s2,
            s0*s0+s1*s1+s2*s2+s3*s3,
            s0*s0+s1*s1+s2*s2+s3*s3+s4*s4,
            s0*s0+s1*s1+s2*s2+s3*s3+s4*s4+s5*s5
        );
    "
};

/// The imported file contains two tables:
/// * `coeffs_const8` for lower quality filtering (table is 7x5x6 vec4)
/// * `coeffs_quad32` for highest quality filtering (table is 7x5x3x24 vec4)
///
/// Currently the filtering shader is only implemented for `quad32`.
/// Support for `const8` is planned.
///
/// Note that while the original reference uses `hlsl float4` types, these
/// tables instead use raw `float`s.
const LIB_TABLE_CUBEMAP_COEFFICIENTS: GlslLib = GlslLib::new(include_str!("cubemap_coeffs.h"));
