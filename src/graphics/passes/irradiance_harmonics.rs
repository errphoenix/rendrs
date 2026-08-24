use ethel::shader::{GlslLib, GlslStruct};

use crate::{
    ComputePass,
    graphics::ShCoeffsBuffer,
    pipeline::{Sampler, SamplerObject},
};

pub type IrradianceHarmonicsPass = ComputePass<IrradianceHarmonicsCtxWrapper, 1, 0>;

#[derive(Debug)]
pub struct IrradianceHarmonicsCtx<'ctx> {
    /// A non-triple-buffered SSBO of second frequency spherical harmonics
    ///
    /// Each entry is an array of 9 floats.
    ///
    /// Currently a single entry is expected.
    pub output_coefficients: &'ctx ShCoeffsBuffer,
}
crate::context_wrapper!(for<'ctx> IrradianceHarmonicsCtx);

pub const fn irradiance_harmonics(
    shader: &ComputeShaderIrradianceHarmonics,
    radiance_map: SamplerObject,
) -> IrradianceHarmonicsPass {
    let handle_view = shader.compute_handle().view();
    IrradianceHarmonicsPass::new(
        handle_view,
        [Sampler::wrap(radiance_map, SAMPLER_UNIT_RADIANCEMAP)],
        [],
        |_, ctx| {
            ctx.output_coefficients
                .bind_shader_storage(SSBO_BINDING_OUTPUT_COEFFS, 0);

            // just one irradiance map
            [1, 1, 1]
        },
    )
}

ethel::shader_glsl_struct! {
    struct ShCoeffs {
        y22 : [f32; 4] => vec4,
        y31 : [f32; 4] => vec4,
        y32 : [f32; 4] => vec4,
        y33 : [f32; 4] => vec4,
        y40 : [f32; 4] => vec4,
        y41 : [f32; 4] => vec4,
        y42 : [f32; 4] => vec4,
        y43 : [f32; 4] => vec4,
        y44 : [f32; 4] => vec4
    }
}

macro_rules! ssbo_binding {
    (_devShCoeffs) => {
        12
    };
}

pub const TYPE_SH_COEFFS: GlslStruct = ShCoeffsGlslStruct::as_definition();
pub const SAMPLER_UNIT_RADIANCEMAP: u32 = 0;
pub const SSBO_BINDING_OUTPUT_COEFFS: u32 = ssbo_binding!(_devShCoeffs);

ethel::shader_glsl_compute! {
    struct IrradianceHarmonics > [460] {
        workgroup [16, 16, 1];

        sampler {
            on SAMPLER_UNIT_RADIANCEMAP => radiance_map : samplerCube;
        };

        type {
            ShCoeffsGlslStruct::as_definition()
        };

        ssbo {
            ethel::shader_glsl_ssbo! {
                buf _devShCoeffs => {
                    ShCoeffs : out_sh_coeffs;
                }
            }
        };

        lib {
            super::LIB_UTIL_CUBEMAP_UV;
        };

        share {
            vec3 sm_coeffs[9][256];
        };

        src() {
            "
            uvec2 id    = gl_LocalInvocationID.xy;
            uint  index = gl_LocalInvocationIndex;

            float S = 16.0;
            vec2 uv = ((vec2(id) + 0.5) / S) * 2.0 - 1.0;

            float d_sq = uv.x*uv.x + uv.y*uv.y + 1.0;
            // angle subtended by the texel
            float d_omega = 4.0 / (sqrt(d_sq) * d_sq * 256.0);

            vec3 thread_coeffs[9] = vec3[9](
                vec3(0.0),vec3(0.0),vec3(0.0),
                vec3(0.0),vec3(0.0),vec3(0.0),
                vec3(0.0),vec3(0.0),vec3(0.0)
            );

            for (uint face = 0; face < 6u; ++face) {
                vec3 r = rendrs_CubemapUV(uv, face);
                vec3 L = textureLod(radiance_map, r, 0).rgb;

                float k_cos[9] = float[9](
                    0.886227,
                    1.023327 * r.y,
                    1.023327 * r.z,
                    1.023327 * r.x,
                    0.858086 * r.x * r.y,
                    0.858086 * r.y * r.z,
                    0.247708 * (3.0 * r.z * r.z - 1.0),
                    0.858086 * r.x * r.z,
                    0.429043 * (r.x * r.x - r.y * r.y)
                );

                vec3 kL = L * d_omega;
                for (uint j = 0; j < 9u; ++j) {
                    thread_coeffs[j] += kL * k_cos[j];
                }
            }

            for (uint j = 0; j < 9; ++j) {
                sm_coeffs[j][index] = thread_coeffs[j];
            }

            // sync all coeffs. writes, then process further
            barrier();

            // shared memory parallel tree reduction
            for (uint stride = 128u; stride > 0u; stride >>= 1u) {
                if (index < stride) {
                    for (uint j = 0; j < 9u; ++j) {
                        sm_coeffs[j][index] += sm_coeffs[j][index + stride];
                    }
                }
                // sync for each halving
                barrier();
            }

            // flush to ssbo (only thread 0)
            if (index == 0u) {
                out_sh_coeffs = ShCoeffs(
                    vec4(sm_coeffs[0][0], 0.0),
                    vec4(sm_coeffs[1][0], 0.0),
                    vec4(sm_coeffs[2][0], 0.0),
                    vec4(sm_coeffs[3][0], 0.0),
                    vec4(sm_coeffs[4][0], 0.0),
                    vec4(sm_coeffs[5][0], 0.0),
                    vec4(sm_coeffs[6][0], 0.0),
                    vec4(sm_coeffs[7][0], 0.0),
                    vec4(sm_coeffs[8][0], 0.0)
                );
            }
            ";
        }
    }
}

/// Evaluate second-frequency spherical harmonics functions.
///
/// Creates the `rendrs_EvalSH_L2` function which takes the following
/// arguments:
/// * the second-frequency spherical harmonics coefficients as [`ShCoeffs`]
/// * a 3d vector to sample the function
///
/// Returns a 3d vector representing the value of the function at the encoded
/// point.
///
/// Requires the [`ShCoeffs`] glsl type.
pub const LIB_EVALUATE_SH_L2: GlslLib = ethel::shader_glsl_lib! {
    vec3 rendrs_EvalSH_L2[
        sh : ShCoeffs,
        r  : vec3
    ] => "
        return max(vec3(0.0), vec3(
            // l0 band
            sh.y22 * 0.282095 +
            // l1 band, linear
            sh.y31 * (0.488602 * r.y) +
            sh.y32 * (0.488602 * r.z) +
            sh.y33 * (0.488602 * r.x) +
            // l2 band, quadratic
            sh.y40 * (1.092548 * r.x * r.y) +
            sh.y41 * (1.092548 * r.y * r.z) +
            sh.y42 * (0.315392 * (3.0 * r.z*r.z - 1.0)) +
            sh.y43 * (1.092548 * r.x * r.z) +
            sh.y44 * (0.546274 * (r.x*r.x - r.z*r.z))
        ));
    "
};
