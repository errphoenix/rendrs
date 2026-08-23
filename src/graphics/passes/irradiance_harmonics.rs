use ethel::{render::buffer::SingleBuffer, shader::GlslStruct};
use janus::texture::Tex;

use crate::{ComputePass, pipeline::SamplerObject};

pub type IrradianceHarmonicsPass = ComputePass<IrradianceHarmonicsCtxWrapper, 1, 0>;

#[derive(Debug)]
pub struct IrradianceHarmonicsCtx<'ctx> {
    /// A non-triple-buffered SSBO of second frequency spherical harmonics
    ///
    /// Each entry is an array of 9 floats.
    ///
    /// Currently a single entry is expected.
    pub output_coefficients: &'ctx SingleBuffer<[f32; 9]>,
}
crate::context_wrapper!(for<'ctx> IrradianceHarmonicsCtx);

/// Will panic if `radiance_map` is not a 16x16 cubemap.
pub fn irradiance_harmonics(
    shader: &ComputeShaderIrradianceHarmonics,
    radiance_map: SamplerObject,
) -> IrradianceHarmonicsPass {
    let size = radiance_map.texture().size().0;
    let mip = radiance_map.mip_view().unwrap_or_default();
    let effective_size = size >> mip;

    assert_eq!(
        effective_size,
        16,
        "radiance map resolution must be 16x16 pixels, but it is {effective_size}; {}",
        "note that a restricted view of the texture at a mip with the correct resolution will also work."
    );

    let handle_view = shader.compute_handle().view();
    IrradianceHarmonicsPass::new(handle_view, [radiance_map], [], |_, ctx| {
        ctx.output_coefficients
            .bind_shader_storage(SSBO_BINDING_OUTPUT_COEFFS, 0);

        // just one irradiance map
        [1, 1, 1]
    })
}

ethel::shader_glsl_struct! {
    struct ShCoeffs {
        y22 : f32 => float,
        y31 : f32 => float,
        y32 : f32 => float,
        y33 : f32 => float,
        y40 : f32 => float,
        y41 : f32 => float,
        y42 : f32 => float,
        y43 : f32 => float,
        y44 : f32 => float,
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
                    out_sh_coeffs: ShCoeffs;
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

            float size = float(textureSize(radiance_map, 0).x);
            vec2 uv    = ((vec2(id) + 0.5) / size) * 2.0 - 1.0;

            float d_sq = uv.x*uv.x + uv.y*uv.y + 1.0;
            // angle subtended by the texel
            float d_omega = 4.0 / (sqrt(d_sq) * d_sq * size*size);

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
                for (uint j = 0; j < 9u; ++i) {
                    thread_coeffs[j] += kL * k_cos[j];
                }
            }

            for (uint j; j < 9; ++j) {
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
                    sm_coeffs[0],
                    sm_coeffs[1],
                    sm_coeffs[2],
                    sm_coeffs[3],
                    sm_coeffs[4],
                    sm_coeffs[5],
                    sm_coeffs[6],
                    sm_coeffs[7],
                    sm_coeffs[8]
                );
            }
            ";
        }
    }
}
