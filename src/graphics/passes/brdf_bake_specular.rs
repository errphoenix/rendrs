use ethel::render::buffer::StorageSection;
use ethel::shader::Constant;
use janus::texture::{Tex, TextureView};

use crate::{
    ComputePass,
    pipeline::{ImageAccessKind, ImageObject, ImageObjectTarget, Pass, RenderPool},
};

pub fn brdf_bake_specular(shader: &ComputeShaderBrdfBakingSpecular, output: TextureView) {
    let image = ImageObjectTarget::new(
        ImageObject::DirectTexture(output),
        ImageAccessKind::WriteOnly,
        IMAGE_BINDING_OUTPUT,
        None,
    );

    #[derive(Debug)]
    struct Ctx {
        resolution: u32,
    }
    impl crate::pipeline::CtxType for Ctx {
        type Ctx<'ctx> = Self;
    }

    let pass =
        ComputePass::<Ctx, 0, 1>::new(shader.compute_handle().view(), [], [image], |_, ctx| {
            let wg_size = ctx.resolution.div_ceil(WORKGROUP_SIZE_XY);
            [wg_size, wg_size, 1]
        });
    pass.execute(
        StorageSection::Back, //unused
        &RenderPool::dummy(), //unused
        &Ctx {
            resolution: output.size().0 as u32,
        },
    );
}

pub fn brdf_bake_specular_compile(
    variant: ComputeShaderBrdfBakingSpecularVariants,
    output: TextureView,
) {
    let shader = ComputeShaderBrdfBakingSpecular::new_compiled_variant(variant);
    brdf_bake_specular(&shader, output);
}

pub const WORKGROUP_SIZE_XY: u32 = 8;
pub const IMAGE_BINDING_OUTPUT: u32 = 0;
pub const SAMPLES: u32 = 1024;

// todo: variants for other lobes
ethel::shader_glsl_compute! {
    struct BrdfBakingSpecular > [460] {
        workgroup [8, 8, 1];

        image {
            on IMAGE_BINDING_OUTPUT => out_brdf : image2D as rg16f writeonly;
        };

        const {
            Constant::new("SAMPLES", SAMPLES)
        };

        lib {
            super::LIB_UTIL_VAN_DER_CORPUT;
            super::LIB_UTIL_HAMMERSLEY_2D;
            super::LIB_GGX_IMPSAMPLE;
            super::LIB_NDF_MASK_SMITH_G2_HEIGHT_GGX_HAMMON_APPROX;
        };

        src() {
            "
            uvec2 id   = gl_GlobalInvocationID.xy;
            ivec2 size = imageSize(out_brdf);

            if (id.x >= size.x || id.y >= size.y) {
                return;
            }

            vec2 uv = vec2(id) / vec2(size);
            float angle = uv.x;
            float roughness = uv.y;

            vec3 V = vec3(
                sqrt(1.0 - angle*angle),
                0.0,
                angle
            );

            float scale = 0.0;
            float bias  = 0.0;

            const vec3 N = vec3(0.0, 0.0, 1.0);

            for (uint i = 0; i < SAMPLES; ++i) {
                vec2 P = rendrs_Hammersley2D(i, SAMPLES);
                vec3 H = rendrs_GGX_ImportanceSample(P, N, roughness);

                float VdotH = dot(V, H);
                vec3 L = normalize(2.0 * VdotH * H - V);

                float NdotL = max(L.z, 0.0);
                float NdotH = max(H.z, 0.0001);
                VdotH = max(VdotH, 0.0);

                if (NdotL > 0.0) {
                    float G = rendrs_ndf_SmithG2_Height(
                        max(angle, 0.0001), NdotL, roughness
                    );

                    float weight = (4.0 * G * NdotL * VdotH) / NdotH;

                    float iVdotH = 1.0 - VdotH;
                    float F = iVdotH*iVdotH*iVdotH*iVdotH*iVdotH;

                    scale += (1.0 - F) * weight;
                    bias  += F * weight;
                }
            }

            scale /= float(SAMPLES);
            bias  /= float(SAMPLES);

            imageStore(out_brdf, ivec2(id), vec4(scale, bias, 0.0, 0.0));
            ";
        }
    }
}
