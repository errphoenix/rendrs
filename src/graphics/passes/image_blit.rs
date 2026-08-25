use std::sync::LazyLock;

use ethel::{
    render::buffer::StorageSection,
    shader::{ComputeShaderHandle, ComputeShaderHandleView},
};
use janus::texture::Tex;

use crate::{
    ComputePass,
    pipeline::{ImageAccessKind, ImageObject, ImageObjectTarget, Pass, RenderPool},
};

pub type ImageBlitPass = ComputePass<ImageBlitCtxWrapper, 0, 2>;

#[derive(Debug, Clone, Copy)]
pub struct ImageBlitCtx {
    pub resolution: (u32, u32),
}
crate::context_wrapper!(ImageBlitCtx);

pub fn image_blit(
    format: ImageTargetFormat,
    src: ImageObject,
    dst: ImageObject,
    src_mip: Option<i32>,
    dst_mip: Option<i32>,
    src_layer: Option<i32>,
    dst_layer: Option<i32>,
) {
    let shader = cache_shader(format);
    let src = ImageObjectTarget::new_with_mip_level(
        src,
        ImageAccessKind::ReadOnly,
        IMAGE_BINDING_SRC,
        src_layer,
        src_mip,
    );
    let dst = ImageObjectTarget::new_with_mip_level(
        dst,
        ImageAccessKind::WriteOnly,
        IMAGE_BINDING_DST,
        dst_layer,
        dst_mip,
    );
    let (w, h) = src.texture().size();
    ImageBlitPass::new(shader, [], [src, dst], |_, ctx| {
        let wg_x = ctx.resolution.0.div_ceil(WORKGROUP_SIZE_XY);
        let wg_y = ctx.resolution.1.div_ceil(WORKGROUP_SIZE_XY);
        [wg_x, wg_y, 1]
    })
    .execute(
        StorageSection::Back,
        &RenderPool::dummy(),
        &ImageBlitCtx {
            resolution: (w as u32, h as u32),
        },
    );
}

pub const IMAGE_BINDING_SRC: u32 = 0;
pub const IMAGE_BINDING_DST: u32 = 1;
pub const WORKGROUP_SIZE_XY: u32 = 8;

macro_rules! image_blit_compute {
    ($format:ident) => {
        paste::paste! {
            ethel::shader_glsl_compute! {
                struct [< ImageBlit $format:camel >] > [460] {
                    workgroup [8, 8, 1];

                    image {
                        on IMAGE_BINDING_SRC => img_src : image2D as [< $format:lower >] readonly;
                        on IMAGE_BINDING_DST => img_dst : image2D as [< $format:lower >] writeonly;
                    };

                    src() {
                        "
                        uvec2 size = imageSize(img_src);
                        uvec2 id   = gl_GlobalInvocationID.xy;
                        if (any(id >= size)) return;
                        uvec2 px = id + offset;
                        vec4 C = imageLoad(img_src, px);
                        imageStore(img_dst, px, C);
                        ";
                    }
                }
            }

        }
    };
}

pub fn cache_shader(format: ImageTargetFormat) -> ComputeShaderHandleView {
    SHADERS_CACHE[format as usize].view()
}

macro_rules! def_shaders {
    ($($fmt:ident $(,)?)*) => {
        $(image_blit_compute!($fmt);)*

        pub const SUPPORTED_FORMATS_COUNT: u32 = def_shaders!(@count $($fmt,)*);

        paste::paste! {
            #[repr(usize)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub enum ImageTargetFormat {
                $([< $fmt:camel >],)*
            }

            pub static SHADERS_CACHE: [LazyLock<ComputeShaderHandle>; SUPPORTED_FORMATS_COUNT as usize] = [
                    $(
                        LazyLock::new(||
                            [< ComputeShaderImageBlit $fmt:camel >]::new_compiled().compute_handle_owned()
                        ),
                    )*
            ];
        }
    };

    (@count $fmt:ident, ) => { 1 };
    (@count $fmt:ident, $($rem:tt)*) => { 1 + def_shaders!(@count $($rem)*) };
}

def_shaders!(
    rgba32ui,
    rgba32i,
    rgba32f,
    rgba32,
    rg32ui,
    rg32i,
    rg32f,
    r32ui,
    f32f,
    rgba16ui,
    rgba16i,
    rgba16f,
    rgba16,
    rg16ui,
    rg16i,
    rg16f,
    rg16,
    r16ui,
    r16f,
    r16,
    rgba8ui,
    rgba8i,
    rgba8,
    rg8ui,
    rg8i,
    rg8,
    r8ui,
    r8,
    r11f_g11f_b10f,
    rgb10_a2ui,
    rgb10_a2,
    rgba16_snorm,
    rgba8_snorm,
    rg16_snorm,
    rg8_snorm,
    r16_snorm,
    r8_snorm
);
