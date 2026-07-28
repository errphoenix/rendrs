use ethel::assets::RawTexture;
use image::{EncodableLayout, Rgb, RgbImage, Rgba};
use janus::{
    StringHash,
    texture::{ImageFormat, ImageType, Tex, Texture, TextureView},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct MaterialParams {
    pub roughness: f32,
    pub metallic: f32,
    pub ambient_occlusion: f32,
    pub _pad: f32,
}

pub const MATERIAL_TEXTURE_FORMAT: ImageFormat = ImageFormat::Rgb;
pub const MATERIAL_TEXTURE_PIXEL_TYPE: ImageType = ImageType::Bits8;

#[derive(Clone, Debug)]
pub struct ValidatedRawTexture(RgbImage);
impl ValidatedRawTexture {
    pub fn from_texture(texture: RawTexture) -> Self {
        Self(texture.0.to_rgb8())
    }

    pub fn width(&self) -> u32 {
        self.0.width()
    }

    pub fn height(&self) -> u32 {
        self.0.height()
    }

    pub fn bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}
impl From<RawTexture> for ValidatedRawTexture {
    fn from(value: RawTexture) -> Self {
        Self::from_texture(value)
    }
}

#[derive(Debug)]
pub struct MaterialGroup {
    array_texture_object: Texture,
    pages: u32,
    size: u32,
}
impl MaterialGroup {
    pub const fn pages(&self) -> u32 {
        self.pages
    }

    pub const fn size(&self) -> u32 {
        self.size
    }

    pub const fn array_texture_object(&self) -> TextureView {
        self.array_texture_object.view()
    }

    pub fn copy_page(&self, texture: impl Into<ValidatedRawTexture>, page_target: u32) {
        let texture = texture.into();
        let bytes = texture.bytes();
        let width = texture.width();
        let height = texture.height();
        self.array_texture_object
            .upload_layer(
                0,
                0,
                0,
                page_target as i32,
                width as i32,
                height as i32,
                bytes,
            )
            .expect("material group texture object is an array texture");
    }
}

#[derive(Debug)]
pub struct GlobalMaterials<const GROUPS: usize> {
    groups: [MaterialGroup; GROUPS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaterialLocation {
    group_index: u32,
    page: u32,
    width: f32,
    height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(StringHash);

pub struct MaterialManager {
    map: janus::StringMap<MaterialLocation>,
}
