use ethel::assets::RawTexture;
use image::{EncodableLayout, RgbImage};
use janus::{
    StringHash, StringMap,
    texture::{ImageFormat, ImageType, Tex, Texture, TextureView},
};

use crate::pipeline::SamplerObject;

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

    #[cfg(feature = "pipeline")]
    pub fn sampler(&self) -> SamplerObject {
        SamplerObject::new(&self.array_texture_object)
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

/// CPU-side representation of material groups.
///
/// Used for staging and uploading onto material textures and binding during
/// rendering.
#[derive(Debug)]
pub struct MaterialGroups<const GROUPS: usize> {
    groups: [MaterialGroup; GROUPS],
}
impl<const GROUPS: usize> MaterialGroups<GROUPS> {
    pub const fn new(groups: [MaterialGroup; GROUPS]) -> Self {
        Self { groups }
    }

    pub const fn group(&self, index: usize) -> &MaterialGroup {
        &self.groups[index]
    }

    #[cfg(feature = "pipeline")]
    pub fn as_samplers(&self) -> [SamplerObject; GROUPS] {
        use janus::texture::TextureKind;
        let mut samplers = [SamplerObject::new(TextureView::null(TextureKind::Dim2D)); GROUPS];
        self.groups
            .iter()
            .enumerate()
            .for_each(|(i, group)| samplers[i] = group.sampler());
        samplers
    }
}

/// Material maps coefficient for roughness, metallic/specular,
/// and ao strength.
///
/// Represents both cpu and gpu data.
///
/// The 4th field `pad_or_extra` is used as padding, but it is public and can
/// be used as a custom field if necessary. It is, as default, initialised
/// as 0 but the other fields are initialised as 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MaterialParams {
    pub roughness: f32,
    pub metallic: f32,
    pub ambient_occlusion: f32,
    pub pad_or_extra: f32,
}
impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            roughness: 1f32,
            metallic: 1f32,
            ambient_occlusion: 1f32,
            pad_or_extra: 0f32,
        }
    }
}

/// Location of material onto the GPU.
///
/// Specifies the material group index, the page of the material group
/// (which is the layer of the texture array), the width and height of
/// the texture as a `[0.0 - 1.0]` range.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaterialLocation {
    group_index: u32,
    page: u32,
    width: f32,
    height: f32,
}
impl MaterialLocation {
    pub const fn group(&self) -> u32 {
        self.group_index
    }

    pub const fn page(&self) -> u32 {
        self.page
    }

    pub const fn width(&self) -> f32 {
        self.width
    }

    pub const fn height(&self) -> f32 {
        self.height
    }
}

/// String-hashed material ID.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(StringHash);

#[derive(Debug, Default, Clone)]
pub struct MaterialLocationRegistry {
    map: StringMap<MaterialLocation>,
}
impl MaterialLocationRegistry {
    pub fn new() -> Self {
        Self {
            map: StringMap::default(),
        }
    }

    pub fn add(&mut self, id: MaterialId, location: MaterialLocation) {
        self.map.insert(id.0, location);
    }

    pub fn get(&self, id: &MaterialId) -> Option<MaterialLocation> {
        self.map.get(&id.0).copied()
    }

    pub fn inner_map(&self) -> &StringMap<MaterialLocation> {
        &self.map
    }
}
