use ethel::assets::{AssetId, RawTexture};
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

/// Location of a full material onto the GPU.
///
/// Specifies the [`MaterialEntryLocation`] of a diffuse and RSO
/// (roughness-specular-occlusion) entry as 2 distinct textures.
///
/// The diffuse entry is a single RGB texture, RSO is also a single RGB
/// texture formed from 3 distinct single-channel textures.
///
/// This also contains the width and height of the texture as a `[0.0 - 1.0]`
/// range, which is equal for both entries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaterialLocation {
    diffuse_entry: MaterialEntryLocation,
    rso_entry: MaterialEntryLocation,
    width: f32,
    height: f32,
}
impl MaterialLocation {
    pub const fn diffuse(&self) -> MaterialEntryLocation {
        self.diffuse_entry
    }

    pub const fn rso(&self) -> MaterialEntryLocation {
        self.rso_entry
    }

    pub const fn width(&self) -> f32 {
        self.width
    }

    pub const fn height(&self) -> f32 {
        self.height
    }
}

/// Location of single material entry onto the GPU.
///
/// Specifies the material group index and the page of the material
/// group (which is the layer of the texture array)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialEntryLocation {
    group_index: u16,
    page: u16,
}
impl MaterialEntryLocation {
    pub const fn group(&self) -> u16 {
        self.group_index
    }

    pub const fn page(&self) -> u16 {
        self.page
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

///
/// group {
///     // group descriptor
///     pages: num;
///     size: num;
///
///     entry(id: str) { // material entry descriptor
///         // single rgb8
///         diffuse(path str OR asset value);
///
///         // RSO
///         // each is single-channel, coalesce into rgb8
///         roughness(path str OR asset value);
///         specular(path str OR asset value);
///         occlusion(path str OR asset value);
///         // OR
///         // optional pre-coalesced rso
///         rso(path str OR asset value);
///
///         // diffuse and RSO are optional, if absent they
///         // must default to a 1x1 blank pixel texture.
///     };
/// };
///

#[macro_export]
macro_rules! material_groups {
    ($(group { $g:tt };)*) => {
        todo!()
    };
}

#[macro_export]
macro_rules! material_groups_internal {
    (@group $pages:expr, $size:expr; $($entries:tt)*) => {
        todo!()
    };
}

pub struct MaterialGroupDescriptor {
    // TODO
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialDescriptor {
    /// RGB8 diffuse component (aka albedo)
    pub diffuse: Option<MaterialComponentSource>,
    /// RSO (roughness-specular-occlusion) entry
    pub rso: Option<MaterialRsoDescriptor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialRsoDescriptor {
    /// Pre-coalesced RSO where each channel represents roughness,
    /// specular, and occlusion, respectively.
    Coalesced(MaterialComponentSource),
    /// Separate non-coalesced RSO where each sub-entry is a separate
    /// single-channel texture.
    ///
    /// This is coalesced into a single RGB texture later.
    Separate {
        roughness: Option<MaterialComponentSource>,
        specular: Option<MaterialComponentSource>,
        occlusion: Option<MaterialComponentSource>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialComponentSource {
    Path(&'static str),
    Asset(AssetId),
}
