pub mod builder;
pub mod shader;

use ethel::assets::RawTexture;
use image::{EncodableLayout, RgbaImage};
use janus::{
    StringHash, StringMap,
    texture::{ImageFormat, ImageType, Tex, Texture, TextureView},
};

use crate::pipeline::SamplerObject;

pub const MATERIAL_TEXTURE_FORMAT: ImageFormat = ImageFormat::Rgba;
pub const MATERIAL_TEXTURE_PIXEL_TYPE: ImageType = ImageType::Bits8;

#[derive(Clone, Debug)]
pub struct ValidatedRawTexture(RgbaImage);
impl ValidatedRawTexture {
    pub fn from_texture(texture: RawTexture) -> Self {
        Self(texture.0.to_rgba8())
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
    #[cfg(not(test))]
    array_texture_object: Texture,
    size: u16,
    pages: u16,
}
impl MaterialGroup {
    pub const fn pages(&self) -> u16 {
        self.pages
    }

    pub const fn size(&self) -> u16 {
        self.size
    }

    #[cfg(not(test))]
    pub const fn array_texture_object(&self) -> TextureView {
        self.array_texture_object.view()
    }

    #[cfg(not(test))]
    #[cfg(feature = "pipeline")]
    pub const fn sampler(&self) -> SamplerObject {
        SamplerObject::new(self.array_texture_object.view())
    }

    #[cfg(not(test))]
    pub fn copy_to_page(&self, texture: impl Into<ValidatedRawTexture>, page_target: u16) {
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

    #[cfg(not(test))]
    #[cfg(feature = "pipeline")]
    pub fn as_samplers(&self) -> [SamplerObject; GROUPS] {
        use janus::texture::TextureKind;
        let mut samplers = [SamplerObject::new(TextureView::null(TextureKind::Dim2DArray)); GROUPS];
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
///
/// These values are not global to a specific material. These are local
/// values for each rendered entity.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MaterialParams {
    pub roughness: f32,
    pub metallic: f32,
    pub ambient_occlusion: f32,
    /// Might be replaced by emissive
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
/// The `diffuse_and_alpha` entry is an RGBA texture, where RGB is the
/// diffuse/albedo property of the material, and the alpha component is the
/// transparency property, where 0 is transparent and 1 is opaque.
///
/// The `normal_and_emissive` entry is an RGBA texture, where RGB is the
/// normal mapping for the material, and the alpha component is the emissive
/// property of the material.
///
/// ORMD is an RGBA texture, where each channel represents a
/// different material property. These are, in order: occlusion, roughness,
/// metallic, and displacement.
///
/// Also stores the width and height of the textures as a normalized
/// `[0.0 - 1.0]` range.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaterialLocation {
    diffuse_and_alpha: MaterialEntryLocation,
    normal_and_emissive: MaterialEntryLocation,
    ormd: MaterialEntryLocation,
    width: f32,
    height: f32,
}
impl MaterialLocation {
    pub const fn diffuse_and_alpha(&self) -> MaterialEntryLocation {
        self.diffuse_and_alpha
    }

    pub const fn normal_and_emissive(&self) -> MaterialEntryLocation {
        self.normal_and_emissive
    }

    pub const fn ormd(&self) -> MaterialEntryLocation {
        self.ormd
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
pub struct MaterialId(pub StringHash);

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

#[macro_export]
macro_rules! material_groups {
    (
        $(
            group $g_name:ident {
                pages: $g_pages:expr;
                size: $g_size:expr;

                $($g_body:tt)*
            }
        )*
    ) => {
        const INTERNAL_GROUP_COUNT: usize = $crate::material_groups_internal!(
            @enumerate $($g_name)*
        );

        $crate::material_groups_internal!(
            @enumerate_groups
            INTERNAL_GROUP_COUNT;
            $(group $g_name)*
        );

        $(
            paste::paste! {
                pub fn [< material_group_ $g_name:lower _builder >](
                ) -> $crate::graphics::material::builder::MaterialGroupDescriptor {
                    $crate::material_groups_internal! {
                        @group $g_pages, $g_size;
                        $($g_body)*
                    }
                }
            }

            paste::paste! {
                pub fn [< material_group_ $g_name:lower >](
                    group_index: u16,
                    texture_registry: &mut ethel::assets::AssetRegistry<
                        ethel::assets::RawTexture,
                        ethel::assets::TextureMetadata
                    >,
                    material_registry: &mut $crate::graphics::material::MaterialLocationRegistry,
                ) -> $crate::graphics::material::MaterialGroup {
                    let mut builder = [< material_group_ $g_name:lower _builder >]();
                    builder.distribute_pages();
                    builder.process_locations(group_index);
                    builder.build(texture_registry, material_registry)
                }
            }
        )*
    };
}

#[macro_export]
macro_rules! material_groups_internal {
    (@enumerate $g_name:ident $($tail:tt)* ) => {
        1 + $crate::material_groups_internal!(@enumerate $($tail)*)
    };
    (@enumerate ) => { 0 };

    (@enumerate_groups $n:expr;) => {};
    (@enumerate_groups $n:expr;
        group $g_name:ident
        $($other_groups:tt)*
    ) => {
        paste::paste! {
            pub const [< MATERIAL_GROUP_INDEX_ $g_name:upper >]: usize =
                $n - $crate::material_groups_internal!(@enumerate $($other_groups)*) - 1;
            $crate::material_groups_internal!(
                @enumerate_groups $n;
                $($other_groups)*
            );
        }
    };

    (@group
        $pages:expr, $size:expr;
        $(
            entry($entry_id:expr) {
                $($entry_body:tt)*
            };
        )*
    ) => {
        let mut gd = $crate::graphics::material::builder::MaterialGroupDescriptor::new(
            $pages, $size
        );
        $(
            let id_raw = janus::hash_string($entry_id);
            let id = $crate::graphics::material::MaterialId(id_raw);
            let group = $crate::material_groups_internal!(@entry_body $($entry_body)*);
            gd.add(id, group);
        )*
        gd
    };

    (@entry_body
        $( $comp_type:ident = $loc_kind:ident( $loc_value:expr ); )*
    ) => {{
        #[allow(unused)]
        // default blank sources
        let mut d_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut a_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut da_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut n_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut e_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut ne_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut o_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut r_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut m_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut di_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut orm_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;
        let mut ormd_source = None::<$crate::graphics::material::builder::MaterialComponentSource>;

        $(
            match stringify!($comp_type) {
                "diffuse" => {
                    d_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value));
                },
                "alpha" => a_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "diffuse_alpha" | "da" | "diffuse_and_alpha" => da_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "normal" => {
                    n_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value));
                },
                "emissive" => e_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "normal_emissive" | "ne" | "normal_and_emissive" => ne_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "occlusion" => o_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "roughness" => r_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "metallic" => m_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "displacement" => di_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "orm" | "ORM" | "lighting" => orm_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                "ormd" | "lighting_and_displacement" => ormd_source = Some($crate::material_groups_internal!(@component_src $loc_kind $loc_value)),
                _ => {},
            }
        )*

        let da_desc = if da_source.is_some() {
            $crate::graphics::material::builder::MaterialDiffuseAlphaDescriptor::Coalesced(da_source)
        } else {
            $crate::graphics::material::builder::MaterialDiffuseAlphaDescriptor::Separate {
                diffuse: d_source,
                alpha: a_source,
            }
        };
        let ne_desc = if ne_source.is_some() {
            $crate::graphics::material::builder::MaterialNormalEmissiveDescriptor::Coalesced(ne_source)
        } else {
            $crate::graphics::material::builder::MaterialNormalEmissiveDescriptor::Separate {
                normal: n_source,
                emissive: e_source,
            }
        };
        let ormd_desc = if ormd_source.is_some() {
            $crate::graphics::material::builder::MaterialOrmdDescriptor::Coalesced(ormd_source)
        } else if orm_source.is_some() {
            $crate::graphics::material::builder::MaterialOrmdDescriptor::OrmAndDisplacement {
                orm: orm_source,
                displacement: di_source,
            }
        } else {
            $crate::graphics::material::builder::MaterialOrmdDescriptor::Separate {
                occlusion: o_source,
                roughness: r_source,
                metallic: m_source,
                displacement: di_source,
            }
        };

        $crate::graphics::material::builder::MaterialDescriptor {
            diffuse_alpha: Some(da_desc),
            normal_emissive: Some(ne_desc),
            ormd: Some(ormd_desc),
        }
    }};

    ( @component_src path $path:expr ) => {
        $crate::graphics::material::builder::MaterialComponentSource::Path($path)
    };
    ( @component_src asset $asset:expr ) => {
        $crate::graphics::material::builder::MaterialComponentSource::Asset($asset)
    };
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use ethel::assets::{AssetId, AssetRegistry, TextureMetadata};

    use super::builder::*;
    use super::*;

    #[allow(unused)]
    #[test]
    fn material_group_composition() {
        const TEST_PAGES: u16 = 128;
        const TEST_SIZE: u16 = 256;

        let mut asset_registry = AssetRegistry::<RawTexture, TextureMetadata>::new();

        const SOURCE_DIFFUSE_0: &'static str = "test_64px_rgb.jpg";
        const SOURCE_ALPHA_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_DIFFUSE_AND_ALPHA_0: &'static str = "test_64px_rgba.png";
        const SOURCE_NORMAL_0: &'static str = "test_64px_rgb.jpg";
        const SOURCE_EMISSIVE_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_NORMAL_AND_EMISSIVE_0: &'static str = "test_64px_rgba.png";
        const SOURCE_OCCLUSION_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_ROUGHNESS_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_METALLIC_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_DISPLACEMENT_0: &'static str = "test_64px_gray.jpg";
        const SOURCE_ORMD_0: &'static str = "test_64px_rgba.png";
        ethel::hashet!(
            const SOURCE_DIFFUSE_1 = "diffuse_1.src";
            const SOURCE_ALPHA_1 = "alpha_1.src";
            const SOURCE_DIFFUSE_AND_ALPHA_1 = "diffuse_and_alpha_1.src";
            const SOURCE_NORMAL_1 = "NORMAL_1.src";
            const SOURCE_EMISSIVE_1 = "emissive_1.src";
            const SOURCE_NORMAL_AND_EMISSIVE_1 = "normal_and_emissive_1.src";
            const SOURCE_OCCLUSION_1 = "occlusion_1.src";
            const SOURCE_ROUGHNESS_1 = "roughness_1.src";
            const SOURCE_METALLIC_1 = "metallic_1.src";
            const SOURCE_DISPLACEMENT_1 = "displacement_1.src";
            const SOURCE_ORMD_1 = "ormd_1.src";

            const MATERIAL_A = "test_material_a";
            const MATERIAL_B = "test_material_b";
            const MATERIAL_C = "test_material_c";
            const MATERIAL_D = "test_material_d";
        );

        let mut process_asset = |id: AssetId, channels: u32| {
            let path = match channels {
                1 => "test_64px_gray.jpg",
                3 => "test_64px_rgb.jpg",
                4 => "test_64px_rgba.png",
                _ => unreachable!(),
            };
            asset_registry.register(id, path);
            asset_registry
                .get_mut(id)
                .unwrap()
                .load_to_memory(&())
                .unwrap();
        };
        {
            process_asset(*SOURCE_DIFFUSE_1, 3);
            process_asset(*SOURCE_ALPHA_1, 1);
            process_asset(*SOURCE_DIFFUSE_AND_ALPHA_1, 4);
            process_asset(*SOURCE_NORMAL_1, 3);
            process_asset(*SOURCE_EMISSIVE_1, 1);
            process_asset(*SOURCE_NORMAL_AND_EMISSIVE_1, 4);
            process_asset(*SOURCE_OCCLUSION_1, 1);
            process_asset(*SOURCE_ROUGHNESS_1, 1);
            process_asset(*SOURCE_METALLIC_1, 1);
            process_asset(*SOURCE_DISPLACEMENT_1, 1);
            process_asset(*SOURCE_ORMD_1, 4);
        }

        let mat_a_diffuse = MaterialComponentSource::Path(SOURCE_DIFFUSE_0);
        let mat_a_alpha = MaterialComponentSource::Path(SOURCE_ALPHA_0);
        let mat_a_normal = MaterialComponentSource::Path(SOURCE_NORMAL_0);
        let mat_a_emissive = MaterialComponentSource::Path(SOURCE_EMISSIVE_0);
        let mat_a_occlusion = MaterialComponentSource::Asset(*SOURCE_OCCLUSION_1);
        let mat_a_roughness = MaterialComponentSource::Asset(*SOURCE_ROUGHNESS_1);
        let mat_a_metallic = MaterialComponentSource::Asset(*SOURCE_METALLIC_1);
        let mat_a_displacement = MaterialComponentSource::Path(SOURCE_DISPLACEMENT_0);

        let mat_b_diffuse_and_alpha = MaterialComponentSource::Asset(*SOURCE_DIFFUSE_AND_ALPHA_1);
        let mat_b_normal_and_emissive =
            MaterialComponentSource::Asset(*SOURCE_NORMAL_AND_EMISSIVE_1);
        let mat_b_roughness = MaterialComponentSource::Asset(*SOURCE_ROUGHNESS_1);
        let mat_b_metallic = MaterialComponentSource::Asset(*SOURCE_METALLIC_1);

        let mat_c_diffuse = MaterialComponentSource::Asset(*SOURCE_DIFFUSE_1);
        let mat_c_alpha = MaterialComponentSource::Asset(*SOURCE_ALPHA_1);
        let mat_c_ormd = MaterialComponentSource::Asset(*SOURCE_ORMD_1);

        let mat_d_diffuse = MaterialComponentSource::Asset(*SOURCE_DIFFUSE_1);
        let mat_d_alpha = MaterialComponentSource::Asset(*SOURCE_ALPHA_1);
        let mat_d_normal_and_emissive =
            MaterialComponentSource::Asset(*SOURCE_NORMAL_AND_EMISSIVE_1);
        let mat_d_ormd = MaterialComponentSource::Asset(*SOURCE_ORMD_1);

        let mut group = MaterialGroupDescriptor::new(TEST_PAGES, TEST_SIZE);
        group.add(
            MaterialId(MATERIAL_A.hash().inner()),
            MaterialDescriptor {
                diffuse_alpha: Some(MaterialDiffuseAlphaDescriptor::Separate {
                    diffuse: Some(mat_a_diffuse),
                    alpha: Some(mat_a_alpha),
                }),
                normal_emissive: Some(MaterialNormalEmissiveDescriptor::Separate {
                    normal: Some(mat_a_normal),
                    emissive: Some(mat_a_emissive),
                }),
                ormd: Some(MaterialOrmdDescriptor::Separate {
                    occlusion: Some(mat_a_occlusion),
                    roughness: Some(mat_a_roughness),
                    metallic: Some(mat_a_metallic),
                    displacement: Some(mat_a_displacement),
                }),
            },
        );
        group.add(
            MaterialId(MATERIAL_B.hash().inner()),
            MaterialDescriptor {
                diffuse_alpha: Some(MaterialDiffuseAlphaDescriptor::Coalesced(Some(
                    mat_b_diffuse_and_alpha,
                ))),
                normal_emissive: Some(MaterialNormalEmissiveDescriptor::Coalesced(Some(
                    mat_b_normal_and_emissive,
                ))),
                ormd: Some(MaterialOrmdDescriptor::Separate {
                    occlusion: None,
                    roughness: Some(mat_b_roughness),
                    metallic: Some(mat_b_metallic),
                    displacement: None,
                }),
            },
        );
        group.add(
            MaterialId(MATERIAL_C.hash().inner()),
            MaterialDescriptor {
                diffuse_alpha: Some(MaterialDiffuseAlphaDescriptor::Separate {
                    diffuse: Some(mat_c_diffuse),
                    alpha: Some(mat_c_alpha),
                }),
                normal_emissive: None,
                ormd: Some(MaterialOrmdDescriptor::Coalesced(Some(mat_c_ormd))),
            },
        );
        group.add(
            MaterialId(MATERIAL_D.hash().inner()),
            MaterialDescriptor {
                diffuse_alpha: Some(MaterialDiffuseAlphaDescriptor::Separate {
                    diffuse: Some(mat_d_diffuse),
                    alpha: Some(mat_d_alpha),
                }),
                normal_emissive: Some(MaterialNormalEmissiveDescriptor::Coalesced(Some(
                    mat_d_normal_and_emissive,
                ))),
                ormd: Some(MaterialOrmdDescriptor::Coalesced(Some(mat_d_ormd))),
            },
        );

        let mut mat_registry = MaterialLocationRegistry::new();

        group.distribute_pages();
        group.process_locations(0);

        {
            let mut out = std::io::stdout().lock();
            group
                .cached_entries
                .iter()
                .enumerate()
                .for_each(|(i, (entry, cache))| {
                    writeln!(&mut out, "#{i}: {entry:?} => {cache:?}").unwrap();
                });
            out.flush().unwrap();
        }

        group.build(&mut asset_registry, &mut mat_registry);

        let mat_a = mat_registry
            .get(&MaterialId(MATERIAL_A.hash().inner()))
            .unwrap();
        let mat_b = mat_registry
            .get(&MaterialId(MATERIAL_B.hash().inner()))
            .unwrap();
        let mat_c = mat_registry
            .get(&MaterialId(MATERIAL_C.hash().inner()))
            .unwrap();
        let mat_d = mat_registry
            .get(&MaterialId(MATERIAL_D.hash().inner()))
            .unwrap();

        assert!(mat_a.width > 0.0 && mat_a.height > 0.0);
        assert!(mat_b.width > 0.0 && mat_b.height > 0.0);
        assert!(mat_c.width > 0.0 && mat_c.height > 0.0);
        assert!(mat_d.width > 0.0 && mat_d.height > 0.0);

        assert_eq!(mat_b.normal_and_emissive, mat_d.normal_and_emissive);
        assert_eq!(mat_c.ormd, mat_d.ormd);
    }
}
