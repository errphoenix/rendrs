use std::collections::HashMap;

use ethel::assets::{AssetId, AssetRegistry, RawTexture, TextureMetadata};
use image::EncodableLayout;
use janus::{
    StringMap,
    texture::{MipLevels, Tex, Texture},
};

use crate::graphics::material::{
    MaterialEntryLocation, MaterialGroup, MaterialId, MaterialLocation, MaterialLocationRegistry,
};

/// Page indexing offset to ignore reserved first-N pages in material groups.
///
/// The 3 reserved faces are, in order:
/// * Fallback for `diffuse + alpha`, full blank rgba.
/// * Fallback for ORMD, a full black rgba image.
/// * Fallback for `normal + emissive`, full blue rgb + `0` alpha.
pub const PAGE_OFFSET_RESERVED: usize = 3;
pub const PAGE_FALLBACK_DIFFUSE_ALPHA: usize = 0;
pub const PAGE_FALLBACK_ORMD: usize = 1;
pub const PAGE_FALLBACK_NORMAL_EMISSIVE: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct MaterialGroupDescriptor {
    pub pages: u16,
    pub size: u16,

    // both mapped by material-id
    pub desciptors: StringMap<MaterialDescriptor>,
    pub materials: StringMap<MaterialLocation>,

    /// Cached page indices allocated to entries that were already processed,
    /// avoiding duplicate entries in the final material texture array.
    ///
    /// The cache also stores the sub- width and height of each entry texture,
    /// which are only set at the final build step.
    pub cached_entries: HashMap<MaterialEntryDescriptor, MaterialEntryCache>,
}
impl MaterialGroupDescriptor {
    pub fn new(pages: u16, size: u16) -> Self {
        assert!(pages > PAGE_OFFSET_RESERVED as u16);
        Self {
            pages,
            size,
            desciptors: StringMap::default(),
            materials: StringMap::default(),
            cached_entries: HashMap::new(),
        }
    }

    pub fn add(&mut self, id: MaterialId, material: MaterialDescriptor) {
        self.desciptors.insert(id.0, material);
    }

    pub fn distribute_pages(&mut self) {
        self.cached_entries.clear();
        // page 0,1,2 are reserved for fallback
        let mut page_i = 3;
        self.desciptors.values().for_each(
            |&MaterialDescriptor {
                 diffuse_alpha,
                 normal_emissive,
                 ormd,
             }| {
                match diffuse_alpha {
                    Some(da) => {
                        let entry = MaterialEntryDescriptor::DiffuseAlpha(da);
                        if !self.cached_entries.contains_key(&entry) {
                            self.cached_entries.insert(
                                entry,
                                MaterialEntryCache {
                                    assigned_page_index: page_i,
                                    norm_sub_width: 0.1,
                                    norm_sub_height: 0.1,
                                },
                            );
                            page_i += 1;
                        }
                    }
                    _ => {}
                }
                match normal_emissive {
                    Some(ne) => {
                        let entry = MaterialEntryDescriptor::NormalEmissive(ne);
                        if !self.cached_entries.contains_key(&entry) {
                            self.cached_entries.insert(
                                entry,
                                MaterialEntryCache {
                                    assigned_page_index: page_i,
                                    norm_sub_width: 0.1,
                                    norm_sub_height: 0.1,
                                },
                            );
                            page_i += 1;
                        }
                    }
                    _ => {}
                }
                match ormd {
                    Some(ormd) => {
                        let entry = MaterialEntryDescriptor::Ormd(ormd);
                        if !self.cached_entries.contains_key(&entry) {
                            self.cached_entries.insert(
                                entry,
                                MaterialEntryCache {
                                    assigned_page_index: page_i,
                                    ..Default::default()
                                },
                            );
                            page_i += 1;
                        }
                    }
                    _ => {}
                }
            },
        );

        assert!(
            page_i < self.pages,
            "the specified number of pages is insufficient to distribute all defined materials: {page_i}/{}",
            self.pages
        );

        tracing::info!(
            "Distributed unique material entries among group: used up {} total pages, {} free.",
            page_i,
            self.pages - page_i
        );
    }

    pub fn process_locations(&mut self, group_index: u16) {
        self.materials.clear();

        let fallback_diffuse_alpha = MaterialEntryLocation {
            page: PAGE_FALLBACK_DIFFUSE_ALPHA as u16,
            group_index,
        };
        let fallback_normal_emissive = MaterialEntryLocation {
            page: PAGE_FALLBACK_NORMAL_EMISSIVE as u16,
            group_index,
        };
        let fallback_ormd = MaterialEntryLocation {
            page: PAGE_FALLBACK_ORMD as u16,
            group_index,
        };

        self.desciptors.iter().for_each(|(&id, descriptor)| {
            let mut material_location = MaterialLocation {
                diffuse_and_alpha: fallback_diffuse_alpha,
                normal_and_emissive: fallback_normal_emissive,
                ormd: fallback_ormd,
                // initialized later
                width: 0f32,
                height: 0f32,
            };

            if let Some(diffuse_alpha) = descriptor.diffuse_alpha {
                let entry = MaterialEntryDescriptor::DiffuseAlpha(diffuse_alpha);
                if let Some(cache) = self.cached_entries.get(&entry) {
                    material_location.diffuse_and_alpha.page = cache.assigned_page_index;
                }
            }
            if let Some(normal_emissive) = descriptor.normal_emissive {
                let entry = MaterialEntryDescriptor::NormalEmissive(normal_emissive);
                if let Some(cache) = self.cached_entries.get(&entry) {
                    material_location.normal_and_emissive.page = cache.assigned_page_index;
                }
            }
            if let Some(ormd) = descriptor.ormd {
                let entry = MaterialEntryDescriptor::Ormd(ormd);
                if let Some(cache) = self.cached_entries.get(&entry) {
                    material_location.ormd.page = cache.assigned_page_index;
                }
            }
            self.materials.insert(id, material_location);
        });
    }

    pub fn build(
        mut self,
        texture_registry: &mut AssetRegistry<RawTexture, TextureMetadata>,
        material_registry: &mut MaterialLocationRegistry,
    ) -> MaterialGroup {
        #[cfg(not(test))]
        janus::assert_gl!();

        let size = self.size as i32;
        #[cfg(not(test))]
        let texture = Texture::new_array(
            size,
            size,
            self.pages as i32,
            MipLevels::default(),
            super::MATERIAL_TEXTURE_PIXEL_TYPE,
            super::MATERIAL_TEXTURE_FORMAT,
        );

        // load default blank texture at page 0
        #[cfg(not(test))]
        {
            let size = self.size as u32;
            let blank_image = image::RgbaImage::from_pixel(size, size, image::Rgba([255u8; 4]));
            texture
                .upload_layer_whole(0, 0, blank_image.as_bytes())
                .expect("texture is always array texture");
        }

        let pixel_count = self.size as usize * self.size as usize;
        let fallback_sc_white = vec![255u8; pixel_count];
        let fallback_sc_black = vec![0u8; pixel_count];
        let fallback_rgb_white = vec![255u8; pixel_count * 3];
        let fallback_rgb_black = vec![0u8; pixel_count * 3];
        let fallback_rgb_blue = {
            let mut vec = Vec::with_capacity(pixel_count * 3);
            for _ in 0..pixel_count {
                vec.extend_from_slice(&[0u8, 0u8, 255u8]);
            }
            vec
        };

        let mut image_load_buffer = Vec::with_capacity(pixel_count * 4);

        self.cached_entries
            .iter_mut()
            .for_each(|(entry, cache)| match entry {
                MaterialEntryDescriptor::Ormd(MaterialOrmdDescriptor::Coalesced(rgba))
                | MaterialEntryDescriptor::DiffuseAlpha(
                    MaterialDiffuseAlphaDescriptor::Coalesced(rgba),
                )
                | MaterialEntryDescriptor::NormalEmissive(
                    MaterialNormalEmissiveDescriptor::Coalesced(rgba),
                ) => {
                    if let Some(rgba) = rgba {
                        let image = rgba.load(texture_registry).0;
                        let img_w = image.width();
                        let img_h = image.height();

                        cache.norm_sub_width = img_w as f32 / size as f32;
                        cache.norm_sub_height = img_h as f32 / size as f32;
                        let page_index = cache.assigned_page_index as i32;

                        #[cfg(not(test))]
                        texture
                            .upload_layer(
                                0,
                                0,
                                0,
                                page_index,
                                img_w as i32,
                                img_h as i32,
                                &image.into_bytes(),
                            )
                            .unwrap();
                    }
                }
                MaterialEntryDescriptor::Ormd(MaterialOrmdDescriptor::Separate {
                    occlusion,
                    roughness,
                    metallic,
                    displacement,
                }) => {
                    if occlusion.is_some()
                        || roughness.is_some()
                        || metallic.is_some()
                        || displacement.is_some()
                    {
                        build_4a(
                            #[cfg(not(test))]
                            &texture,
                            occlusion,
                            roughness,
                            metallic,
                            displacement,
                            cache,
                            texture_registry,
                            size as u32,
                            &fallback_sc_black,
                            &mut image_load_buffer,
                        );
                    }
                }
                MaterialEntryDescriptor::Ormd(MaterialOrmdDescriptor::OrmAndDisplacement {
                    orm,
                    displacement,
                }) => {
                    if orm.is_some() || displacement.is_some() {
                        build_rgb_a(
                            #[cfg(not(test))]
                            &texture,
                            orm,
                            displacement,
                            cache,
                            texture_registry,
                            size as u32,
                            &fallback_rgb_black,
                            &fallback_sc_black,
                            &mut image_load_buffer,
                        );
                    }
                }
                MaterialEntryDescriptor::NormalEmissive(
                    MaterialNormalEmissiveDescriptor::Separate { normal, emissive },
                ) => {
                    if normal.is_some() || emissive.is_some() {
                        build_rgb_a(
                            #[cfg(not(test))]
                            &texture,
                            normal,
                            emissive,
                            cache,
                            texture_registry,
                            size as u32,
                            &fallback_rgb_blue,
                            &fallback_sc_black,
                            &mut image_load_buffer,
                        );
                    }
                }
                MaterialEntryDescriptor::DiffuseAlpha(
                    MaterialDiffuseAlphaDescriptor::Separate { diffuse, alpha },
                ) => {
                    if diffuse.is_some() || alpha.is_some() {
                        build_rgb_a(
                            #[cfg(not(test))]
                            &texture,
                            diffuse,
                            alpha,
                            cache,
                            texture_registry,
                            size as u32,
                            &fallback_rgb_white,
                            &fallback_sc_white,
                            &mut image_load_buffer,
                        );
                    }
                }
            });

        self.desciptors.drain().for_each(|(id, descriptor)| {
            let (width, height) = {
                if let Some(da) = descriptor.diffuse_alpha {
                    let entry = MaterialEntryDescriptor::DiffuseAlpha(da);
                    let cache = &self.cached_entries[&entry];
                    (cache.norm_sub_width, cache.norm_sub_height)
                } else if let Some(ne) = descriptor.normal_emissive {
                    let entry = MaterialEntryDescriptor::NormalEmissive(ne);
                    let cache = &self.cached_entries[&entry];
                    (cache.norm_sub_width, cache.norm_sub_height)
                } else if let Some(ormd) = descriptor.ormd {
                    let entry = MaterialEntryDescriptor::Ormd(ormd);
                    let cache = &self.cached_entries[&entry];
                    (cache.norm_sub_width, cache.norm_sub_height)
                } else {
                    panic!("invalid material {id:?} has zero defined material components");
                }
            };
            let location = self.materials.get_mut(&id).unwrap();
            location.width = width;
            location.height = height;
        });

        self.materials.drain().for_each(|(id, location)| {
            material_registry.add(MaterialId(id), location);
        });

        MaterialGroup {
            #[cfg(not(test))]
            array_texture_object: texture,
            size: self.size,
            pages: self.pages,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaterialDescriptor {
    /// Diffuse + alpha entry
    pub diffuse_alpha: Option<MaterialDiffuseAlphaDescriptor>,
    /// Normal + emissive entry
    pub normal_emissive: Option<MaterialNormalEmissiveDescriptor>,
    /// ORMD (occlusion-roughness-metallic-displacement) entry
    pub ormd: Option<MaterialOrmdDescriptor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialEntryDescriptor {
    DiffuseAlpha(MaterialDiffuseAlphaDescriptor),
    NormalEmissive(MaterialNormalEmissiveDescriptor),
    Ormd(MaterialOrmdDescriptor),
}

#[derive(Clone, Debug, Default)]
pub struct MaterialEntryCache {
    pub norm_sub_width: f32,
    pub norm_sub_height: f32,
    pub assigned_page_index: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialDiffuseAlphaDescriptor {
    /// Pre-coalesced `diffuse + alpha` where the first 3 channels (RGB)
    /// represent the diffuse/albedo properties, and the last alpha channel
    /// represents the transparency.
    Coalesced(Option<MaterialComponentSource>),
    /// Separate non-coalesced `diffuse + alpha` where the diffuse is an RGB
    /// dffuse/albedo texture, and the alpha is a single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    Separate {
        diffuse: Option<MaterialComponentSource>,
        alpha: Option<MaterialComponentSource>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialNormalEmissiveDescriptor {
    /// Pre-coalesced `normal + emissive` where the first 3 channels (RGB)
    /// represent the normal map properties, and the last channel
    /// represents the emissive mapping property.
    Coalesced(Option<MaterialComponentSource>),
    /// Separate non-coalesced `normal + emissive` where the normal map is an RGB
    /// texture, and the last channel is the emissive single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    Separate {
        normal: Option<MaterialComponentSource>,
        emissive: Option<MaterialComponentSource>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialOrmdDescriptor {
    /// Pre-coalesced ORMD where each channel represents occlusion,
    /// roughness, metallic, and displacement, respectively.
    Coalesced(Option<MaterialComponentSource>),
    /// Separate non-coalesced `ORM + displacement` where the ORM map is an RGB
    /// texture, and the displacement map is a single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    OrmAndDisplacement {
        orm: Option<MaterialComponentSource>,
        displacement: Option<MaterialComponentSource>,
    },
    /// Separate non-coalesced RSOD where each sub-entry is a separate
    /// single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    Separate {
        occlusion: Option<MaterialComponentSource>,
        roughness: Option<MaterialComponentSource>,
        metallic: Option<MaterialComponentSource>,
        displacement: Option<MaterialComponentSource>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialComponentSource {
    Path(&'static str),
    Asset(AssetId),
}
impl MaterialComponentSource {
    /// Load material component image from path.
    ///
    /// # Panics
    /// If the material component source is not a
    /// [`Path`](MaterialComponentSource::Path) variant.
    ///
    /// The function will also panic if there was an IO read error or image
    /// load error.
    pub fn load_path(&self) -> RawTexture {
        if let Self::Path(path) = self {
            let mut bytes = Vec::new();
            match std::fs::read(path) {
                Ok(loaded) => bytes = loaded,
                Err(err) => {
                    tracing::error!(
                        "failed to read image data from filesystem from path {path}:\n{err}"
                    )
                }
            }
            RawTexture(image::load_from_memory(&bytes).expect("failed to load image from memory"))
        } else {
            panic!("material component source is not defined by local path");
        }
    }

    /// Load material component image from asset registry.
    ///
    /// # Panics
    /// If the material component source is not an
    /// [`Asset`](MaterialComponentSource::Asset) variant.
    ///
    /// The function will also panic if the asset does not exist on the
    /// given asset registry.
    pub fn load_asset(
        &self,
        registry: &mut AssetRegistry<RawTexture, TextureMetadata>,
    ) -> RawTexture {
        if let Self::Asset(id) = self {
            let handle = registry
                .get_mut(*id)
                .expect("failed to locate asset in asset registry");
            if !handle.is_in_memory() {
                if let Err(err) = handle.load_to_memory(&Default::default()) {
                    tracing::error!("failed to load asset id {id:?} to memory:\n{err}");
                }
            }
            handle
                .take_from_memory()
                .expect("failed to load asset to memory")
        } else {
            panic!("material component source is not defined by asset id");
        }
    }

    pub fn load(&self, registry: &mut AssetRegistry<RawTexture, TextureMetadata>) -> RawTexture {
        match self {
            MaterialComponentSource::Path(_) => self.load_path(),
            MaterialComponentSource::Asset(_) => self.load_asset(registry),
        }
    }
}

fn coalesce_image_rgb_a(rgb: &[u8], a: &[u8], out: &mut Vec<u8>) {
    {
        let c_rgb = rgb.len() / 3;
        let c_a = a.len();
        assert!(
            c_rgb == c_a,
            "cannot coalesce rgb and single-channel image buffers if the number of pixels is not equal"
        );
    }

    a.iter()
        .zip(rgb.as_chunks::<3>().0)
        .for_each(|(&a, &[r, g, b])| {
            out.extend_from_slice(&[r, g, b, a]);
        });
}

fn coalesce_image_4a(r: &[u8], g: &[u8], b: &[u8], a: &[u8], out: &mut Vec<u8>) {
    {
        assert!(
            r.len() == g.len() && g.len() == b.len() && b.len() == a.len(),
            "cannot coalesce 4 single-channel image buffers into rgba if the number of pixels is not equal"
        );
    }

    r.iter()
        .zip(g)
        .zip(b)
        .zip(a)
        .for_each(|(((&r, &g), &b), &a)| {
            out.extend_from_slice(&[r, g, b, a]);
        });
}

fn build_4a(
    #[cfg(not(test))] target_texture: &Texture,
    r: &Option<MaterialComponentSource>,
    g: &Option<MaterialComponentSource>,
    b: &Option<MaterialComponentSource>,
    a: &Option<MaterialComponentSource>,
    cache: &mut MaterialEntryCache,
    texture_registry: &mut AssetRegistry<RawTexture, TextureMetadata>,
    max_size: u32,
    fallback: &[u8],
    coal_buffer: &mut Vec<u8>,
) {
    let r = r.map(|src| src.load(texture_registry));
    let g = g.map(|src| src.load(texture_registry));
    let b = b.map(|src| src.load(texture_registry));
    let a = a.map(|src| src.load(texture_registry));

    let (known_len, width, height) = if let Some(r) = &r {
        let w = r.0.width();
        let h = r.0.height();
        ((w * h) as usize, w, h)
    } else if let Some(g) = &g {
        let w = g.0.width();
        let h = g.0.height();
        ((w * h) as usize, w, h)
    } else if let Some(b) = &b {
        let w = b.0.width();
        let h = b.0.height();
        ((w * h) as usize, w, h)
    } else if let Some(a) = &a {
        let w = a.0.width();
        let h = a.0.height();
        ((w * h) as usize, w, h)
    } else {
        ((max_size * max_size) as usize, max_size, max_size)
    };

    let r = r
        .as_ref()
        .map_or(&fallback[..known_len], |src| src.0.as_bytes());
    let g = g
        .as_ref()
        .map_or(&fallback[..known_len], |src| src.0.as_bytes());
    let b = b
        .as_ref()
        .map_or(&fallback[..known_len], |src| src.0.as_bytes());
    let a = a
        .as_ref()
        .map_or(&fallback[..known_len], |src| src.0.as_bytes());
    coalesce_image_4a(r, g, b, a, coal_buffer);

    cache.norm_sub_width = width as f32 / max_size as f32;
    cache.norm_sub_height = height as f32 / max_size as f32;
    let page_index = cache.assigned_page_index as i32;

    #[cfg(not(test))]
    target_texture
        .upload_layer(
            0,
            0,
            0,
            page_index,
            width as i32,
            height as i32,
            coal_buffer,
        )
        .unwrap();

    coal_buffer.clear();
}

fn build_rgb_a(
    #[cfg(not(test))] target_texture: &Texture,
    rgb: &Option<MaterialComponentSource>,
    alpha: &Option<MaterialComponentSource>,
    cache: &mut MaterialEntryCache,
    texture_registry: &mut AssetRegistry<RawTexture, TextureMetadata>,
    max_size: u32,
    rgb_fallback: &[u8],
    alpha_fallback: &[u8],
    coal_buffer: &mut Vec<u8>,
) {
    let rgb = rgb.map(|src| src.load(texture_registry));
    let alpha = alpha.map(|src| src.load(texture_registry));

    let (known_len, width, height) = if let Some(rgb) = &rgb {
        let w = rgb.0.width();
        let h = rgb.0.height();
        ((w * h) as usize, w, h)
    } else if let Some(alpha) = &alpha {
        let w = alpha.0.width();
        let h = alpha.0.height();
        ((w * h) as usize, w, h)
    } else {
        ((max_size * max_size) as usize, max_size, max_size)
    };

    let rgb = rgb
        .as_ref()
        .map_or(&rgb_fallback[..known_len * 3], |src| src.0.as_bytes());
    let alpha = alpha
        .as_ref()
        .map_or(&alpha_fallback[..known_len], |src| src.0.as_bytes());
    coalesce_image_rgb_a(rgb, alpha, coal_buffer);

    cache.norm_sub_width = width as f32 / max_size as f32;
    cache.norm_sub_height = height as f32 / max_size as f32;
    let page_index = cache.assigned_page_index as i32;

    #[cfg(not(test))]
    target_texture
        .upload_layer(
            0,
            0,
            0,
            page_index,
            width as i32,
            height as i32,
            coal_buffer,
        )
        .unwrap();

    coal_buffer.clear();
}
