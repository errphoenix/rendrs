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
        // page 0 is reserved for default
        let mut page_i = 1;
        self.desciptors.values().for_each(
            |&MaterialDescriptor {
                 diffuse_emissive,
                 rsod,
             }| {
                match diffuse_emissive {
                    Some(de) => {
                        let entry = MaterialEntryDescriptor::DiffuseEmissive(de);
                        self.cached_entries.insert(
                            entry,
                            MaterialEntryCache {
                                assigned_page_index: page_i,
                                ..Default::default()
                            },
                        );
                        page_i += 1;
                    }
                    _ => {}
                }
                match rsod {
                    Some(rsod) => {
                        let entry = MaterialEntryDescriptor::Rsod(rsod);
                        self.cached_entries.insert(
                            entry,
                            MaterialEntryCache {
                                assigned_page_index: page_i,
                                ..Default::default()
                            },
                        );
                        page_i += 1;
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

        let default_entry_loc = MaterialEntryLocation {
            group_index,
            page: 0,
        };

        self.desciptors.iter().for_each(|(&id, descriptor)| {
            let mut material_location = MaterialLocation {
                diffuse_and_emissive: default_entry_loc,
                rsod_entry: default_entry_loc,
                // initialized later
                width: 0f32,
                height: 0f32,
            };

            if let Some(diffuse_emissive) = descriptor.diffuse_emissive {
                let entry = MaterialEntryDescriptor::DiffuseEmissive(diffuse_emissive);
                if let Some(cache) = self.cached_entries.get(&entry) {
                    material_location.diffuse_and_emissive.page = cache.assigned_page_index;
                }
            }
            if let Some(rsod) = descriptor.rsod {
                let entry = MaterialEntryDescriptor::Rsod(rsod);
                if let Some(cache) = self.cached_entries.get(&entry) {
                    material_location.rsod_entry.page = cache.assigned_page_index;
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
        janus::assert_gl!();

        let size = self.size as i32;
        let texture = Texture::new_array(
            size,
            size,
            self.pages as i32,
            MipLevels::default(),
            super::MATERIAL_TEXTURE_PIXEL_TYPE,
            super::MATERIAL_TEXTURE_FORMAT,
        );

        // load default blank texture at page 0
        {
            let size = self.size as u32;
            let blank_image = image::RgbaImage::from_pixel(size, size, image::Rgba([255u8; 4]));
            texture
                .upload_layer_whole(0, 0, blank_image.as_bytes())
                .expect("texture is always array texture");
        }

        let pixel_count = (self.size * self.size) as usize;
        let blank_rgba = vec![255u8; pixel_count * 4];
        let blank_rgb = vec![255u8; pixel_count * 3];
        let blank_sc = vec![255u8; pixel_count];
        let mut image_load_buffer = Vec::with_capacity(pixel_count * 4);
        let mut page_index = 1;

        self.cached_entries
            .iter_mut()
            .for_each(|(entry, cache)| match entry {
                MaterialEntryDescriptor::Rsod(MaterialRsodDescriptor::Coalesced(rgba))
                | MaterialEntryDescriptor::DiffuseEmissive(
                    MaterialDiffuseEmissiveDescriptor::Coalesced(rgba),
                ) => {
                    if let Some(rgba) = rgba {
                        let image = rgba.load(texture_registry).0;
                        let img_w = image.width();
                        let img_h = image.height();
                        cache.norm_sub_width = img_w as f32 / size as f32;
                        cache.norm_sub_height = img_h as f32 / size as f32;
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
                    } else {
                        cache.norm_sub_width = 0.1;
                        cache.norm_sub_height = 0.1;
                        texture
                            .upload_layer(0, 0, 0, page_index, size, size, &blank_rgba)
                            .unwrap();
                    }
                    page_index += 1;
                }
                MaterialEntryDescriptor::Rsod(MaterialRsodDescriptor::Separate {
                    roughness,
                    specular,
                    occlusion,
                    displacement,
                }) => {
                    let roughness = roughness.map(|src| src.load(texture_registry));
                    let specular = specular.map(|src| src.load(texture_registry));
                    let occlusion = occlusion.map(|src| src.load(texture_registry));
                    let displacement = displacement.map(|src| src.load(texture_registry));
                    let r = roughness
                        .as_ref()
                        .map_or(blank_sc.as_slice(), |src| src.0.as_bytes());
                    let s = specular
                        .as_ref()
                        .map_or(blank_sc.as_slice(), |src| src.0.as_bytes());
                    let o = occlusion
                        .as_ref()
                        .map_or(blank_sc.as_slice(), |src| src.0.as_bytes());
                    let d = displacement
                        .as_ref()
                        .map_or(blank_sc.as_slice(), |src| src.0.as_bytes());
                    coalesce_image_4a(r, s, o, d, &mut image_load_buffer);

                    let image = image::load_from_memory(&image_load_buffer)
                        .unwrap()
                        .into_rgba8();
                    let img_w = image.width();
                    let img_h = image.height();
                    cache.norm_sub_width = img_w as f32 / size as f32;
                    cache.norm_sub_height = img_h as f32 / size as f32;
                    texture
                        .upload_layer(
                            0,
                            0,
                            0,
                            page_index,
                            img_w as i32,
                            img_h as i32,
                            &image_load_buffer,
                        )
                        .unwrap();
                    image_load_buffer.clear();
                    page_index += 1;
                }

                MaterialEntryDescriptor::DiffuseEmissive(
                    MaterialDiffuseEmissiveDescriptor::Separate { diffuse, emissive },
                ) => {
                    let diffuse = diffuse.map(|src| src.load(texture_registry));
                    let emissive = emissive.map(|src| src.load(texture_registry));
                    let d = diffuse
                        .as_ref()
                        .map_or(blank_rgb.as_slice(), |src| src.0.as_bytes());
                    let e = emissive
                        .as_ref()
                        .map_or(blank_sc.as_slice(), |src| src.0.as_bytes());
                    coalesce_image_rgb_a(d, e, &mut image_load_buffer);

                    let image = image::load_from_memory(&image_load_buffer)
                        .unwrap()
                        .into_rgba8();
                    let img_w = image.width();
                    let img_h = image.height();
                    cache.norm_sub_width = img_w as f32 / size as f32;
                    cache.norm_sub_height = img_h as f32 / size as f32;
                    texture
                        .upload_layer(
                            0,
                            0,
                            0,
                            page_index,
                            img_w as i32,
                            img_h as i32,
                            &image_load_buffer,
                        )
                        .unwrap();
                    image_load_buffer.clear();
                    page_index += 1;
                }
            });

        self.desciptors.drain().for_each(|(id, descriptor)| {
            let (width, height) = {
                if let Some(de) = descriptor.diffuse_emissive {
                    let entry = MaterialEntryDescriptor::DiffuseEmissive(de);
                    let cache = &self.cached_entries[&entry];
                    (cache.norm_sub_width, cache.norm_sub_height)
                } else if let Some(rsod) = descriptor.rsod {
                    let entry = MaterialEntryDescriptor::Rsod(rsod);
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
            array_texture_object: texture,
            size: self.size,
            pages: self.pages,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaterialDescriptor {
    /// Diffuse + emissive entry
    pub diffuse_emissive: Option<MaterialDiffuseEmissiveDescriptor>,
    /// RSOD (roughness-specular-occlusion-displacement) entry
    pub rsod: Option<MaterialRsodDescriptor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialEntryDescriptor {
    DiffuseEmissive(MaterialDiffuseEmissiveDescriptor),
    Rsod(MaterialRsodDescriptor),
}

#[derive(Clone, Debug, Default)]
pub struct MaterialEntryCache {
    pub norm_sub_width: f32,
    pub norm_sub_height: f32,
    pub assigned_page_index: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialDiffuseEmissiveDescriptor {
    /// Pre-coalesced `diffuse + emissive` where the first 3 channels (RGB)
    /// represent the diffuse/albedo properties, and the last alpha channel
    /// represents the emissive property.
    Coalesced(Option<MaterialComponentSource>),
    /// Separate non-coalesced `diffuse + emissive` where the diffuse is an RGB
    /// dffuse/albedo texture, and the emissive is a single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    Separate {
        diffuse: Option<MaterialComponentSource>,
        emissive: Option<MaterialComponentSource>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialRsodDescriptor {
    /// Pre-coalesced RSOD where each channel represents roughness,
    /// specular, occlusion, and displacement, respectively.
    Coalesced(Option<MaterialComponentSource>),
    /// Separate non-coalesced RSOD where each sub-entry is a separate
    /// single-channel texture.
    ///
    /// This is coalesced into a single RGBA texture later.
    Separate {
        roughness: Option<MaterialComponentSource>,
        specular: Option<MaterialComponentSource>,
        occlusion: Option<MaterialComponentSource>,
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
        .for_each(|(a, [r, g, b])| {
            out.push(*r);
            out.push(*g);
            out.push(*b);
            out.push(*a);
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
        .map(|(((&r, &g), &b), &a)| [r, g, b, a])
        .flatten()
        .for_each(|b| out.push(b));
}
