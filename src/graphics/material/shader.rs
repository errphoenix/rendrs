use ethel::shader::{GlslLib, GlslStruct, WriteValue};

use crate::graphics::material::MaterialEntryLocation;

pub const PACKED: u32 = 0x00ff000f;
pub const PAGE: u16 = (PACKED & 0xffff) as u16;
pub const GROUP: u16 = ((PACKED >> 16) & 0xffff) as u16;
pub const REC: u32 = ((GROUP as u32) << 16) | PAGE as u32;

impl WriteValue for MaterialEntryLocation {
    fn write_value(&self, to: &mut impl std::fmt::Write) -> std::fmt::Result {
        let group = self.group_index as u32;
        let page = self.page as u32;
        let packed = (group << 16) | page;
        write!(to, "MaterialEntryLocation({packed})")
    }
}

/// Unpack a MaterialEntryLocation from 2x packed 16-bit integers
/// to 2x 32-bit integers (`uint`), returning a `uvec2`.
///
/// The first component is the group index field, the second is the page index.
pub const LIB_MATERIAL_ENTRY_UNPACK: GlslLib = ethel::shader_glsl_lib! {
    uvec2 materialLocationEntryUnpack [
        entryLocation : MaterialEntryLocation
    ] => "
        uint inner = entryLocation.inner;
        uint page = inner & 0xffff;
        uint group = (inner >> 16) & 0xffff;
        return uvec2(group, page);
    "
};

pub const TYPE_MATERIAL_ENTRY_LOCATION: GlslStruct =
    MaterialEntryLocationGlslStruct::as_definition();
pub const TYPE_MATERIAL_LOCATION: GlslStruct = MaterialLocationGlslStruct::as_definition();

ethel::shader_glsl_struct! {
    struct MaterialEntryLocation {
        inner: u32 => uint;
    }
}

ethel::shader_glsl_struct! {
    struct MaterialLocation {
        diffuse_and_alpha: MaterialEntryLocation => MaterialEntryLocation;
        normal_and_emissive: MaterialEntryLocation => MaterialEntryLocation;
        ormd: MaterialEntryLocation => MaterialEntryLocation;
        width: f32 => float;
        height: f32 => float;
    }
}
