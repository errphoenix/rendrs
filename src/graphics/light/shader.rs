use ethel::shader::{Constant, GlslLib, GlslStruct};

ethel::shader_glsl_struct! {
    struct Light {
        pos_x: f32 => float;
        pos_y: f32 => float;
        pos_z: f32 => float;
        dir_x: f32 => float;
        dir_y: f32 => float;
        dir_z: f32 => float;
        col_r: f32 => float;
        col_g: f32 => float;
        col_b: f32 => float;
        intensity: f32 => float;
        falloff_exp: f32 => float;
        _pad: f32 => float;
        omni_max_radius: f32 => float;
        spot_inner_size: f32 => float;
        spot_outer_size: f32 => float;
        kind_flag: f32 => float;
    }
}

pub const TYPE_LIGHT_UNIFORM: GlslStruct = LightGlslStruct::as_definition();

pub const CONST_LIGHT_FLAG_DIRECTIONAL: Constant<f32> =
    Constant::new("LIGHT_FLAG_DIRECTIONAL", super::KIND_FLAG_DIRECTIONAL);
pub const CONST_LIGHT_FLAG_OMNI: Constant<f32> =
    Constant::new("LIGHT_FLAG_OMNI", super::KIND_FLAG_OMNI);
pub const CONST_LIGHT_FLAG_SPOTLIGHT: Constant<f32> =
    Constant::new("LIGHT_FLAG_SPOTLIGHT", super::KIND_FLAG_SPOTLIGHT);

pub const LIB_LIGHT_ATTENUATE_ISQ_WINDOWED_CURVE: GlslLib = ethel::shader_glsl_lib! {
    float lightAttenuateISQWindowedCurve [
        sq_light_dist : float,
        light_dist    : float,
        window_max    : float,
        eps           : float
    ] => "
        float t = light_dist / window_max;
        float u = 1.0 - (t * t * t * t);
        float v = max(u, 0.0);
        float w = v * v;
        return w * (sq_light_dist / (sq_light_dist + eps));
    "
};

pub const LIB_LIGHT_ATTENUATE_DISTANCE_FALLOFF: GlslLib = ethel::shader_glsl_lib! {
    float lightAttenuateDistanceFalloff [
        light_dist    : float,
        max_dist    : float
    ] => "
        float t = light_dist / max_dist;
        float u = 1.0 - (t * t);
        float v = max(u, 0.0);
        return v * v;
    "
};
