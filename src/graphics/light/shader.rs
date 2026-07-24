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

/// Light attenuation function through an inverse-squared windowed curve.
///
/// Creates the `lightAttenuate` function with the following parameters:
/// * `float sq_light_dist` the squared distance to the light.
/// * `float light_dist` the distance to the light.
/// * `float window_max` the length of the window or maximum affected
///    distance of the light.
/// * `float eps` a small epsilon to avoid singularities in the curve;
///    depends on scene scale.
///
/// This is exclusive with [`LIB_LIGHT_ATTENUATE_DISTANCE_FALLOFF`] and
/// shaders will fail to compile if they are both defined.
pub const LIB_LIGHT_ATTENUATE_ISQ_WINDOWED_CURVE: GlslLib = ethel::shader_glsl_lib! {
    float lightAttenuate [
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

/// Light attenuation function through a simple distance-falloff curve.
///
/// Creates the `lightAttenuate` function with the following parameters:
/// * `float light_dist` the distance to the light.
/// * `float max_max` the maximum affected distance of the light.
///
/// This is exclusive with [`LIB_LIGHT_ATTENUATE_ISQ_WINDOWED_CURVE`] and
/// shaders will fail to compile if they are both defined.
///
/// It is generally cheaper than [`LIB_LIGHT_ATTENUATE_ISQ_WINDOWED_CURVE`].
pub const LIB_LIGHT_ATTENUATE_DISTANCE_FALLOFF: GlslLib = ethel::shader_glsl_lib! {
    float lightAttenuate [
        light_dist    : float,
        max_dist      : float
    ] => "
        float t = light_dist / max_dist;
        float u = 1.0 - (t * t);
        float v = max(u, 0.0);
        return v * v;
    "
};

/// Spotlight falloff curve through standard penumbra/umbra, squared.
///
/// Creates the `lightSpotlightFalloff` function with the following parameters:
/// * `float penumbra_cos` the cosine angle of the penumbra angle of
///    the spotlight.
/// * `float umbra_cos` the cosine angle of the umbra angle of
///    the spotlight.
/// * `float surfaece_cos` the cosine angle of the angle between the
///    spotlight's direction and vector pointing from the surface to the
///    light.
///
/// This is exclusive with [`LIB_LIGHT_SPOTLIGHT_FALLOFF_SMOOTHED`] and
/// shaders will fail to compile if they are both defined.
///
/// According to "Real-time Rendering", this falloff function is used in the
/// Frostbite game engine.
pub const LIB_LIGHT_SPOTLIGHT_FALLOFF_SQ: GlslLib = ethel::shader_glsl_lib! {
    float lightSpotlightFalloff [
        penumbra_cos : float,
        umbra_cos    : float,
        surface_cos  : float
    ] => "
        float n = surface_cos - umbra_cos;
        float d = penumbra_cos - umbra_cos;
        float t = n / d;
        float u = clamp(t, 0.0, 1.0);
        return u * u;
    "
};

/// Spotlight falloff curve through standard penumbra/umbra, plus a smoothstep.
///
/// Creates the `lightSpotlightFalloff` function with the following parameters:
/// * `float penumbra_cos` the cosine angle of the penumbra angle of
///    the spotlight.
/// * `float umbra_cos` the cosine angle of the umbra angle of
///    the spotlight.
/// * `float surfaece_cos` the cosine angle of the angle between the
///    spotlight's direction and vector pointing from the surface to the
///    light.
///
/// This is exclusive with [`LIB_LIGHT_SPOTLIGHT_FALLOFF_SMOOTHED`] and
/// shaders will fail to compile if they are both defined.
///
/// According to "Real-time Rendering", this falloff function is used in the
/// `three.js` browser graphics library.
pub const LIB_LIGHT_SPOTLIGHT_FALLOFF_SMOOTHED: GlslLib = ethel::shader_glsl_lib! {
    float lightSpotlightFalloff [
        penumbra_cos : float,
        umbra_cos    : float,
        surface_cos  : float
    ] => "
        float n = surface_cos - umbra_cos;
        float d = penumbra_cos - umbra_cos;
        float t = n / d;
        float u = clamp(t, 0.0, 1.0);
        float v = u * u;
        return v * (3.0 - 2.0 * u);
    "
};
