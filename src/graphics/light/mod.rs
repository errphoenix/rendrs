pub mod shader;

#[allow(unused_imports)]
pub use shader::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LightVolume {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Light {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub dir_x: f32,
    pub dir_y: f32,
    pub dir_z: f32,
    pub col_r: f32,
    pub col_g: f32,
    pub col_b: f32,
    pub intensity: f32,
    pub falloff_exp: f32,
    _pad: f32,
    pub params: LightParams,
}
impl Light {
    pub const fn new_const(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        falloff_exp: f32,
        params: LightParams,
    ) -> Self {
        Self {
            pos_x: position[0],
            pos_y: position[1],
            pos_z: position[2],
            dir_x: direction[0],
            dir_y: direction[1],
            dir_z: direction[2],
            col_r: color[0],
            col_g: color[1],
            col_b: color[2],
            intensity,
            falloff_exp,
            _pad: 0f32,
            params,
        }
    }

    pub const fn new_omni_const(
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        falloff_exp: f32,
        max_radius: f32,
    ) -> Self {
        Self::new_const(
            position,
            [0f32; 3],
            color,
            intensity,
            falloff_exp,
            LightParams::omni(max_radius),
        )
    }

    pub const fn new_directional_const(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        falloff_exp: f32,
    ) -> Self {
        Self::new_const(
            position,
            direction,
            color,
            intensity,
            falloff_exp,
            LightParams::directional(),
        )
    }

    pub const fn new_spotlight_const(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        falloff_exp: f32,
        inner_size: f32,
        outer_size: f32,
        max_radius: f32,
    ) -> Self {
        Self::new_const(
            position,
            direction,
            color,
            intensity,
            falloff_exp,
            LightParams::spotlight(inner_size, outer_size, max_radius),
        )
    }

    pub fn new(
        position: impl Into<[f32; 3]>,
        direction: impl Into<[f32; 3]>,
        color: impl Into<[f32; 3]>,
        intensity: f32,
        falloff_exp: f32,
        params: LightParams,
    ) -> Self {
        let position = position.into();
        let direction = direction.into();
        let color = color.into();
        Self {
            pos_x: position[0],
            pos_y: position[1],
            pos_z: position[2],
            dir_x: direction[0],
            dir_y: direction[1],
            dir_z: direction[2],
            col_r: color[0],
            col_g: color[1],
            col_b: color[2],
            intensity,
            falloff_exp,
            _pad: 0f32,
            params,
        }
    }

    pub fn new_omni(
        position: impl Into<[f32; 3]>,
        color: impl Into<[f32; 3]>,
        intensity: f32,
        falloff_exp: f32,
        max_radius: f32,
    ) -> Self {
        let position = position.into();
        let color = color.into();
        Self::new_omni_const(position, color, intensity, falloff_exp, max_radius)
    }

    pub fn new_directional(
        position: impl Into<[f32; 3]>,
        direction: impl Into<[f32; 3]>,
        color: impl Into<[f32; 3]>,
        intensity: f32,
        falloff_exp: f32,
    ) -> Self {
        let position = position.into();
        let direction = direction.into();
        let color = color.into();
        Self::new_directional_const(position, direction, color, intensity, falloff_exp)
    }

    pub fn new_spotlight(
        position: impl Into<[f32; 3]>,
        direction: impl Into<[f32; 3]>,
        color: impl Into<[f32; 3]>,
        intensity: f32,
        falloff_exp: f32,
        inner_size: f32,
        outer_size: f32,
        max_radius: f32,
    ) -> Self {
        let position = position.into();
        let direction = direction.into();
        let color = color.into();
        Self::new_spotlight_const(
            position,
            direction,
            color,
            intensity,
            falloff_exp,
            inner_size,
            outer_size,
            max_radius,
        )
    }

    /// The approximate spherical bounding volume representing the effective
    /// radius of the light.
    ///
    /// Returns `None` if this is a directional light, as it has no defined
    /// boundaries.
    pub const fn volume(&self) -> Option<LightVolume> {
        if self.params.is_directional() {
            None
        } else {
            Some(LightVolume {
                pos_x: self.pos_x,
                pos_y: self.pos_y,
                pos_z: self.pos_z,
                radius: self.params.max_radius,
            })
        }
    }
}

pub const KIND_FLAG_DIRECTIONAL: f32 = 1.0;
pub const KIND_FLAG_OMNI: f32 = 2.0;
pub const KIND_FLAG_SPOTLIGHT: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LightParams {
    pub max_radius: f32,
    pub spot_inner_size: f32,
    pub spot_outer_size: f32,
    pub kind_flag: f32,
}
impl Default for LightParams {
    fn default() -> Self {
        Self::directional()
    }
}
impl LightParams {
    pub const fn directional() -> Self {
        Self {
            max_radius: 0f32,
            spot_inner_size: 0f32,
            spot_outer_size: 0f32,
            kind_flag: KIND_FLAG_DIRECTIONAL,
        }
    }

    pub const fn omni(max_radius: f32) -> Self {
        Self {
            max_radius,
            spot_inner_size: 0f32,
            spot_outer_size: 0f32,
            kind_flag: KIND_FLAG_OMNI,
        }
    }

    pub const fn spotlight(inner_size: f32, outer_size: f32, max_radius: f32) -> Self {
        Self {
            max_radius,
            spot_inner_size: inner_size,
            spot_outer_size: outer_size,
            kind_flag: KIND_FLAG_SPOTLIGHT,
        }
    }

    pub const fn spotlight_no_umbra(size: f32, max_radius: f32) -> Self {
        Self {
            max_radius,
            spot_inner_size: size,
            spot_outer_size: size,
            kind_flag: KIND_FLAG_SPOTLIGHT,
        }
    }

    pub const fn is_directional(&self) -> bool {
        self.kind_flag == KIND_FLAG_DIRECTIONAL
    }

    pub const fn is_omni(&self) -> bool {
        self.kind_flag == KIND_FLAG_OMNI
    }

    pub const fn is_spotlight(&self) -> bool {
        self.kind_flag == KIND_FLAG_SPOTLIGHT
    }
}
