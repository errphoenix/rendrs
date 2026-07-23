pub mod shader;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Oct3x2Float32 {
    pub x: f32,
    pub y: f32,
}
impl Oct3x2Float32 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        let (x, y) = octahedron_encode(x, y, z);
        Self { x, y }
    }

    pub const fn from_array(components: [f32; 3]) -> Self {
        Self::new(components[0], components[1], components[2])
    }

    #[cfg(feature = "glam")]
    pub const fn from_glam(vector: glam::Vec3) -> Self {
        Self::new(vector.x, vector.y, vector.z)
    }

    pub fn decode(self) -> (f32, f32, f32) {
        octahedron_decode(self.x, self.y)
    }

    pub fn decode_array(self) -> [f32; 3] {
        let (x, y, z) = self.decode();
        [x, y, z]
    }

    #[cfg(feature = "glam")]
    pub fn decode_glam(self) -> glam::Vec3 {
        let (x, y, z) = self.decode();
        glam::vec3(x, y, z)
    }
}

const fn octahedron_wrap(x: f32, y: f32) -> (f32, f32) {
    (
        (1.0 - x.abs()) * x.copysign(1f32),
        (1.0 - y.abs()) * y.copysign(1f32),
    )
}

const fn octahedron_encode(x: f32, y: f32, z: f32) -> (f32, f32) {
    let s = x.abs() + y.abs() + z.abs();
    let x = x / s;
    let y = y / s;
    let z = z / s;
    let (x, y) = if z >= 0.0 {
        (x, y)
    } else {
        octahedron_wrap(x, y)
    };
    (x * 0.5 + 0.5, y * 0.5 + 0.5)
}

fn octahedron_decode(x: f32, y: f32) -> (f32, f32, f32) {
    let mut x = x * 2.0 - 1.0;
    let mut y = y * 2.0 - 1.0;
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).clamp(0.0, 1.0);
    let u = if x >= 0.0 { -t } else { t };
    let v = if y >= 0.0 { -t } else { t };
    x += u;
    y += v;
    let l = (x * x + y * y + z * z).sqrt();
    (x / l, y / l, z / l)
}
