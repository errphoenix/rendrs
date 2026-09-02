use ethel::render::buffer::SingleBuffer;

pub use dispatch::GeomPass;
pub use shader::{
    SSBO_BINDING_DOMAINS, SSBO_BINDING_GBANK_GCOUNTER, SSBO_BINDING_GBANK_RENDERVERTEX,
    SSBO_BINDING_GBANK_TRIANGLE, SSBO_BINDING_GBANK_TRIANGLE_ATTRIBS, TYPE_DOMAIN_DATA,
    TYPE_TRIANGLE_ATTRIBS,
};

pub mod dispatch;
pub mod rasterize;
pub mod shader;

const DOMAIN_INDEX_BITSHIFT: u32 = 24;
const DOMAIN_GEOID_BITMASK: u32 = u32::MAX >> (32 - DOMAIN_INDEX_BITSHIFT);

pub const DOMAIN_MAX_INDEX: u32 = 0xff;
pub const DOMAIN_MAX_GEOID: u32 = DOMAIN_GEOID_BITMASK;

/// Max amount of domains submitted in a single geometry dispatch.
pub const MAX_DOMAIN_COUNT: u32 = 131_070;
pub const DOMAIN_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderVertex {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub norm_oct_x: f32,
    pub norm_oct_y: f32,
    pub tan_oct_x: f32,
    pub tan_oct_y: f32,
    pub uv_x: f32,
    pub uv_y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DomainData {
    pub idx8_geoid24: u32,
    pub thread_count: u32,
}
impl DomainData {
    pub const fn new(index: u8, geom_id: u32, thread_count: u32) -> Self {
        let geoid24 = geom_id & DOMAIN_GEOID_BITMASK;
        let idx8 = (index as u32) << DOMAIN_INDEX_BITSHIFT;
        Self {
            idx8_geoid24: idx8 | geoid24,
            thread_count,
        }
    }

    pub const fn index(self) -> u8 {
        (self.idx8_geoid24 >> DOMAIN_INDEX_BITSHIFT) as u8
    }

    pub const fn geom_id(self) -> u32 {
        self.idx8_geoid24 & DOMAIN_GEOID_BITMASK
    }
}

/// todo
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TriangleAttribs {
    pub geometry_id: u32,
}

pub type VertexBuffer = SingleBuffer<RenderVertex>;
pub type TriangleBuffer = SingleBuffer<[u32; 3]>;
pub type TriangleAttribsBuffer = SingleBuffer<TriangleAttribs>;

/// Atomic counters buffer.
///
/// Index 0 = vertex counter
///
/// Index 1 = triangle counter
pub type GCounterBuffer = SingleBuffer<[u32; 2]>;

#[derive(Debug, Default)]
pub struct GeometryBank {
    vertex_cap: usize,
    triangle_cap: usize,
    vert: VertexBuffer,
    triangle: TriangleBuffer,
    triangle_attribs: TriangleAttribsBuffer,
    gcounter: GCounterBuffer,
}
impl GeometryBank {
    pub fn new(vertex_cap: usize, triangle_cap: usize) -> Self {
        Self {
            vertex_cap,
            triangle_cap,
            vert: SingleBuffer::zeroed(vertex_cap),
            triangle: SingleBuffer::zeroed(triangle_cap),
            triangle_attribs: SingleBuffer::zeroed(triangle_cap),
            gcounter: SingleBuffer::zeroed(1),
        }
    }

    pub const fn vertex_cap(&self) -> usize {
        self.vertex_cap
    }

    pub const fn triangle_cap(&self) -> usize {
        self.triangle_cap
    }

    pub const fn vertex_buffer(&self) -> &VertexBuffer {
        &self.vert
    }

    pub const fn triangle_buffer(&self) -> &TriangleBuffer {
        &self.triangle
    }

    pub const fn triangle_attribs_buffer(&self) -> &TriangleAttribsBuffer {
        &self.triangle_attribs
    }

    pub const fn gcounter_buffer(&self) -> &GCounterBuffer {
        &self.gcounter
    }

    pub fn bind_data_buffers_to(&self, v_index: u32, tri_index: u32, trimeta_index: u32) {
        self.vert.bind_shader_storage(v_index, 0);
        self.triangle.bind_shader_storage(tri_index, 0);
        self.triangle_attribs.bind_shader_storage(trimeta_index, 0);
    }

    pub fn bind_data_buffers(&self) {
        self.bind_data_buffers_to(
            SSBO_BINDING_GBANK_RENDERVERTEX,
            SSBO_BINDING_GBANK_TRIANGLE,
            SSBO_BINDING_GBANK_TRIANGLE_ATTRIBS,
        );
    }

    pub fn bind_gcounter_buffer_to(&self, index: u32) {
        self.gcounter.bind_shader_storage(index, 0);
    }

    pub fn bind_gcounter_buffer(&self) {
        self.bind_gcounter_buffer_to(SSBO_BINDING_GBANK_GCOUNTER);
    }
}
