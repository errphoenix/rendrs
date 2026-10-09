use ethel::render::buffer::SingleBuffer;

pub use dispatch::GeomPass;
use janus::GpuResource;
pub use rasterize::{
    AttribInterpolationPass, GeomRasterizePass, geom_attribs_framespace_target,
    geom_attribs_gradients_target, geom_rasterize_target,
};
pub use shader::{
    SSBO_BINDING_DOMAINS, SSBO_BINDING_GBANK_GCOUNTER, SSBO_BINDING_GBANK_INSTANCE_DATA,
    SSBO_BINDING_GBANK_TRIANGLE, SSBO_BINDING_GBANK_VERTEX, SSBO_DOMAINS, SSBO_GBANK_GCOUNTER,
    SSBO_GBANK_INSTANCING_DATA, TYPE_DOMAIN_DATA, TYPE_INSTANCE_TRANSFORM, TYPE_INSTANCELIST,
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

pub const MAX_INSTANCELIST_COUNT: u32 = 1024;
pub const MAX_INSTANCES_GLOBAL_COUNT: u32 = 65535;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TriangleAttribs {
    pub geometry_id: u32,
    pub material_id: u16,
    pub extra: u16,
}

pub trait HasVertexBuffers: GpuResource + std::fmt::Debug + 'static {
    const CAP: usize;
    fn new() -> Self;
    fn bind_positions(&self, index: u32);
    fn bind_normals(&self, index: u32);
    fn bind_uvs(&self, index: u32);
    fn bind_arrays(&self, index: u32);
}
pub trait HasTriangleBuffers: GpuResource + std::fmt::Debug + 'static {
    const CAP: usize;
    fn new() -> Self;
    fn bind_indices(&self, index: u32);
    fn bind_attribs(&self, index: u32);
    fn bind_arrays(&self, index: u32);
}

#[macro_export]
macro_rules! geometry_buffers_impls {
    (
        $vb:ty;
        $tb:ty;
        $valloc:expr;
        $talloc:expr;
    ) => {
        pub type VertexBuffers = $vb;
        pub type TriangleBuffers = $tb;
        impl VertexBuffers {
            pub const SSBO_ARRAYS: ethel::shader::GlslStorage =
                ethel::shader::GlslStorage::new(concat!(
                    "layout(std430, binding = ",
                    0, // must match value defined in shader module
                    ") buffer Rendrs_GBANK_VertexBuffers\n{\n",
                    "    float rendrs_gbank_vertex_positions[",
                    $valloc,
                    "][3];\n",
                    "    float rendrs_gbank_vertex_normals[",
                    $valloc,
                    "][2];\n",
                    "    float rendrs_gbank_vertex_uvs[",
                    $valloc,
                    "][2];\n",
                    "};\n"
                ));
        }
        impl TriangleBuffers {
            pub const SSBO_ARRAYS: ethel::shader::GlslStorage =
                ethel::shader::GlslStorage::new(concat!(
                    "layout(std430, binding = ",
                    1, // must match value defined in shader module
                    ") buffer Rendrs_GBANK_TriangleBuffers\n{\n",
                    "    uint rendrs_gbank_triangle_indices[",
                    $talloc,
                    "][3];\n",
                    "    TriangleAttribs rendrs_gbank_triangle_attribs[",
                    $talloc,
                    "];\n",
                    "};\n"
                ));
        }
        impl $crate::geometry::HasVertexBuffers for VertexBuffers {
            const CAP: usize = $valloc;
            fn new() -> Self {
                <$vb>::new()
            }
            fn bind_positions(&self, index: u32) {
                self.bind_ssbo_positions(Some(index));
            }
            fn bind_normals(&self, index: u32) {
                self.bind_ssbo_normals(Some(index));
            }
            fn bind_uvs(&self, index: u32) {
                self.bind_ssbo_uvs(Some(index));
            }
            fn bind_arrays(&self, index: u32) {
                self.bind_ssbo_arrays(Some(index));
            }
        }
        impl $crate::geometry::HasTriangleBuffers for TriangleBuffers {
            const CAP: usize = $talloc;
            fn new() -> Self {
                <$tb>::new()
            }
            fn bind_indices(&self, index: u32) {
                self.bind_ssbo_indices(Some(index));
            }
            fn bind_attribs(&self, index: u32) {
                self.bind_ssbo_attribs(Some(index));
            }
            fn bind_arrays(&self, index: u32) {
                self.bind_ssbo_arrays(Some(index));
            }
        }
    };
}

#[macro_export]
macro_rules! geometry_buffers {
    (
        vertices  = $valloc:expr;
        triangles = $talloc:expr;
    ) => {
        ethel::typed_part_buffer! {
            const Vertex : 3, {
                enum Positions: $valloc => {
                    type [f32; 3];
                    bind 0;
                };
                enum Normals: $valloc => {
                    type [f32; 2];
                    bind 1;
                };
                enum Uvs: $valloc => {
                    type [f32; 2];
                    bind 2;
                };
            }
        }
        ethel::typed_part_buffer! {
            const Triangle : 2, {
                enum Indices: $talloc => {
                    type [u32; 3];
                    bind 0;
                };
                enum Attribs: $talloc => {
                    type $crate::geometry::TriangleAttribs;
                    bind 1;
                };
            }
        }

        pub const GEOM_ALLOC_VERTEX: usize = $valloc;
        pub const GEOM_ALLOC_TRIANGLE: usize = $talloc;

        pub type GeomRasterizePass =
            $crate::geometry::GeomRasterizePass<VertexBuffers, TriangleBuffers>;
        pub type AttribInterpolationPass =
            $crate::geometry::AttribInterpolationPass<VertexBuffers, TriangleBuffers>;
        pub type GeometryBank = $crate::geometry::GeometryBank<VertexBuffers, TriangleBuffers>;

        $crate::geometry_buffers_impls! {
            VertexPartitionedBuffer;
            TrianglePartitionedBuffer;
            $valloc; $talloc;
        }
    };
}

const GCOUNTER_COUNT: usize = 4;

/// Atomic counters buffer.
///
/// Index 0 = vertex counter
///
/// Index 1 = triangle counter
///
/// Index 2 = instance lists counter
///
/// Index 3 = instances counter
pub type GCounterBuffer = SingleBuffer<[u32; GCOUNTER_COUNT]>;

/// Atomic counters buffer.
///
/// Index 0 = vertex counter
///
/// Index 1 = triangle counter
///
/// Index 2 = instance lists counter
///
/// Index 3 = instances counter
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeoCounters(u32, u32, u32, u32);
impl GeoCounters {
    pub const fn vertices(self) -> u32 {
        self.0
    }

    pub const fn triangles(self) -> u32 {
        self.1
    }

    pub const fn instance_lists(self) -> u32 {
        self.2
    }

    pub const fn instances(self) -> u32 {
        self.3
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct InstanceList {
    pub geometry_id: u32,

    pub tri_base: u32,
    pub tri_count: u32,

    /// packed as uint on gpu
    pub instance_base_count: (u16, u16),
}

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
// pub struct InstanceAttribs {
//     //todo
// }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct InstanceTransform {
    pub px: f32,
    pub py: f32,
    pub pz: f32,

    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub qw: f32,

    pub s: f32,
}

ethel::typed_part_buffer! {
    const InstancingData : 2, {
        enum Lists: MAX_INSTANCELIST_COUNT as usize => {
            type InstanceList;
            bind 0;
        };
        enum Transforms: MAX_INSTANCES_GLOBAL_COUNT as usize => {
            type InstanceTransform;
            bind 1;
        };
    }
}

pub type InstancingBuffers = InstancingDataPartitionedBuffer;
impl InstancingBuffers {
    pub fn bind_lists(&self, index: u32) {
        self.bind_ssbo_lists(Some(index));
    }

    pub fn bind_transforms(&self, index: u32) {
        self.bind_ssbo_transforms(Some(index));
    }

    pub fn bind_arrays(&self, index: u32) {
        self.bind_ssbo_arrays(Some(index));
    }
}

#[derive(Debug, Default)]
pub struct GeometryBank<V: HasVertexBuffers, T: HasTriangleBuffers> {
    vertex: V,
    triangle: T,
    gcounter: GCounterBuffer,
    instancing: InstancingBuffers,
}
impl<V: HasVertexBuffers, T: HasTriangleBuffers> GeometryBank<V, T> {
    pub fn new() -> Self {
        Self {
            vertex: V::new(),
            triangle: T::new(),
            gcounter: SingleBuffer::zeroed(1),
            instancing: InstancingBuffers::new(),
        }
    }

    pub const fn vertex_cap() -> usize {
        V::CAP
    }

    pub const fn triangle_cap() -> usize {
        T::CAP
    }

    pub const fn instance_lists_cap() -> usize {
        MAX_INSTANCELIST_COUNT as usize
    }

    pub const fn instances_cap() -> usize {
        MAX_INSTANCES_GLOBAL_COUNT as usize
    }

    pub const fn vertex_buffers(&self) -> &V {
        &self.vertex
    }

    pub const fn triangle_buffers(&self) -> &T {
        &self.triangle
    }

    pub const fn gcounter_buffer(&self) -> &GCounterBuffer {
        &self.gcounter
    }

    pub const fn instancing_buffers(&self) -> &InstancingBuffers {
        &self.instancing
    }

    /// Atomic counters buffer.
    ///
    /// Index 0 = vertex counter
    ///
    /// Index 1 = triangle counter
    ///
    /// Index 2 = instance lists counter
    ///
    /// Index 3 = instances counter
    pub fn get_gcounters(&self) -> GeoCounters {
        let gcounter_buf = self.gcounter.resource_id();
        let mut dst = GeoCounters::default();
        let dst_ptr = (&raw mut dst).cast();
        unsafe {
            janus::gl::GetNamedBufferSubData(
                gcounter_buf,
                0,
                (size_of::<u32>() * GCOUNTER_COUNT) as isize,
                dst_ptr,
            );
        }
        dst
    }

    /// Use the triangle buffers' internal indices buffer ID to use as EBO.
    pub fn bind_index_buffer(&self) {
        // DrawElements is dispatched with offset 0 (which matches the
        // buffer's layout) and length is guaranted to be lesser than its
        // capacity, as geometry is discarded beyond that range.
        unsafe {
            janus::gl::BindBuffer(janus::gl::ELEMENT_ARRAY_BUFFER, self.triangle.resource_id());
        }
    }

    /// Binds the vertex buffers to a single bind point as arrays.
    ///
    /// Note that the ssbo block layout and array lengths must match the
    /// vertex buffers' internal layout.
    pub fn bind_vertex_buffers_to(&self, index: u32) {
        self.vertex.bind_arrays(index);
    }

    /// Binds the triangle buffers to a single bind point as arrays.
    ///
    /// Note that the ssbo block layout and array lengths must match the
    /// triangle buffers' internal layout.
    pub fn bind_triangle_buffers_to(&self, index: u32) {
        self.triangle.bind_arrays(index);
    }

    /// Binds the vertex buffers to a single bind point as arrays.
    ///
    /// Note that the ssbo block layout and array lengths must match the
    /// vertex buffers' internal layout.
    pub fn bind_vertex_buffers(&self) {
        self.vertex.bind_arrays(SSBO_BINDING_GBANK_VERTEX);
    }

    /// Binds the triangle buffers to a single bind point as arrays.
    ///
    /// Note that the ssbo block layout and array lengths must match the
    /// triangle buffers' internal layout.
    pub fn bind_triangle_buffers(&self) {
        self.triangle.bind_arrays(SSBO_BINDING_GBANK_TRIANGLE);
    }

    pub fn bind_gcounter_buffer_to(&self, index: u32) {
        self.gcounter.bind_shader_storage(index, 0);
    }

    pub fn bind_gcounter_buffer(&self) {
        self.bind_gcounter_buffer_to(SSBO_BINDING_GBANK_GCOUNTER);
    }

    pub fn bind_instancing_buffers_to(&self, index: u32) {
        self.instancing.bind_arrays(index);
    }

    pub fn bind_instancing_buffers(&self) {
        self.instancing
            .bind_arrays(SSBO_BINDING_GBANK_INSTANCE_DATA);
    }
}
