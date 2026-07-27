pub mod light;

use ethel::render::buffer::{TriBuffer, View, ViewMut};
#[allow(unused_imports)]
pub use light::{Light, LightParams};

#[derive(Debug, Default)]
pub struct LightsBuffer {
    mapped: TriBuffer<Light>,
}
impl LightsBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            mapped: TriBuffer::zeroed(capacity),
        }
    }

    pub unsafe fn get(&self, section: usize) -> View<'_, Light> {
        unsafe { self.mapped.view_section(section) }
    }

    /// Get mutable access to a `section` of the underlying triple buffer.
    ///
    /// # Panics
    /// Will panic if section is not within the 0-2 range.
    ///
    /// # Safety
    /// The caller must manually ensure the section of this triple buffer
    /// is currently not being read or modified somewhere else.
    ///
    /// This is meant to be used in accordance to the triple buffer manager
    /// [`ethel::data::cross::Cross`] to ensure safety.
    pub unsafe fn get_mut(&self, section: usize) -> ViewMut<'_, Light> {
        unsafe { self.mapped.view_section_mut(section) }
    }
}

#[derive(Debug)]
pub struct TilesBuffer(TriBuffer<Tile>);
impl TilesBuffer {
    /// Allocate persistent coherent mapped triple buffered gpu memory of
    /// [`tiles`](Tile);
    ///
    /// The `max_allocated_resolution` will be the **maximum** screen
    /// resolution representable.
    pub fn new<const TILE_W: usize, const TILE_H: usize>(
        max_allocated_resolution: PixelResolution,
    ) -> Self {
        let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(max_allocated_resolution);
        let total_tiles = col_count * row_count;
        Self(TriBuffer::zeroed(total_tiles as usize))
    }

    pub fn new_3d<const TILE_W: usize, const TILE_H: usize>(
        max_allocated_resolution: PixelResolution,
        depth_slices: u32,
    ) -> Self {
        let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(max_allocated_resolution);
        let total_tiles = col_count * row_count * depth_slices;
        Self(TriBuffer::zeroed(total_tiles as usize))
    }

    pub unsafe fn get(&self, section: usize) -> View<'_, Tile> {
        unsafe { self.0.view_section(section) }
    }

    /// Get mutable access to a `section` of the underlying triple buffer.
    ///
    /// # Panics
    /// Will panic if section is not within the 0-2 range.
    ///
    /// # Safety
    /// The caller must manually ensure the section of this triple buffer
    /// is currently not being read or modified somewhere else.
    ///
    /// This is meant to be used in accordance to the triple buffer manager
    /// [`ethel::data::cross::Cross`] to ensure safety.
    pub unsafe fn get_mut(&mut self, section: usize) -> ViewMut<'_, Tile> {
        unsafe { self.0.view_section_mut(section) }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterSlice<const W: usize, const H: usize> {
    depth_level: f32,
    inner: TilesSlice<W, H>,
}
impl<const W: usize, const H: usize> ClusterSlice<W, H> {
    /// Get an immutable view of the underlying triple buffer.
    ///
    /// # Returns
    /// `None` if the pointer is not valid after a resolution change.
    pub unsafe fn view(&self, section: usize) -> Option<&[Tile]> {
        // SAFETY: responsability moved to view
        unsafe { self.inner.view(section) }
    }

    pub fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    pub fn depth_level(&self) -> f32 {
        self.depth_level
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ClusterDepthCurve {
    #[default]
    Linear,
    Exponential,
    Quadratic,
    Cubic,
    Logarithmic,
    Custom(fn(f32) -> f32),
}
impl ClusterDepthCurve {
    fn evaluate_inner(&self, x: f32) -> f32 {
        match self {
            ClusterDepthCurve::Linear => x,
            ClusterDepthCurve::Exponential => x.exp(),
            ClusterDepthCurve::Quadratic => x * x,
            ClusterDepthCurve::Cubic => x * x * x,
            ClusterDepthCurve::Logarithmic => (x + 1f32).ln(),
            ClusterDepthCurve::Custom(func) => func(x),
        }
    }

    pub fn evaluate<const D: usize>(&self, length: f32) -> [f32; D] {
        let section = length / D as f32;
        std::array::from_fn(|i| {
            let param = i as f32 * section;
            self.evaluate_inner(param)
        })
    }

    pub fn slice_at_z<const D: usize>(&self, z: f32, length: f32) -> usize {
        let w = self.evaluate_inner(z);
        let t = w / length;
        (D as f32 * t).floor() as usize
    }
}

#[derive(Debug, Clone)]
pub struct Clusters<const W: usize, const H: usize, const D: usize> {
    resolution: PixelResolution,
    per_slice_col_count: u32,
    per_slice_row_count: u32,
    frustum_depth: f32,
    depth_curve: ClusterDepthCurve,
    slices: [ClusterSlice<W, H>; D],
}
impl<const W: usize, const H: usize, const D: usize> Clusters<W, H, D> {
    pub fn new(
        resolution: ethel::render::Resolution,
        buffer: &TriBuffer<Tile>,
        depth_curve: ClusterDepthCurve,
        frustum_depth: f32,
    ) -> Self {
        let resolution = PixelResolution::from(resolution);
        let (col_count, row_count) = screen_div_tiles::<W, H>(resolution);
        let per_slice_tile_count = col_count * row_count;

        let depths = depth_curve.evaluate::<D>(frustum_depth);
        let mut slices = [ClusterSlice::default(); D];
        slices
            .iter_mut()
            .enumerate()
            .zip(depths)
            .for_each(|((i, slice), depth)| {
                *slice = ClusterSlice {
                    depth_level: depth,
                    inner: TilesSlice {
                        pointer: None,
                        buffer_offset: per_slice_tile_count * i as u32,
                    },
                };
            });

        slices
            .iter_mut()
            .for_each(|slice| slice.inner.validate(per_slice_tile_count, buffer));

        Self {
            resolution,
            per_slice_col_count: col_count,
            per_slice_row_count: row_count,
            frustum_depth,
            depth_curve,
            slices,
        }
    }

    pub fn revalidate_pointers(&mut self, buffer: &TriBuffer<Tile>) {
        let pstc = self.per_slice_tile_count();
        self.slices
            .iter_mut()
            .for_each(|slice| slice.inner.validate(pstc, buffer));
    }

    /// Recompute the tile sizes on resolution change.
    ///
    /// This will invalidate the underlying pointer to the triple buffer,
    /// requiring a [`Self::revalidate_pointer`] call.
    pub fn revalidate_resolution(&mut self, resolution: ethel::render::Resolution) {
        if resolution.is_changed() {
            self.resolution = PixelResolution::from(resolution);
            let (col_count, row_count) = screen_div_tiles::<W, H>(self.resolution);
            if col_count == self.per_slice_col_count && row_count == self.per_slice_row_count {
                return;
            }
            self.slices.iter_mut().for_each(ClusterSlice::invalidate);
        }
    }

    pub unsafe fn cluster_at(&self, x: u32, y: u32, z: f32, section: usize) -> Option<&Tile> {
        if z < 0.0 || z > self.frustum_depth {
            return None;
        }
        let slice_index = self.depth_curve.slice_at_z::<D>(z, self.frustum_depth);
        // SAFETY: responsability moved to cluster_at
        unsafe { self.cluster_of_slice_at(x, y, slice_index, section) }
    }

    pub unsafe fn cluster_of_slice_at(
        &self,
        x: u32,
        y: u32,
        depth_slice: usize,
        section: usize,
    ) -> Option<&Tile> {
        assert!(
            section < 3,
            "triple buffer section index must be within 0-2 range"
        );
        assert!(
            depth_slice < D,
            "depth slice index must be within defined bounds"
        );

        if x > self.resolution.width || y > self.resolution.height {
            return None;
        }
        // SAFETY: responsability moved to cluster_of_slice_at
        if let Some(list) = unsafe { self.view(section, depth_slice) } {
            let col = x / W as u32;
            let row = y / H as u32;
            let i = (row * self.per_slice_row_count) + col;
            list.get(i as usize)
        } else {
            None
        }
    }

    pub const fn current_resolution(&self) -> PixelResolution {
        self.resolution
    }

    pub const fn per_slice_column_count(&self) -> u32 {
        self.per_slice_col_count
    }

    pub const fn per_slice_row_count(&self) -> u32 {
        self.per_slice_row_count
    }

    pub const fn per_slice_tile_count(&self) -> u32 {
        self.per_slice_col_count * self.per_slice_row_count
    }

    /// Get an immutable view of the referenced global triple buffer.
    ///
    /// # Returns
    /// `None` if the pointer is not valid after a resolution change.
    pub unsafe fn view(&self, section: usize, depth_slice: usize) -> Option<&[Tile]> {
        assert!(
            depth_slice < D,
            "depth slice index must be within defined bounds"
        );
        // SAFETY: responsability moved to view
        unsafe { self.slices[depth_slice].view(section) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct LightIndex(pub u32);

/// Returns the number of columns and rows of tiles to represent the screen
/// `resolution`.
fn screen_div_tiles<const TILE_W: usize, const TILE_H: usize>(
    resolution: PixelResolution,
) -> (u32, u32) {
    (
        resolution.width.div_ceil(TILE_W as u32),
        resolution.height.div_ceil(TILE_H as u32),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct LightListPtr {
    ptr_buffered: [*mut Tile; 3],
    length: u32,
}
impl LightListPtr {
    pub unsafe fn view(&self, section: usize) -> &[Tile] {
        assert!(
            section < 3,
            "triple buffer section index must be within 0-2 range"
        );

        let ptr = self.ptr_buffered[section];
        unsafe { std::slice::from_raw_parts(ptr as *const Tile, self.length as usize) }
    }

    pub unsafe fn view_mut(&self, section: usize) -> &mut [Tile] {
        assert!(
            section < 3,
            "triple buffer section index must be within 0-2 range"
        );

        let ptr = self.ptr_buffered[section];
        unsafe { std::slice::from_raw_parts_mut(ptr, self.length as usize) }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TilesSlice<const TILE_W: usize, const TILE_H: usize> {
    pointer: Option<LightListPtr>,
    buffer_offset: u32, // n of elem.
}
impl<const TILE_W: usize, const TILE_H: usize> TilesSlice<TILE_W, TILE_H> {
    fn validate(&mut self, tile_count: u32, buffer: &TriBuffer<Tile>) {
        self.pointer = Some(LightListPtr {
            ptr_buffered: std::array::from_fn(|i| unsafe {
                buffer.raw_section(i).add(self.buffer_offset as usize)
            }),
            length: tile_count,
        });
    }

    fn invalidate(&mut self) {
        self.pointer = None;
    }

    unsafe fn view(&self, section: usize) -> Option<&[Tile]> {
        assert!(
            section < 3,
            "triple buffer section index must be within 0-2 range"
        );
        if let Some(ptr) = &self.pointer {
            // SAFETY: responsability moved to view
            Some(unsafe { ptr.view(section) })
        } else {
            None
        }
    }
}

pub type MappedSquareTiles<const TILE_SIZE: usize> = MappedTiles<TILE_SIZE, TILE_SIZE>;

#[derive(Debug, Clone, Default)]
pub struct MappedTiles<const TILE_W: usize, const TILE_H: usize> {
    resolution: PixelResolution,
    col_count: u32,
    row_count: u32,
    slice: TilesSlice<TILE_W, TILE_H>,
}
impl<const TILE_W: usize, const TILE_H: usize> MappedTiles<TILE_W, TILE_H> {
    /// Create a new vertical slice of tiles for screen partitioning.
    ///
    /// Requires the global `buffer` owning all screen [`tiles`](Tile) and a
    /// `offset` into the buffer.
    ///
    /// Computes the size of its view into the `buffer` depending on the
    /// current screen `resolution`.
    ///
    /// The `offset` is intended to be used for clustered screen partitioning,
    /// as it represents the number of all 'previous' tiles that are part of
    /// the preceding depth slices, as managed by [`Clusters`].
    pub fn new(
        resolution: ethel::render::Resolution,
        buffer: &TriBuffer<Tile>,
        offset: u32,
    ) -> Self {
        let resolution = PixelResolution::from(resolution);
        let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(resolution);
        let tile_count = col_count * row_count;

        let mut slice = TilesSlice {
            pointer: None,
            buffer_offset: offset,
        };
        slice.validate(tile_count, buffer);

        Self {
            resolution,
            col_count,
            row_count,
            slice,
        }
    }

    pub fn revalidate_pointer(&mut self, buffer: &TriBuffer<Tile>) {
        self.slice.validate(self.len(), buffer);
    }

    /// Recompute the tile sizes on resolution change.
    ///
    /// This will invalidate the underlying pointer to the triple buffer,
    /// requiring a [`Self::revalidate_pointer`] call.
    pub fn revalidate_resolution(&mut self, resolution: ethel::render::Resolution) {
        if resolution.is_changed() {
            self.resolution = PixelResolution::from(resolution);
            let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(self.resolution);
            if col_count == self.col_count && row_count == self.row_count {
                return;
            }
            self.slice.invalidate();
        }
    }

    pub unsafe fn tile_at(&self, x: u32, y: u32, section: usize) -> Option<&Tile> {
        assert!(
            section < 3,
            "triple buffer section index must be within 0-2 range"
        );
        if x > self.resolution.width || y > self.resolution.height {
            return None;
        }
        // SAFETY: responsability moved to tile_at
        if let Some(list) = unsafe { self.slice.view(section) } {
            let col = x / TILE_W as u32;
            let row = y / TILE_H as u32;
            let i = (row * self.row_count) + col;
            list.get(i as usize)
        } else {
            None
        }
    }

    pub const fn current_resolution(&self) -> PixelResolution {
        self.resolution
    }

    pub const fn column_count(&self) -> u32 {
        self.col_count
    }

    pub const fn row_count(&self) -> u32 {
        self.row_count
    }

    pub const fn len(&self) -> u32 {
        self.col_count * self.row_count
    }

    /// Get an immutable view of the referenced global triple buffer.
    ///
    /// # Returns
    /// `None` if the pointer is not valid after a resolution change.
    pub unsafe fn view(&self, section: usize) -> Option<&[Tile]> {
        // SAFETY: responsability moved to view
        unsafe { self.slice.view(section) }
    }
}

pub const PER_TILE_MAX_LIGHTS: usize = 128;

// GPU-side representation of a tile.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tile(pub [LightIndex; PER_TILE_MAX_LIGHTS]);
impl Default for Tile {
    fn default() -> Self {
        Self(std::array::from_fn(|_| LightIndex::default()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PixelResolution {
    width: u32,
    height: u32,
}
impl Default for PixelResolution {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
        }
    }
}
impl From<ethel::render::Resolution> for PixelResolution {
    fn from(value: ethel::render::Resolution) -> Self {
        Self {
            width: value.width as u32,
            height: value.height as u32,
        }
    }
}
impl PixelResolution {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn preset_qhd_540p() -> Self {
        Self {
            width: 960,
            height: 540,
        }
    }

    pub const fn preset_hd_720p() -> Self {
        Self {
            width: 1280,
            height: 720,
        }
    }

    pub const fn preset_fhd_1080p() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }

    pub const fn preset_2k() -> Self {
        Self {
            width: 2048,
            height: 1080,
        }
    }

    pub const fn preset_uhd_2160p() -> Self {
        Self {
            width: 3840,
            height: 2160,
        }
    }

    pub const fn preset_4k() -> Self {
        Self {
            width: 4096,
            height: 2160,
        }
    }

    pub const fn preset_5k() -> Self {
        Self {
            width: 5120,
            height: 2880,
        }
    }

    pub const fn preset_uhd_8k() -> Self {
        Self {
            width: 7680,
            height: 4320,
        }
    }

    pub fn ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn width(&self) -> u32 {
        self.width
    }
}
