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

    pub fn get(&self, section: usize) -> View<'_, Light> {
        self.mapped.view_section(section)
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
        self.mapped.view_section_mut(section)
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
    pub fn new<const TILE_W: u32, const TILE_H: u32>(
        max_allocated_resolution: PixelResolution,
    ) -> Self {
        let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(max_allocated_resolution);
        let total_tiles = col_count * row_count;
        Self(TriBuffer::zeroed(total_tiles as usize))
    }

    pub fn new_3d<const TILE_W: u32, const TILE_H: u32>(
        max_allocated_resolution: PixelResolution,
        depth_slices: u32,
    ) -> Self {
        let (col_count, row_count) = screen_div_tiles::<TILE_W, TILE_H>(max_allocated_resolution);
        let total_tiles = col_count * row_count * depth_slices;
        Self(TriBuffer::zeroed(total_tiles as usize))
    }

    pub fn get(&self, section: usize) -> View<'_, Tile> {
        self.0.view_section(section)
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
    pub fn get_mut(&mut self, section: usize) -> ViewMut<'_, Tile> {
        self.0.view_section_mut(section)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct LightIndex(pub u32);

/// Returns the number of columns and rows of tiles to represent the screen
/// `resolution`.
fn screen_div_tiles<const TILE_W: u32, const TILE_H: u32>(
    resolution: PixelResolution,
) -> (u32, u32) {
    (
        resolution.width.div_ceil(TILE_W),
        resolution.height.div_ceil(TILE_H),
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
            section > 0 && section < 3,
            "triple buffer section index must be within 0-2 range"
        );

        let ptr = self.ptr_buffered[section];
        unsafe { std::slice::from_raw_parts(ptr as *const Tile, self.length as usize) }
    }

    pub unsafe fn view_mut(&self, section: usize) -> &mut [Tile] {
        assert!(
            section > 0 && section < 3,
            "triple buffer section index must be within 0-2 range"
        );

        let ptr = self.ptr_buffered[section];
        unsafe { std::slice::from_raw_parts_mut(ptr, self.length as usize) }
    }
}

pub type MappedSquareTiles<const TILE_SIZE: u32> = MappedTiles<TILE_SIZE, TILE_SIZE>;

#[derive(Debug, Clone)]
pub struct MappedTiles<const TILE_W: u32, const TILE_H: u32> {
    resolution: PixelResolution,
    col_count: u32,
    row_count: u32,
    pointer: Option<LightListPtr>,
    buffer_offset: u32, // n of elem.
}
impl<const TILE_W: u32, const TILE_H: u32> MappedTiles<TILE_W, TILE_H> {
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
        let pointer = Some(Self::get_pointer(tile_count, buffer, offset));
        Self {
            resolution,
            col_count,
            row_count,
            pointer,
            buffer_offset: offset,
        }
    }

    fn get_pointer(tile_count: u32, buffer: &TriBuffer<Tile>, offset: u32) -> LightListPtr {
        LightListPtr {
            ptr_buffered: std::array::from_fn(|i| unsafe {
                buffer.raw_section(i).add(offset as usize)
            }),
            length: tile_count,
        }
    }

    pub fn revalidate_pointer(&mut self, buffer: &TriBuffer<Tile>) {
        self.pointer = Some(Self::get_pointer(self.len(), buffer, self.buffer_offset));
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
            self.pointer = None;
        }
    }

    pub unsafe fn tile_at(&self, x: u32, y: u32, section: usize) -> Option<&Tile> {
        assert!(
            section > 0 && section < 3,
            "triple buffer section index must be within 0-2 range"
        );
        if x > self.resolution.width || y > self.resolution.height {
            return None;
        }
        if let Some(ptr) = &self.pointer {
            let col = x / TILE_W;
            let row = y / TILE_H;
            let i = (row * self.row_count) + col;
            // SAFETY: responsability moved to tile_at
            let list = unsafe { ptr.view(section) };
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

    /// Get an immutable view of the underlying triple buffer.
    ///
    /// # Returns
    /// `None` if the pointer is not valid after a resolution change.
    pub unsafe fn view(&self, section: usize) -> Option<&[Tile]> {
        assert!(
            section > 0 && section < 3,
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

pub struct Clusters {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PixelResolution {
    width: u32,
    height: u32,
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
