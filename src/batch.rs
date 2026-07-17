use janus::texture::{TextureKey, TextureTarget};

pub const PER_BATCH_UNITS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct BatchUnitIndex(usize);

/// Represents a single draw-call, on 16 concurrent texture units.
///
/// Textures are distributed among `N` amount of texture units, each command
/// samples from a different unit in order to minimize the total number of draw
/// calls.
///
/// This allows up to `N` draw-calls to be submitted concurrently.
#[derive(Debug, Default, Clone)]
pub struct Batch<T: Clone + Copy> {
    array: Vec<T>,
    units: [Option<TextureKey>; PER_BATCH_UNITS],
    head: usize,
}
impl<T: Clone + Copy> Batch<T> {
    pub fn new() -> Self {
        Self {
            array: Vec::new(),
            units: [None; PER_BATCH_UNITS],
            head: 0,
        }
    }

    /// Binds all units' textures as 2D textures.
    pub fn bind_textures(&self) {
        for (i, &texture) in self.units.iter().enumerate() {
            let texture = texture.unwrap_or_default();
            let unit = i as u32;
            janus::texture::bind_without_meta(TextureTarget::Flat, texture, unit);
        }
    }

    /// Returns `true` if the batch is exhausted.
    ///
    /// I.e. if the total amount of texture groups has reached the defined
    /// [`Self::UNITS`] maximum.
    pub fn is_exhausted(&self) -> bool {
        self.head >= PER_BATCH_UNITS
    }

    /// Attempt to insert an `element` attached to the unit bound to `texture`.
    ///
    /// Will return `true` if the operation is successful, otherwise `false`.
    ///
    /// If the operation is not successful, you should attempt to insert
    /// the element in another batch.
    pub fn insert(&mut self, element: T, texture: TextureKey) -> bool {
        if self.fetch_location_or_create(texture).is_some() {
            self.array.push(element);
            true
        } else {
            false
        }
    }

    /// Look up location associated to `texture`, or attach to a new unit if
    /// available.
    ///
    /// If one is not available, the function returns `None`.
    pub fn fetch_location_or_create(&mut self, texture: TextureKey) -> Option<BatchUnitIndex> {
        let existing = self.fetch_location(texture);
        if let Some(existing) = existing {
            Some(existing)
        } else if !self.is_exhausted() {
            let location = BatchUnitIndex(self.units.len());
            self.units[self.head] = Some(texture);
            Some(location)
        } else {
            None
        }
    }

    /// Look up location associated to the given `texture`.
    pub fn fetch_location(&self, texture: TextureKey) -> Option<BatchUnitIndex> {
        self.units
            .iter()
            .position(|key| *key == Some(texture))
            .map(BatchUnitIndex)
    }

    pub fn data(&self) -> &[T] {
        &self.array
    }

    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.array
    }

    pub fn texture(&self, index: usize) -> Option<TextureKey> {
        self.units[index]
    }

    pub fn textures(&self) -> [Option<TextureKey>; PER_BATCH_UNITS] {
        self.units
    }

    pub fn clear(&mut self) {
        self.units.iter_mut().for_each(|opt| *opt = None);
        self.array.clear();
        self.head = 0;
    }
}
