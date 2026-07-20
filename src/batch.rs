use janus::texture::{Tex, TextureView};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct BatchGroupIndex(usize);

/// A dynamic persistent collection of [`batches`](Batch).
#[derive(Debug, Clone, Default)]
pub struct BatchManager<T: Clone + Copy> {
    batches: Vec<Batch<T>>,
}
impl<T: Clone + Copy> BatchManager<T> {
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            batches: Vec::with_capacity(capacity),
        }
    }

    /// Insert a new `element` attached to a `texture`.
    ///
    /// This function will attempt to find an existing [`Batch`] to append the
    /// `element` to, returning its [`batch group index`](BatchGroupIndx) and
    /// its [`unit texture index`](BatchUnitIndex) in that batch.
    ///
    /// If one is not found, a new [`Batch`] is created and appended at the end
    /// of the batches list.
    /// This might trigger an allocation if the batches list needs to expand
    /// its capacity.
    ///
    /// This operation is O(n) depending on the number of batches in the list:
    /// usually the number of batches is not very high, especially if you are
    /// making use of texture atlases, so the overhead is negligible.
    pub fn insert(
        &mut self,
        element: T,
        texture: TextureView,
    ) -> (BatchGroupIndex, BatchUnitIndex) {
        for (i, batch) in self
            .batches
            .iter_mut()
            .enumerate()
            .filter(|(_, batch)| !batch.is_exhausted())
        {
            if let Some(tui) = batch.insert(element, texture) {
                return (BatchGroupIndex(i), tui);
            }
        }

        const BASE_CAPACITY: usize = 1024;
        let new_batch_index = self.len();
        let (new_batch, tui) = {
            let mut batch = Batch::with_array_capacity(BASE_CAPACITY);
            let tui = batch
                .insert(element, texture)
                .expect("new batch must have attachment texture group available");
            (batch, tui)
        };
        self.batches.push(new_batch);

        (BatchGroupIndex(new_batch_index), tui)
    }

    pub fn batches(&self) -> &[Batch<T>] {
        &self.batches
    }

    pub fn batches_mut(&mut self) -> &mut [Batch<T>] {
        &mut self.batches
    }

    pub fn get(&self, index: usize) -> Option<&Batch<T>> {
        self.batches.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Batch<T>> {
        self.batches.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    /// Clear all batching data.
    ///
    /// Does not de-allocate any memory, allowing the memory to be reused and
    /// avoid allocations down the line.
    pub fn clear(&mut self) {
        self.batches.iter_mut().for_each(Batch::clear);
    }
}

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
pub struct Batch<T> {
    array: Vec<T>,
    units: [Option<TextureView>; PER_BATCH_UNITS],
    head: usize,
}
impl<T: Clone + Copy> Batch<T> {
    pub const UNITS: usize = PER_BATCH_UNITS;

    pub fn new() -> Self {
        Self {
            array: Vec::new(),
            units: [None; PER_BATCH_UNITS],
            head: 0,
        }
    }

    pub fn with_array_capacity(capacity: usize) -> Self {
        Self {
            array: Vec::with_capacity(capacity),
            units: [None; PER_BATCH_UNITS],
            head: 0,
        }
    }

    /// Binds all units' textures as 2D textures.
    pub fn bind_unit_textures(&self) {
        for (i, &texture) in self.units.iter().enumerate() {
            if let Some(texture) = texture {
                let unit = i as u32;
                texture.bind(unit);
            }
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
    /// Will return `None` if the operation was not successful, otherwise the
    /// [`unit texture index`](BatchUnitIndex) is returned.
    ///
    /// If the operation is not successful, you should attempt to insert
    /// the element in another batch.
    pub fn insert(&mut self, element: T, texture: TextureView) -> Option<BatchUnitIndex> {
        self.fetch_location_or_create(texture).and_then(|bui| {
            self.array.push(element);
            Some(bui)
        })
    }

    /// Look up location associated to `texture`, or attach to a new unit if
    /// available.
    ///
    /// If one is not available, the function returns `None`.
    pub fn fetch_location_or_create(&mut self, texture: TextureView) -> Option<BatchUnitIndex> {
        let existing = self.fetch_location(texture);
        if let Some(existing) = existing {
            // exists
            Some(existing)
        } else if !self.is_exhausted() {
            // create new and advance head
            let location = BatchUnitIndex(self.head);
            self.units[self.head] = Some(texture);
            self.head += 1;
            Some(location)
        } else {
            // batch exhausted and no matching tex group
            None
        }
    }

    /// Look up location associated to the given `texture`.
    pub fn fetch_location(&self, texture: TextureView) -> Option<BatchUnitIndex> {
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

    pub fn texture(&self, index: usize) -> Option<TextureView> {
        self.units[index]
    }

    pub fn textures(&self) -> [Option<TextureView>; PER_BATCH_UNITS] {
        self.units
    }

    pub fn clear(&mut self) {
        self.units.iter_mut().for_each(|opt| *opt = None);
        self.array.clear();
        self.head = 0;
    }
}
