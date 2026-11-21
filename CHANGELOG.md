# Changelog

## Unreleased

### Added

- OpenGL query objects to measure `GL_TIME_ELAPSED` for pipeline `Pass`es, available optionally under the `pipeline-measure` feature flag

### Changed

- Adjust internal bit representation of rasterizer output (G component):
  * `instanceID`: 21 bits (previously 15 bits)
  * `instanceFlag`: 1 bit (unchanged)
  * `materialID`: 10 bits (previously 16 bits)
  
    The physical order has also changed: `materialID `is now on the most-significant end, while `instanceID `is on the least significant end. `instanceFlag` is on bit 22, between `instanceID` and `materialID`.
  
    With these changes, the total capacity of materials is reduced from 65,535 to 1024. The total capacity of *global* instances is greatly increased, from 65,535 to 2,097,152.
  
- In line with the increase in the maximum count of instances, `InstanceList` no longer packs `instance_base` and `_count` in a single 32 bit integer. `instance_count` and `instance_base` are both 32 bits. 
 
  `InstanceList` is now represented by 4 additional bytes, or one additional word in the internal instances data buffer.

### Fixed

- Prevent dispatching workgroup when workgroup count is 0 in any dimension for `Compute`-based `Pass`es

  This prevents an unnecessary graphics API call and GL error.

## 0.1.0 - 2026/11/21
Base pre-changelog state
