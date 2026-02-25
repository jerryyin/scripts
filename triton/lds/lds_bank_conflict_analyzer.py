#!/usr/bin/env python3
"""
LDS Bank Conflict Analyzer for AMD GPUs

This script analyzes bank conflicts for shared memory (LDS) access patterns,
particularly useful for understanding ds_load_tr* instructions and WMMA operand loads.

=============================================================================
CONCEPTS
=============================================================================

1. LDS Banks:
   - AMD gfx1250 has 64 banks, each 4 bytes wide
   - Bank = (byte_address / 4) % 64
   - Bank conflict occurs when multiple lanes access the same bank simultaneously

2. Linear Layout (from Triton TTGIR):
   - Describes how (register_idx, lane_id, warp_id) maps to tensor [row, col]
   - Format: lane = [[row_contrib, col_contrib], ...] for each bit of lane ID
   - Example: lane = [[1,0], [2,0], [4,0], [8,0], [0,8]]
     - Bits 0-3 contribute to row (1, 2, 4, 8)
     - Bit 4 contributes to column (8)

3. ds_load_tr16_b128:
   - Each lane loads 16 bytes (8 fp16 elements)
   - 8 lanes cooperate for transposed load
   - Accesses 4 consecutive banks per lane

4. Storage Layouts (conflict avoidance strategies):

   a) NONE:    byte_addr = row * stride + col * elem_bytes
               No conflict avoidance.  Baseline for comparison.

   b) PADDING: byte_addr = row * (row_width + pad) * elem_bytes + col * elem_bytes
               Extra elements widen each row to shift bank alignment.
               Wastes LDS space proportional to (pad / row_width).

   c) SWIZZLE: XOR-based column remapping (zero wasted LDS space).
               Parameterised by (vec, perPhase, maxPhase) matching Triton's
               #ttg.swizzled_shared<{vec, perPhase, maxPhase}> encoding.

               phase = (row // perPhase) % maxPhase
               vec_col = col // vec
               swizzled_vec_col = vec_col ^ phase
               swizzled_col = swizzled_vec_col * vec + (col % vec)
               byte_addr = row * stride + swizzled_col * elem_bytes

               Different rows XOR their vector-group column index with a
               different phase, rotating which banks each row hits so that
               simultaneously-accessed rows land on distinct banks.

=============================================================================
USAGE
=============================================================================

    python3 lds_bank_conflict_analyzer.py

Or import as module:

    from lds_bank_conflict_analyzer import analyze_bank_conflicts, LDSConfig, AccessPattern

=============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import argparse
import sys


class SwizzleMode(Enum):
    """LDS storage layout strategy for bank conflict avoidance."""
    NONE = "none"
    PADDING = "padding"
    SWIZZLE = "swizzle"


@dataclass
class LDSConfig:
    """
    Configuration for LDS memory layout.

    Supports three storage strategies controlled by ``mode``:

    - **NONE**: Logical (row, col) maps linearly to byte address.
    - **PADDING**: Extra elements appended after each row widen the stride.
    - **SWIZZLE**: XOR-based column remapping (zero wasted space).

    Swizzle parameters (``vec``, ``per_phase``, ``max_phase``) correspond
    directly to Triton's ``#ttg.swizzled_shared<{vec, perPhase, maxPhase}>``
    encoding and can be supplied manually or derived automatically via
    ``from_shared_encoding()`` / ``auto_swizzle()``.

    The canonical address formula (matching Triton's LinearLayoutConversions.cpp)
    uses vector-group indices.  Let ``num_vec_k = row_width / vec`` be the number
    of vec-groups per tensor row.

    When the tensor row is narrower than one bank row (``per_phase > 1``),
    ``per_phase`` consecutive tensor rows are packed side-by-side into each
    physical LDS row.  The packed vec-group index and the physical LDS row
    are::

        packed_vec = vec_col + (row % per_phase) * num_vec_k
        lds_row    = row // per_phase

    The phase and XOR then operate on the packed coordinate::

        phase = (row // per_phase) % max_phase
        swizzled_vec = packed_vec ^ phase
        swizzled_col = swizzled_vec * vec + (col % vec)
        lds_stride   = per_phase * row_width * element_bytes
        byte_addr    = lds_row * lds_stride + swizzled_col * element_bytes

    When ``per_phase == 1`` (the common case for K >= 64 with fp16) this
    reduces to the simpler per-row XOR formula.  When ``mode != SWIZZLE``
    the phase is effectively 0 (identity XOR).
    """
    num_banks: int = 64
    bytes_per_bank: int = 4
    row_width_elements: int = 128
    padding_elements: int = 0
    element_bytes: int = 2

    mode: SwizzleMode = SwizzleMode.NONE

    # Swizzle parameters (only used when mode == SWIZZLE)
    vec: int = 1
    per_phase: int = 1
    max_phase: int = 1

    @property
    def row_stride_bytes(self) -> int:
        """Total bytes per row including padding."""
        return (self.row_width_elements + self.padding_elements) * self.element_bytes

    def logical_to_byte_addr(self, row: int, col: int) -> int:
        """
        Map a logical (row, col) tensor position to a physical LDS byte address.

        Handles all three modes transparently so callers never need to know
        which conflict-avoidance strategy is active.
        """
        if self.mode == SwizzleMode.SWIZZLE:
            num_vec_k = self.row_width_elements // self.vec
            vec_col = col // self.vec

            # Pack per_phase tensor rows into one physical LDS row
            packed_vec = vec_col + (row % self.per_phase) * num_vec_k
            lds_row = row // self.per_phase

            phase = (row // self.per_phase) % self.max_phase
            swizzled_vec = packed_vec ^ phase
            swizzled_col = swizzled_vec * self.vec + (col % self.vec)

            lds_stride = self.per_phase * self.row_width_elements * self.element_bytes
            return lds_row * lds_stride + swizzled_col * self.element_bytes

        # NONE and PADDING both use the same linear formula;
        # PADDING simply has a wider row_stride_bytes.
        return row * self.row_stride_bytes + col * self.element_bytes

    def describe(self) -> str:
        """One-line human-readable description of the storage layout."""
        if self.mode == SwizzleMode.SWIZZLE:
            return (f"swizzle(vec={self.vec}, perPhase={self.per_phase}, "
                    f"maxPhase={self.max_phase})")
        if self.mode == SwizzleMode.PADDING:
            return f"padding({self.padding_elements} elems, {self.padding_elements * self.element_bytes}B)"
        return "none"

    # -----------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------

    @classmethod
    def with_padding(cls, padding_elements: int,
                     row_width: int = 128,
                     element_bytes: int = 2,
                     num_banks: int = 64) -> 'LDSConfig':
        """Create an LDSConfig with row-padding conflict avoidance."""
        return cls(
            row_width_elements=row_width,
            padding_elements=padding_elements,
            element_bytes=element_bytes,
            num_banks=num_banks,
            mode=SwizzleMode.PADDING if padding_elements > 0 else SwizzleMode.NONE,
        )

    @classmethod
    def with_swizzle(cls, vec: int, per_phase: int, max_phase: int,
                     row_width: int = 128,
                     element_bytes: int = 2,
                     num_banks: int = 64) -> 'LDSConfig':
        """Create an LDSConfig with explicit swizzle parameters.

        Parameters match ``#ttg.swizzled_shared<{vec, perPhase, maxPhase}>``.
        """
        return cls(
            row_width_elements=row_width,
            element_bytes=element_bytes,
            num_banks=num_banks,
            mode=SwizzleMode.SWIZZLE,
            vec=vec,
            per_phase=per_phase,
            max_phase=max_phase,
        )

    @classmethod
    def from_shared_encoding(cls, vec: int, per_phase: int, max_phase: int,
                             row_width: int = 128,
                             element_bytes: int = 2,
                             num_banks: int = 64) -> 'LDSConfig':
        """Alias for ``with_swizzle`` using Triton shared-encoding names."""
        return cls.with_swizzle(vec, per_phase, max_phase,
                                row_width, element_bytes, num_banks)

    @classmethod
    def auto_swizzle(cls, row_width: int, element_bytes: int = 2,
                     k_width: int = 8, num_banks: int = 32,
                     bank_bit_width: int = 32,
                     non_k_dim: int = 16,
                     arch: str = "wmma") -> 'LDSConfig':
        """Derive swizzle parameters the same way the Triton compiler does.

        This mirrors ``AMDWmmaEncodingAttr::composeSharedLayoutForOperand``
        (for arch="wmma") and ``AMDMfmaEncodingAttr::composeSharedLayoutForOperand``
        (for arch="mfma").

        Args:
            row_width:      Inner-dimension length (K for K-contiguous layout).
            element_bytes:  Bytes per element.
            k_width:        Dot operand kWidth (vec group size).
            num_banks:      LDS bank count (32 for RDNA, 64 for CDNA3+).
            bank_bit_width: Bits per bank word (32).
            non_k_dim:      M/N dimension of the matrix instruction (16 typical).
            arch:           "wmma" or "mfma" — selects the derivation formula.
        """
        elem_bit_width = element_bytes * 8

        # vec = min(kWidth * elemBits, 128) / elemBits  (in elements)
        vec = min(k_width * elem_bit_width, 128) // elem_bit_width

        inner_dim_length = row_width
        elems_per_bank_row = (num_banks * bank_bit_width) // elem_bit_width

        per_phase = max(1, elems_per_bank_row // inner_dim_length)

        if arch == "mfma":
            simd_width = 16
            max_phase = max(min(simd_width // per_phase,
                                inner_dim_length // vec), 1)
        else:
            # WMMA formula
            m_dim = non_k_dim
            max_phase = max(min(m_dim // per_phase,
                                inner_dim_length // vec), 1)

        return cls.with_swizzle(vec, per_phase, max_phase,
                                row_width, element_bytes, num_banks)


@dataclass 
class AccessPattern:
    """
    Describes how lanes access tensor positions.
    
    Attributes:
        lanes_per_wave: Number of lanes per wave (32 for AMD)
        elements_per_load: Elements loaded per lane per instruction (8 for ds_load_tr16_b128)
        lane_to_position: List of (row_offset, col_offset) for each lane
                         Derived from linear layout's lane field
        element_offsets: Column offsets for each of the elements_per_load elements
                        Derived from register bits. Default: [0, 1, 2, ..., 7] (consecutive)
                        
    Note on linear layout interpretation:
        - lane bits: Define the STARTING [row, col] for each lane
        - register bits 0-N: Define the ELEMENT positions within one load
        - Higher register bits: Define tiling across the full tensor (not used here)
        
    For ds_load_tr16_b128 with register = [[0,1], [0,2], [0,4], ...]:
        - Bits 0-2 contribute cols 1, 2, 4 -> elements at consecutive cols 0-7
        - This is the "basic unit" that one instruction loads
    """
    lanes_per_wave: int = 32
    elements_per_load: int = 8
    lane_to_position: List[Tuple[int, int]] = None
    element_offsets: List[int] = None  # Column offset for each element within the load
    
    def __post_init__(self):
        if self.lane_to_position is None:
            # Default: simple row-major access
            self.lane_to_position = [(i, 0) for i in range(self.lanes_per_wave)]
        if self.element_offsets is None:
            # Default: consecutive elements [0, 1, 2, ..., elements_per_load-1]
            self.element_offsets = list(range(self.elements_per_load))
    
    @classmethod
    def from_linear_layout(cls, lane_bits: List[Tuple[int, int]], 
                           register_bits: List[Tuple[int, int]] = None) -> 'AccessPattern':
        """
        Create AccessPattern from Triton linear layout.
        
        Args:
            lane_bits: List of [row_contribution, col_contribution] for each lane bit
                      e.g., [[1,0], [2,0], [4,0], [8,0], [0,8]] from TTGIR
                      Number of lanes = 2^len(lane_bits)
            register_bits: List of [row_contribution, col_contribution] for register bits
                          that define element positions within one load.
                          e.g., [[0,1], [0,2], [0,4]] for 8 consecutive elements
                          Number of elements = 2^len(register_bits)
                          If None, assumes 8 consecutive elements (default for ds_load_tr16_b128).
        
        Returns:
            AccessPattern with lane_to_position and element_offsets computed
        
        Note:
            lanes_per_wave and elements_per_load are DERIVED from the bit counts:
            - lanes_per_wave = 2^len(lane_bits)
            - elements_per_load = 2^len(register_bits)
        """
        # Derive counts from bit lengths
        num_lane_bits = len(lane_bits)
        lanes_per_wave = 1 << num_lane_bits  # 2^num_lane_bits
        
        if register_bits is not None:
            num_reg_bits = len(register_bits)
            elements_per_load = 1 << num_reg_bits  # 2^num_reg_bits
        else:
            # Default: 8 elements (3 bits) with consecutive columns
            num_reg_bits = 3
            elements_per_load = 8
            register_bits = [[0, 1], [0, 2], [0, 4]]  # Consecutive cols 0-7
        
        # Compute lane starting positions
        lane_to_position = []
        for lane_id in range(lanes_per_wave):
            row = 0
            col = 0
            for bit_idx in range(num_lane_bits):
                if lane_id & (1 << bit_idx):
                    row += lane_bits[bit_idx][0]
                    col += lane_bits[bit_idx][1]
            lane_to_position.append((row, col))
        
        # Compute element offsets within each load from register bits
        element_offsets = []
        for elem_idx in range(elements_per_load):
            col_offset = 0
            for bit_idx in range(num_reg_bits):
                if elem_idx & (1 << bit_idx):
                    col_offset += register_bits[bit_idx][1]
            element_offsets.append(col_offset)
        
        return cls(lanes_per_wave=lanes_per_wave,
                   elements_per_load=elements_per_load,
                   lane_to_position=lane_to_position,
                   element_offsets=element_offsets)


def compute_banks_accessed(byte_addr: int, 
                           load_size_bytes: int,
                           num_banks: int = 64,
                           bytes_per_bank: int = 4) -> List[int]:
    """
    Compute which banks are accessed for a load starting at byte_addr.
    
    Args:
        byte_addr: Starting byte address in LDS
        load_size_bytes: Number of bytes loaded (16 for ds_load_tr16_b128)
        num_banks: Number of LDS banks
        bytes_per_bank: Bytes per bank
    
    Returns:
        List of bank indices accessed
    """
    num_banks_accessed = load_size_bytes // bytes_per_bank
    start_bank = (byte_addr // bytes_per_bank) % num_banks
    return [(start_bank + i) % num_banks for i in range(num_banks_accessed)]


def compute_lane_accesses(config: LDSConfig, 
                          pattern: AccessPattern) -> List[Dict]:
    """
    Compute LDS access details for each lane.
    
    Args:
        config: LDS configuration (address mapping is delegated to
                ``config.logical_to_byte_addr()``, so swizzle/padding/none
                are handled transparently).
        pattern: Access pattern
    
    Returns:
        List of dicts with lane access info:
        [{lane, row, col, byte_addr, banks, element_addrs}, ...]
    """
    accesses = []
    
    for lane_id in range(pattern.lanes_per_wave):
        row, col_start = pattern.lane_to_position[lane_id]
        
        element_addrs = []
        all_banks = set()
        
        for elem_offset in pattern.element_offsets:
            col = col_start + elem_offset
            byte_addr = config.logical_to_byte_addr(row, col)
            element_addrs.append(byte_addr)
            bank = (byte_addr // config.bytes_per_bank) % config.num_banks
            all_banks.add(bank)
        
        base_addr = element_addrs[0]
        banks = sorted(all_banks)
        
        accesses.append({
            'lane': lane_id,
            'row': row,
            'col': col_start,
            'byte_addr': base_addr,
            'banks': banks,
            'element_addrs': element_addrs
        })
    
    return accesses


def find_bank_conflicts(accesses: List[Dict],
                        bytes_per_bank: int = 4,
                        num_banks: int = 64) -> Tuple[Dict[int, List[int]], int]:
    """
    Find bank conflicts from lane accesses.

    On AMD LDS, two accesses to the *same* dword (aligned 4-byte word) in the
    same bank are served as a broadcast -- NOT a conflict.  Only accesses to
    *different* dwords that map to the same bank produce a conflict.

    For dword-sized or larger elements (f32, vector loads) this distinction
    doesn't matter: each element occupies its own dword.  For sub-dword
    elements (fp16 = 2B, fp8 = 1B) multiple elements share a dword, and
    naive per-bank lane counting overcounts conflicts.

    Args:
        accesses: List of lane access info from compute_lane_accesses()
        bytes_per_bank: Bank granularity in bytes (4 for AMD LDS)
        num_banks: Number of LDS banks

    Returns:
        Tuple of (bank_to_lanes dict, max_conflict_count)
        bank_to_lanes maps bank -> deduplicated list of lanes that cause a
        true conflict (one representative per distinct dword).
        max_conflict_count is the true conflict degree (distinct dwords
        per bank).
    """
    # bank -> { dword_addr: first_lane_that_accessed_it }
    bank_to_dword_lane: Dict[int, Dict[int, int]] = {}

    for access in accesses:
        for byte_addr in access['element_addrs']:
            dword_addr = byte_addr // bytes_per_bank
            bank = dword_addr % num_banks

            if bank not in bank_to_dword_lane:
                bank_to_dword_lane[bank] = {}
            if dword_addr not in bank_to_dword_lane[bank]:
                bank_to_dword_lane[bank][dword_addr] = access['lane']

    # Build bank_to_lanes with one representative lane per distinct dword
    bank_to_lanes: Dict[int, List[int]] = {}
    for bank, dword_map in bank_to_dword_lane.items():
        bank_to_lanes[bank] = list(dword_map.values())

    max_conflict = max(len(lanes) for lanes in bank_to_lanes.values()) if bank_to_lanes else 0

    return bank_to_lanes, max_conflict


def generate_bank_grid(config: LDSConfig, 
                       tile_rows: int, 
                       tile_cols: int) -> List[List[Tuple[int, bool]]]:
    """
    Generate a grid showing which bank each logical element maps to,
    including padding columns (for PADDING mode).

    The grid always shows *logical* (row, col) positions.  The bank
    number at each position reflects whichever storage strategy
    (none / padding / swizzle) the config uses.
    
    Args:
        config: LDS configuration
        tile_rows: Number of rows in the tile
        tile_cols: Number of DATA columns in the tile
    
    Returns:
        2D list of (bank_index, is_padding) tuples [row][col]
        where col covers data columns + padding columns.
    """
    total_cols = tile_cols + config.padding_elements
    grid = []
    for row in range(tile_rows):
        row_banks = []
        for col in range(total_cols):
            byte_addr = config.logical_to_byte_addr(row, col)
            bank = (byte_addr // config.bytes_per_bank) % config.num_banks
            is_pad = col >= tile_cols
            row_banks.append((bank, is_pad))
        grid.append(row_banks)
    return grid


def _compute_covered_cells(pattern: 'AccessPattern') -> set:
    """Return the set of (row, col) cells touched by the pattern's lanes."""
    covered = set()
    for row, col_start in pattern.lane_to_position:
        for off in pattern.element_offsets:
            covered.add((row, col_start + off))
    return covered


def print_bank_grid(grid: List[List[Tuple[int, bool]]], max_cols: int = 64,
                    pattern: Optional['AccessPattern'] = None):
    """
    Print bank grid with three visual zones:

    1. **Covered** by the current access pattern → bank number shown normally
    2. **Data but not covered** (rest of the LDS row) → bank shown dimmed as ·XX
    3. **Padding** columns → bank shown in parentheses (XX)

    When pattern is None, all data cells are shown normally (no coverage
    highlighting).
    """
    num_rows = len(grid)
    num_cols = len(grid[0]) if grid else 0
    show_cols = min(num_cols, max_cols)

    covered = _compute_covered_cells(pattern) if pattern else None

    cw = 4  # column width for all cells

    # Determine the column boundary of the accessed region
    cov_max_col = -1
    if covered is not None:
        cov_max_col = max(c for _, c in covered)

    data_cols = sum(1 for bank, is_pad in grid[0] if not is_pad) if grid else 0

    # Header: column indices
    print("     ", end="")
    for col in range(show_cols):
        print(f"{col:>{cw}d}", end="")
    if num_cols > max_cols:
        print("  ..", end="")
    print("   ← col")

    # Separator line with border markers
    sep = ""
    for col in range(show_cols):
        if col == data_cols:
            sep += "╫" + "─" * (cw - 1)
        elif covered is not None and col == cov_max_col + 1 and col < data_cols:
            sep += "┼" + "─" * (cw - 1)
        else:
            sep += "─" * cw
    print("     " + sep)

    # Grid rows
    for row in range(num_rows):
        print(f"r{row:2d} │", end="")
        for col in range(show_cols):
            bank, is_pad = grid[row][col]
            if col == data_cols:
                print(f"║{bank:3d}", end="")
            elif covered is not None and col == cov_max_col + 1 and col < data_cols:
                print(f"|{bank:3d}", end="")
            else:
                print(f"{bank:>{cw}d}", end="")
        if num_cols > max_cols:
            print("  ..", end="")
        print()

    # Legend
    legend_parts = []
    if covered is not None:
        # Compute covered bounding box for the legend
        cov_rows = sorted(set(r for r, _ in covered))
        cov_cols = sorted(set(c for _, c in covered))
        legend_parts.append(
            f"   XX = accessed by current pattern "
            f"(rows {cov_rows[0]}-{cov_rows[-1]}, cols {cov_cols[0]}-{cov_cols[-1]})"
        )
        legend_parts.append("     | = boundary of accessed region")
    if grid and any(is_pad for _, is_pad in grid[0]):
        legend_parts.append("     ║ = boundary of data / padding")
    if legend_parts:
        print()
        for line in legend_parts:
            print(line)


def print_lane_accesses(accesses: List[Dict]):
    """Print lane access table."""
    print("Lane | Row | Col | Addr (bytes) | Banks accessed")
    print("-----|-----|-----|--------------|----------------")
    for a in accesses:
        print(f" {a['lane']:2d}  |  {a['row']:2d} | {a['col']:2d}  |    {a['byte_addr']:5d}     | {a['banks']}")


def print_conflict_summary(bank_to_lanes: Dict[int, List[int]], max_conflict: int):
    """Print bank conflict summary."""
    if max_conflict <= 1:
        print("✓ No bank conflicts!")
    else:
        print(f"✗ Max conflict: {max_conflict}-way")
        print()
        print("Conflicting banks (first 4):")
        conflict_banks = [(b, lanes) for b, lanes in bank_to_lanes.items() 
                         if len(lanes) == max_conflict]
        for bank, lanes in conflict_banks[:4]:
            print(f"  Bank {bank:2d}: lanes {lanes}")


def analyze_bank_conflicts(config: LDSConfig, 
                           pattern: AccessPattern,
                           tile_rows: int = 16,
                           tile_cols: int = 16,
                           verbose: bool = True) -> int:
    """
    Full bank conflict analysis.
    
    Args:
        config: LDS configuration
        pattern: Access pattern
        tile_rows: Rows in tile to visualize
        tile_cols: Columns in tile to visualize  
        verbose: Print detailed output
    
    Returns:
        Maximum conflict count
    """
    if verbose:
        print("=" * 70)
        print("BANK CONFLICT ANALYSIS")
        print("=" * 70)
        print()
        print(f"LDS Configuration:")
        print(f"  Row width: {config.row_width_elements} elements")
        print(f"  Storage layout: {config.describe()}")
        print(f"  Row stride: {config.row_stride_bytes} bytes")
        print(f"  Element size: {config.element_bytes} bytes")
        print(f"  Banks: {config.num_banks}")
        print()
        print(f"Access Pattern:")
        print(f"  Lanes per wave: {pattern.lanes_per_wave}")
        print(f"  Elements per load: {pattern.elements_per_load}")
        print(f"  Load size: {pattern.elements_per_load * config.element_bytes} bytes")
        print()
    
    if verbose:
        layout_label = f" [{config.describe()}]" if config.mode != SwizzleMode.NONE else ""
        print("=" * 70)
        print(f"BANK MAPPING ({tile_rows}×{tile_cols} tile{layout_label})")
        print("=" * 70)
        print()
        grid = generate_bank_grid(config, tile_rows, tile_cols)
        print_bank_grid(grid, max_cols=tile_cols + config.padding_elements,
                        pattern=pattern)
        print()
    
    # Compute lane accesses
    accesses = compute_lane_accesses(config, pattern)
    
    if verbose:
        print("=" * 70)
        print("LANE ACCESS PATTERN")
        print("=" * 70)
        print()
        print_lane_accesses(accesses)
        print()
    
    # Find conflicts (pass bank geometry for sub-dword broadcast detection)
    bank_to_lanes, max_conflict = find_bank_conflicts(
        accesses, bytes_per_bank=config.bytes_per_bank, num_banks=config.num_banks)
    
    if verbose:
        print("=" * 70)
        print("CONFLICT SUMMARY")
        print("=" * 70)
        print()
        print_conflict_summary(bank_to_lanes, max_conflict)
        print()
    
    return max_conflict


# =============================================================================
# Preset configurations for common use cases
# =============================================================================

def ds_load_tr16_b128_pattern() -> AccessPattern:
    """
    Access pattern for ds_load_tr16_b128 with typical WMMA layout.
    
    From TTGIR #linear layout:
        lane = [[1,0], [2,0], [4,0], [8,0], [0,8]]
        register = [[0,1], [0,2], [0,4], ...]  (first 3 bits for 8 elements)
    
    Lane mapping (derived: 5 bits → 32 lanes):
        - Lanes 0-15: rows 0-15, col 0
        - Lanes 16-31: rows 0-15, col 8
    
    Register mapping (derived: 3 bits → 8 elements):
        - Each lane loads 8 consecutive column elements
        - Element offsets: 0, 1, 2, 3, 4, 5, 6, 7
    """
    return AccessPattern.from_linear_layout(
        lane_bits=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        register_bits=[[0, 1], [0, 2], [0, 4]]
    )


def mfma16_kcontig_pattern(kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for MFMA-16x16 dot operand with K-contiguous LDS layout.
    Non-transposed ds_read_b128: each lane reads kWidth consecutive K elements.

    The layout is symmetric for operand A and B -- the formula only depends
    on nonKDim and warpSize, not opIdx. We name this by the instruction
    geometry, not the operand index.

    From mfmaDotToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim=16, lane, nonK) * identity1D(4, lane, K)

    Lane mapping (6 bits -> 64 lanes):
        - Bits 0-3 (lane & 0xF): nonK position (0-15)
        - Bits 4-5 (lane >> 4):  K group (0-3), K_start = group * kWidth

    Each lane reads kWidth elements starting at (nonK=lane&0xF, K=group*kWidth).

    For K=16 tile with kWidth=8: only K-groups 0,1 are active (lanes 0-31).
    For K=32 tile with kWidth=8: all 4 K-groups active (lanes 0-63).

    LDS layout: row = nonK position, columns = K elements (K contiguous).
    """
    lane_to_position = []
    for lane_id in range(64):
        n = lane_id & 0xF
        k_group = (lane_id >> 4) & 0x3
        k_start = k_group * kWidth
        lane_to_position.append((n, k_start))

    return AccessPattern(
        lanes_per_wave=64,
        elements_per_load=kWidth,
        lane_to_position=lane_to_position,
        element_offsets=list(range(kWidth))
    )


def wmma16_kcontig_pattern(kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for WMMA v3 16x16 dot operand with K-contiguous LDS layout.
    Non-transposed ds_read_b128: each lane reads kWidth consecutive K elements.

    The layout is symmetric for operand A and B -- the formula only depends
    on nonKDim and warpSize, not opIdx. We name this by the instruction
    geometry, not the operand index.

    From wmmaDotOperandToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim=16, lane, nonK) * identity1D(depth=2, lane, K)

    Lane mapping (5 bits -> 32 lanes):
        - Bits 0-3 (lane & 0xF): nonK position (0-15)
        - Bit 4    (lane >> 4):  K group (0-1), K_start = group * kWidth

    Each lane reads kWidth elements starting at (nonK=lane&0xF, K=group*kWidth).

    For K=16 with kWidth=8: both K-groups active (lanes 0-31), full coverage.

    LDS layout: row = nonK position, columns = K elements (K contiguous).
    """
    lane_to_position = []
    for lane_id in range(32):
        n = lane_id & 0xF
        k_group = (lane_id >> 4) & 0x1
        k_start = k_group * kWidth
        lane_to_position.append((n, k_start))

    return AccessPattern(
        lanes_per_wave=32,
        elements_per_load=kWidth,
        lane_to_position=lane_to_position,
        element_offsets=list(range(kWidth))
    )


def wmma16_transposed_scalar_pattern(kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for WMMA v3 16x16 dot operand with TRANSPOSED LDS layout.

    When K is the slow-varying dimension in LDS (loadTransposed=True) and
    there is no ds_load_tr* instruction for this element width (e.g. f32),
    the compiler falls back to scalar ds_load_b32 loads -- one element per
    lane per load cycle.

    Each thread needs kWidth K elements, issued as kWidth separate scalar
    loads.  For a single scalar load (register index r), all 32 lanes fire
    simultaneously.  Since the conflict structure is identical for every r
    (only the absolute row shifts, not the stride), we model r=0 as
    representative.

    From wmmaDotOperandToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim=16, lane, nonK) * identity1D(depth=2, lane, K)

    Lane mapping (5 bits -> 32 lanes):
        - Bits 0-3 (lane & 0xF): nonK position (0-15)  -> LDS column
        - Bit 4    (lane >> 4):   K group (0-1)         -> LDS row offset

    LDS layout (transposed): nonK is contiguous (columns), K is strided (rows).
    For register r=0:
        - Lanes 0-15:  row = 0,       col = lane
        - Lanes 16-31: row = kWidth,  col = lane - 16

    Bank conflict occurs when kWidth * row_stride % num_banks == 0.
    """
    positions = []
    for lane_id in range(32):
        nonk = lane_id & 0xF
        k_group = (lane_id >> 4) & 0x1
        row = k_group * kWidth
        positions.append((row, nonk))

    return AccessPattern(
        lanes_per_wave=32,
        elements_per_load=1,
        lane_to_position=positions,
        element_offsets=[0],
    )


def create_config_with_padding(padding_elements: int,
                                row_width: int = 128,
                                element_bytes: int = 2) -> LDSConfig:
    """Create LDS config with specified padding (legacy helper)."""
    return LDSConfig.with_padding(padding_elements, row_width, element_bytes)


# =============================================================================
# Pattern registry (maps CLI name -> factory function)
# =============================================================================

PATTERN_REGISTRY = {
    "ds_load_tr16_b128":       (ds_load_tr16_b128_pattern, "Transposed 16-bit load, 32 lanes (gfx1250)"),
    "mfma16_kcontig":          (lambda kw=8: mfma16_kcontig_pattern(kWidth=kw), "MFMA-16x16 non-transposed, 64 lanes (CDNA)"),
    "wmma16_kcontig":          (lambda kw=8: wmma16_kcontig_pattern(kWidth=kw), "WMMA-16x16 non-transposed, 32 lanes (gfx1250)"),
    "wmma16_transposed_scalar": (lambda kw=8: wmma16_transposed_scalar_pattern(kWidth=kw), "WMMA-16x16 transposed scalar, 32 lanes (f32 fallback)"),
}


def get_pattern(name: str, kwidth: int = 8) -> AccessPattern:
    """Look up a pattern by name from the registry."""
    if name not in PATTERN_REGISTRY:
        available = ", ".join(PATTERN_REGISTRY.keys())
        raise ValueError(f"Unknown pattern '{name}'. Available: {available}")
    factory, _ = PATTERN_REGISTRY[name]
    import inspect
    sig = inspect.signature(factory)
    if sig.parameters:
        return factory(kwidth)
    return factory()


# =============================================================================
# Main entry point
# =============================================================================

def _cap_tile_dims(tile_rows, tile_cols, row_width, pattern):
    """Cap visualization tile dims to the actual LDS tile covered by the pattern."""
    # Max rows = number of unique rows any lane touches
    max_row = max(r for r, _c in pattern.lane_to_position) + 1
    # Max cols = row_width (data columns, excluding padding)
    tile_rows = min(tile_rows, max_row)
    tile_cols = min(tile_cols, row_width)
    return tile_rows, tile_cols


def _build_config(args) -> LDSConfig:
    """Build an LDSConfig from parsed CLI arguments."""
    if args.layout == 'swizzle':
        if args.swizzle_vec is not None:
            return LDSConfig.with_swizzle(
                vec=args.swizzle_vec,
                per_phase=args.swizzle_per_phase,
                max_phase=args.swizzle_max_phase,
                row_width=args.row_width,
                element_bytes=args.element_bytes,
                num_banks=args.num_banks,
            )
        arch = "mfma" if "mfma" in args.pattern else "wmma"
        return LDSConfig.auto_swizzle(
            row_width=args.row_width,
            element_bytes=args.element_bytes,
            k_width=args.kwidth,
            num_banks=args.num_banks,
            arch=arch,
        )
    return LDSConfig.with_padding(
        padding_elements=args.padding,
        row_width=args.row_width,
        element_bytes=args.element_bytes,
        num_banks=args.num_banks,
    )


def _run_one(config: LDSConfig, pattern: AccessPattern,
             tile_rows: int, tile_cols: int, quiet: bool) -> int:
    """Run a single analysis and return the max conflict count."""
    max_conflict = analyze_bank_conflicts(
        config, pattern,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        verbose=not quiet,
    )
    if quiet:
        import math
        status = "OK" if max_conflict <= 2 else "BAD"
        print(f"row={config.row_width_elements} {config.describe()} "
              f"stride={config.row_stride_bytes}B "
              f"gcd={math.gcd(config.row_stride_bytes, 256)} "
              f"→ {max_conflict}-way {status}")
    return max_conflict


def compare_layouts(pattern: AccessPattern, row_width: int,
                    element_bytes: int, num_banks: int, kwidth: int,
                    tile_rows: int, tile_cols: int, quiet: bool,
                    arch: str = "wmma"):
    """Compare none / padding / swizzle storage layouts side by side."""
    print("=" * 70)
    print("COMPARING STORAGE LAYOUTS")
    print("=" * 70)
    print()

    configs = [
        ("none",    LDSConfig.with_padding(0, row_width, element_bytes, num_banks)),
        ("pad=8",   LDSConfig.with_padding(8, row_width, element_bytes, num_banks)),
        ("pad=16",  LDSConfig.with_padding(16, row_width, element_bytes, num_banks)),
        ("swizzle", LDSConfig.auto_swizzle(row_width, element_bytes,
                                           kwidth, num_banks, arch=arch)),
    ]

    results = []
    for label, config in configs:
        max_conflict = analyze_bank_conflicts(
            config, pattern,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            verbose=not quiet,
        )
        results.append((label, config, max_conflict))
        if not quiet:
            print()

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Layout':>12s} | {'Details':30s} | Stride | Conflict")
    print(f"{'-'*12:s}-+-{'-'*30:s}-+--------+---------")
    for label, config, conflict in results:
        status = "OK" if conflict <= 2 else "BAD"
        print(f"{label:>12s} | {config.describe():30s} | {config.row_stride_bytes:4d} B "
              f"| {conflict:2d}-way {status}")


def main():
    pattern_list = "\n".join(
        f"    {name:25s} {desc}"
        for name, (_, desc) in PATTERN_REGISTRY.items()
    )

    parser = argparse.ArgumentParser(
        description="Analyze LDS bank conflicts for AMD GPU shared memory access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Patterns:
{pattern_list}

Examples:
  # Default: fp16, pad=0, ds_load_tr16_b128, row-width=128
  python3 lds_bank_conflict_analyzer.py --row-width 128

  # With padding (typical flash attention)
  python3 lds_bank_conflict_analyzer.py --row-width 128 --padding 8

  # Auto-derived swizzle layout
  python3 lds_bank_conflict_analyzer.py --layout swizzle --row-width 128

  # Explicit swizzle params from Triton IR
  python3 lds_bank_conflict_analyzer.py --layout swizzle --row-width 64 \\
    --swizzle-vec 8 --swizzle-per-phase 1 --swizzle-max-phase 8

  # Compare none vs padding vs swizzle
  python3 lds_bank_conflict_analyzer.py --row-width 128 --compare --quiet

  # MFMA fp8 with padding
  python3 lds_bank_conflict_analyzer.py --pattern mfma16_kcontig \\
    --row-width 128 --padding 8 --element-bytes 1
        """
    )

    parser.add_argument('--pattern', type=str, default='ds_load_tr16_b128',
                        choices=list(PATTERN_REGISTRY.keys()),
                        help='Access pattern to simulate (default: ds_load_tr16_b128)')
    parser.add_argument('--kwidth', type=int, default=8,
                        help='Elements per lane for non-transposed patterns (default: 8)')
    parser.add_argument('--row-width', type=int, default=128,
                        help='Data elements per LDS row, excl. padding (default: 128)')
    parser.add_argument('--layout', type=str, default='padding',
                        choices=['none', 'padding', 'swizzle'],
                        help='Storage layout: none, padding (default), or swizzle (XOR)')
    parser.add_argument('--padding', type=int, default=0,
                        help='Padding elements per row (default: 0, used with --layout=padding)')
    parser.add_argument('--swizzle-vec', type=int, default=None,
                        help='Swizzle vec; auto-derived from kwidth if omitted')
    parser.add_argument('--swizzle-per-phase', type=int, default=1,
                        help='Swizzle perPhase (default: 1)')
    parser.add_argument('--swizzle-max-phase', type=int, default=1,
                        help='Swizzle maxPhase (default: 1)')
    parser.add_argument('--num-banks', type=int, default=64,
                        help='LDS bank count (default: 64; use 32 for RDNA3)')
    parser.add_argument('--element-bytes', type=int, default=2,
                        help='Bytes per element: 2=fp16, 1=fp8, 4=f32 (default: 2)')
    parser.add_argument('--tile-rows', type=int, default=None,
                        help='Rows in bank grid (default: auto from pattern)')
    parser.add_argument('--tile-cols', type=int, default=None,
                        help='Cols in bank grid (default: auto from row-width)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare none / padding / swizzle side by side')
    parser.add_argument('--quiet', action='store_true',
                        help='One-line summary only')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    pattern = get_pattern(args.pattern, args.kwidth)

    default_rows = max(r for r, _c in pattern.lane_to_position) + 1
    default_cols = args.row_width
    tile_rows = args.tile_rows if args.tile_rows is not None else default_rows
    tile_cols = args.tile_cols if args.tile_cols is not None else default_cols
    tile_rows, tile_cols = _cap_tile_dims(tile_rows, tile_cols, args.row_width, pattern)

    if args.compare:
        arch = "mfma" if "mfma" in args.pattern else "wmma"
        compare_layouts(pattern, args.row_width, args.element_bytes,
                        args.num_banks, args.kwidth, tile_rows, tile_cols,
                        args.quiet, arch)
    else:
        config = _build_config(args)
        _run_one(config, pattern, tile_rows, tile_cols, args.quiet)


if __name__ == "__main__":
    main()
