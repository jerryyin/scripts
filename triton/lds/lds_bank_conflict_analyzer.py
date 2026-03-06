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

2. Storage Layouts (conflict avoidance strategies):

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
        lds_group_size: How many lanes LDS services simultaneously per cycle.
                        LDS processes this many threads' data in one service
                        cycle; bank conflicts are checked within each group.
                        - ds_load_b{32,64,128}: 16 (16 threads/cycle for all)
                        - ds_load_2addr_b64: 32 (all lanes, 1 dword per sub-op)
                        - ds_load_tr*: 16 (transpose shuffle group)
                        Default: None → uses find_bank_conflicts default (all lanes).
        dwords_per_subop: How many dwords per lane are checked in one sub-operation.
                        Only needed for multi-address instructions (ds_load_2addr_b64)
                        where each address is an independent sub-op; set to 1 so
                        dwords from different addresses aren't checked against each
                        other.  For all single-address instructions, leave as None
                        (all dwords are served simultaneously).
                        
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
    lds_group_size: int = None
    dwords_per_subop: int = None
    
    def __post_init__(self):
        if self.lane_to_position is None:
            # Default: simple row-major access
            self.lane_to_position = [(i, 0) for i in range(self.lanes_per_wave)]
        if self.element_offsets is None:
            # Default: consecutive elements [0, 1, 2, ..., elements_per_load-1]
            self.element_offsets = list(range(self.elements_per_load))
    
    @classmethod
    def from_linear_layout(cls, lane_bits: List[Tuple[int, int]], 
                           register_bits: List[Tuple[int, int]] = None,
                           lds_group_size: int = None,
                           dwords_per_subop: int = None) -> 'AccessPattern':
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
            lds_group_size: Override for execution model (see AccessPattern docstring).
            dwords_per_subop: Override for execution model (see AccessPattern docstring).
        
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
                   element_offsets=element_offsets,
                   lds_group_size=lds_group_size,
                   dwords_per_subop=dwords_per_subop)


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
                        num_banks: int = 64,
                        lds_group_size: int = None,
                        dwords_per_subop: int = None) -> Tuple[Dict[int, List[int]], int]:
    """
    Find bank conflicts from lane accesses, respecting LDS service grouping.

    Bank conflicts only occur between lanes in the **same** LDS service
    group AND the **same** sub-operation.

    LDS services a fixed number of threads per cycle (lds_group_size).
    This is distinct from SIMD width (which determines SQ instruction
    issue speed) and from steady-state throughput (which depends on
    bandwidth utilization and pipeline contention across SIMDs).

    **ds_load_b{32,64,128}**: 16 threads per LDS service cycle.
        All dwords per lane are served simultaneously within the group.
        Wave32 → 2 groups executed sequentially.
    **ds_load_2addr_b64**: 32 threads per LDS service cycle, but each
        address is a separate sub-operation (dwords_per_subop=1).
        Dwords from different addresses don't compete for banks.
    **ds_load_tr***: 16 threads per LDS service cycle (transpose group).
        Each lane reads 1 dword.

    On AMD LDS, two accesses to the *same* dword (aligned 4-byte word) in the
    same bank are served as a broadcast — NOT a conflict.  Only accesses to
    *different* dwords that map to the same bank produce a conflict.

    Args:
        accesses: List of lane access info from compute_lane_accesses()
        bytes_per_bank: Bank granularity in bytes (4 for AMD LDS)
        num_banks: Number of LDS banks
        lds_group_size: Threads LDS services simultaneously (default: all)
        dwords_per_subop: Dwords per lane checked in one sub-operation.
                         None means all dwords together (default for single-
                         address instructions).  1 means each dword is an
                         independent sub-op (for multi-address instructions).

    Returns:
        Tuple of (bank_to_lanes dict, max_conflict_count)
        bank_to_lanes maps bank -> deduplicated list of lanes that cause a
        true conflict (one representative per distinct dword).
        max_conflict_count is the worst-case conflict degree across all groups.
    """
    total_lanes = len(accesses)
    if lds_group_size is None:
        lds_group_size = total_lanes
    num_groups = max(1, (total_lanes + lds_group_size - 1) // lds_group_size)

    overall_bank_to_lanes: Dict[int, List[int]] = {}
    overall_max = 0

    for grp in range(num_groups):
        lo = grp * lds_group_size
        hi = min(lo + lds_group_size, total_lanes)
        group_accesses = [a for a in accesses if lo <= a['lane'] < hi]

        if dwords_per_subop is None:
            subop_groups = [group_accesses]
        else:
            # Split each lane's dwords into independent sub-operations so
            # dwords from different sub-ops don't compete for banks.
            sc_map: Dict[int, List[Tuple[int, int]]] = {}
            for access in group_accesses:
                dwords_seen = []
                for ba in access['element_addrs']:
                    dw = ba // bytes_per_bank
                    if dw not in dwords_seen:
                        dwords_seen.append(dw)
                for rank, dw in enumerate(dwords_seen):
                    sc_idx = rank // dwords_per_subop
                    if sc_idx not in sc_map:
                        sc_map[sc_idx] = []
                    sc_map[sc_idx].append((access['lane'], dw))

            subop_groups = list(sc_map.values()) if sc_map else []

        for group in subop_groups:
            bank_to_dword_lane: Dict[int, Dict[int, int]] = {}

            if dwords_per_subop is None:
                for access in group:
                    for byte_addr in access['element_addrs']:
                        dword_addr = byte_addr // bytes_per_bank
                        bank = dword_addr % num_banks
                        if bank not in bank_to_dword_lane:
                            bank_to_dword_lane[bank] = {}
                        if dword_addr not in bank_to_dword_lane[bank]:
                            bank_to_dword_lane[bank][dword_addr] = access['lane']
            else:
                for lane_id, dword_addr in group:
                    bank = dword_addr % num_banks
                    if bank not in bank_to_dword_lane:
                        bank_to_dword_lane[bank] = {}
                    if dword_addr not in bank_to_dword_lane[bank]:
                        bank_to_dword_lane[bank][dword_addr] = lane_id

            for bank, dword_map in bank_to_dword_lane.items():
                lanes = list(dword_map.values())
                if len(lanes) > 1:
                    if bank not in overall_bank_to_lanes or \
                       len(lanes) > len(overall_bank_to_lanes[bank]):
                        overall_bank_to_lanes[bank] = lanes
                grp_max = len(lanes)
                if grp_max > overall_max:
                    overall_max = grp_max

    return overall_bank_to_lanes, overall_max


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

    cw = 3  # column width

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
                print(f"║{bank:{cw - 1}d}", end="")
            elif covered is not None and col == cov_max_col + 1 and col < data_cols:
                print(f"|{bank:{cw - 1}d}", end="")
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
    
    # Find conflicts, using pattern's execution model if specified
    fc_kwargs = dict(bytes_per_bank=config.bytes_per_bank, num_banks=config.num_banks)
    if pattern.lds_group_size is not None:
        fc_kwargs['lds_group_size'] = pattern.lds_group_size
    if pattern.dwords_per_subop is not None:
        fc_kwargs['dwords_per_subop'] = pattern.dwords_per_subop
    bank_to_lanes, max_conflict = find_bank_conflicts(accesses, **fc_kwargs)
    
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
    LDS access pattern for ds_load_tr16_b128 with WMMA v3 layout (gfx1250).

    Each lane reads 4 bytes (1 dword = 2 fp16 elements) from LDS.
    16 threads per LDS service cycle (transpose shuffle boundary).  The cross-lane
    transpose then produces the 128-bit (8 × fp16) result per lane.

    Pre-transpose LDS access (what actually hits the banks):
        lane = [[1,0], [2,0], [4,0], [0,8], [8,0]]   (K, N)
        - Group 0 (lanes 0-7):   rows 0-7,   col_start 0
        - Group 1 (lanes 8-15):  rows 0-7,   col_start 8
        - Group 2 (lanes 16-23): rows 8-15,  col_start 0
        - Group 3 (lanes 24-31): rows 8-15,  col_start 8

    Each lane reads 2 consecutive N elements (= 1 dword) from one K row.
    The 8-lane shuffle group transposes these into 8 K values per lane.
    """
    # Each lane reads 1 dword (2 fp16 = 4 bytes) from LDS.
    # 16 threads per LDS service cycle.
    return AccessPattern.from_linear_layout(
        lane_bits=[[1, 0], [2, 0], [4, 0], [0, 8], [8, 0]],
        register_bits=[[0, 1]],
        lds_group_size=16,
    )


def ds_load_tr8_b64_pattern() -> AccessPattern:
    """
    LDS access pattern for ds_load_tr8_b64 with doubleB8Contiguity (gfx1250).

    For 8-bit data (fp8/i8).  Each lane reads 4 bytes (1 dword = 4 fp8
    elements) from LDS.  16 threads per LDS service cycle.
    The cross-lane transpose produces the 64-bit (8 × fp8) result per lane.

    The doubleB8Contiguity trick changes lane bit 3 from N+8 to K+4,
    expanding DOWN instead of RIGHT.  This compensates for the interleaved
    shuffle groups so the post-transpose result matches WMMA's register layout.

    Pre-transpose LDS access (derived from compiler LLVM IR):
        lane = [[1,0], [2,0], [0,8], [4,0], [8,0]]   (K, N)

        - Lanes 0-3:   K=0..3,   N=0..3   (bit2=0)
        - Lanes 4-7:   K=0..3,   N=8..11  (bit2=1)
        - Lanes 8-11:  K=4..7,   N=0..3   (bit3=1)
        - Lanes 12-15: K=4..7,   N=8..11  (bit2=1, bit3=1)
        - Lanes 16-31: same pattern with K+8 (bit4=1)

    Each lane reads 4 consecutive N elements (= 1 dword) from one K row.
    """
    # Each lane reads 1 dword (4 fp8 = 4 bytes) from LDS.
    # 16 threads per LDS service cycle.
    return AccessPattern.from_linear_layout(
        lane_bits=[[1, 0], [2, 0], [0, 8], [4, 0], [8, 0]],
        register_bits=[[0, 1], [0, 2]],
        lds_group_size=16,
    )


def ds_load_2addr_b64_pattern(kWidth: int = 8) -> AccessPattern:
    """
    LDS access pattern for ds_load_2addr_b64 — WMMA v3 non-transposed (gfx1250).

    Used for the A operand when K is contiguous in LDS (loadTransposed=False).
    Each instruction issues TWO 64-bit (8-byte) loads from non-contiguous
    addresses, with an 8-byte gap between them.

    Instruction encoding (offsets in units of 8 bytes):
        ds_load_2addr_b64  offset0:0  offset1:2   →  base+0  and base+16
        ds_load_2addr_b64  offset0:4  offset1:6   →  base+32 and base+48

    Per-lane base address (from LLVM IR):
        base = M * stride + K_start
    where:
        M       = lane & 0xF                (0-15, row position)
        K_start = ((lane >> 4) & 1) * kWidth (0 or kWidth=8, interleaved K groups)
        stride  = (BLOCK_K + pad) * element_bytes

    The two lane halves INTERLEAVE within each 2*kWidth span:
        Lane 0  (K_start=0):  loads K[0..7]  + K[16..23]
        Lane 16 (K_start=8):  loads K[8..15] + K[24..31]

    Each lane loads 2*kWidth = 16 elements, but NOT consecutively.
    The element_offsets are [0..kWidth-1, 2*kWidth..3*kWidth-1], reflecting
    the 8-byte gap between the two sub-loads within one instruction.

    Execution model:
        All 32 threads serviced together per LDS cycle.
        Each address is an independent sub-operation (dwords_per_subop=1).
        Each lane's 2 addresses produce 2 dwords each; only dwords from the
        same address compete for banks.

    Typical BLOCK_K=64, fp8 (1 byte/elem):
        pad=8:  stride_d=18, gcd(18,64)=2, 2%2=0 → solution exists → 2-way
        pad=16: stride_d=20, gcd(20,64)=4, 2%4≠0 → no solution → conflict-free

    Args:
        kWidth: Elements per sub-load (8 for fp8).
    """
    load_width = 2 * kWidth

    lane_to_position = []
    for lane_id in range(32):
        m = lane_id & 0xF
        k_group = (lane_id >> 4) & 0x1
        k_start = k_group * kWidth
        lane_to_position.append((m, k_start))

    element_offsets = list(range(kWidth)) + list(range(2 * kWidth, 3 * kWidth))

    # All 32 threads serviced together per LDS cycle.
    # dwords_per_subop=1: the two addresses are independent sub-operations;
    # dwords from different addresses don't compete for banks.
    return AccessPattern(
        lanes_per_wave=32,
        elements_per_load=load_width,
        lane_to_position=lane_to_position,
        element_offsets=element_offsets,
        lds_group_size=32,
        dwords_per_subop=1,
    )


def mfma_kcontig_pattern(nonKDim: int = 16, kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for MFMA dot operand with K-contiguous LDS layout.

    Unified implementation for all MFMA geometries. The layout is symmetric
    for operand A and B — the formula only depends on nonKDim and warpSize
    (always 64 for CDNA), not opIdx.

    From mfmaDotToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim, lane, nonK) * identity1D(64/nonKDim, lane, K)

    Lane mapping (6 bits -> 64 lanes):
        - Low bits:  nonK position (0 .. nonKDim-1)
        - High bits: K group (0 .. 64/nonKDim - 1), K_start = group * kWidth

    Common configurations:
        nonKDim=16 (MFMA 16x16):
            4 K groups, default kWidth=8 → ds_read_b128 (fp16/fp8)
        nonKDim=32 (MFMA 32x32):
            2 K groups, default kWidth=2 → ds_read_b64  (f32)
            kWidth=4 → ds_read_b128 (TF32 mfma_f32_32x32x4_xf32)

    Args:
        nonKDim: M/N dimension of the MFMA instruction (16 or 32).
        kWidth:  Elements per lane per load.
    """
    num_k_groups = 64 // nonKDim
    nonk_mask = nonKDim - 1
    k_shift = nonKDim.bit_length() - 1
    k_mask = num_k_groups - 1

    lane_to_position = []
    for lane_id in range(64):
        n = lane_id & nonk_mask
        k_group = (lane_id >> k_shift) & k_mask
        lane_to_position.append((n, k_group * kWidth))

    # 16 threads per LDS service cycle. Wave64 → 4 groups.
    return AccessPattern(
        lanes_per_wave=64,
        elements_per_load=kWidth,
        lane_to_position=lane_to_position,
        element_offsets=list(range(kWidth)),
        lds_group_size=16,
    )


def mfma16_kcontig_pattern(kWidth: int = 8) -> AccessPattern:
    """MFMA-16x16 non-transposed (alias for mfma_kcontig_pattern(nonKDim=16))."""
    return mfma_kcontig_pattern(nonKDim=16, kWidth=kWidth)


def mfma32_kcontig_pattern(kWidth: int = 2) -> AccessPattern:
    """MFMA-32x32 non-transposed (alias for mfma_kcontig_pattern(nonKDim=32))."""
    return mfma_kcontig_pattern(nonKDim=32, kWidth=kWidth)


def wmma_kcontig_pattern(kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for WMMA dot operand with K-contiguous LDS layout.
    Non-transposed ds_read_b128: each lane reads kWidth consecutive K elements.

    WMMA on gfx1250 is always 16x16 (M/N), so nonKDim is fixed at 16 and
    warpSize is 32. The layout is symmetric for operand A and B.

    From wmmaDotOperandToLinearLayout (LinearLayoutConversions.cpp):
        regs  = identity1D(kWidth, register, K)
        lanes = identity1D(nonKDim=16, lane, nonK) * identity1D(depth=2, lane, K)

    Lane mapping (5 bits -> 32 lanes):
        - Bits 0-3 (lane & 0xF): nonK position (0-15)
        - Bit 4    (lane >> 4):  K group (0-1), K_start = group * kWidth

    Each lane reads kWidth elements starting at (nonK=lane&0xF, K=group*kWidth).

    Supported for fp8/fp16 operands (f32 uses wmma_transposed_scalar instead).
    """
    lane_to_position = []
    for lane_id in range(32):
        n = lane_id & 0xF
        k_group = (lane_id >> 4) & 0x1
        k_start = k_group * kWidth
        lane_to_position.append((n, k_start))

    # 16 threads per LDS service cycle.
    # All dwords per lane are served simultaneously within the group.
    return AccessPattern(
        lanes_per_wave=32,
        elements_per_load=kWidth,
        lane_to_position=lane_to_position,
        element_offsets=list(range(kWidth)),
        lds_group_size=16,
    )


def wmma_transposed_scalar_pattern(kWidth: int = 8) -> AccessPattern:
    """
    Access pattern for WMMA dot operand with TRANSPOSED LDS layout.

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

    # 16 threads per LDS service cycle (same as ds_load_b64/b128).
    return AccessPattern(
        lanes_per_wave=32,
        elements_per_load=1,
        lane_to_position=positions,
        element_offsets=[0],
        lds_group_size=16,
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
    "ds_load_tr16_b128":        (ds_load_tr16_b128_pattern, "Transposed 16-bit load, 32 lanes (gfx1250)"),
    "ds_load_tr8_b64":          (ds_load_tr8_b64_pattern, "Transposed 8-bit load, 32 lanes (gfx1250)"),
    "ds_load_2addr_b64":        (lambda kw=8: ds_load_2addr_b64_pattern(kWidth=kw), "Dual-addr 64-bit load, 32 lanes (gfx1250 fp8 A-operand)"),
    "mfma16_kcontig":           (lambda kw=8: mfma_kcontig_pattern(nonKDim=16, kWidth=kw), "MFMA-16x16 non-transposed, 64 lanes (CDNA)"),
    "mfma32_kcontig":           (lambda kw=2: mfma_kcontig_pattern(nonKDim=32, kWidth=kw), "MFMA-32x32 non-transposed, 64 lanes (CDNA)"),
    "wmma_kcontig":             (lambda kw=8: wmma_kcontig_pattern(kWidth=kw), "WMMA non-transposed, 32 lanes (gfx1250)"),
    "wmma_transposed_scalar":   (lambda kw=8: wmma_transposed_scalar_pattern(kWidth=kw), "WMMA transposed scalar, 32 lanes (f32 fallback)"),
}

# =============================================================================
# Valid pattern ↔ element-size combinations
# =============================================================================
#
# The bank conflict math is fully decoupled from data type — any combination
# gives a mathematically valid result.  However, each pattern models a specific
# hardware instruction whose semantics are tied to a particular element width.
# This table captures which combinations correspond to real hardware so that
# the CLI can warn (and optionally reject) unrealistic requests.

_ELEM_BYTES_LABEL = {1: "fp8/i8", 2: "fp16/bf16", 4: "f32/tf32"}

VALID_ELEMENT_BYTES = {
    # Transposed loads — instruction name encodes the element width
    "ds_load_tr16_b128":        {2},       # "tr16" = 16-bit
    "ds_load_tr8_b64":          {1},       # "tr8"  = 8-bit

    # Dual-address 64-bit load: 2×64b = 16 bytes, kWidth=8 → 1 byte/elem
    "ds_load_2addr_b64":        {1},

    # WMMA non-transposed — all operand types (fp8, fp16, f32)
    # f32 uses wmma_f32_16x16x4_f32 (kWidth=2, kBase=2)
    "wmma_kcontig":             {1, 2, 4},

    # WMMA transposed scalar fallback — when K is slow-varying and
    # no ds_load_tr* exists for the element type, compiler emits
    # scalar ds_load_b32 loads. Applicable to all data types.
    "wmma_transposed_scalar":   {1, 2, 4},

    # MFMA non-transposed — all operand types
    "mfma16_kcontig":           {1, 2, 4},
    "mfma32_kcontig":           {2, 4},
}

def _patterns_for_element_bytes(eb: int) -> List[str]:
    """Return pattern names that support the given element_bytes."""
    return [p for p, valid in VALID_ELEMENT_BYTES.items() if eb in valid]


def validate_pattern_element_bytes(pattern_name: str, element_bytes: int,
                                   *, force: bool = False) -> None:
    """Check that pattern + element_bytes is a realistic hardware combination.

    Prints an informative error and exits unless ``force`` is set.
    """
    valid = VALID_ELEMENT_BYTES.get(pattern_name)
    if valid is None or element_bytes in valid:
        return  # OK (or unknown pattern — skip validation)

    dtype_label = _ELEM_BYTES_LABEL.get(element_bytes, f"{element_bytes}B")
    valid_labels = ", ".join(
        f"{_ELEM_BYTES_LABEL.get(v, f'{v}B')} (--element-bytes {v})"
        for v in sorted(valid)
    )
    alt_patterns = _patterns_for_element_bytes(element_bytes)
    alt_labels = ", ".join(alt_patterns) if alt_patterns else "(none)"

    msg = (
        f"Error: --pattern {pattern_name} is not compatible with "
        f"--element-bytes {element_bytes} ({dtype_label})\n"
        f"\n"
        f"  {pattern_name} supports: {valid_labels}\n"
        f"  {dtype_label} is supported by: {alt_labels}\n"
    )

    if force:
        print(f"Warning: {pattern_name} + {dtype_label} is not a real hardware "
              f"combination (--force overrides)\n", file=sys.stderr)
        return

    print(msg, file=sys.stderr)
    print("Use --force to override for hypothetical analysis.", file=sys.stderr)
    sys.exit(1)


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
  # Default (ds_load_tr16_b128, fp16, no padding, row-width=128)
  python3 lds_bank_conflict_analyzer.py

  # Padding layout with all fields
  python3 lds_bank_conflict_analyzer.py --pattern wmma_kcontig \\
    --row-width 64 --element-bytes 2 --padding 16 --num-banks 64

  # Swizzle layout with all fields
  python3 lds_bank_conflict_analyzer.py --pattern wmma_kcontig --layout swizzle \\
    --row-width 64 --element-bytes 2 --swizzle-vec 8 --swizzle-per-phase 1 \\
    --swizzle-max-phase 8 --num-banks 64
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
    parser.add_argument('--force', action='store_true',
                        help='Allow unrealistic pattern + element-bytes combinations')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    validate_pattern_element_bytes(args.pattern, args.element_bytes,
                                   force=args.force)

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
